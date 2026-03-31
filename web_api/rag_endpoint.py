"""
RAG Endpoint — optimized for quality and speed.

Pipeline: multi-query expansion → grouped search → RRF fusion → rerank → MMR → LLM.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cached singletons
# ---------------------------------------------------------------------------
_INIT_LOCK = threading.Lock()
_CACHED_CONFIG: Optional[Dict[str, Any]] = None
_CACHED_DB = None
_CACHED_EMBEDDER = None
_CACHED_LLM = None

_CACHED_RERANKER = None

_CACHED_VARIABLES: List[str] = []
_CACHED_VARIABLES_TS: float = 0.0
_VARIABLES_LOCK = threading.Lock()

_CACHED_INFO: Optional[Dict[str, Any]] = None
_CACHED_INFO_TS: float = 0.0
_INFO_LOCK = threading.Lock()


def _get_reranker():
    """Lazy-load cross-encoder reranker."""
    global _CACHED_RERANKER
    if _CACHED_RERANKER is not None:
        return _CACHED_RERANKER
    with _INIT_LOCK:
        if _CACHED_RERANKER is not None:
            return _CACHED_RERANKER
        from src.climate_embeddings.embeddings.text_models import Reranker
        _CACHED_RERANKER = Reranker()
        return _CACHED_RERANKER


def _get_llm_client():
    """Get LLM client — OpenRouter only."""
    import os
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    from src.llm.openrouter_client import OpenRouterClient
    client = OpenRouterClient()
    logger.info(f"Using OpenRouter: {client.model}")
    return client


def _get_components():
    """Lazy-init heavy components (config, DB, embedder, LLM). Thread-safe."""
    global _CACHED_CONFIG, _CACHED_DB, _CACHED_EMBEDDER, _CACHED_LLM
    if _CACHED_CONFIG and _CACHED_DB and _CACHED_EMBEDDER and _CACHED_LLM:
        return _CACHED_CONFIG, _CACHED_DB, _CACHED_EMBEDDER, _CACHED_LLM

    with _INIT_LOCK:
        if _CACHED_CONFIG and _CACHED_DB and _CACHED_EMBEDDER and _CACHED_LLM:
            return _CACHED_CONFIG, _CACHED_DB, _CACHED_EMBEDDER, _CACHED_LLM

        from src.utils.config_loader import ConfigLoader
        from src.embeddings.database import VectorDatabase
        from src.climate_embeddings.embeddings.text_models import TextEmbedder

        _CACHED_CONFIG = ConfigLoader("config/pipeline_config.yaml").load()
        _CACHED_DB = VectorDatabase(config=_CACHED_CONFIG)
        _CACHED_EMBEDDER = TextEmbedder()
        _CACHED_LLM = _get_llm_client()

        # Warm up embedder
        try:
            _CACHED_EMBEDDER.embed_queries(["warmup"])
        except Exception:
            pass

    return _CACHED_CONFIG, _CACHED_DB, _CACHED_EMBEDDER, _CACHED_LLM


def _get_variable_list(db, force_refresh: bool = False) -> List[str]:
    """Collect distinct variable names via Qdrant scroll; cached 5 min."""
    global _CACHED_VARIABLES, _CACHED_VARIABLES_TS
    now = time.time()
    if not force_refresh and _CACHED_VARIABLES and (now - _CACHED_VARIABLES_TS) < 300:
        return _CACHED_VARIABLES

    with _VARIABLES_LOCK:
        if not force_refresh and _CACHED_VARIABLES and (time.time() - _CACHED_VARIABLES_TS) < 300:
            return _CACHED_VARIABLES

        client = getattr(db, "client", None)
        collection = getattr(db, "collection_name", None)
        seen: set = set()

        if client and collection:
            try:
                offset = None
                for _ in range(50):
                    points, offset = client.scroll(
                        collection_name=collection,
                        limit=500,
                        offset=offset,
                        with_vectors=False,
                        with_payload=True,
                    )
                    for p in points:
                        var = (getattr(p, "payload", None) or {}).get("variable")
                        if var:
                            seen.add(str(var))
                    if offset is None:
                        break
            except Exception as e:
                logger.error(f"Variable list scan error: {e}")

        _CACHED_VARIABLES = sorted(seen)
        _CACHED_VARIABLES_TS = time.time()
        return _CACHED_VARIABLES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VARIABLE_LIST_RE = re.compile(
    r"\b(what|which|list|show|tell|give)\b.*\b(variable|variables|var|data|fields|columns)\b.*\b(available|in|does|are|have|contains|include)\b",
    re.IGNORECASE,
)


def _is_variable_list_question(q: str) -> bool:
    if not q:
        return False
    if _VARIABLE_LIST_RE.search(q):
        return True
    low = q.lower()
    return "variables" in low and ("available" in low or "list" in low)


def _extract_payload(hit) -> tuple:
    """Return (payload_dict, score_float) from a Qdrant hit."""
    if hasattr(hit, "payload"):
        return hit.payload, float(getattr(hit, "score", 0.0))
    if isinstance(hit, dict):
        return hit.get("payload", {}), float(hit.get("score", 0.0))
    return {}, 0.0


def _hit_id(hit) -> Optional[str]:
    if hasattr(hit, "id"):
        return str(hit.id)
    if isinstance(hit, dict):
        return str(hit["id"]) if "id" in hit else None
    return None


def _enforce_diversity(results: list, max_per_source: int = 4,
                       max_per_source_var: int = 2) -> list:
    """Limit results per source_id and per source+variable for diverse coverage.
    Summary chunks (is_dataset_summary=True) bypass the per-variable limit."""
    source_counts: Dict[str, int] = {}
    combo_counts: Dict[str, int] = {}
    out = []
    for hit in sorted(results,
                      key=lambda x: _extract_payload(x)[1],
                      reverse=True):
        meta, _ = _extract_payload(hit)
        sid = meta.get("source_id", "unknown")
        is_summary = meta.get("is_dataset_summary", False)
        var = meta.get("variable", "unknown")
        combo = f"{sid}::{var}"
        if is_summary:
            # Summaries always pass (one per source max)
            if source_counts.get(sid, 0) < max_per_source:
                out.append(hit)
                source_counts[sid] = source_counts.get(sid, 0) + 1
        elif (source_counts.get(sid, 0) < max_per_source
                and combo_counts.get(combo, 0) < max_per_source_var):
            out.append(hit)
            source_counts[sid] = source_counts.get(sid, 0) + 1
            combo_counts[combo] = combo_counts.get(combo, 0) + 1
    return out


def _rrf_fuse(result_lists: List[List], k: int = 60) -> List:
    """Reciprocal Rank Fusion — merge multiple result lists by rank position.

    Each result gets score = sum(1 / (k + rank)) across all lists it appears in.
    Results that appear in multiple lists get boosted.
    """
    scores: Dict[str, float] = {}
    hit_map: Dict[str, Any] = {}

    for results in result_lists:
        for rank, hit in enumerate(results):
            hid = _hit_id(hit)
            if hid is None:
                continue
            scores[hid] = scores.get(hid, 0.0) + 1.0 / (k + rank + 1)
            if hid not in hit_map:
                hit_map[hid] = hit

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [hit_map[hid] for hid in sorted_ids if hid in hit_map]


def _mmr_rerank(results: list, query_vector, embedder,
                lambda_param: float = 0.7, top_k: int = 15) -> list:
    """Maximal Marginal Relevance — balance relevance and diversity.

    Iteratively selects results that are relevant to the query but dissimilar
    to already-selected results. Uses payload similarity (source_id + variable)
    as a proxy for embedding similarity to avoid re-embedding.
    """
    if len(results) <= top_k:
        return results

    # Use payload-based similarity as a fast proxy
    def _payload_sim(hit_a, hit_b) -> float:
        meta_a, _ = _extract_payload(hit_a)
        meta_b, _ = _extract_payload(hit_b)
        sim = 0.0
        if meta_a.get("source_id") == meta_b.get("source_id"):
            sim += 0.5
        if meta_a.get("dataset_name") == meta_b.get("dataset_name"):
            sim += 0.3
        if meta_a.get("variable") == meta_b.get("variable"):
            sim += 0.2
        return sim

    selected = []
    candidates = list(results)

    while len(selected) < top_k and candidates:
        best_score = -1.0
        best_idx = 0

        for i, candidate in enumerate(candidates):
            _, relevance = _extract_payload(candidate)

            # Max similarity to any already-selected result
            max_sim = 0.0
            for sel in selected:
                sim = _payload_sim(candidate, sel)
                if sim > max_sim:
                    max_sim = sim

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(candidates.pop(best_idx))

    return selected


def _generate_query_variants(llm_client, question: str) -> List[str]:
    """Use the LLM to generate search query variants for broader retrieval.

    Returns the original query plus 2-3 variants that capture different
    aspects or phrasings of the question.
    """
    prompt = (
        "Generate 3 alternative search queries for a climate data vector database. "
        "Each query should focus on a DIFFERENT key concept from the original. "
        "If the original mentions multiple topics (e.g. temperature AND drought AND fire), "
        "create one query per topic. Use climate science terminology and dataset names "
        "where applicable (ERA5, MERRA-2, IMERG, GRACE, CAMS, E-OBS, SPEI, etc). "
        "Return ONLY the queries, one per line, no numbering or explanation.\n\n"
        f"Original query: {question}\n\n"
        "Alternative queries:"
    )
    try:
        response = llm_client.generate(
            prompt=prompt,
            temperature=0.1,
            max_tokens=200,
            timeout_s=12,
        )
        variants = [q.strip() for q in response.strip().split("\n") if q.strip()]
        # Filter out empty or too-short variants
        variants = [v for v in variants if len(v) > 10][:3]
        return variants
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RAGMessage(BaseModel):
    role: str
    content: str


class RAGRequest(BaseModel):
    question: str
    top_k: int = 5
    use_llm: bool = True
    use_reranker: bool = True
    temperature: float = 0.1
    source_id: Optional[str] = None
    variable: Optional[str] = None
    timeout: int = 60
    conversation_history: Optional[List[RAGMessage]] = None


class RAGResponse(BaseModel):
    question: str
    answer: str
    chunks: List[Dict[str, Any]]
    references: List[str]
    llm_used: bool
    reranker_used: bool = False
    search_time_ms: float
    llm_time_ms: Optional[float] = None
    conversation_id: Optional[str] = None


# ---------------------------------------------------------------------------
# RAG PIPELINE — single LLM call, 2 parallel Qdrant searches
# ---------------------------------------------------------------------------

async def rag_query(request: RAGRequest) -> RAGResponse:
    """
    Optimized RAG pipeline.

    Flow:
      1. Embed question                    (~100ms)
      2. Semantic search + metadata search  (~200ms, parallel)
      3. Optional reranking                 (~200ms)
      4. Diversity enforcement              (~10ms)
      5. Single LLM call                    (~3-15s)
    """
    question = (request.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    config, db, embedder, llm_client = _get_components()

    # ── 1. EMBED + QUERY EXPANSION ───────────────────────────────────────
    search_start = time.time()
    query_vec = embedder.embed_queries([question])[0]
    embed_ms = (time.time() - search_start) * 1000
    logger.info(f"Embed: {embed_ms:.0f}ms")

    # ── 2. MULTI-QUERY EXPANSION (LLM generates search variants) ──────
    filter_dict = {}
    if request.source_id:
        filter_dict["source_id"] = request.source_id
    if request.variable:
        filter_dict["variable"] = request.variable

    effective_top_k = max(1, min(request.top_k, 20))

    # Generate query variants in parallel with main search
    query_variants: List[str] = []
    try:
        query_variants = await asyncio.wait_for(
            asyncio.to_thread(_generate_query_variants, llm_client, question),
            timeout=12,
        )
        logger.info(f"Query expansion: {len(query_variants)} variants")
    except Exception as e:
        logger.warning(f"Query expansion skipped: {e}")

    # Embed all variants
    variant_vecs = []
    if query_variants:
        try:
            variant_vecs = [
                v.tolist() for v in embedder.embed_queries(query_variants)
            ]
        except Exception as e:
            logger.warning(f"Variant embedding failed: {e}")

    # ── 3. GROUPED SEARCH + MULTI-QUERY RRF FUSION ────────────────────
    reranker_used = False
    all_result_lists: List[List] = []

    # Primary search: grouped by dataset_name for guaranteed diversity
    grouped_results = db.search_grouped(
        query_vector=query_vec.tolist(),
        group_by="dataset_name",
        group_limit=15,
        group_size=2,
        filter_dict=filter_dict or None,
    )
    if grouped_results:
        all_result_lists.append(grouped_results)

    # Also run a standard search (ungrouped) to catch high-relevance hits
    standard_results = db.search(
        query_vector=query_vec.tolist(),
        limit=max(40, effective_top_k * 4),
        filter_dict=filter_dict or None,
    )
    if standard_results:
        all_result_lists.append(standard_results)

    # Search with each query variant for broader recall (fewer results = less RRF influence)
    for vvec in variant_vecs:
        try:
            variant_hits = db.search(
                query_vector=vvec,
                limit=15,
                filter_dict=filter_dict or None,
            )
            if variant_hits:
                all_result_lists.append(variant_hits)
        except Exception as e:
            logger.debug(f"Variant search failed: {e}")

    # RRF fusion across all result lists
    if len(all_result_lists) > 1:
        results = _rrf_fuse(all_result_lists)
    elif all_result_lists:
        results = all_result_lists[0]
    else:
        results = []

    logger.info(f"RRF fused {len(all_result_lists)} lists → {len(results)} results")

    # ── 4. CROSS-ENCODER RERANKING ────────────────────────────────────
    if request.use_reranker and results:
        try:
            reranker = _get_reranker()
            from src.climate_embeddings.schema import generate_human_readable_text as _gen_text

            # Rerank the top candidates
            rerank_candidates = results[:min(len(results), 50)]
            passages = []
            for hit in rerank_candidates:
                payload = hit.payload if hasattr(hit, "payload") else (
                    hit.get("payload", {}) if isinstance(hit, dict) else {}
                )
                passages.append(_gen_text(payload))

            ranked = reranker.rerank(question, passages, top_k=len(rerank_candidates))

            reranked = []
            for entry in ranked:
                # Filter out low-confidence reranker results (noise)
                if entry["score"] < 0.05:
                    continue
                idx = entry["index"]
                hit = rerank_candidates[idx]
                if hasattr(hit, "score"):
                    hit.score = entry["score"]
                elif isinstance(hit, dict):
                    hit["score"] = entry["score"]
                reranked.append(hit)

            # Append any results beyond the rerank window
            reranked_ids = {_hit_id(h) for h in reranked}
            for hit in results[len(rerank_candidates):]:
                if _hit_id(hit) not in reranked_ids:
                    reranked.append(hit)

            results = reranked
            reranker_used = True
        except Exception as e:
            logger.warning(f"Reranker failed ({e}), using RRF order")

    # ── 5. MMR DIVERSITY ──────────────────────────────────────────────
    results = _mmr_rerank(results, query_vec, embedder,
                          lambda_param=0.65, top_k=effective_top_k * 2)

    search_ms = (time.time() - search_start) * 1000
    logger.info(f"Search total: {search_ms:.0f}ms, {len(results)} results")

    # ── 4. FORMAT CHUNKS ──────────────────────────────────────────────────
    from src.climate_embeddings.schema import generate_human_readable_text

    chunks = []
    references: set = set()
    for i, hit in enumerate(results, 1):
        meta, score = _extract_payload(hit)
        source_id = meta.get("source_id", "unknown")
        variable = meta.get("variable", "unknown")
        text = generate_human_readable_text(meta)

        chunks.append({
            "rank": i,
            "score": round(score, 3),
            "source_id": source_id,
            "variable": variable,
            "text": text,
            "metadata": meta,
        })
        references.add(f"{source_id}:{variable}")

    # ── 5. VARIABLE LIST FAST PATH ────────────────────────────────────────
    if _is_variable_list_question(question):
        all_vars = _get_variable_list(db)
        if all_vars:
            answer = f"Available climate variables ({len(all_vars)}):\n" + ", ".join(all_vars)
            return RAGResponse(
                question=question,
                answer=answer,
                chunks=chunks[:10],
                references=sorted(references),
                llm_used=False,
                reranker_used=reranker_used,
                search_time_ms=round(search_ms, 2),
            )

    # ── 6. LLM GENERATION (single call) ──────────────────────────────────
    llm_used = False
    llm_time_ms = None
    answer = ""

    if request.use_llm and chunks:
        try:
            llm_start = time.time()

            # Get variable list (cached, no extra cost)
            all_variables = _get_variable_list(db)

            from web_api.prompt_builder import build_rag_prompt, detect_question_type
            q_type = detect_question_type(question)

            formatted = [
                {"metadata": c["metadata"], "score": c["score"], "text": c["text"]}
                for c in chunks
            ]

            prompt, max_tokens = build_rag_prompt(
                question=question,
                context_chunks=formatted,
                all_variables=all_variables,
                question_type=q_type,
            )

            answer = await asyncio.wait_for(
                asyncio.to_thread(
                    lambda: llm_client.generate(
                        prompt=prompt,
                        temperature=request.temperature,
                        max_tokens=max_tokens,
                        timeout_s=min(request.timeout, 45),
                    )
                ),
                timeout=request.timeout,
            )

            llm_time_ms = (time.time() - llm_start) * 1000
            llm_used = True
            logger.info(f"LLM: {llm_time_ms:.0f}ms, {len(answer)} chars")

        except asyncio.TimeoutError:
            logger.warning(f"LLM timeout after {request.timeout}s")
            answer = f"Found {len(chunks)} relevant results but LLM timed out. Showing search results."
        except Exception as e:
            logger.error(f"LLM error: {e}")
            answer = f"Found {len(chunks)} relevant results. LLM error: {str(e)[:100]}."

    if not answer:
        if not chunks:
            answer = "No relevant climate data found for your query."
        else:
            summary = "\n".join(
                f"- {c['source_id']}: {c['variable']} (score {c['score']})"
                for c in chunks[:5]
            )
            answer = f"Found {len(chunks)} results (no LLM):\n{summary}"

    return RAGResponse(
        question=question,
        answer=answer,
        chunks=chunks,
        references=sorted(references),
        llm_used=llm_used,
        reranker_used=reranker_used,
        search_time_ms=round(search_ms, 2),
        llm_time_ms=round(llm_time_ms, 2) if llm_time_ms else None,
    )


# ---------------------------------------------------------------------------
# COLLECTION INFO (cached, no LLM)
# ---------------------------------------------------------------------------

async def get_collection_info() -> Dict[str, Any]:
    """Return cached collection info: variables, sources, count."""
    global _CACHED_INFO, _CACHED_INFO_TS
    now = time.time()
    if _CACHED_INFO and (now - _CACHED_INFO_TS) < 300:
        return _CACHED_INFO

    with _INFO_LOCK:
        if _CACHED_INFO and (time.time() - _CACHED_INFO_TS) < 300:
            return _CACHED_INFO

        from src.utils.config_loader import ConfigLoader
        from src.embeddings.database import VectorDatabase

        config = ConfigLoader("config/pipeline_config.yaml").load()
        db = VectorDatabase(config=config)

        variables: set = set()
        sources: set = set()
        count = 0

        client = getattr(db, "client", None)
        collection = getattr(db, "collection_name", None)

        if client and collection:
            try:
                info = client.get_collection(collection)
                count = info.points_count

                offset = None
                for _ in range(20):
                    points, offset = client.scroll(
                        collection_name=collection,
                        limit=500,
                        offset=offset,
                        with_vectors=False,
                        with_payload={"include": ["variable", "source_id", "dataset_name"]},
                    )
                    for p in points:
                        pl = getattr(p, "payload", None) or {}
                        if pl.get("variable"):
                            variables.add(pl["variable"])
                        src = pl.get("source_id") or pl.get("dataset_name")
                        if src:
                            sources.add(src)
                    if offset is None:
                        break
            except Exception as e:
                logger.error(f"get_collection_info error: {e}")

        _CACHED_INFO = {
            "total_embeddings": count,
            "variables": sorted(variables),
            "sources": sorted(sources),
            "collection_name": collection or "climate_data",
        }
        _CACHED_INFO_TS = time.time()
        return _CACHED_INFO


# ---------------------------------------------------------------------------
# SIMPLE SEARCH (no LLM, for debugging)
# ---------------------------------------------------------------------------

async def simple_search(query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
    """Fast vector search without LLM."""
    try:
        config, db, embedder, _ = _get_components()
        query_vec = embedder.embed_queries([query])[0]
        results = db.search(
            query_vector=query_vec.tolist(),
            limit=top_k,
            filter_dict=filters,
        )

        chunks = []
        for hit in results:
            meta, score = _extract_payload(hit)
            chunks.append({
                "score": round(score, 3),
                "source_id": meta.get("source_id"),
                "variable": meta.get("variable"),
                "metadata": meta,
            })
        return chunks

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
