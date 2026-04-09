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


def _enforce_diversity(results: list, max_per_source: int = 3,
                       max_per_source_var: int = 2,
                       min_sources: int = 4) -> list:
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

    # Guarantee minimum unique sources
    unique_sources = len(set(
        _extract_payload(h)[0].get("source_id", "?") for h in out
    ))
    if unique_sources < min_sources:
        seen_sources = {_extract_payload(h)[0].get("source_id", "?") for h in out}
        for hit in sorted(results,
                          key=lambda x: _extract_payload(x)[1],
                          reverse=True):
            meta, _ = _extract_payload(hit)
            sid = meta.get("source_id", "unknown")
            if sid not in seen_sources:
                out.append(hit)
                seen_sources.add(sid)
                if len(seen_sources) >= min_sources:
                    break

    return out


def _rrf_fuse(result_lists: List[List], k: int = 60,
              weights: Optional[List[float]] = None) -> List:
    """Weighted Reciprocal Rank Fusion — merge multiple result lists by rank.

    Each result gets score = sum(weight * 1 / (k + rank)) across all lists.
    Primary query gets higher weight than expansion variants.
    Adaptive threshold: drops results below 40% of top score.
    """
    scores: Dict[str, float] = {}
    hit_map: Dict[str, Any] = {}

    if weights is None:
        weights = [1.0] * len(result_lists)

    for results, w in zip(result_lists, weights):
        for rank, hit in enumerate(results):
            hid = _hit_id(hit)
            if hid is None:
                continue
            scores[hid] = scores.get(hid, 0.0) + w / (k + rank + 1)
            if hid not in hit_map:
                hit_map[hid] = hit

    # Adaptive threshold: drop noise below 40% of top score
    if scores:
        max_score = max(scores.values())
        threshold = max_score * 0.4
        scores = {k: v for k, v in scores.items() if v >= threshold}

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


def _extract_topic_keywords(question: str) -> List[str]:
    """Extract distinct topic keywords from the question for per-topic sub-searches.

    Uses a curated list of climate domain terms to identify which distinct topics
    the query covers. Returns a list of topic keywords.
    """
    q_lower = question.lower()
    # Climate domain topics — each tuple is (canonical_term, [synonyms])
    TOPIC_MAP = [
        ("drought", ["drought", "dry spell", "aridity", "spei", "pdsi", "spi "]),
        ("temperature", ["temperature", "warming", "heat", "thermal", "t2m"]),
        ("precipitation", ["precipitation", "rainfall", "rain", "monsoon", "imerg"]),
        ("wildfire", ["wildfire", "fire", "burned area", "fire emission", "gfas"]),
        ("flood", ["flood", "inundation", "submerg"]),
        ("sea level", ["sea level", "sea-level", "tide", "grace"]),
        ("ice", ["ice sheet", "ice loss", "glacier", "cryosphere", "ice mass"]),
        ("aerosol", ["aerosol", "dust", "particulate", "pm2.5", "pm10", "saharan"]),
        ("co2", ["co2", "carbon dioxide", "greenhouse gas", "ghg", "ppm"]),
        ("soil moisture", ["soil moisture", "soil water"]),
        ("water storage", ["water storage", "terrestrial water", "grace-fo"]),
        ("wind", ["wind speed", "wind gust", "cyclone", "hurricane", "storm"]),
        ("marine", ["marine heatwave", "sst", "sea surface temperature", "ocean heat"]),
        ("emissions", ["emission", "cams"]),
    ]
    found_topics = []
    for canonical, synonyms in TOPIC_MAP:
        if any(syn in q_lower for syn in synonyms):
            found_topics.append(canonical)
    return found_topics


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
        # Use fast model (Sonnet) for query expansion — much faster than Opus
        gen_fn = getattr(llm_client, "generate_fast", None) or llm_client.generate
        response = gen_fn(
            prompt=prompt,
            temperature=0.0,
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


def _generate_topic_queries(topics: List[str], original_question: str) -> List[str]:
    """Generate one focused search query per extracted topic.

    These are simple, keyword-dense queries that target each topic individually
    in the vector DB. They complement the LLM-generated variants.
    """
    if len(topics) <= 1:
        return []

    # Dataset name hints per topic for better retrieval
    DATASET_HINTS = {
        "drought": "SPEI drought index climate data",
        "temperature": "ERA5 temperature climate reanalysis data",
        "precipitation": "IMERG GPM precipitation rainfall satellite data",
        "wildfire": "CAMS GFAS wildfire fire emissions burned area",
        "flood": "precipitation flood inundation soil moisture water",
        "sea level": "GRACE sea level rise ice sheet mass loss",
        "ice": "GRACE ice sheet mass loss cryosphere glacier",
        "aerosol": "CAMS MERRA-2 aerosol dust particulate reanalysis",
        "co2": "CAMS CO2 carbon dioxide greenhouse gas atmospheric concentration",
        "soil moisture": "ERA5 Land soil moisture volumetric",
        "water storage": "GRACE-FO terrestrial water storage anomaly",
        "wind": "ERA5 wind speed gust storm cyclone",
        "marine": "sea surface temperature SST marine heatwave ocean heat",
        "emissions": "CAMS emissions wildfire carbon atmospheric composition",
    }
    topic_queries = []
    for topic in topics:
        tq = DATASET_HINTS.get(topic, f"{topic} climate data")
        topic_queries.append(tq)
    return topic_queries


# ---------------------------------------------------------------------------
# CRAG: Corrective RAG — grade retrieval quality, retry if insufficient
# ---------------------------------------------------------------------------

def _grade_retrieval(llm_client, question: str, chunks: List[Dict]) -> str:
    """Fast LLM check: do retrieved chunks actually answer the question?"""
    chunk_summaries = "\n".join(
        f"- [{i+1}] {c.get('source_id','?')}/{c.get('variable','?')}: "
        f"{c.get('text','')[:120]}"
        for i, c in enumerate(chunks[:5])
    )
    prompt = (
        f"Question: {question}\n\n"
        f"Retrieved data:\n{chunk_summaries}\n\n"
        "Do these results contain information relevant to answering the question? "
        "Reply ONLY 'sufficient' or 'insufficient'."
    )
    try:
        gen_fn = getattr(llm_client, "generate_fast", None) or llm_client.generate
        result = gen_fn(prompt=prompt, temperature=0.0, max_tokens=10, timeout_s=8)
        grade = "insufficient" if "insufficient" in result.lower() else "sufficient"
        logger.info(f"CRAG grade: {grade}")
        return grade
    except Exception as e:
        logger.warning(f"CRAG grading failed: {e}")
        return "sufficient"  # Don't block on failure


def _rewrite_query(llm_client, question: str) -> Optional[str]:
    """Rewrite query to improve retrieval on retry."""
    prompt = (
        f"The following search query did not retrieve good results from a climate data database:\n"
        f"Query: {question}\n\n"
        f"Rewrite as a better search query. Use specific climate variable names, "
        f"dataset names (ERA5, E-OBS, IMERG, GRACE, CAMS, MERRA-2, SPEI, etc), "
        f"and technical terms. Return ONLY the rewritten query, nothing else."
    )
    try:
        gen_fn = getattr(llm_client, "generate_fast", None) or llm_client.generate
        result = gen_fn(prompt=prompt, temperature=0.1, max_tokens=100, timeout_s=8)
        rewritten = result.strip().strip('"').strip("'")
        if len(rewritten) > 10 and rewritten.lower() != question.lower():
            logger.info(f"CRAG rewrite: {rewritten[:80]}")
            return rewritten
    except Exception as e:
        logger.warning(f"CRAG rewrite failed: {e}")
    return None


# ---------------------------------------------------------------------------
# LLM-based query analysis for hybrid search
# ---------------------------------------------------------------------------

def _analyze_query(llm_client, question: str) -> Dict[str, Any]:
    """Use fast LLM to extract search metadata from the query.

    Returns dict with optional keys: datasets, variables, hazards.
    This replaces hardcoded keyword maps — the LLM understands context.
    """
    prompt = (
        "Analyze this climate data query and extract search filters.\n\n"
        f"Query: {question}\n\n"
        "Return ONLY a JSON object with these optional fields:\n"
        '- "datasets": list of likely dataset names (e.g. ["ERA5", "IMERG", "GRACE"])\n'
        '- "variables": list of likely variable names (e.g. ["t2m", "tp", "sst"])\n'
        '- "hazards": list of hazard categories (e.g. ["Extreme heat", "Drought"])\n'
        "Only include fields you are confident about. Be concise."
    )
    try:
        gen_fn = getattr(llm_client, "generate_fast", None) or llm_client.generate
        result = gen_fn(prompt=prompt, temperature=0.0, max_tokens=150, timeout_s=8)
        import json as _json
        # Extract JSON from response (handle markdown code blocks)
        text = result.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = _json.loads(text.strip())
        logger.info(f"Query analysis: {parsed}")
        return parsed
    except Exception as e:
        logger.debug(f"Query analysis failed: {e}")
        return {}


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
    timings: Dict[str, float] = {}

    # ── 1. EMBED + QUERY EXPANSION ───────────────────────────────────────
    search_start = time.time()
    t0 = time.time()
    query_vec = embedder.embed_queries([question])[0]
    timings["embed"] = (time.time() - t0) * 1000
    logger.info(f"⏱ Embed: {timings['embed']:.0f}ms")

    # ── 2. MULTI-QUERY EXPANSION (LLM generates search variants) ──────
    filter_dict = {}
    if request.source_id:
        filter_dict["source_id"] = request.source_id
    if request.variable:
        filter_dict["variable"] = request.variable

    effective_top_k = max(1, min(request.top_k, 20))

    # Generate query variants in parallel with main search
    query_variants: List[str] = []
    t0 = time.time()
    try:
        query_variants = await asyncio.wait_for(
            asyncio.to_thread(_generate_query_variants, llm_client, question),
            timeout=12,
        )
        logger.info(f"Query expansion: {len(query_variants)} variants")
    except Exception as e:
        logger.warning(f"Query expansion skipped: {e}")
    timings["query_expansion"] = (time.time() - t0) * 1000
    logger.info(f"⏱ Query expansion: {timings['query_expansion']:.0f}ms")

    # Generate per-topic focused queries for multi-topic questions
    topics = _extract_topic_keywords(question)
    topic_queries = _generate_topic_queries(topics, question)
    if topic_queries:
        logger.info(f"Multi-topic detected: {topics} → {len(topic_queries)} topic queries")

    # Embed all variants + topic queries
    all_extra_queries = query_variants + topic_queries
    variant_vecs = []
    topic_vec_start = len(query_variants)  # index where topic vecs start
    if all_extra_queries:
        t0 = time.time()
        try:
            all_vecs = embedder.embed_queries(all_extra_queries)
            variant_vecs = [v.tolist() for v in all_vecs]
        except Exception as e:
            logger.warning(f"Variant embedding failed: {e}")
        timings["variant_embed"] = (time.time() - t0) * 1000
        logger.info(f"⏱ Variant embed: {timings['variant_embed']:.0f}ms ({len(query_variants)} variants + {len(topic_queries)} topics)")

    # ── 3. BATCH SEARCH + RRF FUSION (single gRPC call) ────────────────
    reranker_used = False
    t0 = time.time()

    # Build all search queries for batch execution
    batch_queries = [
        # Primary: broad search for grouping + diversity
        {"vector": query_vec.tolist(), "limit": 60, "filter_dict": filter_dict or None},
    ]
    # Add variant searches — each variant captures a different aspect
    for vvec in variant_vecs:
        batch_queries.append({"vector": vvec, "limit": 20, "filter_dict": filter_dict or None})

    # Execute all searches in ONE gRPC call
    batch_results = await asyncio.to_thread(db.search_batch, batch_queries)

    # Client-side grouping on primary results for diversity
    all_result_lists: List[List] = []
    if batch_results and batch_results[0]:
        primary = batch_results[0]
        all_result_lists.append(primary)

        # Group by dataset_name from primary results
        groups: Dict[str, List] = {}
        for hit in primary:
            meta, _ = _extract_payload(hit)
            ds = meta.get("dataset_name", "unknown")
            if ds not in groups:
                groups[ds] = []
            if len(groups[ds]) < 2:
                groups[ds].append(hit)
        grouped = []
        for hits in sorted(groups.values(),
                           key=lambda g: max(_extract_payload(h)[1] for h in g),
                           reverse=True)[:15]:
            grouped.extend(hits)
        if grouped:
            all_result_lists.append(grouped)

    # Add variant results
    for vr in batch_results[1:]:
        if vr:
            all_result_lists.append(vr)

    # Weighted RRF fusion — primary + grouped = 1.0, LLM variants = 0.6, topic queries = 0.9
    if len(all_result_lists) > 1:
        weights = []
        # all_result_lists: [primary, grouped, variant_0..N, topic_0..M]
        # variant searches start at index 2, topic searches start at 2 + len(query_variants)
        n_variant_searches = len(query_variants) if query_variants else 0
        for i in range(len(all_result_lists)):
            if i <= 1:
                weights.append(1.0)  # primary + grouped
            elif i < 2 + n_variant_searches:
                weights.append(0.6)  # LLM-generated variants
            else:
                weights.append(0.9)  # topic-focused queries (high weight — targeted)
        results = _rrf_fuse(all_result_lists, weights=weights)
    elif all_result_lists:
        results = all_result_lists[0]
    else:
        results = []

    timings["search_total"] = (time.time() - t0) * 1000
    logger.info(f"⏱ Search + RRF: {timings['search_total']:.0f}ms, {len(all_result_lists)} lists → {len(results)} results")

    # ── 4. BOOST DATA CHUNKS FOR MULTI-TOPIC QUERIES ONLY ───────────
    # For cross-domain queries, metadata-only entries dominate top-5 and
    # push out real data chunks. Only apply boost for multi-topic queries.
    if len(topics) >= 2:
        data_chunks = []
        meta_only = []
        for hit in results:
            payload, _ = _extract_payload(hit)
            if payload.get("is_metadata_only") or payload.get("is_dataset_summary"):
                meta_only.append(hit)
            else:
                data_chunks.append(hit)
        results = data_chunks + meta_only
        logger.info(f"Data boost: {len(data_chunks)} data + {len(meta_only)} metadata-only")

    # ── 5. DIVERSITY ENFORCEMENT + MMR ──────────────────────────────
    results = _enforce_diversity(results)

    # Guarantee minimum 3 unique sources: if we have < 3, relax filters
    # and pull from further down the ranked list
    _unique_srcs = set()
    for h in results:
        meta, _ = _extract_payload(h)
        _unique_srcs.add(meta.get("source_id") or meta.get("dataset_name", "unknown"))
    if len(_unique_srcs) < 3 and len(all_result_lists) > 0:
        # Gather all candidate hits across all lists
        _all_candidates = []
        _seen_ids = {_hit_id(h) for h in results}
        for rl in all_result_lists:
            for h in rl:
                hid = _hit_id(h)
                if hid and hid not in _seen_ids:
                    _all_candidates.append(h)
                    _seen_ids.add(hid)
        # Sort by score descending, pick hits from new sources
        _all_candidates.sort(key=lambda x: _extract_payload(x)[1], reverse=True)
        for h in _all_candidates:
            meta, _ = _extract_payload(h)
            src = meta.get("source_id") or meta.get("dataset_name", "unknown")
            if src not in _unique_srcs:
                results.append(h)
                _unique_srcs.add(src)
                if len(_unique_srcs) >= 3:
                    break

    results = _mmr_rerank(results, query_vec, embedder,
                          lambda_param=0.55, top_k=effective_top_k * 2)

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

    # ── 5b. CRAG: Grade retrieval quality, retry if insufficient ─────────
    # CRAG: only when retrieval looks very weak (< 3 chunks or very low scores)
    t0 = time.time()
    top_score = chunks[0]["score"] if chunks else 0
    crag_needed = len(chunks) < 3 or top_score < 0.2
    if request.use_llm and crag_needed and chunks:
        try:
            grade = await asyncio.wait_for(
                asyncio.to_thread(_grade_retrieval, llm_client, question, chunks),
                timeout=10,
            )
            if grade == "insufficient":
                # Rewrite query and do one more search pass
                rewritten = await asyncio.wait_for(
                    asyncio.to_thread(_rewrite_query, llm_client, question),
                    timeout=10,
                )
                if rewritten:
                    retry_vec = embedder.embed_queries([rewritten])[0]
                    retry_results = db.search(
                        query_vector=retry_vec.tolist(),
                        limit=40,
                        filter_dict=filter_dict or None,
                    )
                    if retry_results:
                        # Rerank retry results
                        if request.use_reranker:
                            try:
                                reranker = _get_reranker()
                                from src.climate_embeddings.schema import generate_human_readable_text as _gen_text2
                                retry_passages = [_gen_text2(
                                    h.payload if hasattr(h, "payload") else h.get("payload", {})
                                ) for h in retry_results[:30]]
                                retry_ranked = reranker.rerank(rewritten, retry_passages, top_k=len(retry_passages))
                                for entry in retry_ranked:
                                    if entry["score"] >= 0.10:
                                        hit = retry_results[entry["index"]]
                                        if hasattr(hit, "score"):
                                            hit.score = entry["score"]
                                        # Add to chunks if not duplicate
                                        meta, score = _extract_payload(hit)
                                        sid = meta.get("source_id", "unknown")
                                        var = meta.get("variable", "unknown")
                                        ref_key = f"{sid}:{var}"
                                        if ref_key not in references:
                                            chunks.append({
                                                "rank": len(chunks) + 1,
                                                "score": round(score, 3),
                                                "source_id": sid,
                                                "variable": var,
                                                "text": generate_human_readable_text(meta),
                                                "metadata": meta,
                                            })
                                            references.add(ref_key)
                            except Exception as e:
                                logger.warning(f"CRAG retry rerank failed: {e}")

                        crag_ms = (time.time() - search_start) * 1000
                        logger.info(f"CRAG retry added chunks, total: {len(chunks)}, {crag_ms:.0f}ms")
        except Exception as e:
            logger.warning(f"CRAG step failed: {e}")

    timings["crag"] = (time.time() - t0) * 1000
    logger.info(f"⏱ CRAG: {timings['crag']:.0f}ms")

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
                        timeout_s=min(request.timeout, 180),
                    )
                ),
                timeout=request.timeout,
            )

            llm_time_ms = (time.time() - llm_start) * 1000
            timings["llm"] = llm_time_ms
            llm_used = True
            logger.info(f"⏱ LLM: {llm_time_ms:.0f}ms, {len(answer)} chars")

        except asyncio.TimeoutError:
            logger.warning(f"LLM timeout after {request.timeout}s")
            answer = f"Found {len(chunks)} relevant results but LLM timed out. Showing search results."
        except Exception as e:
            logger.warning(f"LLM error (attempt 1): {e}, retrying...")
            try:
                import time as _time
                _time.sleep(1)
                answer = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda: llm_client.generate(
                            prompt=prompt,
                            temperature=request.temperature,
                            max_tokens=max_tokens,
                            timeout_s=min(request.timeout, 180),
                        )
                    ),
                    timeout=request.timeout,
                )
                llm_time_ms = (time.time() - llm_start) * 1000
                timings["llm"] = llm_time_ms
                llm_used = True
                logger.info(f"⏱ LLM retry OK: {llm_time_ms:.0f}ms")
            except Exception as e2:
                logger.error(f"LLM error (attempt 2): {e2}")
                answer = f"Found {len(chunks)} relevant results. LLM error: {str(e2)[:100]}."

    if not answer:
        if not chunks:
            answer = "No relevant climate data found for your query."
        else:
            summary = "\n".join(
                f"- {c['source_id']}: {c['variable']} (score {c['score']})"
                for c in chunks[:5]
            )
            answer = f"Found {len(chunks)} results (no LLM):\n{summary}"

    total_ms = (time.time() - search_start) * 1000
    timings["total"] = total_ms
    timing_summary = " | ".join(f"{k}={v:.0f}ms" for k, v in timings.items())
    logger.info(f"⏱ PIPELINE TOTAL: {timing_summary}")

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

                # Use facets for accurate counts instead of slow scroll
                try:
                    var_facets = client.facet(
                        collection_name=collection, key="variable", limit=500,
                    )
                    variables = {h.value for h in var_facets.hits}
                except Exception:
                    pass

                try:
                    src_facets = client.facet(
                        collection_name=collection, key="source_id", limit=500,
                    )
                    sources = {h.value for h in src_facets.hits}
                except Exception:
                    pass
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
