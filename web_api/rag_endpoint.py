"""
Optimized RAG Endpoint - Best Practices
Fast, reliable RAG with proper error handling and timeouts
"""
from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import HTTPException
import asyncio
import threading
import time
import re

logger = logging.getLogger(__name__)


_INIT_LOCK = threading.Lock()
_CACHED_CONFIG: Optional[Dict[str, Any]] = None
_CACHED_DB = None
_CACHED_EMBEDDER = None
_CACHED_LLM = None
_CACHED_VARIABLES: List[str] = []
_CACHED_VARIABLES_TS: float = 0.0
_VARIABLES_LOCK = threading.Lock()


def _get_llm_client():
    """Get LLM client - OpenRouter only."""
    import os
    
    # OpenRouter only - no fallback to slow Ollama
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not set! LLM will not work.")
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    
    try:
        from src.llm.openrouter_client import OpenRouterClient
        client = OpenRouterClient()
        logger.info(f"Using OpenRouter API with model: {client.model}")
        return client, "openrouter"
    except Exception as e:
        logger.error(f"OpenRouter client init failed: {e}")
        raise ValueError(f"Failed to initialize OpenRouter: {e}")


def _get_components():
    """Lazy init + cache heavy components (config, DB client, embedder, LLM)."""
    global _CACHED_CONFIG, _CACHED_DB, _CACHED_EMBEDDER, _CACHED_LLM
    if _CACHED_CONFIG is not None and _CACHED_DB is not None and _CACHED_EMBEDDER is not None and _CACHED_LLM is not None:
        return _CACHED_CONFIG, _CACHED_DB, _CACHED_EMBEDDER, _CACHED_LLM

    with _INIT_LOCK:
        if _CACHED_CONFIG is not None and _CACHED_DB is not None and _CACHED_EMBEDDER is not None and _CACHED_LLM is not None:
            return _CACHED_CONFIG, _CACHED_DB, _CACHED_EMBEDDER, _CACHED_LLM

        from src.utils.config_loader import ConfigLoader
        from src.embeddings.database import VectorDatabase
        from src.climate_embeddings.embeddings.text_models import TextEmbedder

        config_loader = ConfigLoader("config/pipeline_config.yaml")
        _CACHED_CONFIG = config_loader.load()
        _CACHED_DB = VectorDatabase(config=_CACHED_CONFIG)
        _CACHED_EMBEDDER = TextEmbedder()
        _CACHED_LLM, llm_type = _get_llm_client()
        
        # Warm up embedder with a dummy query
        try:
            _CACHED_EMBEDDER.embed_queries(["warmup"])
            logger.info("Embedder warmed up")
        except Exception as e:
            logger.warning(f"Embedder warmup failed: {e}")

    return _CACHED_CONFIG, _CACHED_DB, _CACHED_EMBEDDER, _CACHED_LLM


def _get_variable_list(db, force_refresh: bool = False, max_vars: int = 1000) -> List[str]:
    """Collect distinct variable names quickly using Qdrant scroll; cache results.
    
    This function scrolls through ALL points in the collection to get a complete
    list of unique variables, not just from search results.
    """
    global _CACHED_VARIABLES, _CACHED_VARIABLES_TS
    now = time.time()
    if not force_refresh and _CACHED_VARIABLES and (now - _CACHED_VARIABLES_TS) < 300:
        return _CACHED_VARIABLES

    with _VARIABLES_LOCK:
        if not force_refresh and _CACHED_VARIABLES and (time.time() - _CACHED_VARIABLES_TS) < 300:
            return _CACHED_VARIABLES

        client_attr = getattr(db, "client", None)
        collection = getattr(db, "collection_name", None)
        seen = set()

        if client_attr is not None and collection:
            try:
                offset = None
                rounds = 0
                # Increase rounds to scan more of the database
                while rounds < 50 and len(seen) < max_vars:
                    points, offset = client_attr.scroll(
                        collection_name=collection,
                        limit=500,
                        offset=offset,
                        with_vectors=False,
                        with_payload=True,
                    )
                    rounds += 1
                    for p in points:
                        payload = getattr(p, "payload", None) or {}
                        var = payload.get("variable")
                        if var:
                            seen.add(str(var))
                    if offset is None:
                        break
                logger.info(f"Scanned {rounds} rounds, found {len(seen)} unique variables")
            except Exception as e:
                logger.error(f"Error getting variable list: {e}")

        _CACHED_VARIABLES = sorted(seen)[:max_vars]
        _CACHED_VARIABLES_TS = time.time()
        logger.info(f"Cached {len(_CACHED_VARIABLES)} variables")
        return _CACHED_VARIABLES


_VARIABLE_LIST_RE = re.compile(
    r"\b(what|which|list|show|tell|give)\b.*\b(variable|variables|var|data|fields|columns)\b.*\b(available|in|does|are|have|contains|include)\b",
    re.IGNORECASE,
)


def _is_variable_list_question(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    if _VARIABLE_LIST_RE.search(q):
        return True
    # Common short variants
    lowered = q.lower()
    return "variables" in lowered and ("available" in lowered or "list" in lowered)


def _format_hit_summary(meta: Dict[str, Any], score: float) -> str:
    """Create a compact, stable summary for LLM context (avoid huge prompts)."""
    source_id = meta.get("source_id", "unknown")
    dataset = meta.get("dataset_name") or source_id
    variable = meta.get("variable", "unknown")
    unit = meta.get("unit") or meta.get("units") or ""
    time_start = meta.get("time_start")
    time_end = meta.get("time_end")

    parts = [f"source={dataset}", f"var={variable}", f"score={score:.3f}"]
    if unit:
        parts.append(f"unit={unit}")
    if time_start:
        if time_end and time_end != time_start:
            parts.append(f"time={time_start}..{time_end}")
        else:
            parts.append(f"time={time_start}")

    # Only include a tiny bit of stats if present
    for key in ("stats_mean", "stats_min", "stats_max"):
        if key in meta and meta[key] is not None:
            try:
                parts.append(f"{key.replace('stats_', '')}={float(meta[key]):.3g}")
            except Exception:
                pass

    return ", ".join(parts)


def _default_max_tokens(question: str) -> int:
    # Keep answers SHORT on slow CPU servers
    q = (question or "").strip()
    if _is_variable_list_question(q):
        return 60
    if len(q) <= 50:
        return 80
    if len(q) <= 100:
        return 100
    return 120

# ====================================================================================
# MODELS
# ====================================================================================

class RAGMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class RAGRequest(BaseModel):
    question: str
    top_k: int = 5
    use_llm: bool = True
    temperature: float = 0.7
    source_id: Optional[str] = None
    variable: Optional[str] = None
    timeout: int = 180  # LLM timeout in seconds (increased for slow servers)
    conversation_history: Optional[List[RAGMessage]] = None  # For multi-turn conversations

class RAGResponse(BaseModel):
    question: str
    answer: str
    chunks: List[Dict[str, Any]]
    references: List[str]
    llm_used: bool
    search_time_ms: float
    llm_time_ms: Optional[float] = None
    conversation_id: Optional[str] = None  # For tracking conversations

# ====================================================================================
# RAG PIPELINE
# ====================================================================================

async def rag_query(request: RAGRequest) -> RAGResponse:
    """
    Optimized RAG pipeline with proper timeout handling.
    """
    try:
        question_text = (request.question or "").strip()
        if not question_text:
            raise HTTPException(status_code=400, detail="Question is required")

        _config, db, embedder, llm_client = _get_components()

        # Fast path: variable listing without embeddings/LLM
        if _is_variable_list_question(question_text) and request.use_llm is False:
            vars_cached = _get_variable_list(db)
            if vars_cached:
                answer = "Available climate variables (cached):\n" + ", ".join(vars_cached[:80])
                return RAGResponse(
                    question=question_text,
                    answer=answer,
                    chunks=[],
                    references=[],
                    llm_used=False,
                    search_time_ms=0,
                    llm_time_ms=None,
                )
        
        # 1. VECTOR SEARCH (fast - should take <500ms)
        search_start = time.time()
        
        # Embed query
        embed_start = time.time()
        query_vec = embedder.embed_queries([question_text])[0]
        embed_time = (time.time() - embed_start) * 1000
        logger.info(f"Embedding took {embed_time:.0f}ms")
        
        # Build filter
        filter_dict = {}
        if request.source_id:
            filter_dict["source_id"] = request.source_id
        if request.variable:
            filter_dict["variable"] = request.variable
        
        # Search Qdrant
        effective_top_k = max(1, min(request.top_k, 8))
        results = db.search(
            query_vector=query_vec.tolist(),
            limit=effective_top_k,
            filter_dict=filter_dict if filter_dict else None
        )
        
        search_time = (time.time() - search_start) * 1000  # Convert to ms
        logger.info(f"Total search (embed+qdrant) took {search_time:.0f}ms")
        
        # 2. FORMAT CONTEXT
        chunks = []
        references = set()
        # Keep LLM context compact; UI can still show full chunk text.
        llm_context_lines: List[str] = []
        
        for i, hit in enumerate(results, 1):
            # Extract payload
            if hasattr(hit, 'payload'):
                meta = hit.payload
                score = hit.score
            else:
                meta = hit.get('payload', {})
                score = hit.get('score', 0.0)
            
            # Build context entry
            source_id = meta.get('source_id', 'unknown')
            variable = meta.get('variable', 'unknown')
            
            # Generate human-readable text from metadata
            from src.climate_embeddings.schema import generate_human_readable_text
            text = generate_human_readable_text(meta)
            
            chunks.append({
                "rank": i,
                "score": round(score, 3),
                "source_id": source_id,
                "variable": variable,
                "text": text,
                "metadata": meta
            })
            
            references.add(f"{source_id}:{variable}")
            llm_context_lines.append(f"[{i}] {_format_hit_summary(meta, float(score))}")

        # Fast path: variable listing questions - ALWAYS get ALL variables from database
        if _is_variable_list_question(question_text):
            # Get ALL variables from database, not just from search results
            all_variables = _get_variable_list(db, force_refresh=False)
            if not all_variables:
                # Fallback: try to get from collection info
                try:
                    collection_info = await get_collection_info()
                    all_variables = collection_info.get("variables", [])
                except Exception as e:
                    logger.warning(f"Failed to get collection info: {e}")
                    # Last resort: use variables from chunks
                    all_variables = sorted({c.get("variable") for c in chunks if c.get("variable")})
            
            # Get sources info (reuse collection_info if we already have it, otherwise fetch)
            sources_text = ""
            try:
                collection_info = await get_collection_info()
                sources = collection_info.get("sources", [])
                if sources:
                    sources_text = f" from {len(sources)} source(s): {', '.join(sources)}"
            except Exception as e:
                logger.warning(f"Failed to get sources info: {e}")
            
            answer = f"Available climate variables{sources_text}:\n" + ", ".join(all_variables)
            return RAGResponse(
                question=question_text,
                answer=answer,
                chunks=chunks[:10],  # Include some chunks for context
                references=sorted(list(references)),
                llm_used=False,
                search_time_ms=round(search_time, 2),
                llm_time_ms=None,
            )

        context_text = "\n".join(llm_context_lines[:5])
        if len(context_text) > 4000:
            context_text = context_text[:4000] + "\n..."
        
        # 3. LLM GENERATION (with timeout and conversation history)
        llm_used = False
        llm_time = None
        answer = ""
        
        if request.use_llm and chunks:
            try:
                llm_start = time.time()
                
                # For variable list questions, include ALL variables in context
                if _is_variable_list_question(question_text):
                    all_variables = _get_variable_list(db, force_refresh=False)
                    try:
                        collection_info = await get_collection_info()
                        sources = collection_info.get("sources", [])
                        sources_text = f" from {len(sources)} source(s): {', '.join(sources)}" if sources else ""
                    except:
                        sources_text = ""
                    
                    # Build comprehensive prompt with ALL variables
                    prompt = f"""You are a climate data assistant. Answer the question using the complete dataset information.

AVAILABLE VARIABLES{sources_text.upper()}: {', '.join(all_variables)}

SEARCH RESULTS (sample data):
{context_text}

Q: {question_text}
A: List ALL available variables from the dataset above."""
                    max_tokens = min(200, len(all_variables) * 3)  # Enough tokens for all variables
                else:
                    # Build minimal prompt - every token counts on slow CPU
                    prompt = f"""Answer in 1-2 sentences using ONLY this data:
{context_text}

Q: {question_text}
A:"""
                    max_tokens = _default_max_tokens(question_text)
                
                # Async timeout wrapper
                answer = await asyncio.wait_for(
                    asyncio.to_thread(
                        lambda: llm_client.generate(
                            prompt=prompt,
                            temperature=0.3,  # Lower temp = faster, more focused
                            max_tokens=max_tokens,
                            timeout_s=min(request.timeout, 60),
                        )
                    ),
                    timeout=request.timeout
                )
                
                llm_time = (time.time() - llm_start) * 1000
                llm_used = True
                
            except asyncio.TimeoutError:
                logger.warning(f"LLM timeout after {request.timeout}s")
                answer = f"⏱️ Found {len(chunks)} relevant results but LLM timed out. Try: 1) Simpler question, 2) Reduce top_k, or 3) Use search-only mode."
            except Exception as e:
                logger.error(f"LLM error: {e}")
                answer = f"⚠️ Found {len(chunks)} relevant results. LLM error: {str(e)[:100]}. Showing search results instead."
        
        # Fallback answer if no LLM
        if not answer:
            if not chunks:
                answer = "No relevant climate data found for your query."
            else:
                answer = f"Found {len(chunks)} relevant climate data chunks:\n{context_text[:500]}"
        
        return RAGResponse(
            question=request.question,
            answer=answer,
            chunks=chunks,
            references=sorted(list(references)),
            llm_used=llm_used,
            search_time_ms=round(search_time, 2),
            llm_time_ms=round(llm_time, 2) if llm_time else None
        )
        
    except Exception as e:
        logger.error(f"RAG pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")


# ====================================================================================
# FAST INFO ENDPOINT (no embedding, no LLM)
# ====================================================================================

_CACHED_INFO: Optional[Dict[str, Any]] = None
_CACHED_INFO_TS: float = 0.0
_INFO_LOCK = threading.Lock()


async def get_collection_info() -> Dict[str, Any]:
    """
    Return cached collection info: variables, sources, count.
    No embeddings, no LLM - instant response.
    """
    global _CACHED_INFO, _CACHED_INFO_TS
    now = time.time()
    
    # Return cache if fresh (5 min)
    if _CACHED_INFO and (now - _CACHED_INFO_TS) < 300:
        return _CACHED_INFO
    
    with _INFO_LOCK:
        if _CACHED_INFO and (time.time() - _CACHED_INFO_TS) < 300:
            return _CACHED_INFO
        
        _config, db, _embedder, _llm = _get_components()
        
        variables = set()
        sources = set()
        count = 0
        
        client = getattr(db, "client", None)
        collection = getattr(db, "collection_name", None)
        
        if client and collection:
            try:
                # Get count
                info = client.get_collection(collection)
                count = info.points_count
                
                # Scroll to get unique variables/sources
                offset = None
                rounds = 0
                while rounds < 20 and len(variables) < 200:
                    points, offset = client.scroll(
                        collection_name=collection,
                        limit=500,
                        offset=offset,
                        with_vectors=False,
                        with_payload={"include": ["variable", "source_id", "dataset_name"]},
                    )
                    rounds += 1
                    for p in points:
                        payload = getattr(p, "payload", None) or {}
                        if payload.get("variable"):
                            variables.add(payload["variable"])
                        src = payload.get("source_id") or payload.get("dataset_name")
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


# ====================================================================================
# SIMPLE SEARCH (no LLM)
# ====================================================================================

async def simple_search(query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict]:
    """
    Fast vector search without LLM - for debugging and quick queries.
    """
    from src.climate_embeddings.embeddings.text_models import TextEmbedder
    from src.embeddings.database import VectorDatabase
    from src.utils.config_loader import ConfigLoader
    
    try:
        config_loader = ConfigLoader("config/pipeline_config.yaml")
        config = config_loader.load()
        
        db = VectorDatabase(config=config)
        embedder = TextEmbedder()
        
        # Search
        query_vec = embedder.embed_queries([query])[0]
        results = db.search(
            query_vector=query_vec.tolist(),
            limit=top_k,
            filter_dict=filters
        )
        
        # Format
        chunks = []
        for hit in results:
            if hasattr(hit, 'payload'):
                meta = hit.payload
                score = hit.score
            else:
                meta = hit.get('payload', {})
                score = hit.get('score', 0.0)
            
            chunks.append({
                "score": round(score, 3),
                "source_id": meta.get('source_id'),
                "variable": meta.get('variable'),
                "metadata": meta
            })
        
        return chunks
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
