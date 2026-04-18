"""RAG (Retrieval-Augmented Generation) endpoints."""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from web_api.models import RAGChatRequest, RAGChatResponse, RAGChunk
from web_api.rag_endpoint import rag_query, simple_search, get_collection_info, RAGRequest, RAGResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=RAGResponse)
async def rag_query_endpoint(request: RAGRequest):
    """Optimized RAG endpoint with timeout handling."""
    return await rag_query(request)


@router.get("/search")
async def rag_search_only(
    query: str,
    top_k: int = 5,
    source_id: Optional[str] = None,
    variable: Optional[str] = None,
):
    """Fast vector search without LLM - for testing and debugging."""
    filters = {}
    if source_id:
        filters["source_id"] = source_id
    if variable:
        filters["variable"] = variable
    return await simple_search(query, top_k, filters if filters else None)


@router.get("/info")
async def rag_info():
    """Get collection info: variables, sources, count."""
    return await get_collection_info()


@router.post("/chat", response_model=RAGChatResponse)
async def rag_chat_legacy(request: RAGChatRequest):
    """Legacy chat endpoint - delegates to optimized rag_query()."""
    if not (request.question or "").strip():
        raise HTTPException(400, "Question is required")

    # Honour the runtime default for `use_reranker` (toggled in Settings)
    # rather than hardcoding True — that was why turning it off in the UI
    # didn't actually speed up the chat.
    from web_api.routes.admin import _runtime_settings
    default_reranker = bool(_runtime_settings.get("use_reranker", False))

    try:
        rag_req = RAGRequest(
            question=request.question,
            top_k=request.top_k or 10,
            use_llm=request.use_llm if request.use_llm is not None else True,
            use_reranker=default_reranker,
            temperature=request.temperature or 0.3,
            source_id=getattr(request, "source_id", None),
            variable=getattr(request, "variable", None),
        )
        result = await rag_query(rag_req)
    except HTTPException:
        # Preserve original status codes raised by rag_query (e.g. 400 on empty query).
        raise
    except Exception as e:
        logger.error(f"RAG chat error: {e}", exc_info=True)
        raise HTTPException(500, str(e))

    chunks_model = []
    for c in result.chunks:
        meta = c.get("metadata", {})
        chunks_model.append(
            RAGChunk(
                source_id=c.get("source_id", "unknown"),
                variable=c.get("variable", "unknown"),
                similarity=c.get("score", 0.0),
                text=c.get("text", "")[:200],
                metadata=meta,
            )
        )

    return RAGChatResponse(
        question=result.question,
        answer=result.answer,
        references=result.references,
        chunks=chunks_model,
        llm_used=result.llm_used,
    )
