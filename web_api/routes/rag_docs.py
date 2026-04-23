"""
RAG endpoint for unstructured documents (PDF/DOCX/PPTX/MD/TXT).

Routes queries to the rag-mendelu AgenticRAG (LangGraph + Ollama).
Documents are stored in the `climate_docs` Qdrant collection (separate from
the `climate_data` scientific raster collection).
"""

import os
import sys
import logging
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag/docs", tags=["rag-docs"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SUBMODULE = str(PROJECT_ROOT / "rag-mendelu")

# ---------------------------------------------------------------------------
# Cached singletons — the embedding model and Qdrant client are expensive to
# initialise so we keep them alive for the process lifetime.  DuckDB is
# intentionally NOT cached here: it must be opened per-request (read-only) and
# closed immediately after so the Dagster ETL container can always acquire the
# write lock when it needs to ingest a new document.
# ---------------------------------------------------------------------------
_SINGLETON_LOCK = threading.Lock()
_CACHED_EMBEDDING: Optional[Any] = None
_CACHED_QDRANT: Optional[Any] = None
_CACHED_COLLECTION: Optional[str] = None


def _ensure_submodule_path():
    if _SUBMODULE not in sys.path:
        sys.path.insert(0, _SUBMODULE)


def _get_singletons():
    """Initialise and cache the embedding service and Qdrant repo (once)."""
    global _CACHED_EMBEDDING, _CACHED_QDRANT, _CACHED_COLLECTION

    collection = os.getenv("DOCS_COLLECTION_NAME", "climate_data_documents")

    if _CACHED_EMBEDDING is not None and _CACHED_COLLECTION == collection:
        return _CACHED_EMBEDDING, _CACHED_QDRANT, collection

    with _SINGLETON_LOCK:
        if _CACHED_EMBEDDING is not None and _CACHED_COLLECTION == collection:
            return _CACHED_EMBEDDING, _CACHED_QDRANT, collection

        _ensure_submodule_path()

        from database.qdrant_db_repository import QdrantDbRepository
        from text_embedding import TextEmbeddingService

        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_REST_PORT", "6333"))
        qdrant_grpc = int(os.getenv("QDRANT_GRPC_PORT", "6334"))

        embedding_service = TextEmbeddingService()
        vector_size = embedding_service.get_embedding_dim()

        db_repository = QdrantDbRepository(
            ip=qdrant_host,
            port=qdrant_port,
            grpc_port=qdrant_grpc,
            collection_name=collection,
            metadata={"vector_size": vector_size, "distance": "DOT"},
        )

        existing = [c.name for c in db_repository.client.get_collections().collections]
        if collection not in existing:
            db_repository.create_collection()

        _CACHED_EMBEDDING = embedding_service
        _CACHED_QDRANT = db_repository
        _CACHED_COLLECTION = collection
        logger.info(f"RAG singletons initialised: collection={collection}")

    return _CACHED_EMBEDDING, _CACHED_QDRANT, collection


def _build_rag_for_request(embedding_service, db_repository, collection: str):
    """Build AgenticRAG with a fresh read-only DuckDB connection.

    Called once per HTTP request so DuckDB is never held open between requests,
    allowing the Dagster ETL (separate container) to acquire the write lock
    whenever it needs to ingest a new document.

    Returns (rag, sql_db) — the caller must call sql_db.close() in a
    try/finally block.
    """
    from database.duck_db_repository import DuckDbRepository
    from rag.agentic_rag import AgenticRAG

    model_name = os.getenv("OLLAMA_MODEL", "ministral-3:3b")
    duckdb_path = str(PROJECT_ROOT / "data" / "sql" / f"{collection}.duckdb")

    # Open DuckDB read-only.  If the ETL is currently writing (write lock held
    # in the dagster container), fall back to an in-memory instance so the
    # query still works via vector search only.
    try:
        sql_db = DuckDbRepository(db_path=duckdb_path, read_only=True)
    except Exception as e:
        logger.warning(f"Could not open DuckDB read-only ({e}); falling back to in-memory (vector-only RAG)")
        sql_db = DuckDbRepository(db_path=":memory:")

    rag = AgenticRAG(
        database_service=db_repository,
        embedding_service=embedding_service,
        sql_db_repo=sql_db,
        model_name=model_name,
    )
    return rag, sql_db


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class DocsQueryRequest(BaseModel):
    question: str
    model: Optional[str] = None  # override OLLAMA_MODEL for this request


class DocsQueryResponse(BaseModel):
    answer: str
    sources: list
    rewritten_queries: list
    collection: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=DocsQueryResponse)
async def chat_docs(request: DocsQueryRequest):
    """Query unstructured documents via AgenticRAG (LangGraph + Ollama).

    Requires:
    - Ollama running at OLLAMA_HOST (default http://localhost:11434)
    - At least one document indexed in the DOCS_COLLECTION_NAME collection
    """
    if not request.question.strip():
        raise HTTPException(400, "question must not be empty")

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    import urllib.request
    import urllib.error
    try:
        urllib.request.urlopen(f"{ollama_host}/api/tags", timeout=3)
    except (urllib.error.URLError, OSError):
        raise HTTPException(
            503,
            f"Ollama is not reachable at {ollama_host}. "
            "Start Ollama and set OLLAMA_HOST in .env (for Docker: http://host.docker.internal:11434).",
        )

    try:
        _ensure_submodule_path()
        embedding_service, db_repository, collection = _get_singletons()
    except Exception as e:
        logger.error(f"RAG singleton init failed: {e}", exc_info=True)
        raise HTTPException(503, f"Could not initialise document RAG: {e}")

    sql_db = None
    try:
        rag, sql_db = _build_rag_for_request(embedding_service, db_repository, collection)
        result = rag.chat(request.question)
    except Exception as e:
        logger.error(f"AgenticRAG.chat failed: {e}", exc_info=True)
        raise HTTPException(500, f"RAG pipeline error: {e}")
    finally:
        if sql_db is not None:
            sql_db.close()

    sources = []
    for item in result.get("sources", []):
        if hasattr(item, "metadata"):
            item.metadata['text'] = item.text
            sources.append(item.metadata)
        elif isinstance(item, dict):
            sources.append(item)

    return DocsQueryResponse(
        answer=result.get("response", ""),
        sources=sources,
        rewritten_queries=result.get("rewritten_queries", []),
        collection=collection,
    )


@router.get("/info")
async def docs_info():
    """Return basic info about the docs collection (chunk count, collection name)."""
    _ensure_submodule_path()
    collection = os.getenv("DOCS_COLLECTION_NAME", "climate_data_documents")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_REST_PORT", "6333"))

    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=5)
        collections = [c.name for c in client.get_collections().collections]
        if collection not in collections:
            return {"collection": collection, "exists": False, "chunk_count": 0}
        count = client.count(collection_name=collection).count
        return {"collection": collection, "exists": True, "chunk_count": count}
    except Exception as e:
        return {"collection": collection, "exists": False, "error": str(e)}
