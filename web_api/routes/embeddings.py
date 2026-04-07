"""Embedding management endpoints (Qdrant stats, sample, clear)."""

import os
import time
import logging
from typing import List

from fastapi import APIRouter, HTTPException

from web_api.models import EmbeddingResponse, EmbeddingStatsResponse
from web_api.dependencies import get_qdrant_client, get_collection_name, get_vector_database, clear_rag_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


@router.get("/stats", response_model=EmbeddingStatsResponse)
async def get_embeddings_stats():
    """Get stats from Qdrant."""
    try:
        db = get_vector_database()

        collections = db.client.get_collections().collections
        exists = any(c.name == db.collection for c in collections)

        if not exists:
            return EmbeddingStatsResponse(
                total_embeddings=0, variables=[], date_range=None, sources=[], collection_name=db.collection
            )

        count_res = db.client.count(collection_name=db.collection)

        points, _ = db.client.scroll(
            collection_name=db.collection, limit=100, with_payload=True, with_vectors=False
        )

        variables = set()
        sources = set()
        timestamps = []

        for point in points:
            payload = point.payload or {}
            if "variable" in payload:
                variables.add(payload["variable"])
            if "source_id" in payload:
                sources.add(payload["source_id"])
            if "timestamp" in payload:
                timestamps.append(payload["timestamp"])

        date_range = None
        if timestamps:
            date_range = {"earliest": min(timestamps), "latest": max(timestamps)}

        return EmbeddingStatsResponse(
            total_embeddings=count_res.count,
            variables=sorted(list(variables)),
            date_range=date_range,
            sources=sorted(list(sources)),
            collection_name=db.collection,
        )
    except Exception as e:
        logger.error(f"Failed to get embedding stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sample", response_model=List[EmbeddingResponse])
async def get_sample_embeddings(limit: int = 10):
    """Get sample embeddings from Qdrant."""
    try:
        db = get_vector_database()

        points, _ = db.client.scroll(
            collection_name=db.collection, limit=limit, with_payload=True, with_vectors=True
        )

        samples = []
        for point in points:
            meta = point.payload or {}
            samples.append(
                EmbeddingResponse(
                    id=str(point.id),
                    variable=meta.get("variable", "unknown"),
                    timestamp=meta.get("timestamp"),
                    location=None,
                    metadata=meta,
                    embedding_preview=point.vector[:10] if point.vector else [],
                )
            )
        return samples
    except Exception:
        return []


@router.post("/clear")
async def clear_embeddings(confirm: bool = False, delete_sources: bool = False):
    """Clear all embeddings from Qdrant. Optionally also delete all sources."""
    if not confirm:
        raise HTTPException(400, "Set confirm=true to clear embeddings")

    try:
        collection_name = get_collection_name()
        client = get_qdrant_client()

        logger.info(f"Attempting to delete collection: {collection_name}")

        try:
            collections = client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)

            points_before = 0
            if exists:
                try:
                    count_res = client.count(collection_name=collection_name)
                    points_before = count_res.count if hasattr(count_res, "count") else 0
                except Exception as e:
                    logger.warning(f"Could not count points: {e}")

                client.delete_collection(collection_name)
                logger.info(f"Deleted collection: {collection_name} (had {points_before} points)")

                time.sleep(0.5)

                collections_after = client.get_collections().collections
                still_exists = any(c.name == collection_name for c in collections_after)
                if still_exists:
                    logger.error(f"Collection {collection_name} still exists after deletion attempt!")
                    raise HTTPException(500, f"Failed to delete collection {collection_name}")
            else:
                logger.warning(f"Collection {collection_name} does not exist")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting collection: {e}", exc_info=True)
            raise HTTPException(500, f"Failed to delete collection: {str(e)}")

        clear_rag_cache()

        sources_deleted = 0
        if delete_sources:
            from src.sources import get_source_store

            store = get_source_store()
            all_sources = store.get_all_sources(active_only=False)
            for source in all_sources:
                if store.hard_delete_source(source.source_id):
                    sources_deleted += 1
            logger.info(f"Deleted {sources_deleted} sources")

        return {
            "status": "cleared",
            "collection": collection_name,
            "points_deleted": points_before if exists else 0,
            "sources_deleted": sources_deleted if delete_sources else None,
            "message": f"Collection {collection_name} deleted successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in clear_embeddings: {e}", exc_info=True)
        raise HTTPException(500, f"Unexpected error: {str(e)}")
