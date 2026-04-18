"""Qdrant dataset discovery endpoints.

Provides per-dataset stats directly from Qdrant, including chunk counts,
variables, hazard types, and sample metadata (links, time ranges, etc.).
Used by the frontend to show which catalog sources have data in the vector DB.
"""

import logging
import time
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException

from web_api.dependencies import get_qdrant_client, get_collection_name

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/qdrant", tags=["qdrant"])

# Cache for dataset listing (expensive to compute)
_DATASETS_CACHE: Optional[List[Dict[str, Any]]] = None
_DATASETS_CACHE_TS: float = 0
_COUNTS_CACHE: Optional[Dict[str, int]] = None
_COUNTS_CACHE_TS: float = 0
CACHE_TTL = 300  # 5 minutes


def _get_embedding_counts() -> Dict[str, int]:
    """Get per-source_id embedding counts using Qdrant facets. Cached."""
    global _COUNTS_CACHE, _COUNTS_CACHE_TS
    now = time.time()
    if _COUNTS_CACHE and (now - _COUNTS_CACHE_TS) < CACHE_TTL:
        return _COUNTS_CACHE

    client = get_qdrant_client()
    collection = get_collection_name()
    counts: Dict[str, int] = {}

    try:
        # Facet by source_id to get counts per source
        from qdrant_client.http.models import FacetRequest
        result = client.facet(
            collection_name=collection,
            key="source_id",
            limit=500,
        )
        for hit in result.hits:
            counts[hit.value] = hit.count
    except Exception as e:
        logger.warning(f"Facet by source_id failed: {e}")

    _COUNTS_CACHE = counts
    _COUNTS_CACHE_TS = time.time()
    return counts


@router.get("/datasets")
async def list_qdrant_datasets():
    """List all datasets in Qdrant with per-dataset stats.

    Uses Qdrant facet API for fast aggregation, then fetches one sample
    point per dataset to extract metadata (links, hazard type, etc.).
    """
    global _DATASETS_CACHE, _DATASETS_CACHE_TS
    now = time.time()
    if _DATASETS_CACHE and (now - _DATASETS_CACHE_TS) < CACHE_TTL:
        return _DATASETS_CACHE

    client = get_qdrant_client()
    collection = get_collection_name()

    try:
        info = client.get_collection(collection)
        total_points = info.points_count
    except Exception as e:
        raise HTTPException(500, f"Cannot reach Qdrant: {e}")

    # Get dataset_name facets (chunk counts per dataset)
    try:
        from qdrant_client.http.models import FacetRequest
        dataset_facets = client.facet(
            collection_name=collection,
            key="dataset_name",
            limit=500,
        )
    except Exception as e:
        raise HTTPException(500, f"Facet query failed: {e}")

    datasets = []
    for hit in dataset_facets.hits:
        dataset_name = hit.value
        chunk_count = hit.count

        # Fetch one sample point to get metadata
        sample_meta = {}
        try:
            from qdrant_client.http.models import Filter, FieldCondition, MatchValue
            points, _ = client.scroll(
                collection_name=collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="dataset_name", match=MatchValue(value=dataset_name))]
                ),
                limit=1,
                with_payload=True,
                with_vectors=False,
            )
            if points:
                sample_meta = points[0].payload or {}
        except Exception as e:
            logger.warning(f"Sample fetch failed for {dataset_name}: {e}")

        # Get per-dataset variable facets
        variables = []
        try:
            var_facets = client.facet(
                collection_name=collection,
                key="variable",
                limit=50,
                exact=False,
                filter=Filter(
                    must=[FieldCondition(key="dataset_name", match=MatchValue(value=dataset_name))]
                ),
            )
            variables = [{"name": v.value, "count": v.count} for v in var_facets.hits]
        except Exception:
            # Older qdrant versions may not support filter on facet
            if sample_meta.get("variable"):
                variables = [{"name": sample_meta["variable"], "count": chunk_count}]

        datasets.append({
            "dataset_name": dataset_name,
            "chunk_count": chunk_count,
            "source_id": sample_meta.get("source_id"),
            "hazard_type": sample_meta.get("hazard_type"),
            "variables": variables,
            "link": sample_meta.get("link") or sample_meta.get("dataset_webpage") or sample_meta.get("dataset_References"),
            "catalog_source": sample_meta.get("catalog_source"),
            "location_name": sample_meta.get("location_name"),
            "impact_sector": sample_meta.get("impact_sector"),
            "spatial_coverage": sample_meta.get("spatial_coverage"),
            "is_metadata_only": sample_meta.get("is_metadata_only", False),
            "time_start": sample_meta.get("time_start"),
            "time_end": sample_meta.get("time_end"),
        })

    # Sort by chunk count descending
    datasets.sort(key=lambda d: d["chunk_count"], reverse=True)

    _DATASETS_CACHE = datasets
    _DATASETS_CACHE_TS = time.time()

    return datasets


@router.get("/datasets/{dataset_name}/variables")
async def get_dataset_variables(dataset_name: str):
    """Get detailed variable breakdown for a specific dataset."""
    client = get_qdrant_client()
    collection = get_collection_name()

    try:
        from qdrant_client.http.models import Filter, FieldCondition, MatchValue
        var_facets = client.facet(
            collection_name=collection,
            key="variable",
            limit=200,
            facet_filter=Filter(
                must=[FieldCondition(key="dataset_name", match=MatchValue(value=dataset_name))]
            ),
        )
        return [{"name": v.value, "count": v.count} for v in var_facets.hits]
    except Exception as e:
        # An unknown dataset or a transient Qdrant hiccup should not 500 — the
        # caller (UI) just wants a list. Log and return empty.
        logger.warning(f"Variable facet query failed for dataset '{dataset_name}': {e}")
        return []


@router.post("/cache/clear")
async def clear_qdrant_cache():
    """Clear the datasets cache to force a refresh."""
    global _DATASETS_CACHE, _DATASETS_CACHE_TS, _COUNTS_CACHE, _COUNTS_CACHE_TS
    _DATASETS_CACHE = None
    _DATASETS_CACHE_TS = 0
    _COUNTS_CACHE = None
    _COUNTS_CACHE_TS = 0
    return {"status": "cleared"}
