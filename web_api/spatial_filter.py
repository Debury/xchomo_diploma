"""
Spatial Intent Extraction and Qdrant Filter Builder for Climate RAG.

Implements the Spatial-RAG approach (arXiv:2502.18470): extract geographic
constraints from natural language queries and apply them as Qdrant payload
filters BEFORE vector similarity search.

This eliminates the "Location: Europe" mismatch problem where text-based
location names in chunk embeddings cause spatially incorrect retrieval.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Region bounding boxes (lat/lon)
# ---------------------------------------------------------------------------

REGION_BOUNDS: dict[str, dict[str, float]] = {
    # Continents / large regions
    "europe": {"lat_min": 35.0, "lat_max": 72.0, "lon_min": -25.0, "lon_max": 45.0},
    "africa": {"lat_min": -35.0, "lat_max": 37.0, "lon_min": -18.0, "lon_max": 52.0},
    "asia": {"lat_min": -10.0, "lat_max": 77.0, "lon_min": 25.0, "lon_max": 180.0},
    "north america": {"lat_min": 15.0, "lat_max": 72.0, "lon_min": -170.0, "lon_max": -50.0},
    "south america": {"lat_min": -56.0, "lat_max": 13.0, "lon_min": -82.0, "lon_max": -34.0},
    "australia": {"lat_min": -45.0, "lat_max": -10.0, "lon_min": 110.0, "lon_max": 155.0},
    "antarctica": {"lat_min": -90.0, "lat_max": -60.0, "lon_min": -180.0, "lon_max": 180.0},
    # Hemispheres
    "northern hemisphere": {"lat_min": 0.0, "lat_max": 90.0, "lon_min": -180.0, "lon_max": 180.0},
    "southern hemisphere": {"lat_min": -90.0, "lat_max": 0.0, "lon_min": -180.0, "lon_max": 180.0},
    "global": {"lat_min": -90.0, "lat_max": 90.0, "lon_min": -180.0, "lon_max": 180.0},
    # European sub-regions
    "central europe": {"lat_min": 45.0, "lat_max": 55.0, "lon_min": 5.0, "lon_max": 25.0},
    "western europe": {"lat_min": 42.0, "lat_max": 60.0, "lon_min": -10.0, "lon_max": 10.0},
    "eastern europe": {"lat_min": 42.0, "lat_max": 60.0, "lon_min": 20.0, "lon_max": 45.0},
    "southern europe": {"lat_min": 35.0, "lat_max": 46.0, "lon_min": -10.0, "lon_max": 30.0},
    "northern europe": {"lat_min": 54.0, "lat_max": 72.0, "lon_min": -25.0, "lon_max": 32.0},
    "scandinavia": {"lat_min": 55.0, "lat_max": 72.0, "lon_min": 4.0, "lon_max": 32.0},
    "mediterranean": {"lat_min": 30.0, "lat_max": 46.0, "lon_min": -6.0, "lon_max": 36.0},
    "iberian peninsula": {"lat_min": 36.0, "lat_max": 43.8, "lon_min": -9.5, "lon_max": 3.3},
    "balkans": {"lat_min": 39.0, "lat_max": 47.0, "lon_min": 13.0, "lon_max": 30.0},
    "british isles": {"lat_min": 49.5, "lat_max": 61.0, "lon_min": -11.0, "lon_max": 2.0},
    "alpine region": {"lat_min": 44.0, "lat_max": 48.5, "lon_min": 5.0, "lon_max": 17.0},
    # Countries (key ones for climate research)
    "czech republic": {"lat_min": 48.5, "lat_max": 51.1, "lon_min": 12.1, "lon_max": 18.9},
    "czechia": {"lat_min": 48.5, "lat_max": 51.1, "lon_min": 12.1, "lon_max": 18.9},
    "germany": {"lat_min": 47.3, "lat_max": 55.1, "lon_min": 5.9, "lon_max": 15.0},
    "france": {"lat_min": 41.3, "lat_max": 51.1, "lon_min": -5.1, "lon_max": 9.6},
    "spain": {"lat_min": 36.0, "lat_max": 43.8, "lon_min": -9.3, "lon_max": 3.3},
    "italy": {"lat_min": 36.6, "lat_max": 47.1, "lon_min": 6.6, "lon_max": 18.5},
    "united kingdom": {"lat_min": 49.9, "lat_max": 60.9, "lon_min": -8.2, "lon_max": 1.8},
    "uk": {"lat_min": 49.9, "lat_max": 60.9, "lon_min": -8.2, "lon_max": 1.8},
    "poland": {"lat_min": 49.0, "lat_max": 54.8, "lon_min": 14.1, "lon_max": 24.2},
    "austria": {"lat_min": 46.4, "lat_max": 49.0, "lon_min": 9.5, "lon_max": 17.2},
    "switzerland": {"lat_min": 45.8, "lat_max": 47.8, "lon_min": 5.9, "lon_max": 10.5},
    "slovakia": {"lat_min": 47.7, "lat_max": 49.6, "lon_min": 16.8, "lon_max": 22.6},
    "hungary": {"lat_min": 45.7, "lat_max": 48.6, "lon_min": 16.1, "lon_max": 22.9},
    "netherlands": {"lat_min": 50.8, "lat_max": 53.5, "lon_min": 3.4, "lon_max": 7.2},
    "sweden": {"lat_min": 55.3, "lat_max": 69.1, "lon_min": 11.1, "lon_max": 24.2},
    "norway": {"lat_min": 58.0, "lat_max": 71.2, "lon_min": 4.6, "lon_max": 31.2},
    "finland": {"lat_min": 59.8, "lat_max": 70.1, "lon_min": 20.5, "lon_max": 31.6},
    "greece": {"lat_min": 34.8, "lat_max": 41.7, "lon_min": 19.4, "lon_max": 29.6},
    "portugal": {"lat_min": 37.0, "lat_max": 42.2, "lon_min": -9.5, "lon_max": -6.2},
    "romania": {"lat_min": 43.6, "lat_max": 48.3, "lon_min": 20.3, "lon_max": 29.7},
    # Climate-relevant zones
    "arctic": {"lat_min": 66.5, "lat_max": 90.0, "lon_min": -180.0, "lon_max": 180.0},
    "tropics": {"lat_min": -23.5, "lat_max": 23.5, "lon_min": -180.0, "lon_max": 180.0},
    "sahel": {"lat_min": 10.0, "lat_max": 20.0, "lon_min": -18.0, "lon_max": 40.0},
}

# Aliases — maps common variants to canonical region names
REGION_ALIASES: dict[str, str] = {
    "central european": "central europe",
    "western european": "western europe",
    "eastern european": "eastern europe",
    "southern european": "southern europe",
    "northern european": "northern europe",
    "european": "europe",
    "african": "africa",
    "asian": "asia",
    "mediterranean region": "mediterranean",
    "mediterranean basin": "mediterranean",
    "alps": "alpine region",
    "alpine": "alpine region",
    "britain": "british isles",
    "great britain": "british isles",
    "england": "british isles",
    "scotland": "british isles",
    "czech": "czech republic",
    "cr": "czech republic",
    "cz": "czech republic",
}


@dataclass
class SpatialIntent:
    """Extracted spatial constraints from a user query."""

    region_name: Optional[str] = None
    lat_min: Optional[float] = None
    lat_max: Optional[float] = None
    lon_min: Optional[float] = None
    lon_max: Optional[float] = None
    dataset_name: Optional[str] = None
    variable_hint: Optional[str] = None


def extract_spatial_intent(query: str) -> SpatialIntent:
    """
    Extract spatial intent from a natural language query using rule-based
    region matching. Fast, deterministic, no LLM call needed.

    Falls back to no spatial filter if no region is detected (safe default).
    """
    query_lower = query.lower().strip()
    intent = SpatialIntent()

    # 1. Try to match a region name (longest match first to prefer specificity)
    best_match = None
    best_match_len = 0

    # Check aliases first
    for alias, canonical in REGION_ALIASES.items():
        if alias in query_lower and len(alias) > best_match_len:
            best_match = canonical
            best_match_len = len(alias)

    # Check canonical region names
    for region in REGION_BOUNDS:
        if region in query_lower and len(region) > best_match_len:
            best_match = region
            best_match_len = len(region)

    if best_match and best_match != "global":
        bounds = REGION_BOUNDS[best_match]
        intent.region_name = best_match
        intent.lat_min = bounds["lat_min"]
        intent.lat_max = bounds["lat_max"]
        intent.lon_min = bounds["lon_min"]
        intent.lon_max = bounds["lon_max"]
        logger.info(
            f"Spatial intent: '{best_match}' → "
            f"lat [{bounds['lat_min']}, {bounds['lat_max']}], "
            f"lon [{bounds['lon_min']}, {bounds['lon_max']}]"
        )

    # 2. Try to extract explicit coordinates (e.g., "latitude 50 to 55")
    lat_match = re.search(
        r'lat(?:itude)?\s+(-?\d+\.?\d*)\s*(?:to|-)\s*(-?\d+\.?\d*)',
        query_lower,
    )
    if lat_match:
        intent.lat_min = float(lat_match.group(1))
        intent.lat_max = float(lat_match.group(2))

    lon_match = re.search(
        r'lon(?:gitude)?\s+(-?\d+\.?\d*)\s*(?:to|-)\s*(-?\d+\.?\d*)',
        query_lower,
    )
    if lon_match:
        intent.lon_min = float(lon_match.group(1))
        intent.lon_max = float(lon_match.group(2))

    # 3. Try to detect dataset name
    dataset_patterns = {
        "cmip6": "CMIP6",
        "cmip 6": "CMIP6",
        "cru": "CRU",
        "e-obs": "E-OBS",
        "eobs": "E-OBS",
        "era5": "ERA5",
        "era 5": "ERA5",
        "gistemp": "GISTEMP",
        "euro-cordex": "EURO-CORDEX",
        "cordex": "EURO-CORDEX",
    }
    for pattern, name in dataset_patterns.items():
        if pattern in query_lower:
            intent.dataset_name = name
            break

    return intent


def build_qdrant_filter(intent: SpatialIntent, extra_filters: Optional[dict] = None):
    """
    Build a Qdrant Filter from spatial intent and optional extra filters.

    Uses bounding-box overlap logic: a chunk overlaps the target region if
    chunk.lat_max >= target.lat_min AND chunk.lat_min <= target.lat_max
    (and same for longitude).

    Returns None if no filters are needed.
    """
    from qdrant_client.models import (
        FieldCondition,
        Filter,
        MatchValue,
        Range,
    )

    conditions = []

    # Spatial bounding box overlap on latitude.
    # Most chunks have numeric latitude_min/latitude_max fields.
    # Longitude fields (longitude_min/max) are missing on many chunks,
    # so we only filter on latitude to avoid excluding valid results.
    # This alone fixes the main "Southern Hemisphere for Europe query" issue.
    if intent.lat_min is not None:
        conditions.append(
            FieldCondition(key="latitude_max", range=Range(gte=intent.lat_min))
        )
    if intent.lat_max is not None:
        conditions.append(
            FieldCondition(key="latitude_min", range=Range(lte=intent.lat_max))
        )

    # Dataset filter
    if intent.dataset_name:
        conditions.append(
            FieldCondition(key="dataset_name", match=MatchValue(value=intent.dataset_name))
        )

    # Extra filters (source_id, variable, etc.)
    if extra_filters:
        for key, value in extra_filters.items():
            conditions.append(
                FieldCondition(key=key, match=MatchValue(value=value))
            )

    return Filter(must=conditions) if conditions else None


def create_payload_indexes(client, collection_name: str = "climate_data"):
    """
    Create payload indexes for efficient filtered search. Run once.

    Creates:
    - Float indexes on lat/lon fields for spatial range filtering
    - Keyword indexes on categorical fields for exact match filtering
    """
    float_fields = [
        "latitude_min", "latitude_max",
        "longitude_min", "longitude_max",
    ]
    keyword_fields = [
        "dataset_name", "variable", "source_id",
    ]

    for f in float_fields:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=f,
                field_schema="float",
            )
            logger.info(f"Created float index on '{f}'")
        except Exception as e:
            logger.debug(f"Index on '{f}' may already exist: {e}")

    for f in keyword_fields:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=f,
                field_schema="keyword",
            )
            logger.info(f"Created keyword index on '{f}'")
        except Exception as e:
            logger.debug(f"Index on '{f}' may already exist: {e}")
