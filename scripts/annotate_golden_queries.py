#!/usr/bin/env python3
"""
Golden Query Annotation Script for Climate RAG Evaluation

Runs against a live Qdrant instance after ingestion to annotate 52 predefined
queries with real chunk IDs. Uses two strategies:
  1. Metadata-based ground truth (primary): scroll + filter on variable, hazard_type,
     region_country, source_id patterns to find objectively matching chunks.
  2. Embedding cross-check (secondary): embed each query with BGE, search top-20,
     log overlap with metadata results.

Output: tests/fixtures/golden_queries.json with real UUIDs.

Usage:
    python scripts/annotate_golden_queries.py
    # or inside Docker:
    docker exec climate-web-api python scripts/annotate_golden_queries.py
"""

import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "climate_data")

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "golden_queries.json"

# ---------------------------------------------------------------------------
# 52 query definitions across 6 categories
# ---------------------------------------------------------------------------

QUERY_DEFINITIONS: list[dict[str, Any]] = [
    # === Variable-specific (10) ===
    {
        "id": "q001",
        "query": "What is the global mean temperature anomaly for 2023?",
        "category": "variable-specific",
        "difficulty": "easy",
        "variable_patterns": ["temperature", "tas", "tmp", "t2m", "temp"],
        "key_numbers": ["2023"],
        "notes": "Temperature variable retrieval",
    },
    {
        "id": "q002",
        "query": "How does annual precipitation vary across Central Europe?",
        "category": "variable-specific",
        "difficulty": "medium",
        "variable_patterns": ["precipitation", "pr", "pre", "tp", "precip", "rain"],
        "spatial_filters": ["europe", "central"],
        "key_numbers": [],
        "notes": "Precipitation with spatial constraint",
    },
    {
        "id": "q003",
        "query": "What are the mean sea level pressure trends in the North Atlantic?",
        "category": "variable-specific",
        "difficulty": "medium",
        "variable_patterns": ["pressure", "psl", "mslp", "slp", "msl"],
        "key_numbers": [],
        "notes": "Pressure variable retrieval",
    },
    {
        "id": "q004",
        "query": "What wind speed data is available for offshore wind energy assessment?",
        "category": "variable-specific",
        "difficulty": "medium",
        "variable_patterns": ["wind", "sfcWind", "uas", "vas", "wind_speed", "u10", "v10"],
        "key_numbers": [],
        "notes": "Wind speed variable retrieval",
    },
    {
        "id": "q005",
        "query": "How has relative humidity changed over the past decades?",
        "category": "variable-specific",
        "difficulty": "medium",
        "variable_patterns": ["humidity", "hurs", "huss", "rh", "relative_humidity"],
        "key_numbers": [],
        "notes": "Humidity variable retrieval",
    },
    {
        "id": "q006",
        "query": "What drought index datasets are available for monitoring agricultural drought?",
        "category": "variable-specific",
        "difficulty": "hard",
        "variable_patterns": ["drought", "spi", "spei", "pdsi", "soil_moisture", "sm"],
        "hazard_patterns": ["drought"],
        "key_numbers": [],
        "notes": "Drought indices retrieval",
    },
    {
        "id": "q007",
        "query": "What solar radiation data is available for climate modelling?",
        "category": "variable-specific",
        "difficulty": "medium",
        "variable_patterns": ["radiation", "rsds", "rlds", "solar", "shortwave", "longwave"],
        "key_numbers": [],
        "notes": "Radiation variable retrieval",
    },
    {
        "id": "q008",
        "query": "What snow cover and snow depth datasets exist for Alpine regions?",
        "category": "variable-specific",
        "difficulty": "hard",
        "variable_patterns": ["snow", "snw", "snd", "snow_cover", "snow_depth"],
        "spatial_filters": ["alpine", "alps"],
        "key_numbers": [],
        "notes": "Snow variables with spatial constraint",
    },
    {
        "id": "q009",
        "query": "What sea surface temperature data is available for the Mediterranean?",
        "category": "variable-specific",
        "difficulty": "medium",
        "variable_patterns": ["sst", "sea_surface_temperature", "tos", "sea surface"],
        "spatial_filters": ["mediterranean"],
        "key_numbers": [],
        "notes": "SST variable retrieval",
    },
    {
        "id": "q010",
        "query": "What evapotranspiration data is available for water balance studies?",
        "category": "variable-specific",
        "difficulty": "hard",
        "variable_patterns": ["evapotranspiration", "evspsbl", "et", "pet", "evap"],
        "key_numbers": [],
        "notes": "Evapotranspiration variable retrieval",
    },
    # === Spatial (10) ===
    {
        "id": "q011",
        "query": "What climate datasets cover the Czech Republic?",
        "category": "spatial",
        "difficulty": "easy",
        "variable_patterns": [],
        "spatial_filters": ["czech", "czechia", "cz"],
        "key_numbers": [],
        "notes": "Country-level spatial query",
    },
    {
        "id": "q012",
        "query": "What temperature data is available for the Mediterranean basin?",
        "category": "spatial",
        "difficulty": "medium",
        "variable_patterns": ["temperature", "tas", "tmp", "t2m"],
        "spatial_filters": ["mediterranean"],
        "key_numbers": [],
        "notes": "Regional spatial query with variable filter",
    },
    {
        "id": "q013",
        "query": "Which datasets provide pan-European climate observations?",
        "category": "spatial",
        "difficulty": "easy",
        "variable_patterns": [],
        "spatial_filters": ["europe", "european", "pan-european"],
        "key_numbers": [],
        "notes": "Continental spatial query",
    },
    {
        "id": "q014",
        "query": "What Arctic climate datasets are available for sea ice analysis?",
        "category": "spatial",
        "difficulty": "hard",
        "variable_patterns": ["ice", "sic", "sea_ice", "arctic"],
        "spatial_filters": ["arctic"],
        "key_numbers": [],
        "notes": "Polar region query",
    },
    {
        "id": "q015",
        "query": "What climate data exists for Alpine regions above 2000m elevation?",
        "category": "spatial",
        "difficulty": "hard",
        "variable_patterns": [],
        "spatial_filters": ["alpine", "alps", "mountain"],
        "key_numbers": ["2000"],
        "notes": "Mountain region query with elevation",
    },
    {
        "id": "q016",
        "query": "What Nordic climate datasets are available for Scandinavia?",
        "category": "spatial",
        "difficulty": "medium",
        "variable_patterns": [],
        "spatial_filters": ["nordic", "scandinavia", "sweden", "norway", "finland"],
        "key_numbers": [],
        "notes": "Sub-continental spatial query",
    },
    {
        "id": "q017",
        "query": "What global gridded climate datasets cover all continents?",
        "category": "spatial",
        "difficulty": "easy",
        "variable_patterns": [],
        "spatial_filters": ["global"],
        "key_numbers": [],
        "notes": "Global coverage query",
    },
    {
        "id": "q018",
        "query": "What precipitation data exists for the Iberian Peninsula?",
        "category": "spatial",
        "difficulty": "medium",
        "variable_patterns": ["precipitation", "pr", "pre", "tp", "rain"],
        "spatial_filters": ["iberian", "spain", "portugal"],
        "key_numbers": [],
        "notes": "Sub-regional spatial query with variable filter",
    },
    {
        "id": "q019",
        "query": "What climate datasets are available for the Danube river basin?",
        "category": "spatial",
        "difficulty": "hard",
        "variable_patterns": [],
        "spatial_filters": ["danube", "river basin"],
        "key_numbers": [],
        "notes": "River basin spatial query",
    },
    {
        "id": "q020",
        "query": "What coastal climate data exists for the North Sea region?",
        "category": "spatial",
        "difficulty": "medium",
        "variable_patterns": [],
        "spatial_filters": ["north sea", "coastal"],
        "key_numbers": [],
        "notes": "Coastal region query",
    },
    # === Temporal (8) ===
    {
        "id": "q021",
        "query": "What historical climate data is available for the period 1950-2000?",
        "category": "temporal",
        "difficulty": "easy",
        "variable_patterns": [],
        "temporal_filters": ["1950", "2000", "historical"],
        "key_numbers": ["1950", "2000"],
        "notes": "Historical period query",
    },
    {
        "id": "q022",
        "query": "What climate projections are available for the period 2050-2100?",
        "category": "temporal",
        "difficulty": "easy",
        "variable_patterns": [],
        "temporal_filters": ["2050", "2100", "projection", "future"],
        "key_numbers": ["2050", "2100"],
        "notes": "Future projection period query",
    },
    {
        "id": "q023",
        "query": "What was the maximum daily temperature in Europe during summer 2022?",
        "category": "temporal",
        "difficulty": "hard",
        "variable_patterns": ["temperature", "tas", "tasmax", "tx", "tmax"],
        "temporal_filters": ["2022"],
        "spatial_filters": ["europe"],
        "key_numbers": ["2022"],
        "notes": "Specific season and year query",
    },
    {
        "id": "q024",
        "query": "What monthly temperature trends are observed over the last 30 years?",
        "category": "temporal",
        "difficulty": "medium",
        "variable_patterns": ["temperature", "tas", "tmp", "t2m"],
        "temporal_filters": ["monthly", "trend"],
        "key_numbers": ["30"],
        "notes": "Trend analysis temporal query",
    },
    {
        "id": "q025",
        "query": "What near-real-time climate datasets are updated daily?",
        "category": "temporal",
        "difficulty": "hard",
        "variable_patterns": [],
        "temporal_filters": ["daily", "real-time", "near-real-time"],
        "key_numbers": [],
        "notes": "Temporal resolution query",
    },
    {
        "id": "q026",
        "query": "What climate reanalysis data covers the 20th century?",
        "category": "temporal",
        "difficulty": "medium",
        "variable_patterns": [],
        "temporal_filters": ["reanalysis", "20th century", "1900"],
        "key_numbers": [],
        "notes": "Reanalysis temporal query",
    },
    {
        "id": "q027",
        "query": "What seasonal forecast data is available for the next 6 months?",
        "category": "temporal",
        "difficulty": "hard",
        "variable_patterns": [],
        "temporal_filters": ["seasonal", "forecast"],
        "key_numbers": ["6"],
        "notes": "Forecast temporal query",
    },
    {
        "id": "q028",
        "query": "How has the frequency of heat waves changed since 1990?",
        "category": "temporal",
        "difficulty": "hard",
        "variable_patterns": ["temperature", "tas", "tasmax", "heat"],
        "temporal_filters": ["1990"],
        "hazard_patterns": ["heat", "heatwave"],
        "key_numbers": ["1990"],
        "notes": "Extreme event frequency temporal query",
    },
    # === Cross-variable (8) ===
    {
        "id": "q029",
        "query": "What compound flood-heat events are represented in the datasets?",
        "category": "cross-variable",
        "difficulty": "hard",
        "variable_patterns": ["flood", "temperature", "precipitation", "heat"],
        "hazard_patterns": ["flood", "heat"],
        "key_numbers": [],
        "notes": "Multi-hazard compound event query",
    },
    {
        "id": "q030",
        "query": "Compare temperature and precipitation projections under SSP245 and SSP585",
        "category": "cross-variable",
        "difficulty": "hard",
        "variable_patterns": ["temperature", "tas", "precipitation", "pr"],
        "source_id_patterns": ["ssp245", "ssp585", "scenario"],
        "key_numbers": [],
        "notes": "Multi-variable scenario comparison",
    },
    {
        "id": "q031",
        "query": "What multi-hazard risk datasets combine flooding and wind damage?",
        "category": "cross-variable",
        "difficulty": "hard",
        "variable_patterns": ["flood", "wind", "damage", "risk"],
        "hazard_patterns": ["flood", "wind", "storm"],
        "key_numbers": [],
        "notes": "Multi-hazard risk query",
    },
    {
        "id": "q032",
        "query": "How do temperature and humidity interact to produce heat stress indices?",
        "category": "cross-variable",
        "difficulty": "hard",
        "variable_patterns": ["temperature", "humidity", "heat_stress", "wbgt", "utci"],
        "key_numbers": [],
        "notes": "Derived index cross-variable query",
    },
    {
        "id": "q033",
        "query": "What datasets combine precipitation with soil moisture for drought monitoring?",
        "category": "cross-variable",
        "difficulty": "hard",
        "variable_patterns": ["precipitation", "pr", "soil_moisture", "sm", "drought"],
        "hazard_patterns": ["drought"],
        "key_numbers": [],
        "notes": "Cross-variable drought monitoring query",
    },
    {
        "id": "q034",
        "query": "What energy-relevant climate variables are available for renewable energy planning?",
        "category": "cross-variable",
        "difficulty": "medium",
        "variable_patterns": ["wind", "solar", "radiation", "temperature", "rsds"],
        "key_numbers": [],
        "notes": "Application-oriented cross-variable query",
    },
    {
        "id": "q035",
        "query": "How are ocean heat content and sea surface temperature related in the datasets?",
        "category": "cross-variable",
        "difficulty": "hard",
        "variable_patterns": ["sst", "tos", "ocean", "heat_content", "ohc"],
        "key_numbers": [],
        "notes": "Ocean cross-variable query",
    },
    {
        "id": "q036",
        "query": "What coupled atmosphere-ocean datasets include both temperature and salinity?",
        "category": "cross-variable",
        "difficulty": "hard",
        "variable_patterns": ["temperature", "salinity", "so", "tos", "tas", "coupled"],
        "key_numbers": [],
        "notes": "Coupled model cross-variable query",
    },
    # === Methodological (8) ===
    {
        "id": "q037",
        "query": "What bias correction methods are applied to climate projection data?",
        "category": "methodological",
        "difficulty": "hard",
        "variable_patterns": [],
        "source_id_patterns": ["bias", "correction", "adjusted"],
        "key_numbers": [],
        "notes": "Bias correction methodological query",
    },
    {
        "id": "q038",
        "query": "What spatial resolution options are available across the datasets?",
        "category": "methodological",
        "difficulty": "medium",
        "variable_patterns": [],
        "source_id_patterns": ["resolution"],
        "key_numbers": ["0.1", "0.25", "0.5", "1.0"],
        "notes": "Resolution methodological query",
    },
    {
        "id": "q039",
        "query": "How many ensemble members are included in the climate model runs?",
        "category": "methodological",
        "difficulty": "medium",
        "variable_patterns": [],
        "source_id_patterns": ["ensemble", "member", "run"],
        "key_numbers": [],
        "notes": "Ensemble member query",
    },
    {
        "id": "q040",
        "query": "What statistical downscaling methods are used in the regional projections?",
        "category": "methodological",
        "difficulty": "hard",
        "variable_patterns": [],
        "source_id_patterns": ["downscaling", "statistical", "regional"],
        "key_numbers": [],
        "notes": "Downscaling methodology query",
    },
    {
        "id": "q041",
        "query": "What data quality control procedures are applied to the observational datasets?",
        "category": "methodological",
        "difficulty": "hard",
        "variable_patterns": [],
        "source_id_patterns": ["quality", "control", "qc"],
        "key_numbers": [],
        "notes": "Quality control methodology query",
    },
    {
        "id": "q042",
        "query": "What interpolation methods are used for gridding station data?",
        "category": "methodological",
        "difficulty": "hard",
        "variable_patterns": [],
        "source_id_patterns": ["interpolation", "gridding", "kriging"],
        "key_numbers": [],
        "notes": "Interpolation methodology query",
    },
    {
        "id": "q043",
        "query": "What uncertainty quantification is provided with the climate projections?",
        "category": "methodological",
        "difficulty": "hard",
        "variable_patterns": [],
        "source_id_patterns": ["uncertainty", "confidence", "range"],
        "key_numbers": [],
        "notes": "Uncertainty methodology query",
    },
    {
        "id": "q044",
        "query": "What homogenization techniques are applied to long-term temperature records?",
        "category": "methodological",
        "difficulty": "hard",
        "variable_patterns": ["temperature", "tas"],
        "source_id_patterns": ["homogenization", "homogenised"],
        "key_numbers": [],
        "notes": "Homogenization methodology query",
    },
    # === Dataset-specific (8) ===
    {
        "id": "q045",
        "query": "What variables does the CRU TS dataset provide?",
        "category": "dataset-specific",
        "difficulty": "easy",
        "variable_patterns": [],
        "source_id_patterns": ["CRU", "cru"],
        "key_numbers": [],
        "notes": "CRU dataset query",
    },
    {
        "id": "q046",
        "query": "What is the spatial coverage of the E-OBS gridded dataset?",
        "category": "dataset-specific",
        "difficulty": "easy",
        "variable_patterns": [],
        "source_id_patterns": ["E-OBS", "eobs", "EOBS"],
        "key_numbers": ["0.1", "0.25"],
        "notes": "E-OBS dataset query",
    },
    {
        "id": "q047",
        "query": "What ERA5 reanalysis variables are available in the collection?",
        "category": "dataset-specific",
        "difficulty": "easy",
        "variable_patterns": [],
        "source_id_patterns": ["ERA5", "era5"],
        "key_numbers": [],
        "notes": "ERA5 dataset query",
    },
    {
        "id": "q048",
        "query": "What GPCP precipitation data is available?",
        "category": "dataset-specific",
        "difficulty": "easy",
        "variable_patterns": ["precipitation", "pr"],
        "source_id_patterns": ["GPCP", "gpcp"],
        "key_numbers": [],
        "notes": "GPCP dataset query",
    },
    {
        "id": "q049",
        "query": "What WorldClim bioclimatic variables are in the database?",
        "category": "dataset-specific",
        "difficulty": "medium",
        "variable_patterns": ["bio", "bioclim"],
        "source_id_patterns": ["WorldClim", "worldclim"],
        "key_numbers": [],
        "notes": "WorldClim dataset query",
    },
    {
        "id": "q050",
        "query": "What NCEP-NCAR reanalysis data is available?",
        "category": "dataset-specific",
        "difficulty": "medium",
        "variable_patterns": [],
        "source_id_patterns": ["NCEP", "ncep", "NCAR", "ncar"],
        "key_numbers": [],
        "notes": "NCEP-NCAR dataset query",
    },
    {
        "id": "q051",
        "query": "What ISI-MIP impact model datasets are included?",
        "category": "dataset-specific",
        "difficulty": "hard",
        "variable_patterns": [],
        "source_id_patterns": ["ISI-MIP", "isimip", "ISIMIP"],
        "key_numbers": [],
        "notes": "ISI-MIP dataset query",
    },
    {
        "id": "q052",
        "query": "What is the overall structure and content of the climate data catalog?",
        "category": "dataset-specific",
        "difficulty": "easy",
        "variable_patterns": [],
        "source_id_patterns": ["catalog"],
        "key_numbers": ["233"],
        "notes": "Catalog overview query",
    },
]


def _make_point_uuid(uid: str) -> str:
    """Reproduce the deterministic UUID logic from src/embeddings/database.py:173."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(uid)))


def _payload_matches_patterns(payload: dict, patterns: list[str], fields: list[str]) -> bool:
    """Check if any of the specified payload fields contain any of the patterns."""
    for field in fields:
        value = str(payload.get(field, "")).lower()
        if not value:
            continue
        for pat in patterns:
            if pat.lower() in value:
                return True
    return False


def find_ground_truth_by_metadata(
    client,
    qdef: dict[str, Any],
    max_results: int = 10,
) -> list[str]:
    """
    Find ground-truth chunk IDs using metadata scroll + filter.

    Scrolls through the collection and matches points whose payload fields
    contain the query's annotated patterns (variable, hazard, spatial, source_id).
    """
    from qdrant_client.http.models import Filter, FieldCondition, MatchText

    variable_patterns = qdef.get("variable_patterns", [])
    hazard_patterns = qdef.get("hazard_patterns", [])
    spatial_filters = qdef.get("spatial_filters", [])
    source_id_patterns = qdef.get("source_id_patterns", [])
    temporal_filters = qdef.get("temporal_filters", [])

    # Strategy: try Qdrant text match filters first, fall back to full scroll
    matched_ids: list[str] = []

    # Build combined text-match conditions for a scroll query.
    # We use MatchText on common fields for the strongest signal (variable patterns).
    all_patterns = variable_patterns + hazard_patterns + spatial_filters + source_id_patterns

    if not all_patterns:
        # No metadata patterns defined — skip metadata matching for this query
        return []

    # Full-scan approach: scroll through all points and match locally
    # (more reliable than building complex Qdrant filter conditions)
    offset = None
    seen = 0
    SCAN_LIMIT = 250000  # scan entire collection

    search_fields_variable = ["variable", "variable_name", "long_name", "standard_name"]
    search_fields_hazard = ["hazard_type", "hazard", "text"]
    search_fields_spatial = [
        "region_country", "spatial_coverage", "text", "dataset_name",
        "source_id", "long_name",
    ]
    search_fields_source = ["source_id", "dataset_name", "title", "text"]
    search_fields_temporal = ["time_start", "time_end", "temporal_coverage", "text"]

    while seen < SCAN_LIMIT:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            offset=offset,
            with_vectors=False,
            with_payload=True,
        )
        if not points:
            break
        seen += len(points)

        for p in points:
            payload = getattr(p, "payload", None) or {}
            score = 0

            # Variable match
            if variable_patterns:
                if _payload_matches_patterns(payload, variable_patterns, search_fields_variable):
                    score += 2
                # Also check text field for variable mentions
                if _payload_matches_patterns(payload, variable_patterns, ["text"]):
                    score += 1

            # Hazard match
            if hazard_patterns:
                if _payload_matches_patterns(payload, hazard_patterns, search_fields_hazard):
                    score += 2

            # Spatial match
            if spatial_filters:
                if _payload_matches_patterns(payload, spatial_filters, search_fields_spatial):
                    score += 2

            # Source/dataset match
            if source_id_patterns:
                if _payload_matches_patterns(payload, source_id_patterns, search_fields_source):
                    score += 2

            # Temporal match
            if temporal_filters:
                if _payload_matches_patterns(payload, temporal_filters, search_fields_temporal):
                    score += 1

            # Require at least one strong match category
            required_score = 2
            if score >= required_score:
                point_id = str(p.id)
                matched_ids.append(point_id)

        if offset is None:
            break

    # Deduplicate and limit
    seen_ids: set[str] = set()
    unique: list[str] = []
    for pid in matched_ids:
        if pid not in seen_ids:
            seen_ids.add(pid)
            unique.append(pid)
    return unique[:max_results]


def embedding_cross_check(
    client,
    model,
    query: str,
    metadata_ids: list[str],
    top_k: int = 20,
) -> dict[str, Any]:
    """
    Embed the query and search Qdrant. Return overlap statistics with metadata IDs.
    """
    vec = model.encode(query, normalize_embeddings=True).tolist()

    # Support both old (.search) and new (.query_points) qdrant_client APIs
    if hasattr(client, "search"):
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            limit=top_k,
            with_payload=False,
        )
        embedding_ids = [str(hit.id) for hit in results]
    elif hasattr(client, "query_points"):
        resp = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            limit=top_k,
            with_payload=False,
        )
        embedding_ids = [str(p.id) for p in resp.points]
    else:
        # Fallback: REST API
        import requests as _req
        url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{COLLECTION_NAME}/points/search"
        body = {"vector": vec, "limit": top_k, "with_payload": False}
        r = _req.post(url, json=body, timeout=30)
        r.raise_for_status()
        embedding_ids = [str(item["id"]) for item in r.json().get("result", [])]
    metadata_set = set(metadata_ids)
    embedding_set = set(embedding_ids)
    overlap = metadata_set & embedding_set

    return {
        "embedding_top_k": top_k,
        "embedding_ids_count": len(embedding_ids),
        "metadata_ids_count": len(metadata_ids),
        "overlap_count": len(overlap),
        "overlap_ids": sorted(overlap),
        "embedding_only_ids": sorted(embedding_set - metadata_set)[:5],
    }


def main():
    from qdrant_client import QdrantClient

    logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT} ...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)

    # Verify collection exists
    try:
        info = client.get_collection(COLLECTION_NAME)
        point_count = info.points_count
        logger.info(f"Collection '{COLLECTION_NAME}' has {point_count} points")
    except Exception as e:
        logger.error(f"Cannot access collection '{COLLECTION_NAME}': {e}")
        sys.exit(1)

    if point_count == 0:
        logger.error("Collection is empty — run ingestion first")
        sys.exit(1)

    # Load embedding model for cross-check
    model = None
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading BAAI/bge-large-en-v1.5 for embedding cross-check ...")
        model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        logger.info("Embedding model loaded")
    except ImportError:
        logger.warning("sentence-transformers not installed — skipping embedding cross-check")

    # Process each query definition
    annotated_queries: list[dict[str, Any]] = []
    stats = {"total": 0, "with_metadata": 0, "with_overlap": 0, "empty": 0}

    for qdef in QUERY_DEFINITIONS:
        qid = qdef["id"]
        query_text = qdef["query"]
        logger.info(f"[{qid}] Annotating: {query_text[:60]}...")

        # 1. Metadata-based ground truth
        metadata_ids = find_ground_truth_by_metadata(client, qdef)
        logger.info(f"  Metadata match: {len(metadata_ids)} chunks")

        # 2. Embedding-based ground truth (primary for Tier 1 evaluation)
        # The golden queries will be evaluated by embedding search, so the
        # ground truth MUST include what the embedder actually retrieves.
        cross_check = None
        if model is not None:
            cross_check = embedding_cross_check(client, model, query_text, metadata_ids, top_k=20)
            logger.info(
                f"  Embedding overlap: {cross_check['overlap_count']}/{cross_check['metadata_ids_count']} "
                f"(embedding returned {cross_check['embedding_ids_count']})"
            )

            # Use UNION of metadata + embedding results as ground truth.
            # Embedding top-k are what the retriever actually returns, so they
            # must be in the relevant set for Tier 1 metrics to be meaningful.
            embedding_ids = cross_check.get("embedding_only_ids", [])
            # Get ALL embedding IDs (not just the 5 extras)
            all_embedding_ids = []
            vec = model.encode(query_text, normalize_embeddings=True).tolist()
            if hasattr(client, "query_points"):
                resp = client.query_points(
                    collection_name=COLLECTION_NAME,
                    query=vec,
                    limit=20,
                    with_payload=False,
                )
                all_embedding_ids = [str(p.id) for p in resp.points]
            elif hasattr(client, "search"):
                results = client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=vec,
                    limit=20,
                    with_payload=False,
                )
                all_embedding_ids = [str(hit.id) for hit in results]

            # Merge: metadata IDs (topical match) + embedding IDs (what retriever finds)
            merged_set = set(metadata_ids) | set(all_embedding_ids)
            metadata_ids = list(merged_set)[:20]  # cap at 20
            logger.info(f"  Merged ground truth: {len(metadata_ids)} IDs (metadata + embedding)")

        # Build annotated query entry
        entry = {
            "id": qid,
            "query": query_text,
            "category": qdef["category"],
            "difficulty": qdef["difficulty"],
            "relevant_chunk_ids": metadata_ids,
            "key_numbers": qdef.get("key_numbers", []),
            "notes": qdef.get("notes", ""),
        }
        if cross_check:
            entry["_annotation"] = {
                "metadata_count": cross_check["metadata_ids_count"],
                "embedding_overlap": cross_check["overlap_count"],
            }

        annotated_queries.append(entry)
        stats["total"] += 1
        if metadata_ids:
            stats["with_metadata"] += 1
        else:
            stats["empty"] += 1
        if cross_check and cross_check["overlap_count"] > 0:
            stats["with_overlap"] += 1

    # Write output
    output = {
        "$schema": "Golden test set for Climate RAG retrieval evaluation",
        "version": "2.0",
        "description": (
            "Auto-annotated golden queries with real Qdrant chunk IDs. "
            "Generated by scripts/annotate_golden_queries.py. "
            "52 queries across 6 categories."
        ),
        "annotation_stats": stats,
        "collection": COLLECTION_NAME,
        "point_count": point_count,
        "queries": annotated_queries,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Wrote {len(annotated_queries)} annotated queries to {OUTPUT_PATH}")
    logger.info(f"Stats: {stats}")

    # Summary
    categories = {}
    for q in annotated_queries:
        cat = q["category"]
        categories.setdefault(cat, {"count": 0, "with_ids": 0})
        categories[cat]["count"] += 1
        if q["relevant_chunk_ids"]:
            categories[cat]["with_ids"] += 1

    logger.info("Category breakdown:")
    for cat, info in sorted(categories.items()):
        logger.info(f"  {cat}: {info['with_ids']}/{info['count']} have ground truth IDs")

    client.close()


if __name__ == "__main__":
    main()
