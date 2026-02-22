"""
Phase 0: Metadata-only pipeline.

Embeds Excel catalog metadata as Qdrant points — no data download required.
This gives RAG immediate awareness of all 234 climate data sources.
"""

import logging
import traceback
import uuid
from typing import List, Optional, Dict, Any

from src.catalog.excel_reader import CatalogEntry
from src.catalog.location_enricher import enrich_location

logger = logging.getLogger(__name__)


def _build_metadata_text(entry: CatalogEntry) -> str:
    """
    Build a rich text description from catalog metadata for embedding.

    Example output:
        "Climate dataset: ERA5. Hazard: Temperature. Type: Reanalysis data.
         Coverage: Global at 0.25°. Period: 1940-Present, hourly.
         Sectors: Agriculture, Energy. Region: Global."
    """
    parts = []

    if entry.dataset_name:
        parts.append(f"Climate dataset: {entry.dataset_name}")

    if entry.hazard:
        parts.append(f"Hazard: {entry.hazard}")

    if entry.data_type:
        parts.append(f"Type: {entry.data_type}")

    # Spatial info
    spatial_parts = []
    if entry.spatial_coverage:
        spatial_parts.append(entry.spatial_coverage)
    if entry.spatial_resolution:
        spatial_parts.append(f"at {entry.spatial_resolution}")
    if spatial_parts:
        parts.append(f"Coverage: {' '.join(spatial_parts)}")

    # Temporal info
    temporal_parts = []
    if entry.temporal_coverage:
        temporal_parts.append(entry.temporal_coverage)
    if entry.temporal_resolution:
        temporal_parts.append(entry.temporal_resolution)
    if temporal_parts:
        parts.append(f"Period: {', '.join(temporal_parts)}")

    if entry.impact_sector:
        parts.append(f"Sectors: {entry.impact_sector}")

    # Location (from enricher)
    location = enrich_location(entry)
    if location and location != "Unknown":
        parts.append(f"Region: {location}")

    if entry.bias_corrected and entry.bias_corrected.lower() == "yes":
        parts.append("Bias-corrected version available")

    if entry.notes:
        parts.append(f"Notes: {entry.notes}")

    return ". ".join(parts) + "."


def _build_payload(entry: CatalogEntry) -> Dict[str, Any]:
    """Build Qdrant payload from catalog entry."""
    location = enrich_location(entry)

    payload = {
        "source_id": entry.source_id,
        "dataset_name": entry.dataset_name or "unknown",
        "variable": f"catalog_{(entry.hazard or 'general').lower().replace(' ', '_')}",
        "is_metadata_only": True,
        "catalog_source": "D1.1.xlsx",
        "catalog_row_index": entry.row_index,
    }

    # Add all available metadata fields
    if entry.hazard:
        payload["hazard_type"] = entry.hazard
    if entry.data_type:
        payload["data_type"] = entry.data_type
    if entry.spatial_coverage:
        payload["spatial_coverage"] = entry.spatial_coverage
    if entry.region_country:
        payload["region_country"] = entry.region_country
    if entry.spatial_resolution:
        payload["spatial_resolution"] = entry.spatial_resolution
    if entry.temporal_coverage:
        payload["temporal_coverage_text"] = entry.temporal_coverage
    if entry.temporal_resolution:
        payload["temporal_frequency"] = entry.temporal_resolution
    if entry.bias_corrected:
        payload["bias_corrected"] = entry.bias_corrected
    if entry.access:
        payload["access_type"] = entry.access
    if entry.link:
        payload["link"] = entry.link
    if entry.impact_sector:
        payload["impact_sector"] = entry.impact_sector
    if entry.notes:
        payload["notes"] = entry.notes
    if location and location != "Unknown":
        payload["location_name"] = location

    return payload


def process_metadata_only(
    entry: CatalogEntry,
    text_embedder,
    vector_db,
) -> bool:
    """
    Process a single catalog entry as metadata-only (Phase 0).

    Creates a Qdrant point from Excel metadata without downloading any data.

    Args:
        entry: CatalogEntry from Excel
        text_embedder: TextEmbedder instance (BAAI/bge-large-en-v1.5)
        vector_db: VectorDatabase instance (Qdrant)

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Build text and embed
        text = _build_metadata_text(entry)
        embedding = text_embedder.embed_documents([text])[0]

        # Build payload
        payload = _build_payload(entry)

        # Generate deterministic ID
        point_id = entry.source_id

        # Store in Qdrant
        vector_db.add_embeddings(
            ids=[point_id],
            embeddings=[embedding.tolist()],
            metadatas=[payload],
        )

        # Auto-create source in shelve DB for Sources tab integration
        try:
            from src.sources import get_source_store
            store = get_source_store()
            if store.get_source(entry.source_id) is None:
                tags = [t for t in [entry.hazard, entry.data_type, entry.access] if t]
                # Use data_type from Excel catalog; fall back to URL extension guess
                fmt = entry.data_type if entry.data_type else None
                if not fmt and entry.link:
                    link_lower = entry.link.lower()
                    for ext in ["nc", "tif", "tiff", "grib", "csv", "zip"]:
                        if f".{ext}" in link_lower:
                            fmt = ext
                            break

                desc_parts = [entry.dataset_name or ""]
                if entry.hazard:
                    desc_parts.append(entry.hazard)
                if entry.data_type:
                    desc_parts.append(entry.data_type)

                store.create_source({
                    "source_id": entry.source_id,
                    "url": entry.link or "",
                    "format": fmt,
                    "variables": [entry.hazard] if entry.hazard else [],
                    "is_active": True,
                    "description": " — ".join(desc_parts),
                    "tags": tags + ["catalog:D1.1"],
                    "processing_status": "metadata_only",
                    "embedding_model": "BAAI/bge-large-en-v1.5",
                })
                logger.info(f"Created source entry for {entry.dataset_name} in SourceStore")
        except Exception as store_err:
            logger.warning(f"Could not create source for {entry.dataset_name}: {store_err}")

        logger.info(f"Stored metadata for {entry.dataset_name} (row {entry.row_index})")
        return True

    except Exception as e:
        logger.error(f"Failed to process metadata for {entry.dataset_name}: {e}\n{traceback.format_exc()}")
        return False


def process_metadata_batch(
    entries: List[CatalogEntry],
    text_embedder,
    vector_db,
    batch_size: int = 32,
) -> Dict[str, int]:
    """
    Process multiple catalog entries as metadata-only (Phase 0) in batches.

    Args:
        entries: List of CatalogEntry objects
        text_embedder: TextEmbedder instance
        vector_db: VectorDatabase instance
        batch_size: Number of entries to embed at once

    Returns:
        Dict with counts: {"processed": N, "failed": M, "total": T}
    """
    processed = 0
    failed = 0
    total = len(entries)
    succeeded_ids: List[str] = []
    failed_entries: List[tuple] = []  # (source_id, error_message)

    for i in range(0, total, batch_size):
        batch = entries[i : i + batch_size]

        texts = []
        ids = []
        payloads = []
        batch_entries = []

        for entry in batch:
            try:
                texts.append(_build_metadata_text(entry))
                ids.append(entry.source_id)
                payloads.append(_build_payload(entry))
                batch_entries.append(entry)
            except Exception as e:
                logger.error(f"Failed to prepare entry {entry.dataset_name}: {e}")
                failed += 1
                failed_entries.append((entry.source_id, str(e)))

        if not texts:
            continue

        try:
            # Batch embed
            embeddings = text_embedder.embed_documents(texts)

            # Store batch
            vector_db.add_embeddings(
                ids=ids,
                embeddings=[e.tolist() for e in embeddings],
                metadatas=payloads,
            )

            processed += len(texts)
            succeeded_ids.extend(ids)
            logger.info(f"Batch {i // batch_size + 1}: stored {len(texts)} entries ({processed}/{total})")

        except Exception as e:
            logger.error(f"Batch {i // batch_size + 1} failed: {e}")
            failed += len(texts)
            for entry in batch_entries:
                failed_entries.append((entry.source_id, str(e)))

    result = {
        "processed": processed,
        "failed": failed,
        "total": total,
        "succeeded_ids": succeeded_ids,
        "failed_entries": failed_entries,
    }
    logger.info(f"Phase 0 complete: processed={processed}, failed={failed}, total={total}")
    return result
