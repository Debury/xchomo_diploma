"""
Phase 0: Metadata-only pipeline.

Embeds Excel catalog metadata as Qdrant points — no data download required.
This gives RAG immediate awareness of all 234 climate data sources.

Context enrichment is FULLY DYNAMIC — uses LLM to generate dataset descriptions
and synonym expansions. No hardcoded dataset dictionaries.
"""

import logging
import traceback
import uuid
from typing import List, Optional, Dict, Any

from src.catalog.excel_reader import CatalogEntry
from src.catalog.location_enricher import enrich_location

logger = logging.getLogger(__name__)

# Module-level cache: LLM-generated contexts keyed by "dataset::hazard"
_CONTEXT_CACHE: Dict[str, str] = {}


def _get_llm_client():
    """Get LLM client for context generation. Returns None if unavailable."""
    import os
    backend = os.getenv("LLM_BACKEND", "").lower().strip()
    if not backend:
        backend = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "ollama"
    try:
        if backend == "ollama":
            from src.llm.ollama_client import OllamaClient
            client = OllamaClient()
            return client if client.check_health() else None
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        from src.llm.openrouter_client import OpenRouterClient
        return OpenRouterClient()
    except Exception:
        return None


def _generate_dynamic_context(entry: CatalogEntry, llm_client=None) -> str:
    """
    Use LLM to generate a rich, searchable description from catalog fields.
    Cached per unique dataset_name to avoid redundant LLM calls.
    Returns empty string if LLM is unavailable or fails.
    """
    cache_key = entry.dataset_name or ''
    if cache_key in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[cache_key]

    if llm_client is None:
        return ""

    # Build a compact summary of all available fields for the LLM
    fields = []
    if entry.dataset_name:
        fields.append(f"Dataset name: {entry.dataset_name}")
    if entry.hazard:
        fields.append(f"Hazard/variable type: {entry.hazard}")
    if entry.data_type:
        fields.append(f"Data type: {entry.data_type}")
    if entry.spatial_coverage:
        fields.append(f"Spatial coverage: {entry.spatial_coverage}")
    if entry.spatial_resolution:
        fields.append(f"Resolution: {entry.spatial_resolution}")
    if entry.temporal_coverage:
        fields.append(f"Time coverage: {entry.temporal_coverage}")
    if entry.temporal_resolution:
        fields.append(f"Temporal resolution: {entry.temporal_resolution}")
    if entry.impact_sector:
        fields.append(f"Sectors: {entry.impact_sector}")
    if entry.region_country:
        fields.append(f"Region: {entry.region_country}")
    if entry.link:
        fields.append(f"URL: {entry.link}")
    if entry.notes:
        fields.append(f"Notes: {entry.notes}")

    metadata_summary = "\n".join(fields)

    prompt = f"""Given this climate dataset metadata, write a 3-4 sentence description for semantic search indexing.

METADATA:
{metadata_summary}

REQUIREMENTS:
1. Explain what the dataset measures and what it is used for
2. Include related scientific terms, synonyms, and common search phrases
3. Mention typical scientific applications and use cases
4. Do NOT invent specific data values or statistics
5. Write in plain English, suitable for embedding-based retrieval

DESCRIPTION:"""

    try:
        result = llm_client.generate(
            prompt=prompt,
            temperature=0.3,
            max_tokens=200,
            timeout_s=15,
        )
        context = result.strip()
        _CONTEXT_CACHE[cache_key] = context
        logger.debug(f"LLM context generated for {cache_key}: {context[:80]}...")
        return context
    except Exception as e:
        logger.warning(f"LLM context generation failed for {cache_key}: {e}")
        _CONTEXT_CACHE[cache_key] = ""
        return ""


def _build_metadata_text(entry: CatalogEntry, llm_client=None) -> str:
    """
    Build a rich text description from catalog metadata for embedding.
    Fully dynamic — uses LLM to expand context when available,
    falls back to structured text from Excel fields.
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

    # LLM-generated context enrichment (fully dynamic, cached)
    context = _generate_dynamic_context(entry, llm_client)
    if context:
        parts.append(context)

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
    llm_client=None,
) -> bool:
    """
    Process a single catalog entry as metadata-only (Phase 0).

    Creates a Qdrant point from Excel metadata without downloading any data.

    Args:
        entry: CatalogEntry from Excel
        text_embedder: TextEmbedder instance (BAAI/bge-large-en-v1.5)
        vector_db: VectorDatabase instance (Qdrant)
        llm_client: Optional LLM client for context enrichment

    Returns:
        True if successful, False otherwise.
    """
    try:
        # Build text and embed
        text = _build_metadata_text(entry, llm_client=llm_client)
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
    llm_client=None,
) -> Dict[str, int]:
    """
    Process multiple catalog entries as metadata-only (Phase 0) in batches.

    Args:
        entries: List of CatalogEntry objects
        text_embedder: TextEmbedder instance
        vector_db: VectorDatabase instance
        batch_size: Number of entries to embed at once
        llm_client: Optional LLM client for context enrichment

    Returns:
        Dict with counts: {"processed": N, "failed": M, "total": T}
    """
    # Try to get LLM client if not provided
    if llm_client is None:
        llm_client = _get_llm_client()
        if llm_client:
            logger.info("LLM client available — will generate dynamic context for metadata")
        else:
            logger.warning("No LLM client available — metadata will use basic text only")

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
                texts.append(_build_metadata_text(entry, llm_client=llm_client))
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

    logger.info(f"LLM context cache: {len(_CONTEXT_CACHE)} unique contexts generated")

    result = {
        "processed": processed,
        "failed": failed,
        "total": total,
        "succeeded_ids": succeeded_ids,
        "failed_entries": failed_entries,
    }
    logger.info(f"Phase 0 complete: processed={processed}, failed={failed}, total={total}")
    return result
