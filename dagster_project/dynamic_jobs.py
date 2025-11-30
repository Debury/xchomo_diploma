"""
Dynamic Source-Driven ETL Jobs for Phase 5

Complete pipeline: Download → Process → Generate Embeddings → Store in Qdrant
Uses climate_embeddings package for all format handling.
"""

import sys
import time
from pathlib import Path
import requests
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

# --- PATH SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dagster import job, op, Out, OpExecutionContext
from dagster_project.resources import ConfigLoaderResource, LoggerResource, DataPathResource

# --- IMPORTS ---
from src.climate_embeddings.loaders.raster_pipeline import load_raster_auto, raster_to_embeddings
from src.climate_embeddings.loaders.detect_format import detect_format_from_url
from src.climate_embeddings.embeddings.text_models import TextEmbedder
from src.embeddings.database import VectorDatabase
from src.sources import get_source_store


@op(
    description="Complete pipeline: download → process → embeddings → Qdrant (memory-safe)",
    out=Out(dagster_type=List[Dict[str, Any]]),
    tags={"phase": "5", "type": "complete_pipeline"},
    required_resource_keys={"logger", "data_paths"}
)
def process_all_sources(context: OpExecutionContext) -> List[Dict[str, Any]]:
    logger = context.resources.logger
    data_paths = context.resources.data_paths
    
    logger.info("=" * 80)
    logger.info("DYNAMIC SOURCE ETL - Complete Pipeline")
    logger.info("=" * 80)
    
    # 1. Initialize Resources
    store = get_source_store()
    
    # --- FIX 1: Filter Logic ---
    # Check if a specific source_id was passed in the run tags
    target_source_id = context.run.tags.get("source_id")
    
    if target_source_id:
        logger.info(f"Targeting SINGLE source: {target_source_id}")
        # Get just the one source
        specific_source = store.get_source(target_source_id)
        if specific_source and specific_source.is_active:
            sources = [specific_source]
        else:
            logger.error(f"Source {target_source_id} not found or inactive.")
            return []
    else:
        # Fallback to processing EVERYTHING (e.g. scheduled runs)
        logger.info("Targeting ALL active sources")
        sources = store.get_all_sources(active_only=True)

    logger.info(f"Processing {len(sources)} source(s)")
    
    if not sources:
        return []

    # Initialize heavy models once
    logger.info("Initializing Text Embedder (BGE-Large)...")
    text_embedder = TextEmbedder()
    
    logger.info("Connecting to Vector Database...")
    vector_db = VectorDatabase()

    results = []

    # 2. Iterate Sources
    for source in sources:
        source_id = source.source_id
        logger.info(f"\n{'='*70}")
        logger.info(f"SOURCE: {source_id}")
        logger.info(f"{'='*70}")
        
        try:
            store.update_processing_status(source_id, "processing")

            # --- STEP 1: DOWNLOAD (With User-Agent Fix) ---
            format_hint = source.format or detect_format_from_url(source.url)
            output_dir = data_paths.get_raw_path()
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext_map = {'netcdf': 'nc', 'geotiff': 'tif', 'csv': 'csv', 'grib': 'grib', 'zarr': 'zarr', 'zip': 'zip'}
            ext = ext_map.get(format_hint, 'dat')
            filename = f"{source_id}_{timestamp}.{ext}"
            filepath = output_dir / filename

            # Fake User-Agent to prevent 403 Forbidden on data sites
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            if not filepath.exists():
                logger.info(f"[1/4] Downloading {source.url}...")
                
                with requests.get(source.url, headers=headers, stream=True, timeout=(10, 600)) as response:
                    response.raise_for_status()
                    
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded_size = 0
                    last_log_time = time.time()
                    
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                downloaded_size += len(chunk)
                                
                                # Heartbeat log
                                current_time = time.time()
                                if current_time - last_log_time > 5:
                                    mb_down = downloaded_size / 1024 / 1024
                                    logger.info(f"Downloading... {mb_down:.1f} MB")
                                    last_log_time = current_time
                                    
                logger.info(f"✓ Downloaded to {filepath.name}")
            else:
                logger.info(f"[1/4] File exists: {filepath.name}")

            # --- STEP 2: LOAD & CHUNK ---
            logger.info(f"[2/4] Loading with Raster Pipeline...")
            
            raster_result = load_raster_auto(
                filepath,
                chunks="auto",
                variables=source.variables if source.variables else None,
                bbox=tuple(source.spatial_bbox) if source.spatial_bbox and len(source.spatial_bbox) == 4 else None,
            )
            
            detected_format = raster_result.metadata.get("format", "unknown")
            logger.info(f"✓ Detected format: {detected_format}")

            # --- STEP 3: STATISTICAL EMBEDDINGS ---
            logger.info(f"[3/4] Generating Statistical Embeddings...")
            
            stat_embeddings = raster_to_embeddings(
                raster_result,
                normalization="zscore",
                spatial_pooling=True,
                seed=42,
            )
            
            if not stat_embeddings:
                logger.warning(f"No valid data found in {source_id}")
                store.update_processing_status(source_id, "failed", error_message="No numeric data found")
                results.append({"source_id": source_id, "status": "empty"})
                continue
            
            total_chunks = len(stat_embeddings)
            logger.info(f"✓ Generated {total_chunks} chunks")

            # --- STEP 4: BATCHED STORAGE ---
            logger.info(f"[4/4] Semantic Embedding & Storage...")

            BATCH_SIZE = 50 
            
            for i in range(0, total_chunks, BATCH_SIZE):
                batch_slice = stat_embeddings[i : i + BATCH_SIZE]
                if i % 100 == 0:
                    logger.info(f"Processing batch {i}/{total_chunks}...")
                
                # 1. Generate text
                batch_texts = []
                for item in batch_slice:
                    meta = item["metadata"]
                    vec = item["vector"]
                    variable = meta.get("variable", "unknown_var")
                    
                    parts = [f"Climate variable: {variable}"]
                    if "lat_min" in meta:
                        parts.append(f"Lat {meta['lat_min']:.1f} to {meta['lat_max']:.1f}")
                    if "time_start" in meta:
                        parts.append(f"Time: {meta['time_start']}")
                    if len(vec) >= 4:
                        parts.append(f"Stats: Mean={vec[0]:.2f}, Max={vec[3]:.2f}")
                    
                    batch_texts.append(" | ".join(parts))

                # 2. Embed
                batch_vectors = text_embedder.embed_documents(batch_texts)
                
                # 3. Upload
                ids, embeddings, metadatas, documents = [], [], [], []
                
                for j, (sem_vec, stat_item, desc) in enumerate(zip(batch_vectors, batch_slice, batch_texts)):
                    global_idx = i + j
                    uid = f"{source_id}_{timestamp}_{global_idx}"
                    
                    ids.append(uid)
                    embeddings.append(sem_vec.tolist())
                    documents.append(desc)
                    
                    stats_vec = stat_item["vector"]
                    payload = {
                        **stat_item["metadata"],
                        "source_id": source_id,
                        "format": detected_format,
                        "timestamp": timestamp,
                        "stat_mean": float(stats_vec[0]) if len(stats_vec) > 0 else 0.0,
                        "stat_max": float(stats_vec[3]) if len(stats_vec) > 3 else 0.0,
                        "stat_min": float(stats_vec[2]) if len(stats_vec) > 2 else 0.0,
                    }
                    metadatas.append(payload)

                vector_db.add_embeddings(ids, embeddings, metadatas, documents)

            logger.info(f"✓ All {total_chunks} stored successfully.")

            store.update_processing_status(source_id, "completed")
            result = {
                "source_id": source_id,
                "status": "success",
                "chunks": total_chunks,
                "file": str(filepath)
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"✗ FAILED {source_id}: {e}")
            logger.error(traceback.format_exc())
            store.update_processing_status(source_id, "failed", error_message=str(e))
            results.append({"source_id": source_id, "status": "failed", "error": str(e)})

    logger.info("=" * 80)
    return results


@job(
    resource_defs={
        "config_loader": ConfigLoaderResource(config_path="config/pipeline_config.yaml"),
        "logger": LoggerResource(log_file="logs/dagster_dynamic_etl.log", log_level="INFO"),
        "data_paths": DataPathResource(
            raw_data_dir="data/raw",
            processed_data_dir="data/processed",
            embeddings_dir="qdrant_db" 
        )
    },
    tags={"pipeline": "dynamic_etl", "phase": "5"}
)
def dynamic_source_etl_job():
    """
    🚀 Complete Dynamic Source ETL Pipeline
    """
    process_all_sources()


__all__ = ["dynamic_source_etl_job"]