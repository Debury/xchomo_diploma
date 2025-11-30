"""
Dynamic Source-Driven ETL Jobs for Phase 5
"""

import sys
import time
from pathlib import Path
import requests
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np  # <--- Added missing import

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
    
    store = get_source_store()
    target_source_id = context.run.tags.get("source_id")
    if target_source_id:
        specific_source = store.get_source(target_source_id)
        if specific_source and specific_source.is_active:
            sources = [specific_source]
        else:
            logger.error(f"Source {target_source_id} not found/inactive.")
            return []
    else:
        sources = store.get_all_sources(active_only=True)

    if not sources: return []

    logger.info("Initializing Models...")
    text_embedder = TextEmbedder()
    vector_db = VectorDatabase()
    results = []

    for source in sources:
        source_id = source.source_id
        logger.info(f"\nSOURCE: {source_id}")
        
        try:
            store.update_processing_status(source_id, "processing")

            # --- 1. DOWNLOAD ---
            format_hint = source.format or detect_format_from_url(source.url)
            output_dir = data_paths.get_raw_path().resolve()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext_map = {'netcdf': 'nc', 'geotiff': 'tif', 'csv': 'csv', 'grib': 'grib', 'zip': 'zip'}
            ext = ext_map.get(format_hint, 'dat')
            filepath = output_dir / f"{source_id}_{timestamp}.{ext}"
            headers = {"User-Agent": "Mozilla/5.0"}

            if not filepath.exists():
                logger.info(f"Downloading {source.url}...")
                with requests.get(source.url, headers=headers, stream=True, timeout=(10, 600)) as response:
                    response.raise_for_status()
                    ctype = response.headers.get('content-type', '').lower()
                    if 'html' in ctype: raise Exception(f"Invalid content type: {ctype}")
                    
                    downloaded = 0
                    last_log = time.time()
                    with open(filepath, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if time.time() - last_log > 5:
                                logger.info(f"Downloading... {downloaded / 1024 / 1024:.1f} MB")
                                last_log = time.time()
                                
                if filepath.stat().st_size < 500: raise Exception("File too small")
                logger.info(f"✓ Downloaded {filepath.name}")
            
            # --- 2. LOAD & STATS ---
            logger.info("Loading & Calculating Stats...")
            raster_result = load_raster_auto(
                filepath,
                chunks="auto",
                variables=source.variables,
                bbox=tuple(source.spatial_bbox) if source.spatial_bbox else None,
            )
            
            stat_embeddings = raster_to_embeddings(
                raster_result,
                normalization="zscore",
                spatial_pooling=True,
                seed=42,
            )
            
            total_chunks = len(stat_embeddings)
            if not stat_embeddings:
                store.update_processing_status(source_id, "failed", error_message="No numeric data")
                continue
            
            logger.info(f"✓ Generated {total_chunks} chunks")

            # --- 3. ENRICHMENT & STORAGE ---
            logger.info("Semantic Embedding & Storage...")
            BATCH_SIZE = 50 
            
            for i in range(0, total_chunks, BATCH_SIZE):
                batch_slice = stat_embeddings[i : i + BATCH_SIZE]
                if i % 100 == 0: logger.info(f"Processing {i}/{total_chunks}...")
                
                batch_texts = []
                for item in batch_slice:
                    meta = item["metadata"]
                    vec = item["vector"] # [Mean, Std, Min, Max, P10, Median, P90, Range]
                    variable = meta.get("variable", "unknown")
                    
                    parts = [f"Climate variable: {variable}"]
                    
                    # --- NEW: Smart Date Handling ---
                    if "time_start" in meta:
                        t_str = str(meta['time_start'])
                        parts.append(f"Time: {t_str}")
                        # Extract Month Name for better RAG
                        try:
                            # Handle typical numpy/iso formats
                            clean_date = t_str.split('T')[0]
                            dt_obj = datetime.strptime(clean_date, '%Y-%m-%d')
                            month_name = dt_obj.strftime('%B')
                            parts.append(f"Month: {month_name}")
                        except: pass

                    if "lat_min" in meta: parts.append(f"Lat {meta['lat_min']:.1f}")
                    
                    # --- NEW: Rich Stats for LLM ---
                    if len(vec) >= 8: 
                        parts.append(f"Mean={vec[0]:.1f}")
                        parts.append(f"Std={vec[1]:.1f}") # Added Standard Deviation
                        parts.append(f"Max={vec[3]:.1f}")
                        parts.append(f"P90={vec[6]:.1f} (High Extreme)") # 90th percentile
                        parts.append(f"Range={vec[7]:.1f}")             # Variability
                    
                    batch_texts.append(" | ".join(parts))

                batch_vectors = text_embedder.embed_documents(batch_texts)
                
                ids, embeddings, metadatas, documents = [], [], [], []
                
                for j, (sem_vec, stat_item, desc) in enumerate(zip(batch_vectors, batch_slice, batch_texts)):
                    uid = f"{source_id}_{timestamp}_{i+j}"
                    ids.append(uid)
                    embeddings.append(sem_vec.tolist())
                    documents.append(desc)
                    
                    v = stat_item["vector"]
                    meta = stat_item["metadata"]
                    meta_clean = {k: (float(v) if isinstance(v, (np.float32, np.float64)) else v) for k,v in meta.items()}
                    
                    # Store rich payload for filtering
                    payload = {
                        **meta_clean,
                        "source_id": source_id,
                        "timestamp": timestamp,
                        "stat_mean": float(v[0]),
                        "stat_std": float(v[1]), # Added Standard Deviation
                        "stat_max": float(v[3]),
                        "stat_p90": float(v[6]),  # Storing for sorting
                        "stat_range": float(v[7]) # Storing for filtering
                    }
                    metadatas.append(payload)

                vector_db.add_embeddings(ids, embeddings, metadatas, documents)

            logger.info(f"✓ Stored {total_chunks} vectors.")
            store.update_processing_status(source_id, "completed")
            results.append({"source_id": source_id, "status": "success"})
            
        except Exception as e:
            logger.error(f"✗ FAILED {source_id}: {e}")
            logger.error(traceback.format_exc())
            store.update_processing_status(source_id, "failed", error_message=str(e))

    return results

@job(
    resource_defs={
        "config_loader": ConfigLoaderResource(config_path="config/pipeline_config.yaml"),
        "logger": LoggerResource(log_file="logs/dagster_dynamic_etl.log", log_level="INFO"),
        "data_paths": DataPathResource(raw_data_dir="data/raw", processed_data_dir="data/processed", embeddings_dir="qdrant_db")
    },
    tags={"pipeline": "dynamic_etl", "phase": "5"}
)
def dynamic_source_etl_job():
    process_all_sources()

__all__ = ["dynamic_source_etl_job"]