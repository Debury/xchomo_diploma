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
import numpy as np

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
    required_resource_keys={"logger", "data_paths", "config_loader"}
)
def process_all_sources(context: OpExecutionContext) -> List[Dict[str, Any]]:
    logger = context.resources.logger
    data_paths = context.resources.data_paths
    config_loader = context.resources.config_loader
    
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
    # Initialize VectorDatabase with config for proper vector size
    pipeline_config = config_loader.load()
    vector_db = VectorDatabase(config=pipeline_config)
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
                
                # Use the new text generation module for consistent, configurable descriptions
                from src.climate_embeddings.text_generation import generate_batch_descriptions
                
                batch_metadata = [item["metadata"] for item in batch_slice]
                batch_stats = [item["vector"] for item in batch_slice]
                
                # Add source_id to metadata if not present
                for meta in batch_metadata:
                    if "source_id" not in meta:
                        meta["source_id"] = source_id
                
                # Generate normalized metadata using schema
                from src.climate_embeddings.schema import ClimateChunkMetadata, generate_human_readable_text
                
                ids, embeddings, metadatas, texts_for_embedding = [], [], [], []
                
                for j, stat_item in enumerate(batch_slice):
                    uid = f"{source_id}_{timestamp}_{i+j}"
                    ids.append(uid)
                    
                    stats_vector = stat_item["vector"]
                    raw_meta = stat_item["metadata"]
                    
                    # Create normalized metadata
                    normalized_meta = ClimateChunkMetadata.from_chunk_metadata(
                        raw_metadata=raw_meta,
                        stats_vector=stats_vector,
                        source_id=source_id,
                        dataset_name=source_id  # Can be customized per source
                    )
                    
                    # Convert to dict for storage
                    payload = normalized_meta.to_dict()
                    metadatas.append(payload)
                    
                    # Generate text ONLY for embedding (not stored in DB)
                    text_for_embedding = generate_human_readable_text(payload)
                    texts_for_embedding.append(text_for_embedding)
                
                # Generate embeddings from human-readable text
                batch_vectors = text_embedder.embed_documents(texts_for_embedding)
                
                # Convert to list format
                embeddings = [vec.tolist() for vec in batch_vectors]
                
                # Store in vector DB (without text_content - it's generated dynamically)
                vector_db.add_embeddings(ids, embeddings, metadatas, [])  # Empty documents list

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