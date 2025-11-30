"""
Dynamic Source-Driven ETL Jobs for Phase 5

Complete pipeline: Download → Process → Generate Embeddings → Store in Qdrant
Uses climate_embeddings package for all format handling.
"""

from dagster import job, op, Out, OpExecutionContext
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path
import requests
import traceback

from dagster_project.resources import ConfigLoaderResource, LoggerResource, DataPathResource

# ==============================================================================
# IMPORTS FROM NEW CLIMATE_EMBEDDINGS PACKAGE
# ==============================================================================
from climate_embeddings.loaders.raster_pipeline import load_raster_auto, raster_to_embeddings
from climate_embeddings.loaders.detect_format import detect_format_from_url
from climate_embeddings.embeddings.text_models import TextEmbedder

# ==============================================================================
# IMPORTS FROM EXISTING SRC
# ==============================================================================
from src.embeddings.database import VectorDatabase
from src.sources import get_source_store


@op(
    description="Complete pipeline: download → process → embeddings → Qdrant (memory-safe)",
    out=Out(dagster_type=List[Dict[str, Any]]),
    tags={"phase": "5", "type": "complete_pipeline"},
    required_resource_keys={"logger", "data_paths"}
)
def process_all_sources(context: OpExecutionContext) -> List[Dict[str, Any]]:
    """
    Process all active sources:
    1. Download file
    2. Stream chunks via RasterPipeline
    3. Generate Statistical Embeddings (mean, max, etc.)
    4. Generate Semantic Embeddings (Text description of stats)
    5. Store in Qdrant
    """
    logger = context.resources.logger
    data_paths = context.resources.data_paths
    
    logger.info("=" * 80)
    logger.info("DYNAMIC SOURCE ETL - Complete Pipeline (Memory-Safe)")
    logger.info("=" * 80)
    
    # 1. Initialize Resources
    store = get_source_store()
    sources = store.get_all_sources(active_only=True)
    logger.info(f"Found {len(sources)} active source(s)")
    
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

            # --- STEP 1: DOWNLOAD ---
            format_hint = source.format or detect_format_from_url(source.url)
            logger.info(f"Format hint: {format_hint} | URL: {source.url[:60]}...")
            
            logger.info(f"[1/4] Downloading...")
            output_dir = data_paths.get_raw_path()
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Simple extension mapping
            ext_map = {'netcdf': 'nc', 'geotiff': 'tif', 'csv': 'csv', 'grib': 'grib', 'zarr': 'zarr', 'zip': 'zip'}
            ext = ext_map.get(format_hint, 'dat')
            filename = f"{source_id}_{timestamp}.{ext}"
            filepath = output_dir / filename

            # Stream download
            response = requests.get(source.url, stream=True, timeout=120)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size_mb = filepath.stat().st_size / 1024 / 1024
            logger.info(f"✓ Downloaded {file_size_mb:.2f} MB -> {filepath.name}")

            # --- STEP 2: LOAD & CHUNK ---
            logger.info(f"[2/4] Loading with Raster Pipeline...")
            
            # This uses the fixed, memory-safe logic
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
            
            # Converts raw pixels -> [mean, std, min, max...] vectors
            stat_embeddings = raster_to_embeddings(
                raster_result,
                normalization="zscore",
                spatial_pooling=True,
                seed=42,
            )
            
            if not stat_embeddings:
                logger.warning(f"No valid data found in {source_id}")
                store.update_processing_status(source_id, "completed") # technically success, just empty
                results.append({"source_id": source_id, "status": "empty"})
                continue
            
            logger.info(f"✓ Generated {len(stat_embeddings)} chunks")

            # --- STEP 4: SEMANTIC EMBEDDINGS & STORAGE ---
            logger.info(f"[4/4] Semantic Embedding & Storage...")

            # Create text descriptions for the LLM to search against
            text_descriptions = []
            valid_stats = []
            
            for stat_emb in stat_embeddings:
                meta = stat_emb["metadata"]
                vec = stat_emb["vector"]
                
                # Construct a sentence describing this chunk
                variable = meta.get("variable", "unknown_var")
                parts = [f"Climate variable: {variable}"]
                
                if "lat_min" in meta:
                    parts.append(f"Location: Lat {meta['lat_min']:.1f} to {meta['lat_max']:.1f}, Lon {meta['lon_min']:.1f} to {meta['lon_max']:.1f}")
                if "time_start" in meta:
                    parts.append(f"Time: {meta['time_start']}")
                
                # Add the stats to the text so the LLM "sees" the values
                # vec = [mean, std, min, max, ...]
                if len(vec) >= 4:
                    parts.append(f"Statistics: Mean={vec[0]:.2f}, Min={vec[2]:.2f}, Max={vec[3]:.2f}")
                
                text = " | ".join(parts)
                text_descriptions.append(text)
                valid_stats.append(stat_emb)

            # Generate vectors for these descriptions (Batch Process)
            # This uses the BGE/GTE model you configured
            semantic_vectors = text_embedder.embed_documents(text_descriptions)
            
            # Prepare for Qdrant
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for i, (desc, sem_vec, stat_item) in enumerate(zip(text_descriptions, semantic_vectors, valid_stats)):
                unique_id = f"{source_id}_{timestamp}_{i}"
                stats_vec = stat_item["vector"]
                
                ids.append(unique_id)
                embeddings.append(sem_vec.tolist())
                documents.append(desc)
                
                # Flatten metadata + stats for Qdrant payload
                # IMPORTANT: Convert numpy floats to native python floats for JSON serialization
                payload = {
                    **stat_item["metadata"],
                    "source_id": source_id,
                    "format": detected_format,
                    "timestamp": timestamp,
                    "stat_mean": float(stats_vec[0]) if len(stats_vec) > 0 else 0.0,
                    "stat_std": float(stats_vec[1]) if len(stats_vec) > 1 else 0.0,
                    "stat_min": float(stats_vec[2]) if len(stats_vec) > 2 else 0.0,
                    "stat_max": float(stats_vec[3]) if len(stats_vec) > 3 else 0.0,
                }
                metadatas.append(payload)

            # Batch Upsert
            vector_db.add_embeddings(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            logger.info(f"✓ Stored {len(ids)} vectors in Qdrant")

            # --- FINALIZE ---
            store.update_processing_status(source_id, "completed")
            result = {
                "source_id": source_id,
                "status": "success",
                "chunks": len(ids),
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
    description="Dynamic ETL: all active sources → download → process → embeddings → Qdrant",
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
    
    1. Reads sources from DB.
    2. Downloads data (NetCDF, GRIB, GeoTIFF, etc).
    3. Processes via memory-safe raster pipeline.
    4. Generates semantic embeddings using BGE-Large.
    5. Stores results in Qdrant for RAG.
    """
    process_all_sources()


# Make the job available to Dagster repository
__all__ = ["dynamic_source_etl_job"]