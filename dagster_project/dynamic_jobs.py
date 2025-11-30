"""
Dynamic Source-Driven ETL Jobs for Phase 5

Complete pipeline: Download → Process → Generate Embeddings → Store in Qdrant
Uses climate_embeddings package for all format handling.
"""

from dagster import job, op, Out, OpExecutionContext
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

from dagster_project.resources import ConfigLoaderResource, LoggerResource, DataPathResource

# NEW: Use climate_embeddings package
from climate_embeddings.loaders import load_raster_auto, raster_to_embeddings
from climate_embeddings.loaders.detect_format import detect_format

# Keep Qdrant integration
from src.embeddings.database import VectorDatabase
from src.embeddings.generator import EmbeddingGenerator


@op(
    description="Complete pipeline: download → process → embeddings → Qdrant (memory-safe)",
    out=Out(dagster_type=List[Dict[str, Any]]),
    tags={"phase": "5", "type": "complete_pipeline"},
    required_resource_keys={"logger", "data_paths"}
)
def process_all_sources(context: OpExecutionContext) -> List[Dict[str, Any]]:
    """Process all active sources with memory-safe raster pipeline."""
    from src.sources import get_source_store
    from dagster_project.ops.dynamic_source_ops import detect_format_from_url
    import requests
    
    logger = context.resources.logger
    data_paths = context.resources.data_paths
    
    logger.info("=" * 80)
    logger.info("DYNAMIC SOURCE ETL - Complete Pipeline (Memory-Safe)")
    logger.info("=" * 80)
    
    store = get_source_store()
    sources = store.get_all_sources(active_only=True)
    
    logger.info(f"Found {len(sources)} active source(s)")
    
    results = []
    embedding_generator = EmbeddingGenerator()
    vector_db = VectorDatabase()

    for source in sources:
        source_id = source.source_id
        logger.info(f"\n{'='*70}")
        logger.info(f"SOURCE: {source_id}")
        logger.info(f"{'='*70}")
        
        try:
            store.update_processing_status(source_id, "processing")

            format_hint = source.format or detect_format_from_url(source.url)
            logger.info(f"Format hint: {format_hint} | URL: {source.url[:60]}...")

            # STEP 1: DOWNLOAD
            logger.info(f"\n[1/3] DOWNLOADING...")
            output_dir = data_paths.get_raw_path()
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext_map = {'netcdf': 'nc', 'geotiff': 'tif', 'csv': 'csv', 'grib': 'grib'}
            ext = ext_map.get(format_hint, 'dat')
            filename = f"{source_id}_{timestamp}.{ext}"
            filepath = output_dir / filename

            response = requests.get(source.url, stream=True, timeout=120)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size_mb = filepath.stat().st_size / 1024 / 1024
            logger.info(f"✓ Downloaded {file_size_mb:.2f} MB -> {filepath.name}")

            # STEP 2: LOAD WITH CHUNKING (memory-safe)
            logger.info(f"\n[2/3] LOADING WITH RASTER PIPELINE (chunked)...")
            
            raster_result = load_raster_auto(
                filepath,
                chunks="auto",
                variables=source.variables if source.variables else None,
                bbox=tuple(source.spatial_bbox) if source.spatial_bbox and len(source.spatial_bbox) == 4 else None,
            )
            
            detected_format = raster_result.metadata.get("format", "unknown")
            logger.info(f"✓ Detected format: {detected_format}")
            logger.info(f"  Dataset: {raster_result.dataset is not None}")
            logger.info(f"  Iterator: {raster_result.chunk_iterator is not None}")

            # Generate stat-based embeddings from chunks
            logger.info(f"\n[3/3] GENERATING EMBEDDINGS (streaming)...")
            
            stat_embeddings = raster_to_embeddings(
                raster_result,
                normalization="zscore",
                spatial_pooling=True,
                seed=42,
            )
            
            if not stat_embeddings:
                logger.warning(f"No embeddings generated for {source_id}")
                store.update_processing_status(source_id, "completed")
                results.append({
                    "source_id": source_id,
                    "status": "no_embeddings",
                    "format": detected_format,
                })
                continue
            
            logger.info(f"✓ Generated {len(stat_embeddings)} stat-based embeddings")

            # Convert to text embeddings for semantic search
            text_embeddings_data = []
            for idx, stat_emb in enumerate(stat_embeddings):
                meta = stat_emb["metadata"]
                variable = meta.get("variable", "data")
                
                # Build descriptive text
                text_parts = [f"Variable: {variable}"]
                if "lat_min" in meta and "lat_max" in meta:
                    text_parts.append(f"Latitude: {meta['lat_min']:.2f} to {meta['lat_max']:.2f}")
                if "lon_min" in meta and "lon_max" in meta:
                    text_parts.append(f"Longitude: {meta['lon_min']:.2f} to {meta['lon_max']:.2f}")
                if "time_start" in meta:
                    text_parts.append(f"Time: {meta['time_start']}")
                
                # Add stats from vector (mean, std, min, max, p10, p50, p90)
                stats_vec = stat_emb["vector"]
                if len(stats_vec) >= 4:
                    text_parts.append(f"Stats: mean={stats_vec[0]:.3f}, std={stats_vec[1]:.3f}, min={stats_vec[2]:.3f}, max={stats_vec[3]:.3f}")
                
                text = " | ".join(text_parts)
                
                # Generate semantic embedding from text
                semantic_emb = embedding_generator.generate_embeddings([text])[0]
                
                text_embeddings_data.append({
                    "id": f"{source_id}_{timestamp}_{idx}",
                    "embedding": semantic_emb.tolist(),
                    "text": text,
                    "metadata": {
                        **meta,
                        "source_id": source_id,
                        "format": detected_format,
                        "timestamp": timestamp,
                        "stat_mean": float(stats_vec[0]) if len(stats_vec) > 0 else None,
                        "stat_std": float(stats_vec[1]) if len(stats_vec) > 1 else None,
                        "stat_min": float(stats_vec[2]) if len(stats_vec) > 2 else None,
                        "stat_max": float(stats_vec[3]) if len(stats_vec) > 3 else None,
                    }
                })

            # STEP 4: STORE IN QDRANT
            logger.info(f"\n[4/4] STORING IN QDRANT...")
            ids = [item['id'] for item in text_embeddings_data]
            embeddings = [item['embedding'] for item in text_embeddings_data]
            metadatas = [item['metadata'] for item in text_embeddings_data]
            documents = [item['text'] for item in text_embeddings_data]

            vector_db.add_embeddings(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )

            logger.info(f"✓ Stored {len(text_embeddings_data)} embeddings in Qdrant")

            result = {
                "source_id": source_id,
                "status": "success",
                "format": detected_format,
                "raw_file": str(filepath),
                "file_size_mb": file_size_mb,
                "embeddings_count": len(text_embeddings_data),
            }
            
            store.update_processing_status(source_id, "completed")
            logger.info(f"\n✓ COMPLETE: {source_id}")
            results.append(result)
            
        except Exception as e:
            logger.error(f"\n✗ FAILED: {source_id} - {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            store.update_processing_status(source_id, "failed", error_message=str(e))
            result = {
                "source_id": source_id,
                "status": "error",
                "error": str(e)
            }
        
        results.append(result)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"PIPELINE COMPLETE: {len(results)} sources processed")
    logger.info("=" * 80)
    
    return results


@job(
    description="Dynamic ETL: all active sources → download → process → embeddings → ChromaDB",
    resource_defs={
        "config_loader": ConfigLoaderResource(config_path="config/pipeline_config.yaml"),
        "logger": LoggerResource(log_file="logs/dagster_dynamic_etl.log", log_level="INFO"),
        "data_paths": DataPathResource(
            raw_data_dir="data/raw",
            processed_data_dir="data/processed",
            embeddings_dir="chroma_db"
        )
    },
    tags={"pipeline": "dynamic_etl", "phase": "5"}
)
def dynamic_source_etl_job():
    """
    🚀 Complete Dynamic Source ETL Pipeline
    
    For each active source:
    1. 📥 Download from URL
    2. ⚙️ Process & transform data
    3. 🧠 Generate embeddings
    4. 💾 Store in ChromaDB
    
    Usage:
        POST /sources/{source_id}/trigger
    """
    process_all_sources()


__all__ = ["dynamic_source_etl_job"]
