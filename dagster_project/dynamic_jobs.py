"""
Dynamic Source-Driven ETL Jobs for Phase 5

Complete pipeline: Download → Process → Generate Embeddings → Store in ChromaDB
"""

from dagster import job, op, Out, OpExecutionContext
from typing import Dict, Any, List
from datetime import datetime

from dagster_project.resources import ConfigLoaderResource, LoggerResource, DataPathResource
from src.embeddings.generator import EmbeddingGenerator
from src.utils.format_processing import (
    prepare_file_for_processing,
    generate_embeddings_for_file,
)


@op(
    description="Complete pipeline: download → process → embeddings → ChromaDB",
    out=Out(dagster_type=List[Dict[str, Any]]),
    tags={"phase": "5", "type": "complete_pipeline"},
    required_resource_keys={"logger", "data_paths"}
)
def process_all_sources(context: OpExecutionContext) -> List[Dict[str, Any]]:
    """Process all active sources with embeddings generation."""
    from src.sources import get_source_store
    from dagster_project.ops.dynamic_source_ops import detect_format_from_url
    import requests
    
    logger = context.resources.logger
    data_paths = context.resources.data_paths
    
    logger.info("=" * 80)
    logger.info("DYNAMIC SOURCE ETL - Complete Pipeline")
    logger.info("=" * 80)
    
    store = get_source_store()
    sources = store.get_all_sources(active_only=True)
    
    logger.info(f"Found {len(sources)} active source(s)")
    
    results = []
    
    embedding_generator = EmbeddingGenerator()

    for source in sources:
        source_id = source.source_id
        logger.info(f"\n{'='*70}")
        logger.info(f"SOURCE: {source_id}")
        logger.info(f"{'='*70}")
        
        try:
            store.update_processing_status(source_id, "processing")

            format = source.format or detect_format_from_url(source.url)
            logger.info(f"Format: {format} | URL: {source.url[:60]}...")

            # STEP 1: DOWNLOAD
            logger.info(f"\n[1/4] DOWNLOADING...")
            output_dir = data_paths.get_raw_path()
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = format.replace('netcdf', 'nc') if format else 'bin'
            filename = f"{source_id}_{timestamp}.{ext}"
            filepath = output_dir / filename

            response = requests.get(source.url, stream=True, timeout=120)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            file_size_mb = filepath.stat().st_size / 1024 / 1024
            logger.info(f"✓ Downloaded {file_size_mb:.2f} MB")

            # STEP 2: PROCESS
            logger.info(f"\n[2/4] PREPARING FILE...")
            prepared_file, canonical_format = prepare_file_for_processing(filepath, format, logger)

            processed_dir = data_paths.get_processed_path()
            processed_dir.mkdir(parents=True, exist_ok=True)

            # STEP 3: GENERATE EMBEDDINGS
            logger.info(f"\n[3/4] GENERATING EMBEDDINGS FOR {canonical_format.upper()}...")
            processing_result = generate_embeddings_for_file(
                embedding_generator,
                canonical_format,
                prepared_file,
                source_id,
                timestamp,
                processed_dir,
                logger,
            )

            embeddings_data = processing_result.embeddings

            if not embeddings_data:
                logger.warning(f"No embeddings generated for {source_id}; skipping DB upsert.")
                store.update_processing_status(source_id, "completed", error_message=None)
                results.append({
                    "source_id": source_id,
                    "status": "no_embeddings",
                    "format": canonical_format,
                    "message": "No numeric content detected for embedding generation.",
                })
                continue

            # STEP 4: STORE IN VECTOR DB
            logger.info(f"\n[4/4] STORING IN QDRANT...")
            from src.embeddings.database import VectorDatabase
            db = VectorDatabase()
            ids = [item['id'] for item in embeddings_data]
            embeddings = [item['embedding'] for item in embeddings_data]
            metadatas = [item['metadata'] for item in embeddings_data]
            documents = [item['metadata']['text'] for item in embeddings_data]

            db.add_embeddings(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )

            logger.info(f"✓ Stored {len(embeddings_data)} embeddings")

            result = {
                "source_id": source_id,
                "status": "success",
                "format": canonical_format,
                "raw_file": str(filepath),
                "processed_file": str(processing_result.processed_file) if processing_result.processed_file else None,
                "file_size_mb": file_size_mb,
                "embeddings_count": len(embeddings_data),
            }
            result.update(processing_result.artifacts)
            
            store.update_processing_status(source_id, "completed")
            logger.info(f"\n✓ COMPLETE: {source_id}")
            
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
