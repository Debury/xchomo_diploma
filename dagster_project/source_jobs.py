"""
Single-source processing job for Dagster.

Reads source_id from run tags, loads from PostgreSQL, runs the ETL pipeline,
and records processing history.
"""

import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from dagster import (
    job, op, Out, Output, OpExecutionContext, AssetMaterialization,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dagster_project.resources import ConfigLoaderResource, LoggerResource, DataPathResource


@op(
    description="Process a single source: download → load → embed → store",
    out=Out(dagster_type=Dict[str, Any]),
    tags={"type": "single_source_etl"},
    required_resource_keys={"logger", "data_paths", "config_loader"},
)
def process_single_source_op(context: OpExecutionContext) -> Dict[str, Any]:
    """
    Reads source_id from run tags, loads the source from the store,
    runs the processing pipeline, and records the run in PostgreSQL.
    """
    logger = context.resources.logger
    source_id = context.run.tags.get("source_id")

    if not source_id:
        logger.error("No source_id in run tags")
        return {"status": "error", "message": "No source_id provided"}

    logger.info(f"Single source ETL: {source_id}")

    # Load source
    from src.sources import get_source_store
    store = get_source_store()
    source = store.get_source(source_id)

    if not source:
        logger.error(f"Source {source_id} not found")
        return {"status": "error", "source_id": source_id, "message": "Source not found"}

    if not source.is_active:
        logger.warning(f"Source {source_id} is inactive, skipping")
        return {"status": "skipped", "source_id": source_id, "message": "Source inactive"}

    # Record processing run
    run_record_id = None
    trigger_type = context.run.tags.get("trigger_type", "manual")
    try:
        from src.database.source_store import SourceStore as PgStore
        pg_store = PgStore()
        run_record_id = pg_store.record_processing_run(
            source_id=source_id,
            job_name="single_source_etl_job",
            dagster_run_id=context.run_id,
            trigger_type=trigger_type,
        )
    except Exception as e:
        logger.warning(f"Could not record processing run: {e}")

    # Process
    start_time = time.time()
    try:
        store.update_processing_status(source_id, "processing")

        # Import heavy dependencies lazily
        from src.climate_embeddings.loaders.raster_pipeline import load_raster_auto
        from src.climate_embeddings.loaders.detect_format import detect_format_from_url
        from src.climate_embeddings.embeddings.text_models import TextEmbedder
        from src.climate_embeddings.schema import ClimateChunkMetadata, generate_human_readable_text
        from src.embeddings.database import VectorDatabase
        from src.utils.config_loader import ConfigLoader
        import numpy as np
        import requests
        import tempfile

        config = context.resources.config_loader.load()
        embedder = TextEmbedder()
        db = VectorDatabase(config=config)

        # Download
        format_hint = source.format or detect_format_from_url(source.url)
        ext_map = {'netcdf': '.nc', 'geotiff': '.tif', 'csv': '.csv', 'grib': '.grib', 'zip': '.zip'}
        ext = ext_map.get(format_hint, '.dat')

        tmp_path = None
        try:
            if source.url.startswith(('http://', 'https://')):
                logger.info(f"Downloading {source.url}")
                resp = requests.get(source.url, stream=True, timeout=(10, 600),
                                    headers={"User-Agent": "ClimateRAG/1.0"})
                resp.raise_for_status()

                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    for chunk in resp.iter_content(chunk_size=8192):
                        tmp.write(chunk)
                    tmp_path = tmp.name

                logger.info(f"Downloaded to {tmp_path}")
            else:
                # Local file
                local_path = Path(source.url.replace('file://', ''))
                if not local_path.exists():
                    local_path = PROJECT_ROOT / source.url.replace('file://', '').lstrip('/')
                tmp_path = str(local_path)

            # Load raster
            logger.info("Loading raster data...")
            raster_result = load_raster_auto(
                tmp_path,
                variables=source.variables,
                bbox=tuple(source.spatial_bbox) if source.spatial_bbox else None,
            )

            # Stream chunks
            BATCH_SIZE = 2000
            batch_ids, batch_texts, batch_metas = [], [], []
            total_chunks = 0
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            def _flush():
                if not batch_ids:
                    return
                vecs = embedder.embed_documents(batch_texts)
                db.add_embeddings(
                    ids=batch_ids,
                    embeddings=[v.tolist() for v in vecs],
                    metadatas=batch_metas,
                )
                batch_ids.clear()
                batch_texts.clear()
                batch_metas.clear()

            for chunk in raster_result.chunk_iterator:
                data = chunk.data
                valid = data[np.isfinite(data)]
                if valid.size == 0:
                    continue

                mn, mx = float(np.min(valid)), float(np.max(valid))
                stats = [
                    float(np.mean(valid)), float(np.std(valid)),
                    mn, mx,
                    float(np.percentile(valid, 10)),
                    float(np.percentile(valid, 50)),
                    float(np.percentile(valid, 90)),
                    mx - mn,
                ]

                meta = ClimateChunkMetadata.from_chunk_metadata(
                    raw_metadata=chunk.metadata,
                    stats_vector=stats,
                    source_id=source_id,
                    dataset_name=source_id,
                )
                meta_dict = meta.to_dict()
                text = generate_human_readable_text(meta_dict)

                batch_ids.append(f"{source_id}_{timestamp}_{total_chunks}")
                batch_texts.append(text)
                batch_metas.append(meta_dict)
                total_chunks += 1

                if len(batch_ids) >= BATCH_SIZE:
                    _flush()
                    logger.info(f"Processed {total_chunks} chunks...")

            _flush()

            if total_chunks == 0:
                raise ValueError("No data chunks produced")

            duration = time.time() - start_time
            store.update_processing_status(source_id, "completed")

            # Record completion
            if run_record_id:
                try:
                    pg_store.complete_processing_run(
                        run_id=run_record_id,
                        status="completed",
                        chunks_processed=total_chunks,
                    )
                except Exception:
                    pass

            context.log_event(AssetMaterialization(
                asset_key=f"source_{source_id}",
                metadata={
                    "source_id": source_id,
                    "chunks": total_chunks,
                    "duration_seconds": round(duration, 1),
                    "status": "completed",
                },
            ))

            logger.info(f"Completed {source_id}: {total_chunks} chunks in {duration:.1f}s")
            return {
                "status": "completed",
                "source_id": source_id,
                "chunks": total_chunks,
                "duration": round(duration, 1),
            }

        finally:
            # Clean up temp file if we downloaded it
            if tmp_path and source.url.startswith(('http://', 'https://')):
                Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        logger.error(f"Failed {source_id}: {error_msg}")
        logger.error(traceback.format_exc())
        store.update_processing_status(source_id, "failed", error_message=error_msg)

        if run_record_id:
            try:
                pg_store.complete_processing_run(
                    run_id=run_record_id,
                    status="failed",
                    error_message=error_msg,
                )
            except Exception:
                pass

        return {
            "status": "failed",
            "source_id": source_id,
            "error": error_msg,
            "duration": round(duration, 1),
        }


@job(
    resource_defs={
        "config_loader": ConfigLoaderResource(config_path="config/pipeline_config.yaml"),
        "logger": LoggerResource(log_file="logs/dagster_single_source.log", log_level="INFO"),
        "data_paths": DataPathResource(
            raw_data_dir="data/raw",
            processed_data_dir="data/processed",
            embeddings_dir="qdrant_db",
        ),
    },
    tags={"pipeline": "single_source_etl"},
    description="Process a single data source (source_id from run tags)",
)
def single_source_etl_job():
    process_single_source_op()
