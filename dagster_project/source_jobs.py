"""
Single-source processing job for Dagster.

Reads source_id from run tags, loads from PostgreSQL, runs the ETL pipeline,
and records processing history.
"""

import os
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

    # Propagate the Dagster run id to downstream helpers (portal adapters,
    # chunk assembly) so every payload we write is tagged and can be atomically
    # swept after a successful re-ingest.
    from src.utils.ingestion_context import set_ingestion_run_id
    ingestion_run_id = context.run_id
    set_ingestion_run_id(ingestion_run_id)

    logger.info(f"Single source ETL: {source_id} (run_id={ingestion_run_id})")

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
        try:
            store.update_processing_status(source_id, "processing")
        except Exception as status_err:
            # Never let a DB hiccup here orphan the run before we've started
            # real work. The failure handler below also wraps updates so the
            # source can still be marked "failed" on downstream errors.
            logger.warning(f"Could not set processing status for {source_id}: {status_err}")

        # Import heavy dependencies lazily
        from src.climate_embeddings.loaders.raster_pipeline import load_raster_auto
        from src.climate_embeddings.loaders.detect_format import detect_format_from_url
        from src.climate_embeddings.embeddings.text_models import TextEmbedder
        from src.climate_embeddings.schema import ClimateChunkMetadata, generate_human_readable_text
        from src.embeddings.database import VectorDatabase
        from src.utils.config_loader import ConfigLoader
        import numpy as np
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry as Urllib3Retry
        import tempfile

        config = context.resources.config_loader.load()
        embedder = TextEmbedder()
        db = VectorDatabase(config=config)

        # Load per-source credentials from app_settings.json
        _settings_path = PROJECT_ROOT / "data" / "app_settings.json"
        source_creds = {}
        portal = getattr(source, "portal", None)
        if _settings_path.exists():
            import json as _json
            try:
                _app_settings = _json.loads(_settings_path.read_text())
                source_creds = _app_settings.get("source_credentials", {}).get(source_id, {})
                # Also inject global portal credentials into env
                for key, env_var in {
                    "cds_api_key": "CDS_API_KEY",
                    "nasa_earthdata_user": "NASA_EARTHDATA_USER",
                    "nasa_earthdata_password": "NASA_EARTHDATA_PASSWORD",
                    "cmems_username": "CMEMS_USERNAME",
                    "cmems_password": "CMEMS_PASSWORD",
                }.items():
                    val = _app_settings.get("credentials", {}).get(key) or _app_settings.get(key)
                    if val and not os.environ.get(env_var):
                        os.environ[env_var] = val
            except Exception:
                pass

        if not portal:
            portal = source_creds.get("portal")

        # Download
        format_hint = source.format or detect_format_from_url(source.url)
        ext_map = {'netcdf': '.nc', 'geotiff': '.tif', 'csv': '.csv', 'grib': '.grib', 'zip': '.zip'}
        ext = ext_map.get(format_hint, '.dat')

        tmp_path = None
        is_downloaded = False
        try:
            if not source.url.startswith(('http://', 'https://', 'file://')):
                # Local file
                local_path = Path(source.url)
                if not local_path.exists():
                    local_path = PROJECT_ROOT / source.url.lstrip('/')
                tmp_path = str(local_path)
            elif portal and portal.upper() in ("CDS", "NASA", "MARINE", "ESGF"):
                # Use portal adapter — it handles download + embed + upsert.
                # Adapter tags every chunk's payload with ingestion_run_id via
                # the ingestion_context ContextVar (see _process_file).
                _process_via_portal(
                    portal=portal.upper(),
                    source=source,
                    logger=logger,
                )
                # Verify the adapter actually wrote chunks for this run. Portal
                # adapters may return success even after partial/silent failures
                # (e.g. token refresh errors caught internally). Counting points
                # tagged with THIS run_id is the authoritative signal.
                portal_chunks = db.count_by_source(source_id, ingestion_run_id=ingestion_run_id)
                if portal_chunks == 0:
                    raise RuntimeError(
                        f"{portal} adapter reported success but wrote 0 chunks for "
                        f"{source_id} (run_id={ingestion_run_id}). Treating as failure."
                    )

                # Atomic versioned swap: new chunks are live, remove previous-run leftovers.
                try:
                    swept = db.delete_by_source(source_id, exclude_run_id=ingestion_run_id)
                    if swept and swept > 0:
                        logger.info(f"Swept {swept} stale chunks for {source_id}")
                except Exception as sweep_err:
                    # Sweep failure is non-fatal — new data is already live. Log loudly.
                    logger.error(f"Post-ingest sweep failed for {source_id}: {sweep_err}")

                duration = time.time() - start_time
                store.update_processing_status(source_id, "completed")
                if run_record_id:
                    try:
                        pg_store.complete_processing_run(
                            run_id=run_record_id, status="completed", chunks_processed=portal_chunks,
                        )
                    except Exception:
                        pass
                logger.info(
                    f"Completed {source_id} via {portal} adapter in {duration:.1f}s "
                    f"({portal_chunks} chunks)"
                )
                return {
                    "status": "completed",
                    "source_id": source_id,
                    "portal": portal,
                    "chunks": portal_chunks,
                    "duration": round(duration, 1),
                }
            else:
                # Direct HTTP download with auth support. Wrap the session in
                # try/finally so a failed download doesn't leak sockets — under
                # repeated user add-then-fail cycles the pool would otherwise
                # exhaust file descriptors.
                http_session = requests.Session()
                try:
                    _retry = Urllib3Retry(
                        total=3, backoff_factor=2,
                        status_forcelist=[502, 503, 504],
                        allowed_methods=["GET"], raise_on_status=False,
                    )
                    http_session.mount("https://", HTTPAdapter(max_retries=_retry))
                    http_session.mount("http://", HTTPAdapter(max_retries=_retry))
                    http_session.headers["User-Agent"] = "ClimateRAG/1.0"

                    # Apply auth credentials
                    auth_method = source_creds.get("auth_method") or getattr(source, "auth_method", None)
                    creds = source_creds.get("credentials", {})
                    if auth_method == "api_key" and creds.get("api_key"):
                        http_session.headers["X-API-Key"] = creds["api_key"]
                    elif auth_method == "bearer_token" and creds.get("token"):
                        http_session.headers["Authorization"] = f"Bearer {creds['token']}"
                    elif auth_method == "basic" and creds.get("username"):
                        http_session.auth = (creds["username"], creds.get("password", ""))

                    logger.info(f"Downloading {source.url}" + (f" (auth: {auth_method})" if auth_method else ""))
                    try:
                        resp = http_session.get(source.url, stream=True, timeout=(30, 600))
                        resp.raise_for_status()
                    except requests.exceptions.SSLError:
                        resp = http_session.get(source.url, stream=True, timeout=(30, 600), verify=False)
                        resp.raise_for_status()

                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        for chunk in resp.iter_content(chunk_size=65536):
                            tmp.write(chunk)
                        tmp_path = tmp.name
                    is_downloaded = True

                    logger.info(f"Downloaded to {tmp_path}")
                finally:
                    http_session.close()

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

                # Tag every chunk with the Dagster run id so we can atomically
                # sweep previous-run leftovers after this run succeeds.
                meta_dict["ingestion_run_id"] = ingestion_run_id

                # Inject user-provided keywords and custom metadata
                src_keywords = getattr(source, "keywords", None)
                src_custom = getattr(source, "custom_metadata", None)
                if src_keywords:
                    meta_dict["keywords"] = src_keywords
                if src_custom:
                    for k, v in src_custom.items():
                        if k not in meta_dict:
                            meta_dict[k] = v

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

            # Atomic versioned swap: new chunks are live and tagged with this
            # run's ingestion_run_id. Remove any leftovers from previous runs of
            # the same source (covers Reprocess and scheduled refresh; no-op on
            # first ingest). Failures here are non-fatal — new data is already
            # live — but we log them loudly.
            try:
                swept = db.delete_by_source(source_id, exclude_run_id=ingestion_run_id)
                if swept and swept > 0:
                    logger.info(f"Swept {swept} stale chunks for {source_id}")
            except Exception as sweep_err:
                logger.error(f"Post-ingest sweep failed for {source_id}: {sweep_err}")

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
            if tmp_path and is_downloaded:
                Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        logger.error(f"Failed {source_id}: {error_msg}")
        logger.error(traceback.format_exc())
        try:
            store.update_processing_status(source_id, "failed", error_message=error_msg)
        except Exception as status_err:
            # Last-ditch attempt to avoid leaving the source stuck in
            # "processing" — log loudly and carry on.
            logger.error(
                f"Could not mark {source_id} as failed after error ({error_msg}): "
                f"{status_err}"
            )

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


def _process_via_portal(portal: str, source, logger) -> int:
    """Download + embed via portal adapter. Returns chunk count."""
    from src.catalog.portal_adapters import (
        CDSAdapter, NASAAdapter, MarineCopernicusAdapter, ESGFAdapter,
    )

    adapter_map = {
        "CDS": CDSAdapter,
        "NASA": NASAAdapter,
        "MARINE": MarineCopernicusAdapter,
        "ESGF": ESGFAdapter,
    }

    adapter_cls = adapter_map.get(portal)
    if not adapter_cls:
        raise ValueError(f"Unknown portal: {portal}")

    adapter = adapter_cls()
    entry = _source_to_catalog_entry(source)
    logger.info(f"Using {portal} adapter for {source.source_id}")
    success = adapter.download_and_process(entry)
    if not success:
        raise RuntimeError(f"{portal} adapter failed for {source.source_id}")
    # Adapters log chunk counts but don't return them — return 1 as minimum
    return 1


def _source_to_catalog_entry(source):
    """Convert a Source object to a CatalogEntry-like object for portal adapters."""
    from dataclasses import dataclass, field
    from typing import Optional, List, Dict

    @dataclass
    class SourceEntry:
        row_index: int = 0
        hazard: str = ""
        dataset_name: str = ""
        data_type: str = ""
        spatial_coverage: str = ""
        spatial_resolution: str = ""
        region_country: str = ""
        temporal_coverage: str = ""
        temporal_resolution: str = ""
        bias_corrected: str = ""
        access: str = ""
        link: str = ""
        impact_sector: str = ""
        notes: str = ""
        keywords: Optional[List[str]] = None
        custom_metadata: Optional[Dict[str, str]] = None

        @property
        def source_id(self):
            return self.dataset_name.lower().replace(" ", "_") if self.dataset_name else "user_source"

    variables = getattr(source, "variables", None) or []
    time_range = getattr(source, "time_range", None) or {}

    return SourceEntry(
        dataset_name=getattr(source, "source_id", "") or "",
        hazard=getattr(source, "hazard_type", "") or "",
        link=getattr(source, "url", "") or "",
        region_country=getattr(source, "region_country", "") or "",
        spatial_coverage=getattr(source, "spatial_coverage", "") or "",
        impact_sector=getattr(source, "impact_sector", "") or "",
        data_type=getattr(source, "format", "") or "",
        temporal_coverage=f"{time_range.get('start', '')}/{time_range.get('end', '')}" if time_range else "",
        notes=", ".join(variables) if variables else "",
        keywords=getattr(source, "keywords", None),
        custom_metadata=getattr(source, "custom_metadata", None),
    )


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
