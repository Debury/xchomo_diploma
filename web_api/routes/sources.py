"""Source CRUD and lifecycle endpoints."""

import os
import logging
import re
import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from web_api.models import SourceCreate, SourceResponse, SourceScheduleRequest, RunResponse
from web_api.config import load_settings, save_settings
from web_api.dependencies import launch_dagster_run, get_qdrant_client, get_collection_name, clear_rag_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sources", tags=["sources"])

# File-upload root — bind-mounted at `./data` in docker-compose, so uploaded
# files are visible both to web-api and to the Dagster workers.
_UPLOAD_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "uploads"


def _upload_max_bytes() -> int:
    """Resolve the per-upload size cap from env at call time.

    Driven by `UPLOAD_MAX_MB` (default 5000 MB = 5 GB). Reading at call time
    instead of import time means operators can bump the cap by editing `.env`
    and restarting the container, without a rebuild. Silently falls back to
    the default if the value is unparseable or non-positive — climate datasets
    legitimately hit multi-GB territory, so misconfiguration should err on
    the side of "accept" rather than "reject with a cryptic error".
    """
    raw = os.getenv("UPLOAD_MAX_MB", "5000")
    try:
        mb = int(raw)
    except ValueError:
        logger.warning(f"UPLOAD_MAX_MB='{raw}' is not an integer; using default 5000 MB")
        return 5000 * 1024 * 1024
    if mb <= 0:
        logger.warning(f"UPLOAD_MAX_MB={mb} is non-positive; using default 5000 MB")
        return 5000 * 1024 * 1024
    return mb * 1024 * 1024
_ALLOWED_UPLOAD_EXT = {
    ".nc", ".nc4", ".cdf",
    ".hdf", ".hdf5", ".h5", ".he5",
    ".tif", ".tiff",
    ".grib", ".grib2", ".grb", ".grb2",
    ".csv", ".tsv", ".txt",
    ".zip", ".gz", ".tar",
    ".zarr",
    ".parquet",
}
# Whitelist the safe portion of the user-provided filename so a malicious
# upload can't escape _UPLOAD_ROOT via path traversal or shell metachars.
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

# Portal → global credential keys that must be present before the ETL job can
# be launched. Kept in sync with the frontend PORTAL_CREDENTIAL_KEYS map
# (views/CreateSource.vue). Empty list means the portal is fully public.
PORTAL_REQUIRED_CREDS = {
    "CDS": ["cds_api_key"],
    "NASA": ["nasa_earthdata_user", "nasa_earthdata_password"],
    "MARINE": ["cmems_username", "cmems_password"],
    "ESGF": [],
    "NOAA": [],
}


def _missing_portal_credentials(portal: Optional[str], per_source_creds: Optional[dict]) -> List[str]:
    """Return credential keys a portal source needs but doesn't have.

    Checks per-source credentials first, then global credentials in
    app_settings.json. Returns an empty list if the portal is unknown
    (treated as public) or fully configured.
    """
    if not portal:
        return []
    required = PORTAL_REQUIRED_CREDS.get(portal.upper())
    if not required:
        return []
    # Per-source auth satisfies everything: a user who pasted a bearer token
    # or api_key directly into the form should not be blocked by global creds.
    if per_source_creds:
        return []
    global_creds = load_settings().get("credentials", {}) or {}
    return [k for k in required if not global_creds.get(k)]


@router.post("/", response_model=SourceResponse, status_code=201)
async def create_source(source: SourceCreate):
    """Create a new data source."""
    if not (source.source_id or "").strip():
        raise HTTPException(400, "source_id is required")
    if not (source.url or "").strip():
        raise HTTPException(400, "url is required")

    try:
        from src.sources import get_source_store

        store = get_source_store()

        if store.get_source(source.source_id):
            raise HTTPException(409, f"Source '{source.source_id}' already exists")

        if not source.format:
            from src.climate_embeddings.loaders.detect_format import detect_format_from_url

            source.format = detect_format_from_url(source.url)

        auth_method = source.auth_method
        auth_credentials = source.auth_credentials
        portal = source.portal

        source_data = source.dict()
        source_data.pop("auth_credentials", None)

        new_source = store.create_source(source_data)
        source_dict = new_source.to_dict()

        # Store auth credentials separately
        if auth_credentials and auth_method and auth_method != "none":
            persisted = load_settings()
            source_creds = persisted.get("source_credentials", {})
            source_creds[source.source_id] = {
                "auth_method": auth_method,
                "credentials": auth_credentials,
                "portal": portal,
            }
            persisted["source_credentials"] = source_creds
            save_settings(persisted)

        # Set up schedule if requested
        if source.schedule_cron:
            try:
                from src.database.source_store import SourceStore as PgStore

                pg_store = PgStore()
                pg_store.set_schedule(source.source_id, source.schedule_cron)
            except Exception as sched_err:
                logger.warning(f"Could not set schedule: {sched_err}")

        source_dict["auth_method"] = auth_method
        source_dict["portal"] = portal

        # Auto-trigger ETL job if requested
        if source.auto_embed:
            # 1. Validate portal credentials up front. Running the job only to
            #    have it fail 30s later with a cryptic portal error is a known
            #    bad UX — prefer a clear 422-style signal persisted on the row.
            missing = _missing_portal_credentials(portal, auth_credentials)
            if missing:
                msg = (
                    f"Cannot start ETL: {portal} source requires credentials "
                    f"that are not configured: {', '.join(missing)}. "
                    f"Add them on the Settings page, or set per-source "
                    f"credentials when creating the source."
                )
                logger.warning(f"Blocking auto-embed for {source.source_id}: {msg}")
                try:
                    store.update_processing_status(
                        source.source_id, "failed", error_message=msg
                    )
                except Exception as status_err:
                    logger.error(
                        f"Could not persist credential-missing error for "
                        f"{source.source_id}: {status_err}"
                    )
                source_dict["etl_error"] = msg
                source_dict["processing_status"] = "failed"
                source_dict["error_message"] = msg
                return source_dict

            # 2. Launch Dagster run. If the launch itself fails (daemon down,
            #    queue-full, graphql error), persist the error so it survives a
            #    page refresh — previously it was only echoed in this response.
            try:
                run = await launch_dagster_run(
                    "single_source_etl_job",
                    {},
                    tags={"source_id": source.source_id, "trigger_type": "auto_embed"},
                )
                store.update_processing_status(source.source_id, "processing")
                source_dict["etl_run_id"] = run.get("runId")
                source_dict["processing_status"] = "processing"
                logger.info(f"Auto-embed triggered for {source.source_id}: {run.get('runId')}")
            except Exception as etl_err:
                err_text = f"Auto-embed trigger failed: {etl_err}"
                logger.warning(f"{err_text} (source={source.source_id})")
                try:
                    store.update_processing_status(
                        source.source_id, "failed", error_message=err_text
                    )
                except Exception as status_err:
                    logger.error(
                        f"Could not persist trigger failure for {source.source_id}: "
                        f"{status_err}"
                    )
                source_dict["etl_error"] = err_text
                source_dict["processing_status"] = "failed"
                source_dict["error_message"] = err_text

        return source_dict
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create source {source.source_id}: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@router.get("/")
async def list_sources(active_only: bool = False):
    """List all sources. Qdrant is the truth for what data exists,
    PostgreSQL provides management metadata (schedules, auth, processing history)."""

    # 1. Get all datasets from Qdrant (the truth)
    qdrant_datasets = []
    try:
        from web_api.routes.qdrant_datasets import list_qdrant_datasets
        qdrant_datasets = await list_qdrant_datasets()
    except Exception as e:
        logger.warning(f"Could not fetch Qdrant datasets: {e}")

    # 2. Get PostgreSQL management data
    db_sources = {}
    try:
        from src.sources import get_source_store
        for s in get_source_store().get_all_sources(active_only=False):
            d = s.to_dict()
            db_sources[d["source_id"]] = d
    except Exception:
        pass

    schedules_map = {}
    try:
        from src.database.connection import get_db_session
        from src.database.models import SourceSchedule
        with get_db_session() as session:
            for sched in session.query(SourceSchedule).all():
                schedules_map[sched.source_id] = sched.to_dict()
    except Exception:
        pass

    persisted = load_settings()
    source_creds = persisted.get("source_credentials", {})

    # 3. Build unified list: Qdrant datasets enriched with DB management info
    seen_source_ids = set()
    results = []

    for qd in qdrant_datasets:
        source_id = qd.get("source_id") or ""
        dataset_name = qd.get("dataset_name") or source_id
        seen_source_ids.add(source_id)

        # Start with Qdrant data
        entry = {
            "source_id": source_id,
            "dataset_name": dataset_name,
            "url": qd.get("link") or "",
            "format": None,
            "variables": [v["name"] for v in qd.get("variables", [])],
            "is_active": True,
            "processing_status": "completed" if qd.get("chunk_count", 0) > 10 else "metadata_only",
            "embedding_count": qd.get("chunk_count", 0),
            "hazard_type": qd.get("hazard_type"),
            "location_name": qd.get("location_name"),
            "impact_sector": qd.get("impact_sector"),
            "spatial_coverage": qd.get("spatial_coverage"),
            "catalog_source": qd.get("catalog_source"),
            "is_metadata_only": qd.get("is_metadata_only", False),
            "time_start": qd.get("time_start"),
            "time_end": qd.get("time_end"),
        }

        # Enrich with DB management data if available
        db = db_sources.get(source_id, {})
        if db:
            entry["url"] = db.get("url") or entry["url"]
            entry["format"] = db.get("format")
            entry["description"] = db.get("description")
            entry["is_active"] = db.get("is_active", True)
            entry["error_message"] = db.get("error_message")
            entry["last_processed"] = db.get("last_processed")
            entry["tags"] = db.get("tags")
            entry["created_at"] = db.get("created_at")
            # DB processing_status overrides if it's more specific
            db_status = db.get("processing_status")
            if db_status and db_status != "pending":
                entry["processing_status"] = db_status

        # Auth & schedule
        cred_info = source_creds.get(source_id, {})
        entry["auth_method"] = db.get("auth_method") or cred_info.get("auth_method")
        entry["portal"] = db.get("portal") or cred_info.get("portal")
        sched = schedules_map.get(source_id)
        if sched:
            entry["schedule"] = sched

        results.append(entry)

    # 4. Add DB-only sources not yet in Qdrant (e.g., just created, not processed yet)
    for sid, db in db_sources.items():
        if sid not in seen_source_ids:
            db["embedding_count"] = 0
            cred_info = source_creds.get(sid, {})
            db["auth_method"] = db.get("auth_method") or cred_info.get("auth_method")
            db["portal"] = db.get("portal") or cred_info.get("portal")
            sched = schedules_map.get(sid)
            if sched:
                db["schedule"] = sched
            results.append(db)

    # Sort: most data first, then by name
    results.sort(key=lambda r: (-r.get("embedding_count", 0), r.get("dataset_name", r.get("source_id", ""))))

    return results


@router.post("/sync-from-qdrant")
async def sync_sources_from_qdrant():
    """Create PostgreSQL source records for all datasets in Qdrant.
    Ensures every dataset has management metadata (for scheduling, history, etc.).
    Skips sources that already have a DB record."""
    from src.sources import get_source_store

    store = get_source_store()
    created = 0
    skipped = 0

    try:
        from web_api.routes.qdrant_datasets import list_qdrant_datasets
        datasets = await list_qdrant_datasets()
    except Exception as e:
        raise HTTPException(500, f"Could not fetch Qdrant datasets: {e}")

    for ds in datasets:
        source_id = ds.get("source_id")
        if not source_id:
            continue

        existing = store.get_source(source_id)
        if existing:
            skipped += 1
            continue

        variables = [v["name"] for v in ds.get("variables", [])]
        tags = []
        if ds.get("catalog_source"):
            tags.append(f"catalog:{ds['catalog_source']}")
        if ds.get("hazard_type"):
            tags.append(ds["hazard_type"])

        store.create_source({
            "source_id": source_id,
            "url": ds.get("link") or "",
            "format": None,
            "variables": variables,
            "is_active": True,
            "description": ds.get("dataset_name") or source_id,
            "tags": tags,
            "processing_status": "completed" if ds.get("chunk_count", 0) > 10 else "metadata_only",
            "embedding_model": "BAAI/bge-large-en-v1.5",
        })
        created += 1

    # Clear caches
    clear_rag_cache()

    return {"created": created, "skipped": skipped, "total": created + skipped}


@router.put("/{source_id}/metadata")
async def update_source_metadata(source_id: str, updates: dict):
    """Update metadata on both PostgreSQL and Qdrant payload.
    Updates Qdrant payload fields on ALL points matching this source_id,
    improving RAG retrieval without re-embedding."""

    # Fields that go to Qdrant payload
    qdrant_fields = {
        "hazard_type", "keywords", "impact_sector", "location_name",
        "spatial_coverage", "description", "dataset_name",
    }
    qdrant_updates = {k: v for k, v in updates.items() if k in qdrant_fields and v is not None}

    # Update Qdrant payloads
    qdrant_updated = 0
    if qdrant_updates:
        try:
            import httpx
            import os
            qdrant_host = os.getenv("QDRANT_HOST", "localhost")
            qdrant_port = os.getenv("QDRANT_REST_PORT", "6333")
            qdrant_base = f"http://{qdrant_host}:{qdrant_port}"
            collection = get_collection_name()

            # Use Qdrant REST API directly (avoids gRPC timeout issues on bulk ops)
            set_resp = httpx.post(
                f"{qdrant_base}/collections/{collection}/points/payload",
                json={
                    "payload": qdrant_updates,
                    "filter": {"must": [{"key": "source_id", "match": {"value": source_id}}]},
                },
                timeout=120.0,
            )
            set_resp.raise_for_status()

            count_resp = httpx.post(
                f"{qdrant_base}/collections/{collection}/points/count",
                json={"filter": {"must": [{"key": "source_id", "match": {"value": source_id}}]}},
                timeout=30.0,
            )
            if count_resp.status_code == 200:
                qdrant_updated = count_resp.json().get("result", {}).get("count", 0)

            logger.info(f"Updated Qdrant payload for {source_id}: {qdrant_updated} points, fields: {list(qdrant_updates.keys())}")
        except Exception as e:
            logger.error(f"Qdrant payload update failed for {source_id}: {e}")
            raise HTTPException(500, f"Qdrant update failed: {e}")

    # Update PostgreSQL source record
    db_updated = False
    try:
        from src.sources import get_source_store
        store = get_source_store()
        source = store.get_source(source_id)
        if source:
            db_fields = {k: v for k, v in updates.items() if k in {
                "url", "format", "description", "is_active", "variables", "tags",
                "keywords", "hazard_type", "region_country", "spatial_coverage", "impact_sector",
            }}
            if db_fields:
                store.update_source(source_id, db_fields)
                db_updated = True
    except Exception as e:
        logger.warning(f"DB update failed for {source_id}: {e}")

    # Clear caches
    clear_rag_cache()
    try:
        from web_api.routes.qdrant_datasets import _DATASETS_CACHE, _COUNTS_CACHE
        import web_api.routes.qdrant_datasets as qd_mod
        qd_mod._DATASETS_CACHE = None
        qd_mod._DATASETS_CACHE_TS = 0
    except Exception:
        pass

    return {
        "source_id": source_id,
        "qdrant_points_updated": qdrant_updated,
        "db_updated": db_updated,
        "fields_updated": list(qdrant_updates.keys()) + ([k for k in updates if k not in qdrant_fields] if db_updated else []),
    }


@router.post("/{source_id}/trigger", response_model=RunResponse)
async def trigger_source_etl(source_id: str, job_name: str = "dynamic_source_etl_job"):
    """Trigger ETL job for a specific source."""
    from src.sources import get_source_store

    store = get_source_store()
    source = store.get_source(source_id)

    if not source:
        raise HTTPException(404, "Source not found")

    run = await launch_dagster_run(job_name, {}, tags={"source_id": source_id})
    store.update_processing_status(source_id, "processing")

    return RunResponse(
        run_id=run["runId"], job_name=run["jobName"], status=run["status"], message="Job triggered"
    )


@router.put("/{source_id}")
async def update_source(source_id: str, updates: dict):
    """Update source metadata."""
    from src.sources import get_source_store

    store = get_source_store()
    source = store.get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source '{source_id}' not found")

    auth_method = updates.pop("auth_method", None)
    auth_credentials = updates.pop("auth_credentials", None)
    portal = updates.pop("portal", None)

    allowed_fields = {
        "url", "format", "description", "is_active", "variables", "tags",
        "keywords", "custom_metadata", "hazard_type", "region_country",
        "spatial_coverage", "impact_sector",
    }
    filtered = {k: v for k, v in updates.items() if k in allowed_fields}

    # Store/update auth credentials
    if auth_method is not None or auth_credentials is not None or portal is not None:
        persisted = load_settings()
        source_creds = persisted.get("source_credentials", {})
        existing = source_creds.get(source_id, {})
        if auth_method is not None:
            existing["auth_method"] = auth_method
        if auth_credentials is not None:
            existing["credentials"] = auth_credentials
        if portal is not None:
            existing["portal"] = portal
        source_creds[source_id] = existing
        persisted["source_credentials"] = source_creds
        save_settings(persisted)

    if not filtered and auth_method is None and auth_credentials is None and portal is None:
        raise HTTPException(400, "No valid fields to update")

    if filtered:
        try:
            updated = store.update_source(source_id, filtered)
            result = updated.to_dict() if hasattr(updated, "to_dict") else {"source_id": source_id, "updated": True}
        except AttributeError:
            source_dict = source.to_dict() if hasattr(source, "to_dict") else source.__dict__
            source_dict.update(filtered)
            store.hard_delete_source(source_id)
            new_source = store.create_source(source_dict)
            result = new_source.to_dict() if hasattr(new_source, "to_dict") else {"source_id": source_id, "updated": True}
    else:
        result = {"source_id": source_id, "updated": True}

    persisted = load_settings()
    cred_info = persisted.get("source_credentials", {}).get(source_id, {})
    result["auth_method"] = cred_info.get("auth_method")
    result["portal"] = cred_info.get("portal")
    return result


@router.delete("/{source_id}", status_code=204)
async def delete_source(source_id: str):
    """Delete a source. Idempotent — returns 204 even if the source was already gone.

    Also best-effort deletes matching points from Qdrant and clears any stored
    per-source credentials. Without the Qdrant sweep a deleted source would
    reappear in `GET /sources/` on the next refresh because that endpoint
    merges DB sources with Qdrant payload groups.
    """
    from src.sources import get_source_store

    store = get_source_store()
    try:
        store.hard_delete_source(source_id)
    except Exception as e:
        # hard_delete_source can raise if the backing store is unreachable; log
        # and surface as 500 so callers don't silently think the delete worked.
        logger.warning(f"Delete of {source_id} encountered: {e}")
        raise HTTPException(500, f"Failed to delete source: {e}")

    # Drop Qdrant points for this source. Best-effort — a Qdrant hiccup here
    # shouldn't fail the DB delete the user already saw succeed.
    try:
        from qdrant_client import models as qmodels

        client = get_qdrant_client()
        collection_name = get_collection_name()
        collections = client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            client.delete(
                collection_name=collection_name,
                points_selector=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="source_id", match=qmodels.MatchValue(value=source_id))]
                ),
            )
            logger.info(f"Deleted Qdrant points for source: {source_id}")
    except Exception as e:
        logger.warning(f"Failed to delete Qdrant points for {source_id}: {e}")

    # Drop persisted per-source credentials so a later source with the same id
    # doesn't silently inherit them.
    try:
        persisted = load_settings()
        if source_id in persisted.get("source_credentials", {}):
            persisted["source_credentials"].pop(source_id, None)
            save_settings(persisted)
    except Exception as e:
        logger.warning(f"Failed to clear stored credentials for {source_id}: {e}")

    return None


@router.delete("/", status_code=200)
async def delete_all_sources(confirm: bool = False, delete_embeddings: bool = False):
    """Delete all sources. Optionally also delete embeddings from Qdrant."""
    if not confirm:
        raise HTTPException(400, "Set confirm=true to delete all sources")

    from src.sources import get_source_store

    store = get_source_store()
    all_sources = store.get_all_sources(active_only=False)
    source_ids = [s.source_id for s in all_sources]

    embeddings_deleted_count = 0
    if delete_embeddings:
        try:
            from qdrant_client import models

            client = get_qdrant_client()
            collection_name = get_collection_name()

            collections = client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)

            if exists and source_ids:
                for sid in source_ids:
                    try:
                        client.delete(
                            collection_name=collection_name,
                            points_selector=models.Filter(
                                must=[models.FieldCondition(key="source_id", match=models.MatchValue(value=sid))]
                            ),
                        )
                        embeddings_deleted_count += 1
                        logger.info(f"Deleted embeddings for source: {sid}")
                    except Exception as e:
                        logger.warning(f"Failed to delete embeddings for {sid}: {e}")
            elif exists:
                try:
                    count_res = client.count(collection_name=collection_name)
                    points_before = count_res.count if hasattr(count_res, "count") else 0
                    client.delete_collection(collection_name)
                    embeddings_deleted_count = points_before
                except Exception as e:
                    logger.error(f"Failed to delete collection: {e}")
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}", exc_info=True)

    deleted_count = 0
    for sid in source_ids:
        if store.hard_delete_source(sid):
            deleted_count += 1

    clear_rag_cache()

    return {
        "status": "deleted",
        "sources_deleted": deleted_count,
        "embeddings_deleted": embeddings_deleted_count if delete_embeddings else None,
        "message": f"Deleted {deleted_count} source(s)"
        + (f" and {embeddings_deleted_count} embedding(s)" if delete_embeddings else ""),
    }


# --- Source Lifecycle ---


@router.post("/{source_id}/test-connection")
async def test_source_connection(source_id: str):
    """Test connectivity to a source's URL."""
    from src.sources import get_source_store

    store = get_source_store()
    source = store.get_source(source_id)
    if not source:
        raise HTTPException(404, "Source not found")

    from src.sources.connection_tester import test_connection

    result = test_connection(source.url)
    result["source_id"] = source_id
    return result


@router.post("/scan-metadata")
async def scan_source_metadata(data: dict):
    """Read file metadata remotely (variables, time range, spatial extent, attributes).
    For NetCDF/HDF5 files, reads only headers without downloading the full file."""
    url = data.get("url", "")
    if not url:
        raise HTTPException(400, "URL is required")

    # Uploaded files live under _UPLOAD_ROOT and are referenced by their
    # local container path. Resolve and verify the path is inside the upload
    # root before treating it as a trusted local file — no SSRF, no download.
    uploaded_local_path: Optional[Path] = None
    if not url.lower().startswith(("http://", "https://")):
        try:
            candidate = Path(url).resolve()
            if candidate.is_relative_to(_UPLOAD_ROOT.resolve()) and candidate.is_file():
                uploaded_local_path = candidate
        except (OSError, ValueError):
            pass
        if uploaded_local_path is None:
            raise HTTPException(400, "URL must be http(s) or a path to an uploaded file")
    else:
        # SSRF guard — same reasoning as in /test-connection. Runs before any
        # outbound request so we don't get tricked into fetching cloud metadata
        # or internal service ports via user-supplied URLs.
        from src.sources.connection_tester import validate_public_url, UnsafeURLError
        try:
            validate_public_url(url)
        except UnsafeURLError as e:
            raise HTTPException(400, f"URL rejected: {e}")

    import asyncio
    loop = asyncio.get_event_loop()

    def _scan():
        result = {
            "variables": [],
            "time_range": {},
            "spatial_extent": {},
            "attributes": {},
            "description": None,
            "error": None,
        }
        try:
            import xarray as xr
            import tempfile
            import requests as req

            # For uploaded files, skip the download and read the local file
            # path directly. Otherwise, stream the remote file to a temp path.
            downloaded_tmp: Optional[str] = None
            if uploaded_local_path is not None:
                tmp_path = str(uploaded_local_path)
            else:
                tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
                downloaded_tmp = tmp.name
                tmp_path = tmp.name
                try:
                    resp = req.get(url, stream=True, timeout=(10, 60), headers={"User-Agent": "ClimateRAG/1.0"})
                    resp.raise_for_status()
                    # Check size from Content-Length, skip files over 200MB
                    content_length = int(resp.headers.get("Content-Length", 0))
                    if content_length > 200 * 1024 * 1024:
                        tmp.close()
                        import os
                        os.unlink(downloaded_tmp)
                        result["error"] = f"File too large for scan ({content_length // 1024 // 1024}MB). Add the source and let the pipeline process it."
                        return result

                    for chunk in resp.iter_content(chunk_size=65536):
                        tmp.write(chunk)
                    tmp.close()
                except Exception as dl_err:
                    tmp.close()
                    import os
                    try:
                        os.unlink(downloaded_tmp)
                    except Exception:
                        pass
                    result["error"] = f"Download failed: {dl_err}"
                    return result

            ds = xr.open_dataset(tmp_path)

            # Variables
            for var_name in ds.data_vars:
                var = ds[var_name]
                info = {"name": var_name}
                if hasattr(var, "long_name"):
                    info["long_name"] = str(var.attrs.get("long_name", ""))
                if hasattr(var, "units"):
                    info["units"] = str(var.attrs.get("units", ""))
                result["variables"].append(info)

            # Time range
            if "time" in ds.coords:
                times = ds.coords["time"]
                try:
                    result["time_range"]["start"] = str(times.values[0])[:10]
                    result["time_range"]["end"] = str(times.values[-1])[:10]
                except Exception:
                    pass

            # Spatial extent
            for lat_name in ["lat", "latitude", "y"]:
                if lat_name in ds.coords:
                    lats = ds.coords[lat_name].values
                    result["spatial_extent"]["lat_min"] = float(lats.min())
                    result["spatial_extent"]["lat_max"] = float(lats.max())
                    break
            for lon_name in ["lon", "longitude", "x"]:
                if lon_name in ds.coords:
                    lons = ds.coords[lon_name].values
                    result["spatial_extent"]["lon_min"] = float(lons.min())
                    result["spatial_extent"]["lon_max"] = float(lons.max())
                    break

            # Global attributes
            for key in ["title", "institution", "source", "history", "references", "Conventions"]:
                if key in ds.attrs:
                    result["attributes"][key] = str(ds.attrs[key])[:500]

            # Build description from attributes
            parts = []
            if ds.attrs.get("title"):
                parts.append(str(ds.attrs["title"]))
            if ds.attrs.get("institution"):
                parts.append(f"by {ds.attrs['institution']}")
            var_names = [v["name"] for v in result["variables"]]
            if var_names:
                parts.append(f"Variables: {', '.join(var_names[:10])}")
            if result["time_range"].get("start"):
                parts.append(f"Period: {result['time_range']['start']} to {result['time_range'].get('end', 'present')}")
            result["description"] = ". ".join(parts) if parts else None

            ds.close()
            # Only remove the temp file for remote downloads — don't touch the
            # user's uploaded file, that's still referenced by the source.
            if downloaded_tmp:
                import os
                try:
                    os.unlink(downloaded_tmp)
                except Exception:
                    pass
        except Exception as e:
            import traceback
            logger.warning(f"scan-file failed for {url}: {e}\n{traceback.format_exc()}")
            result["error"] = str(e)[:2000]
            if downloaded_tmp:
                try:
                    import os
                    os.unlink(downloaded_tmp)
                except Exception:
                    pass
        return result

    result = await loop.run_in_executor(None, _scan)
    return result


@router.post("/upload")
async def upload_source_file(file: UploadFile = File(...)):
    """Save an uploaded raster/NetCDF/CSV file to `data/uploads/` and return a
    local path the caller can then use as the `url` of a new source.

    Why this exists: the source wizard previously required a public URL. For
    one-off files (a raster the user has locally, private/proprietary data,
    etc.) they now have a way in without spinning up an HTTP host. The saved
    file lives on the shared `./data` bind-mount so the Dagster workers can
    read it directly via its file path.
    """
    if not file or not file.filename:
        raise HTTPException(400, "No file provided")

    # Validate extension — blocks `.py`, `.sh`, etc. at the door.
    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED_UPLOAD_EXT:
        raise HTTPException(
            400,
            f"File type '{ext or 'unknown'}' is not allowed. "
            f"Supported: {', '.join(sorted(_ALLOWED_UPLOAD_EXT))}",
        )

    # Build a safe filename and a per-upload subdir so concurrent uploads of
    # the same name don't collide.
    safe_name = _SAFE_NAME.sub("_", file.filename).strip("._") or "upload"
    upload_id = uuid.uuid4().hex[:12]
    target_dir = _UPLOAD_ROOT / upload_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / safe_name

    # Stream to disk with a running size check so a huge upload can't exhaust
    # the container's tmp space. We abort & clean up the partial file if the
    # cap is exceeded.
    max_bytes = _upload_max_bytes()
    size = 0
    try:
        with open(target_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    out.close()
                    shutil.rmtree(target_dir, ignore_errors=True)
                    raise HTTPException(
                        413,
                        f"Upload exceeds {max_bytes // (1024 * 1024)} MB limit "
                        f"(UPLOAD_MAX_MB env var)",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(target_dir, ignore_errors=True)
        logger.error(f"upload failed for {file.filename}: {e}")
        raise HTTPException(500, f"Upload failed: {e}")

    # Path that both web-api and Dagster workers can resolve inside the
    # container. The frontend pastes this into the `url` field, the raster
    # pipeline opens it directly from disk.
    return {
        "upload_id": upload_id,
        "filename": safe_name,
        "size_bytes": size,
        "file_path": str(target_path),
        "message": "File uploaded. Use `file_path` as the source URL.",
    }


@router.post("/analyze-url")
async def analyze_source_url(data: dict):
    """Auto-detect format, portal, auth, and suggest dataset grouping from a URL."""
    url = data.get("url", "")
    if not url:
        raise HTTPException(400, "URL is required")

    from src.sources.connection_tester import analyze_url, validate_public_url, UnsafeURLError
    try:
        validate_public_url(url)
    except UnsafeURLError as e:
        raise HTTPException(400, f"URL rejected: {e}")

    # Get existing datasets for grouping suggestions (use cache, don't block)
    existing = []
    try:
        from web_api.routes.qdrant_datasets import _DATASETS_CACHE
        if _DATASETS_CACHE:
            existing = _DATASETS_CACHE
        else:
            # Trigger cache population in background, use empty for now
            from web_api.routes.qdrant_datasets import list_qdrant_datasets
            try:
                existing = await list_qdrant_datasets()
            except Exception:
                pass
    except Exception:
        pass

    return analyze_url(url, existing_datasets=existing)


@router.get("/{source_id}/history")
async def get_source_history(source_id: str, limit: int = 20):
    """Get processing history for a source."""
    from src.sources import get_source_store
    if not get_source_store().get_source(source_id):
        raise HTTPException(404, "Source not found")
    try:
        from src.database.source_store import SourceStore

        store = SourceStore()
        history = store.get_source_history(source_id, limit=limit)
        return {"source_id": source_id, "runs": history}
    except ImportError:
        return {"source_id": source_id, "runs": [], "message": "PostgreSQL store not available"}


@router.get("/{source_id}/freshness")
async def get_source_freshness(source_id: str):
    """Get freshness info for a source."""
    from src.sources import get_source_store

    store = get_source_store()
    source = store.get_source(source_id)
    if not source:
        raise HTTPException(404, "Source not found")

    result = {
        "source_id": source_id,
        "processing_status": source.processing_status,
        "last_processed": None,
        "is_stale": False,
        "schedule": None,
    }

    try:
        from src.database.connection import get_db_session
        from src.database.models import Source as SourceModel, SourceSchedule
        from datetime import datetime, timedelta

        with get_db_session() as session:
            row = session.query(SourceModel).filter(SourceModel.source_id == source_id).first()
            if row and row.last_processed_at:
                result["last_processed"] = row.last_processed_at.isoformat()
                result["is_stale"] = row.last_processed_at < datetime.utcnow() - timedelta(days=30)

            sched = session.query(SourceSchedule).filter(SourceSchedule.source_id == source_id).first()
            if sched:
                result["schedule"] = sched.to_dict()
    except ImportError:
        pass

    return result


# --- Source Schedule CRUD ---


@router.get("/{source_id}/schedule")
async def get_source_schedule(source_id: str):
    """Get the schedule for a source."""
    try:
        from src.database.source_store import SourceStore

        store = SourceStore()
        schedule = store.get_schedule(source_id)
        if not schedule:
            return {"source_id": source_id, "schedule": None}
        return {"source_id": source_id, "schedule": schedule}
    except ImportError:
        raise HTTPException(503, "PostgreSQL store not available")


@router.put("/{source_id}/schedule")
async def set_source_schedule(source_id: str, request: SourceScheduleRequest):
    """Create or update a schedule for a source."""
    try:
        from croniter import croniter
    except ImportError:
        raise HTTPException(503, "croniter not installed")

    try:
        croniter(request.cron_expression)
    except (ValueError, KeyError) as e:
        raise HTTPException(400, f"Invalid cron expression: {e}")

    try:
        from src.database.source_store import SourceStore

        store = SourceStore()
        source = store.get_source(source_id)
        if not source:
            raise HTTPException(404, "Source not found")

        schedule = store.set_schedule(
            source_id=source_id,
            cron_expression=request.cron_expression,
            is_enabled=request.is_enabled,
        )
        return {"source_id": source_id, "schedule": schedule}
    except ImportError:
        raise HTTPException(503, "PostgreSQL store not available")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/{source_id}/schedule")
async def delete_source_schedule(source_id: str):
    """Delete the schedule for a source."""
    try:
        from src.database.source_store import SourceStore

        store = SourceStore()
        deleted = store.delete_schedule(source_id)
        if not deleted:
            raise HTTPException(404, "Schedule not found")
        return {"source_id": source_id, "deleted": True}
    except ImportError:
        raise HTTPException(503, "PostgreSQL store not available")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/{source_id}/embeddings", status_code=200)
async def delete_source_embeddings(source_id: str, confirm: bool = False):
    """Delete all embeddings for a specific source from Qdrant."""
    if not confirm:
        raise HTTPException(400, "Set confirm=true to delete embeddings")

    from src.sources import get_source_store

    store = get_source_store()
    source = store.get_source(source_id)
    if not source:
        raise HTTPException(404, "Source not found")

    try:
        from qdrant_client import models

        client = get_qdrant_client()
        collection_name = get_collection_name()

        client.delete(
            collection_name=collection_name,
            points_selector=models.Filter(
                must=[models.FieldCondition(key="source_id", match=models.MatchValue(value=source_id))]
            ),
        )

        logger.info(f"Deleted embeddings for source {source_id} from collection {collection_name}")
        clear_rag_cache()

        return {
            "status": "deleted",
            "source_id": source_id,
            "message": f"Embeddings for source {source_id} deleted from collection {collection_name}",
        }
    except Exception as e:
        logger.error(f"Error deleting embeddings for {source_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to delete embeddings: {str(e)}")
