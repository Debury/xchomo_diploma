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
from web_api.dependencies import (
    launch_dagster_run,
    get_qdrant_client,
    get_collection_name,
    clear_rag_cache,
    get_active_runs_for_source,
    execute_graphql_query,
)

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

        # When the user is appending to an existing dataset (via the wizard's
        # "Add to this dataset" button), refuse if the same URL is already
        # registered under that dataset — that's a true duplicate, not an
        # append. Different URL under the same dataset_name is fine and is
        # the intended append flow.
        if source.dataset_name:
            try:
                existing = [
                    s for s in store.get_all_sources(active_only=False)
                    if (s.dataset_name or s.source_id) == source.dataset_name
                ]
            except Exception:
                existing = []
            url_norm = (source.url or "").strip().rstrip("/").lower()
            for s in existing:
                if (s.url or "").strip().rstrip("/").lower() == url_norm:
                    raise HTTPException(
                        409,
                        f"This URL is already in dataset '{source.dataset_name}' "
                        f"as source '{s.source_id}'. Nothing to append.",
                    )

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


@router.get("/runs/activity")
async def source_runs_activity(limit: int = 20):
    """Recent + active per-source ETL runs for the ETL Monitor.

    Combines two signals so the UI doesn't lie about what's happening:
      - ``active``: Dagster runs tagged ``source_id=<id>`` that are still
        QUEUED / STARTING / STARTED. Read straight from the Dagster
        GraphQL API so a source stuck in the coordinator without a
        ``processing_runs`` row yet still appears on the monitor.
      - ``recent``: last ``limit`` rows of ``processing_runs`` (DESC by
        ``started_at``). These are the completed / failed runs after the
        op's ``complete_run()`` call landed.

    Returns ``{active: [...], recent: [...]}``. Both lists are sorted
    newest-first and include ``dagster_run_id`` so the UI can deep-link
    into Dagit.
    """
    from sqlalchemy import desc
    from src.database.connection import get_db_session
    from src.database.models import ProcessingRun

    # --- active runs from Dagster ------------------------------------
    active: list[dict] = []
    try:
        # We deliberately DO NOT filter by pipelineName — the app spawns
        # runs across two job names (`single_source_etl_job` for auto-embed
        # after POST /sources, `dynamic_source_etl_job` for the Reprocess
        # button). A single-job filter was silently hiding the auto-embed
        # path. Instead we fetch all non-terminal runs and keep only ones
        # that carry a `source_id` tag — that's the invariant for anything
        # relevant to this view.
        query = """
        query ActiveSourceRuns {
            runsOrError(
                filter: { statuses: [STARTED, STARTING, QUEUED] }
                limit: 100
            ) {
                __typename
                ... on Runs {
                    results {
                        runId
                        status
                        startTime
                        creationTime
                        pipelineName
                        tags { key value }
                    }
                }
                ... on PythonError { message }
            }
        }
        """
        data = await execute_graphql_query(query)
        runs_or_error = data.get("runsOrError") or {}
        if runs_or_error.get("__typename") == "Runs":
            for r in runs_or_error.get("results") or []:
                tags = {t["key"]: t["value"] for t in (r.get("tags") or [])}
                sid = tags.get("source_id")
                if not sid:
                    continue  # skip catalog-batch + untagged runs
                active.append({
                    "dagster_run_id": r.get("runId"),
                    "source_id": sid,
                    "status": r.get("status"),
                    "job_name": r.get("pipelineName"),
                    "trigger_type": tags.get("trigger_type"),
                    "start_time": r.get("startTime"),
                    "creation_time": r.get("creationTime"),
                })
        # Sort newest-first by creation time
        active.sort(key=lambda x: x.get("creation_time") or 0, reverse=True)
    except Exception as e:
        logger.warning(f"/sources/runs/activity: active-run fetch failed: {e}")

    # --- recent history from processing_runs -------------------------
    recent: list[dict] = []
    try:
        # Clamp limit into a sane range so a client can't ask for 10_000 rows.
        lim = max(1, min(int(limit), 100))
        with get_db_session() as session:
            rows = (
                session.query(ProcessingRun)
                .order_by(desc(ProcessingRun.started_at))
                .limit(lim)
                .all()
            )
            recent = [r.to_dict() for r in rows]
    except Exception as e:
        logger.warning(f"/sources/runs/activity: recent-run fetch failed: {e}")

    return {"active": active, "recent": recent}


@router.get("/")
async def list_sources(active_only: bool = False):
    """List all data sources, one row per source_id.

    Sources is the superset of "data we have": catalog (Excel) rows that
    were processed AND anything the user added via Add Sources / zip upload.
    Catalog is read-only — it shows what was *in* the Excel. Sources shows
    what we *have* (catalog-derived + user-added). Adding a new source in
    the UI lands here; it does NOT modify the Excel.

    Per-source_id chunk counts come from the same Qdrant facet the Catalog
    page reads, so a row's status / count is identical on both pages.
    """
    # Per-source-id chunk count from Qdrant — matches the Catalog page.
    from web_api.routes.qdrant_datasets import _get_embedding_counts, list_qdrant_datasets
    chunk_by_source: dict = {}
    try:
        chunk_by_source = _get_embedding_counts()
    except Exception as e:
        logger.warning(f"Could not fetch per-source chunk counts: {e}")

    # Dataset-level aggregate (variables, hazard, link) keyed by dataset_name.
    ds_aggregate: dict = {}
    ds_chunk_total: dict = {}
    try:
        for qd in await list_qdrant_datasets():
            dn = qd.get("dataset_name") or ""
            if not dn:
                continue
            ds_aggregate[dn] = qd
            ds_chunk_total[dn] = qd.get("chunk_count", 0)
    except Exception as e:
        logger.warning(f"Could not fetch dataset aggregate: {e}")

    # Catalog Excel — every row here is a "source" too, just one with a
    # known origin. Sources page shows them so the user has admin handles
    # on everything we have data for.
    catalog_entries = []
    try:
        from src.catalog.excel_reader import read_catalog
        excel_path = os.getenv("CATALOG_EXCEL_PATH", "Kopie souboru D1.1.xlsx")
        if not Path(excel_path).exists():
            excel_path = str(Path(__file__).resolve().parents[2] / excel_path)
        catalog_entries = read_catalog(excel_path)
    except Exception as e:
        logger.warning(f"Could not read catalog: {e}")

    # Postgres management state — schedules, history, error_message,
    # description for any row (catalog or user-added).
    db_sources: dict = {}
    try:
        from src.sources import get_source_store
        for s in get_source_store().get_all_sources(active_only=active_only):
            d = s.to_dict()
            db_sources[d["source_id"]] = d
    except Exception as e:
        logger.warning(f"Could not fetch DB sources: {e}")

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

    def _build_entry(source_id: str, dataset_name: str, ce=None, db=None) -> dict:
        """Compose a row by merging Qdrant counts (truth) with catalog
        Excel metadata and Postgres management state. Status derivation
        mirrors /catalog/ exactly so the same row reads identically on
        both pages."""
        chunks = chunk_by_source.get(source_id, 0)
        ds_total = ds_chunk_total.get(dataset_name or "", 0)
        agg = ds_aggregate.get(dataset_name or "", {}) if dataset_name else {}
        db = db or {}
        ce_link = ce.link if ce else None

        if chunks > 1 or ds_total > 5:
            processing_status = "completed"
        elif chunks > 0:
            processing_status = "metadata_only"
        else:
            processing_status = "pending"
        db_status = db.get("processing_status")
        if processing_status == "pending" and db_status and db_status != "pending":
            processing_status = db_status

        cred_info = source_creds.get(source_id, {})
        entry = {
            "source_id": source_id,
            "dataset_name": dataset_name,
            "url": db.get("url") or ce_link or agg.get("link") or "",
            "format": db.get("format"),
            "variables": [v["name"] for v in agg.get("variables", [])],
            "is_active": db.get("is_active", True),
            "processing_status": processing_status,
            "embedding_count": chunks,
            "hazard_type": db.get("hazard_type") or agg.get("hazard_type") or (ce.hazard if ce else None),
            "location_name": agg.get("location_name"),
            "impact_sector": db.get("impact_sector") or agg.get("impact_sector") or (ce.impact_sector if ce else None),
            "spatial_coverage": db.get("spatial_coverage") or agg.get("spatial_coverage") or (ce.spatial_coverage if ce else None),
            "catalog_source": agg.get("catalog_source") or ("D1.1.xlsx" if ce else None),
            "is_metadata_only": (chunks <= 1 and processing_status != "completed"),
            "time_start": agg.get("time_start"),
            "time_end": agg.get("time_end"),
            "description": db.get("description"),
            "error_message": db.get("error_message"),
            "last_processed": db.get("last_processed"),
            "tags": db.get("tags"),
            "created_at": db.get("created_at"),
            "auth_method": db.get("auth_method") or cred_info.get("auth_method"),
            "portal": db.get("portal") or cred_info.get("portal"),
            "catalog_row_index": ce.row_index if ce else db.get("catalog_row_index"),
            "from_catalog": ce is not None,
        }
        sched = schedules_map.get(source_id)
        if sched:
            entry["schedule"] = sched
        return entry

    # Group catalog rows by dataset_name so SYNOP / E-OBS / CMIP6 each
    # appear as ONE administratable source even though the Excel lists
    # them under multiple hazards. The Catalog page still shows the raw
    # 233 rows; Sources is the deduplicated view.
    by_dataset: dict = {}
    for ce in catalog_entries:
        ds = ce.dataset_name or ce.source_id
        by_dataset.setdefault(ds, []).append(ce)

    results = []
    seen_db = set()
    for ds, group in by_dataset.items():
        # Pick the catalog row with the most useful link (prefer one that
        # has a real http URL) as the representative for the group.
        rep = next(
            (e for e in group if e.link and e.link.startswith(("http://", "https://"))),
            group[0],
        )
        # Use a stable group source_id: the dataset_name itself is the
        # natural identifier for administration. Catalog page keeps the
        # row-specific catalog_<DS>_<row> ids; Sources collapses them.
        # If a DB row keyed exactly on the dataset_name exists (manual
        # ingest like catalog_SPEI-GD_manual or a user-added one), prefer
        # it for management metadata.
        group_db = next(
            (db_sources.get(e.source_id) for e in group if db_sources.get(e.source_id)),
            None,
        )
        if group_db:
            seen_db.add(group_db["source_id"])
        # Surface ALL hazards observed across the group as a comma-joined
        # string so the user sees "Drought, Flood, Heat" instead of just
        # the first one — that was the bug the user flagged.
        hazards = sorted({e.hazard for e in group if e.hazard})
        regions = sorted({e.region_country for e in group if e.region_country})
        sectors = sorted({e.impact_sector for e in group if e.impact_sector})

        # Sum chunk counts across all source_ids in the group so the
        # number reflects "all data we have for this dataset".
        group_chunks = sum(chunk_by_source.get(e.source_id, 0) for e in group)

        entry = _build_entry(
            source_id=group_db["source_id"] if group_db else (rep.source_id),
            dataset_name=ds,
            ce=rep,
            db=group_db,
        )
        entry["hazard_type"] = ", ".join(hazards) if hazards else entry.get("hazard_type")
        entry["region_country"] = ", ".join(regions) if regions else None
        entry["impact_sector"] = ", ".join(sectors) if sectors else entry.get("impact_sector")
        # Override with the dataset-aggregate count, not just the rep row's,
        # so SYNOP shows ~420 chunks (60 × 7 rows fanned out) not 60.
        entry["embedding_count"] = max(group_chunks, entry["embedding_count"])
        entry["catalog_row_count"] = len(group)  # how many Excel rows folded in
        results.append(entry)

    # User-added rows in the DB that aren't catalog-derived. These don't
    # need grouping — they're already one row per source_id.
    for sid, db in db_sources.items():
        if sid in seen_db or sid.startswith("catalog_"):
            continue
        seen_db.add(sid)
        results.append(_build_entry(
            source_id=sid,
            dataset_name=db.get("dataset_name") or sid,
            db=db,
        ))

    # Sort: most data first, then by name. Coerce the secondary key to a
    # string explicitly because ``r.get(..., default)`` only kicks in when
    # the key is absent — if dataset_name is present but ``None`` (which
    # SQLAlchemy returns for a NULL column), the sort comparator hits
    # `None < "some-id"` and raises TypeError on the request thread,
    # taking the whole /sources/ page down with a 500.
    def _name_key(r):
        return (r.get("dataset_name") or r.get("source_id") or "")
    results.sort(key=lambda r: (-r.get("embedding_count", 0), _name_key(r)))

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


@router.post("/{source_id}/cancel")
async def cancel_source_etl(source_id: str):
    """Cancel any active Dagster run for this source.

    Returns the number of runs actually terminated. Uses Dagster's
    ``terminateRuns(terminatePolicy: MARK_AS_CANCELED_IMMEDIATELY)`` so a
    dead subprocess doesn't leave the run stuck in CANCELING and block the
    concurrency slot for the next trigger. Safe to call when no run is
    active (returns ``{"cancelled": 0}``); the caller can reset the
    source's ``processing_status`` to ``pending`` separately if desired.
    """
    from src.sources import get_source_store

    store = get_source_store()
    source = store.get_source(source_id)
    if not source:
        raise HTTPException(404, "Source not found")

    active = await get_active_runs_for_source(source_id)
    if not active:
        # No run in flight — also reset the source status if it was left
        # in "processing" by a previous crashed run.
        if getattr(source, "processing_status", None) == "processing":
            store.update_processing_status(source_id, "pending")
        return {"cancelled": 0, "source_id": source_id}

    run_ids = [r.get("runId") for r in active if r.get("runId")]
    mutation = """
    mutation CancelRuns($runIds: [String!]!) {
        terminateRuns(runIds: $runIds, terminatePolicy: MARK_AS_CANCELED_IMMEDIATELY) {
            __typename
            ... on TerminateRunsResult {
                terminateRunResults {
                    __typename
                    ... on TerminateRunSuccess { run { runId status } }
                    ... on TerminateRunFailure { message }
                }
            }
            ... on PythonError { message }
        }
    }
    """
    data = await execute_graphql_query(mutation, {"runIds": run_ids})
    body = data.get("terminateRuns") or {}
    if body.get("__typename") == "PythonError":
        raise HTTPException(502, f"Dagster error: {body.get('message', 'unknown')}")

    results = body.get("terminateRunResults") or []
    cancelled = sum(1 for r in results if r.get("__typename") == "TerminateRunSuccess")

    # Reset the source so the UI doesn't keep showing it as "processing"
    # after the user already clicked Cancel.
    store.update_processing_status(
        source_id,
        "pending",
        error_message="Cancelled by user." if cancelled else None,
    )

    return {
        "cancelled": cancelled,
        "run_ids": run_ids,
        "source_id": source_id,
    }


@router.post("/{source_id}/trigger", response_model=RunResponse)
async def trigger_source_etl(source_id: str, job_name: str = "dynamic_source_etl_job"):
    """Trigger ETL job for a specific source.

    Refuses with 409 if a run tagged ``source_id=<source_id>`` is already
    STARTED/STARTING/QUEUED, so a double-click on "Reprocess" doesn't spawn two
    concurrent runs writing to the same Qdrant ``source_id``. A Postgres
    advisory lock further closes the TOCTOU window between this API path and
    the ``source_schedule_sensor`` in the dagster-daemon container.
    """
    from src.sources import get_source_store
    from src.database.connection import acquire_source_lock

    store = get_source_store()
    source = store.get_source(source_id)

    if not source:
        raise HTTPException(404, "Source not found")

    with acquire_source_lock(source_id) as acquired:
        if not acquired:
            raise HTTPException(
                status_code=409,
                detail=f"Another trigger for source {source_id} is in flight. Retry shortly.",
            )

        active = await get_active_runs_for_source(source_id)
        if active:
            existing = active[0]
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Source {source_id} already has an active run "
                    f"({existing.get('runId')}, status={existing.get('status')})."
                ),
            )

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


def _scan_tabular(path: str, result: dict) -> None:
    """Populate `result` from a CSV/TSV file using pandas. Mirrors the keys the
    NetCDF branch produces so the frontend renders both the same way.

    Heuristics: first matching name in TIME_NAMES becomes the time axis (min/max
    parsed as dates if possible); LAT/LON name matches feed spatial_extent."""
    import pandas as pd

    TIME_NAMES = {"time", "date", "datetime", "year", "month", "timestamp"}
    LAT_NAMES = {"lat", "latitude", "y"}
    LON_NAMES = {"lon", "long", "longitude", "x"}

    try:
        head = pd.read_csv(path, nrows=5000, sep=None, engine="python", comment="#")
    except Exception:
        head = pd.read_csv(path, nrows=5000, comment="#")

    for col in head.columns:
        info = {"name": str(col), "units": str(head[col].dtype)}
        result["variables"].append(info)

    cols_lower = {str(c).strip().lower(): c for c in head.columns}

    time_col = next((cols_lower[k] for k in TIME_NAMES if k in cols_lower), None)
    if time_col is not None:
        try:
            parsed = pd.to_datetime(head[time_col], errors="coerce").dropna()
            if not parsed.empty:
                result["time_range"]["start"] = str(parsed.min())[:10]
                result["time_range"]["end"] = str(parsed.max())[:10]
        except Exception:
            pass

    lat_col = next((cols_lower[k] for k in LAT_NAMES if k in cols_lower), None)
    if lat_col is not None:
        try:
            lats = pd.to_numeric(head[lat_col], errors="coerce").dropna()
            if not lats.empty:
                result["spatial_extent"]["lat_min"] = float(lats.min())
                result["spatial_extent"]["lat_max"] = float(lats.max())
        except Exception:
            pass

    lon_col = next((cols_lower[k] for k in LON_NAMES if k in cols_lower), None)
    if lon_col is not None:
        try:
            lons = pd.to_numeric(head[lon_col], errors="coerce").dropna()
            if not lons.empty:
                result["spatial_extent"]["lon_min"] = float(lons.min())
                result["spatial_extent"]["lon_max"] = float(lons.max())
        except Exception:
            pass

    result["attributes"]["format"] = "CSV"
    result["attributes"]["columns"] = str(len(head.columns))
    result["attributes"]["rows_sampled"] = str(len(head))

    parts = [f"Tabular CSV with {len(head.columns)} columns"]
    var_names = [v["name"] for v in result["variables"]]
    if var_names:
        parts.append(f"Columns: {', '.join(var_names[:10])}")
    if result["time_range"].get("start"):
        parts.append(f"Period: {result['time_range']['start']} to {result['time_range'].get('end', 'present')}")
    result["description"] = ". ".join(parts)


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

    # Pick scan backend by file extension. Gridded formats go through xarray;
    # tabular CSV/TSV through pandas so the wizard surfaces columns/time-range
    # for non-NetCDF sources too (HadCRUT5, GISTEMP, etc.).
    def _detect_format(name: str) -> str:
        from urllib.parse import urlparse
        path = urlparse(name).path if name.lower().startswith(("http://", "https://")) else name
        ext = Path(path).suffix.lower()
        if ext in {".csv", ".tsv"}:
            return "tabular"
        return "gridded"

    scan_format = _detect_format(str(uploaded_local_path) if uploaded_local_path else url)
    tmp_suffix = ".csv" if scan_format == "tabular" else ".nc"

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
            import tempfile
            import requests as req

            # For uploaded files, skip the download and read the local file
            # path directly. Otherwise, stream the remote file to a temp path.
            downloaded_tmp: Optional[str] = None
            if uploaded_local_path is not None:
                tmp_path = str(uploaded_local_path)
            else:
                tmp = tempfile.NamedTemporaryFile(suffix=tmp_suffix, delete=False)
                downloaded_tmp = tmp.name
                tmp_path = tmp.name
                try:
                    # allow_redirects=False: SSRF guard only validated the
                    # original URL; a redirect can point back to 169.254.x.x
                    # or RFC1918 after we've cleared the pre-flight check.
                    resp = req.get(
                        url,
                        stream=True,
                        timeout=(10, 60),
                        allow_redirects=False,
                        headers={"User-Agent": "ClimateRAG/1.0"},
                    )
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

            if scan_format == "tabular":
                _scan_tabular(tmp_path, result)
                if downloaded_tmp:
                    import os
                    try:
                        os.unlink(downloaded_tmp)
                    except Exception:
                        pass
                return result

            import xarray as xr
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


@router.post("/upload-multi")
async def upload_zip_multi_ingest(
    file: UploadFile = File(...),
    source_id: str = "",
    dataset_name: str = "",
    hazard_type: str = "",
    glob_pattern: str = "*.nc",
):
    """Upload a zip of rasters and kick off a generic multi-file ingest.

    This is the future-proof version of the SPEI-GD bespoke script:
    any user can upload a zip via the UI, the helper extracts it once
    (idempotently), iterates the rasters, embeds each as Qdrant
    chunks, and writes ``ingest_state.json`` so the Catalog "Resume"
    badge works automatically if the run is killed.

    The launcher script is generated on the fly under the per-upload
    directory; the Resume button re-invokes it via ``POST
    /catalog/resume-ingest``.
    """
    if not file or not file.filename:
        raise HTTPException(400, "No file provided")
    if not source_id or not dataset_name:
        raise HTTPException(400, "source_id and dataset_name are required")

    ext = Path(file.filename).suffix.lower()
    if ext != ".zip":
        raise HTTPException(400, "upload-multi only accepts .zip — single files use /upload")

    safe_name = _SAFE_NAME.sub("_", file.filename).strip("._") or "upload.zip"
    upload_id = uuid.uuid4().hex[:12]
    target_dir = _UPLOAD_ROOT / upload_id
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / safe_name
    extracted_dir = target_dir / "extracted"
    done_marker = target_dir / "done.txt"
    launcher = target_dir / "_run_ingest.py"

    max_bytes = _upload_max_bytes()
    size = 0
    try:
        with open(zip_path, "wb") as out:
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
                        f"Upload exceeds {max_bytes // (1024 * 1024)} MB limit",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        shutil.rmtree(target_dir, ignore_errors=True)
        logger.error(f"upload-multi failed for {file.filename}: {e}")
        raise HTTPException(500, f"Upload failed: {e}")

    # Sanitize the strings before writing them into a Python source file.
    # repr() is safer than f-string concatenation here — it escapes quotes
    # and backslashes, so an attacker can't break out of the literal.
    launcher.write_text(
        "import logging, sys\n"
        "logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')\n"
        "sys.path.insert(0, '/app')\n"
        "from src.catalog.multi_file_ingester import ingest_directory\n"
        "if __name__ == '__main__':\n"
        f"    ingest_directory(\n"
        f"        directory={str(extracted_dir)!r},\n"
        f"        zip_path={str(zip_path)!r},\n"
        f"        source_id={source_id!r},\n"
        f"        dataset_name={dataset_name!r},\n"
        f"        hazard_type={hazard_type!r},\n"
        f"        glob_pattern={glob_pattern!r},\n"
        f"        catalog_source='manual_upload',\n"
        f"        script_path=__file__,\n"
        f"        done_marker={str(done_marker)!r},\n"
        f"    )\n"
    )

    # Register a Postgres row so the source shows up in the Sources page
    # (same place the user added it from). Without this, the upload would
    # silently chunk into Qdrant but Sources would never list it. We use
    # idempotent get-or-create so re-running the same upload doesn't 409.
    try:
        from src.sources import get_source_store
        from src.sources.climate_source import ClimateDataSource
        store = get_source_store()
        if store.get_source(source_id) is None:
            store.add_source(ClimateDataSource(
                source_id=source_id,
                dataset_name=dataset_name,
                url=str(zip_path),
                format="zip-multi",
                description=f"Multi-file zip upload: {safe_name}",
                hazard_type=hazard_type or None,
                is_active=True,
            ))
            logger.info(f"upload-multi: registered Sources row for {source_id}")
    except Exception as e:
        # Non-fatal: ingest still proceeds, but the source won't show in
        # Sources until a manual /sources/sync-from-qdrant call.
        logger.warning(f"upload-multi: could not register Sources row: {e}")

    # Launch detached so the upload response can return immediately.
    import subprocess
    import sys as _sys
    proc = subprocess.Popen(
        [_sys.executable, str(launcher)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    logger.info(
        f"upload-multi: launched {launcher} as pid {proc.pid} for "
        f"source_id={source_id} dataset={dataset_name}"
    )

    return {
        "upload_id": upload_id,
        "size_bytes": size,
        "zip_path": str(zip_path),
        "launcher": str(launcher),
        "pid": proc.pid,
        "source_id": source_id,
        "dataset_name": dataset_name,
        "message": (
            "Zip uploaded and ingest launched in background. "
            "Watch the Catalog page — the row will show ⏳ Running and update "
            "as files complete. If killed, click the row → Resume ingest."
        ),
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
