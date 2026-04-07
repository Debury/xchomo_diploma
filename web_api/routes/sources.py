"""Source CRUD and lifecycle endpoints."""

import os
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from web_api.models import SourceCreate, SourceResponse, SourceScheduleRequest, RunResponse
from web_api.config import load_settings, save_settings
from web_api.dependencies import launch_dagster_run, get_qdrant_client, get_collection_name, clear_rag_cache

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sources", tags=["sources"])


@router.post("/", response_model=SourceResponse, status_code=201)
async def create_source(source: SourceCreate):
    """Create a new data source."""
    try:
        from src.sources import get_source_store

        store = get_source_store()

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
            try:
                run = await launch_dagster_run(
                    "single_source_etl_job",
                    {},
                    tags={"source_id": source.source_id, "trigger_type": "auto_embed"},
                )
                store.update_processing_status(source.source_id, "processing")
                source_dict["etl_run_id"] = run.get("runId")
                logger.info(f"Auto-embed triggered for {source.source_id}: {run.get('runId')}")
            except Exception as etl_err:
                logger.warning(f"Auto-embed trigger failed for {source.source_id}: {etl_err}")
                source_dict["etl_error"] = str(etl_err)

        return source_dict
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/", response_model=List[SourceResponse])
async def list_sources(active_only: bool = True):
    """List all sources."""
    from src.sources import get_source_store

    persisted = load_settings()
    source_creds = persisted.get("source_credentials", {})

    schedules_map = {}
    try:
        from src.database.connection import get_db_session
        from src.database.models import SourceSchedule

        with get_db_session() as session:
            for sched in session.query(SourceSchedule).all():
                schedules_map[sched.source_id] = sched.to_dict()
    except Exception:
        pass

    results = []
    for s in get_source_store().get_all_sources(active_only):
        d = s.to_dict()
        cred_info = source_creds.get(d.get("source_id", ""), {})
        d["auth_method"] = d.get("auth_method") or cred_info.get("auth_method")
        d["portal"] = d.get("portal") or cred_info.get("portal")
        sched = schedules_map.get(d.get("source_id", ""))
        if sched:
            d["schedule"] = sched
        results.append(d)
    return results


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
    """Delete a source."""
    from src.sources import get_source_store

    get_source_store().hard_delete_source(source_id)
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


@router.post("/analyze-url")
async def analyze_source_url(data: dict):
    """Auto-detect format, portal, and auth requirements from a URL."""
    url = data.get("url", "")
    if not url:
        raise HTTPException(400, "URL is required")

    from src.sources.connection_tester import analyze_url

    return analyze_url(url)


@router.get("/{source_id}/history")
async def get_source_history(source_id: str, limit: int = 20):
    """Get processing history for a source."""
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
