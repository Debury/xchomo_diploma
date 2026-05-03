"""Catalog batch processing endpoints."""

import os
import logging
import threading
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException

from web_api.models import CatalogEntryResponse, CatalogProcessRequest, CatalogProgressResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/catalog", tags=["catalog"])

CATALOG_EXCEL_PATH = os.getenv("CATALOG_EXCEL_PATH", "Kopie souboru D1.1.xlsx")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not Path(CATALOG_EXCEL_PATH).exists():
    CATALOG_EXCEL_PATH = str(_PROJECT_ROOT / CATALOG_EXCEL_PATH)

# Module-level batch thread tracking
_batch_thread: Optional[threading.Thread] = None
_batch_thread_error: Optional[str] = None
_batch_thread_phases: Optional[List[int]] = None


@router.get("/", response_model=List[CatalogEntryResponse])
async def list_catalog():
    """List all catalog entries with phase classification, processing status,
    and the *reason* a row is in its current state.

    Status derivation:
      - 'completed'      → Qdrant has real data chunks (>1, i.e. anything
                           past the single Phase-0 metadata embedding)
      - 'metadata_only'  → only the Phase-0 metadata chunk exists. Either
                           Phase 1 hasn't run yet for this row, or it ran
                           and failed. The ``error`` field tells you which.
      - 'pending'        → not in Qdrant at all (Phase 0 hasn't run)

    The companion ``error`` field carries the failure reason from
    ``catalog_progress.error`` so the UI can show "Zenodo 403", "URL not
    in catalog", etc. without making the user dig through logs.
    """
    try:
        from src.catalog.excel_reader import read_catalog
        from src.catalog.phase_classifier import classify_source
        from src.catalog.batch_orchestrator import DIRECT_DOWNLOAD_URLS, SKIP_PHASE1
        from src.database.models import CatalogProgress
        from src.database.connection import get_db_session

        entries = read_catalog(CATALOG_EXCEL_PATH)

        # Per-source Qdrant chunk count
        qdrant_counts: dict = {}
        try:
            from web_api.routes.qdrant_datasets import _get_embedding_counts
            qdrant_counts = _get_embedding_counts()
        except Exception as e:
            logger.warning(f"Could not fetch Qdrant counts for catalog: {e}")

        # Manual-ingest progress markers — written by long-running scripts
        # (e.g. SPEI-GD multi-file ingest) so the Catalog UI can show a
        # "partial: N/M files" badge and let the user resume after a kill.
        ingest_state: dict = {}
        try:
            import json as _json
            STATE_PATH = "/app/data/ingest_state.json"
            if os.path.exists(STATE_PATH):
                with open(STATE_PATH, "r") as f:
                    ingest_state = _json.load(f) or {}
        except Exception as e:
            logger.warning(f"Could not read ingest_state: {e}")

        # Dataset-name aggregate from Qdrant directly — covers the case
        # where chunks were ingested under a source_id that doesn't match
        # any catalog row (manual upload, dataset_name="MOLOCH" but
        # source_id="catalog_MOLOCH_manual"). Summing per-entry source_id
        # counts only would miss those.
        dataset_chunk_totals: dict = {}
        try:
            from web_api.dependencies import get_qdrant_client, get_collection_name
            client = get_qdrant_client()
            collection = get_collection_name()
            facet_res = client.facet(collection_name=collection, key="dataset_name", limit=500)
            for hit in getattr(facet_res, "hits", []) or []:
                v = getattr(hit, "value", None)
                c = getattr(hit, "count", 0)
                if isinstance(v, str):
                    dataset_chunk_totals[v] = c
        except Exception as e:
            logger.warning(f"Could not facet dataset_name from Qdrant: {e}")
            # Fallback: per-entry sum (only catches rows whose source_ids
            # are themselves the chunk owners — i.e. catalog-batch ingestions).
            for entry in entries:
                ds = entry.dataset_name
                if not ds:
                    continue
                dataset_chunk_totals[ds] = dataset_chunk_totals.get(ds, 0) + qdrant_counts.get(entry.source_id, 0)

        # Latest progress row per source_id (highest phase wins). One DB
        # query, then dict-indexed in the loop.
        progress_by_source: dict = {}
        with get_db_session() as session:
            for row in session.query(CatalogProgress).all():
                cur = progress_by_source.get(row.source_id)
                if cur is None or row.phase > cur["phase"]:
                    progress_by_source[row.source_id] = {
                        "phase": row.phase,
                        "status": row.status,
                        "error": row.error,
                        "started_at": row.started_at.isoformat() if row.started_at else None,
                        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                    }

        result = []
        for entry in entries:
            phase = classify_source(entry)
            count = qdrant_counts.get(entry.source_id, 0)
            ds_total = dataset_chunk_totals.get(entry.dataset_name or "", 0)
            # Real-data check: this row needs ACTUAL chunks tagged with its
            # source_id (or at least with its dataset_name). The orchestrator
            # writes chunks per catalog row, so the source_id count is the
            # honest answer; dataset_name aggregate covers the case where
            # one catalog dataset spans multiple source_ids legitimately.
            if count > 1 or ds_total > 5:
                processing_status = "completed"
            elif count > 0:
                processing_status = "metadata_only"
            else:
                processing_status = "pending"

            prog = progress_by_source.get(entry.source_id) or {}
            db_error = prog.get("error")

            # If the DB doesn't know why this row didn't progress, infer the
            # reason from the catalog itself so the UI never shows an empty
            # "metadata_only" with no explanation.
            inferred_reason = None
            if processing_status != "completed":
                ds = entry.dataset_name or ""
                if ds in SKIP_PHASE1:
                    inferred_reason = f"Skipped by design ({ds} requires manual access)"
                elif phase == 4:
                    inferred_reason = "Manual / restricted access — needs portal registration"
                elif phase == 3 and processing_status != "completed":
                    inferred_reason = "Portal API (e.g. ESGF) — needs credentials in Settings + retry"
                elif not entry.link \
                        and entry.source_id not in DIRECT_DOWNLOAD_URLS \
                        and ds not in DIRECT_DOWNLOAD_URLS:
                    inferred_reason = "No download URL in catalog and no override defined"
                else:
                    # We have a URL, the dataset isn't blocked, the row just
                    # hasn't been attempted yet in this batch (or the previous
                    # attempt was killed before it ran).
                    inferred_reason = "Pending — click Process catalog to download"

            # Resolve any active ingest state for this row's dataset.
            # Manual scripts may use any key (source_id like
            # catalog_SPEI-GD_manual, or just the dataset name). We try
            # exact key match first, then fall through to scanning state
            # values for a `dataset_name` field that matches this row.
            ingest_progress = None
            candidate_keys = [k for k in (entry.source_id, entry.dataset_name) if k]
            state_entry = None
            for k in candidate_keys:
                if k in ingest_state:
                    state_entry = ingest_state[k]
                    break
            if state_entry is None and entry.dataset_name:
                for v in ingest_state.values():
                    if isinstance(v, dict) and v.get("dataset_name") == entry.dataset_name:
                        state_entry = v
                        break
            if state_entry and state_entry.get("done_files") is not None and state_entry.get("total_files"):
                done = int(state_entry["done_files"])
                total = int(state_entry["total_files"])
                is_partial = done < total and not state_entry.get("finished")
                # Auto-detect "killed mid-run": PID is dead OR heartbeat
                # is older than a few minutes. The user doesn't need to
                # tell us — we figure it out from process state + last
                # update timestamp.
                is_alive = False
                if is_partial:
                    pid = state_entry.get("pid")
                    if pid:
                        try:
                            import os as _os
                            _os.kill(int(pid), 0)  # signal 0 = liveness check
                            is_alive = True
                        except (OSError, ValueError):
                            is_alive = False
                    # Even if PID is somehow alive, treat as killed if no
                    # heartbeat in 5 min — script is stuck.
                    if is_alive and state_entry.get("updated_at"):
                        try:
                            from datetime import datetime, timezone, timedelta
                            ts = datetime.fromisoformat(state_entry["updated_at"].replace("Z", ""))
                            if datetime.utcnow() - ts > timedelta(minutes=5):
                                is_alive = False
                        except Exception:
                            pass
                ingest_progress = {
                    "done_files": done,
                    "total_files": total,
                    "updated_at": state_entry.get("updated_at"),
                    "script": state_entry.get("script"),
                    "is_partial": is_partial,
                    "is_alive": bool(is_alive),
                    "is_killed": is_partial and not is_alive,
                    "finished": bool(state_entry.get("finished", False)),
                }

            result.append(
                CatalogEntryResponse(
                    row_index=entry.row_index,
                    source_id=entry.source_id,
                    hazard=entry.hazard,
                    dataset_name=entry.dataset_name,
                    data_type=entry.data_type,
                    spatial_coverage=entry.spatial_coverage,
                    region_country=entry.region_country,
                    spatial_resolution=entry.spatial_resolution,
                    temporal_coverage=entry.temporal_coverage,
                    temporal_resolution=entry.temporal_resolution,
                    bias_corrected=entry.bias_corrected,
                    access=entry.access,
                    link=entry.link,
                    impact_sector=entry.impact_sector,
                    notes=entry.notes,
                    phase=phase,
                    processing_status=processing_status,
                    chunk_count=count,
                    error=db_error or inferred_reason,
                    last_phase=prog.get("phase"),
                    last_status=prog.get("status"),
                    ingest_progress=ingest_progress,
                )
            )
        return result
    except Exception as e:
        logger.error(f"Failed to list catalog: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@router.post("/process")
async def trigger_catalog_processing(request: CatalogProcessRequest):
    """Trigger batch processing of catalog entries in a background thread."""
    global _batch_thread, _batch_thread_error, _batch_thread_phases

    try:
        from src.catalog.batch_orchestrator import run_batch_pipeline

        if request.dry_run:
            result = run_batch_pipeline(
                excel_path=CATALOG_EXCEL_PATH,
                phases=request.phases,
                dry_run=True,
            )
            return result

        if _batch_thread is not None and _batch_thread.is_alive():
            raise HTTPException(
                409,
                "Batch processing is already running. Check /catalog/progress for status.",
            )

        _batch_thread_error = None
        _batch_thread_phases = request.phases

        def _run_in_background():
            global _batch_thread_error
            try:
                run_batch_pipeline(
                    excel_path=CATALOG_EXCEL_PATH,
                    phases=request.phases,
                    dry_run=False,
                    resume=not request.force_reprocess,
                )
            except Exception as e:
                import traceback

                tb = traceback.format_exc()
                _batch_thread_error = f"{e}\n{tb}"
                logger.error(f"Background catalog processing failed: {e}\n{tb}")

        _batch_thread = threading.Thread(target=_run_in_background, daemon=True, name="catalog-batch")
        _batch_thread.start()

        return {
            "status": "started",
            "phases": request.phases,
            "message": f"Catalog processing started for phases {request.phases}",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/progress", response_model=CatalogProgressResponse)
async def get_catalog_progress():
    """Get current batch processing progress."""
    try:
        from src.catalog.batch_orchestrator import get_progress

        data = get_progress()

        thread_alive = _batch_thread is not None and _batch_thread.is_alive()
        thread_crashed = _batch_thread is not None and not _batch_thread.is_alive() and _batch_thread_error is not None

        data["thread_alive"] = thread_alive
        data["thread_crashed"] = thread_crashed
        # Never leak internal tracebacks / file paths via the public API.
        # The full error is already in the server logs; the UI only needs to
        # know *that* the batch crashed so it can surface the "Restart" button.
        if thread_crashed:
            logger.error(f"Batch thread crashed: {_batch_thread_error}")
            data["thread_error"] = "Batch processing crashed. Check server logs for details."
        else:
            data["thread_error"] = None

        return data
    except Exception as e:
        logger.error(f"Failed to get progress: {e}")
        return CatalogProgressResponse()


@router.post("/classify")
async def classify_catalog():
    """Run classifier on all entries, return phase distribution."""
    try:
        from src.catalog.excel_reader import read_catalog
        from src.catalog.phase_classifier import classify_all

        entries = read_catalog(CATALOG_EXCEL_PATH)
        grouped = classify_all(entries)

        return {
            "total": len(entries),
            "phases": {str(phase): len(items) for phase, items in grouped.items()},
            "phase_descriptions": {
                "0": "Metadata-only (all entries)",
                "1": "Direct download, open access",
                "2": "Registration-required",
                "3": "API-based portals (CDS, ESGF)",
                "4": "Manual / contact-required",
            },
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/retry-failed")
async def retry_failed_catalog():
    """Re-run all failed catalog sources."""
    global _batch_thread, _batch_thread_error, _batch_thread_phases

    if _batch_thread is not None and _batch_thread.is_alive():
        raise HTTPException(409, "Batch processing is already running")

    from src.catalog.batch_orchestrator import retry_failed

    _batch_thread_error = None
    _batch_thread_phases = [3]

    def _retry():
        global _batch_thread_error
        try:
            retry_failed(excel_path=CATALOG_EXCEL_PATH)
        except Exception as e:
            import traceback

            _batch_thread_error = f"{e}\n{traceback.format_exc()}"
            logger.error(f"Retry failed: {e}")

    _batch_thread = threading.Thread(target=_retry, daemon=True)
    _batch_thread.start()

    return {"status": "started", "message": "Retrying failed sources in background thread"}


@router.post("/cancel")
async def cancel_catalog_batch():
    """Stop the current batch run.

    There is no clean Python signal we can send to a daemon thread blocked
    inside ``requests.get(...)``, so we set a stop flag the per-entry loop
    polls AND reset any rows stuck in ``status='processing'`` so the next
    /catalog/ render no longer pretends a long-dead run is still going.
    """
    global _batch_thread, _batch_thread_error

    from src.catalog.batch_orchestrator import set_cancel_flag, BatchProgress

    set_cancel_flag(True)
    progress = BatchProgress()
    reset = progress.mark_interrupted()

    alive = _batch_thread is not None and _batch_thread.is_alive()
    return {
        "cancelled": True,
        "thread_alive_at_call": alive,
        "rows_reset": reset,
        "note": (
            "Stop flag set. The active download will finish current chunk, "
            "then the loop exits. Rows in 'processing' have been moved to 'failed'."
        ),
    }


@router.post("/resume-ingest")
async def resume_ingest(payload: dict):
    """Re-launch a multi-file ingest script that was killed mid-run.

    The Catalog UI shows a "⚠ Killed N/M files" badge when the heartbeat
    in ``ingest_state.json`` is stale and the recorded PID no longer
    exists. Clicking Resume posts ``{source_id?, dataset_name?}`` here
    and we look up the script path the run registered, then spawn it
    again — the script's own ``done.txt`` resume marker takes over from
    there. No state is duplicated; this is just the docker-exec the user
    would otherwise have to type by hand.
    """
    import json as _json
    import subprocess
    import sys

    state_path = _PROJECT_ROOT / "data" / "ingest_state.json"
    if not state_path.exists():
        raise HTTPException(404, "No ingest_state.json — nothing to resume")

    try:
        state = _json.loads(state_path.read_text()) or {}
    except (OSError, _json.JSONDecodeError) as e:
        raise HTTPException(500, f"Could not read ingest_state.json: {e}")

    source_id = (payload or {}).get("source_id")
    dataset_name = (payload or {}).get("dataset_name")

    entry = None
    if source_id and source_id in state:
        entry = state[source_id]
    if entry is None and dataset_name:
        for v in state.values():
            if isinstance(v, dict) and v.get("dataset_name") == dataset_name:
                entry = v
                break
    if entry is None:
        raise HTTPException(
            404,
            "No matching ingest state. Pass source_id or dataset_name "
            "matching a row that has a Killed badge.",
        )

    script = entry.get("script")
    if not script or not Path(script).exists():
        raise HTTPException(
            400,
            f"Script path missing or not found inside the container: {script}. "
            "If you uploaded the source via Add Sources, re-upload the zip — "
            "older entries don't carry a resumable script path.",
        )

    # Already running? PID liveness check protects against double-launch.
    pid = entry.get("pid")
    if pid:
        try:
            os.kill(int(pid), 0)
            raise HTTPException(409, f"Ingest already running (pid {pid})")
        except (OSError, ValueError):
            pass

    # Detached so the subprocess survives this request returning.
    proc = subprocess.Popen(
        [sys.executable, script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    logger.info(f"resume-ingest: launched {script} as pid {proc.pid}")
    return {
        "status": "launched",
        "pid": proc.pid,
        "script": script,
        "dataset_name": entry.get("dataset_name"),
        "done_files": entry.get("done_files"),
        "total_files": entry.get("total_files"),
    }


@router.post("/{row_index}/retry")
async def retry_catalog_row(row_index: int):
    """Retry a single catalog row.

    Drops any prior ``catalog_progress`` entry for this source_id (so the
    resume guard doesn't skip it) and kicks off a fresh single-row Phase 1
    run in the same background-thread harness as ``/catalog/process``.
    """
    global _batch_thread, _batch_thread_error

    from src.catalog.excel_reader import read_catalog
    from src.catalog.batch_orchestrator import (
        run_batch_pipeline,
        clear_progress_for_source,
    )

    if _batch_thread is not None and _batch_thread.is_alive():
        raise HTTPException(409, "A batch is already running — cancel it first.")

    entries = read_catalog(CATALOG_EXCEL_PATH)
    target = next((e for e in entries if e.row_index == row_index), None)
    if target is None:
        raise HTTPException(404, f"Catalog row {row_index} not found")

    clear_progress_for_source(target.source_id)
    _batch_thread_error = None

    def _run_in_background():
        global _batch_thread_error
        try:
            run_batch_pipeline(
                excel_path=CATALOG_EXCEL_PATH,
                phases=[1],
                resume=True,
                row_filter={target.row_index},
            )
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            _batch_thread_error = f"{e}\n{tb}"
            logger.error(f"Single-row retry failed for row {row_index}: {e}\n{tb}")

    _batch_thread = threading.Thread(target=_run_in_background, daemon=True, name=f"catalog-retry-{row_index}")
    _batch_thread.start()
    return {
        "status": "retrying",
        "row_index": row_index,
        "source_id": target.source_id,
        "dataset_name": target.dataset_name,
    }


@router.post("/auto-restart")
async def auto_restart_catalog():
    """Detect crashed batch, reset interrupted entries, and restart processing."""
    global _batch_thread, _batch_thread_error, _batch_thread_phases

    if _batch_thread is not None and _batch_thread.is_alive():
        raise HTTPException(409, "Batch thread is still alive - no restart needed.")

    try:
        from src.catalog.batch_orchestrator import BatchProgress, run_batch_pipeline

        progress = BatchProgress()
        reset_count = progress.mark_interrupted()

        phases = _batch_thread_phases or [0, 1]
        if progress.current_phase is not None and progress.current_phase not in phases:
            phases = sorted(set(phases) | {progress.current_phase})

        _batch_thread_error = None

        def _run_in_background():
            global _batch_thread_error
            try:
                run_batch_pipeline(
                    excel_path=CATALOG_EXCEL_PATH,
                    phases=phases,
                    resume=True,
                )
            except Exception as e:
                import traceback

                tb = traceback.format_exc()
                _batch_thread_error = f"{e}\n{tb}"
                logger.error(f"Auto-restart batch failed: {e}\n{tb}")

        _batch_thread = threading.Thread(target=_run_in_background, daemon=True, name="catalog-batch-restart")
        _batch_thread.start()

        return {
            "status": "restarted",
            "entries_reset": reset_count,
            "phases": phases,
            "message": f"Reset {reset_count} interrupted entries and restarted phases {phases}",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/{row_index}")
async def get_catalog_entry(row_index: int):
    """Get a single catalog entry by row index."""
    try:
        from src.catalog.excel_reader import read_catalog
        from src.catalog.phase_classifier import classify_source
        from src.catalog.batch_orchestrator import BatchProgress

        entries = read_catalog(CATALOG_EXCEL_PATH)
        progress = BatchProgress()

        for entry in entries:
            if entry.row_index == row_index:
                phase = classify_source(entry)
                status_info = progress.get_source_info(entry.source_id)
                return {
                    **entry.to_dict(),
                    "phase": phase,
                    "processing_status": status_info.get("status", "pending"),
                    "processing_error": status_info.get("error"),
                }
        raise HTTPException(404, f"Catalog entry {row_index} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
