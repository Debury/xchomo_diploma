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
    """List all catalog entries with phase classification and processing status."""
    try:
        from src.catalog.excel_reader import read_catalog
        from src.catalog.phase_classifier import classify_source
        from src.catalog.batch_orchestrator import BatchProgress

        entries = read_catalog(CATALOG_EXCEL_PATH)
        progress = BatchProgress()

        result = []
        for entry in entries:
            phase = classify_source(entry)
            processing_status = progress.get_overall_status(entry.source_id, phase)

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
                )
            )
        return result
    except Exception as e:
        logger.error(f"Failed to list catalog: {e}")
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
        data["thread_error"] = _batch_thread_error if thread_crashed else None

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
