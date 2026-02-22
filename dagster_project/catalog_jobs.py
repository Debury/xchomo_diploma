"""
Dagster job definitions for batch catalog processing.

Provides granular per-phase ops and composite jobs for the D1.1.xlsx catalog.
Each op wraps the existing batch_orchestrator functions to preserve retry
logic and memory guards.
"""

import os
from pathlib import Path

from dagster import (
    job, op, In, Out, Nothing, OpExecutionContext,
    AssetMaterialization, MetadataValue,
)


EXCEL_PATH = os.getenv("CATALOG_EXCEL_PATH", "Kopie souboru D1.1.xlsx")


def _resolve_excel_path(hint: str = None) -> str:
    """Resolve the Excel path, trying several locations."""
    path = Path(hint or EXCEL_PATH)
    if path.exists():
        return str(path)
    project_root = Path(__file__).resolve().parents[1]
    candidate = project_root / EXCEL_PATH
    if candidate.exists():
        return str(candidate)
    # Docker mount
    docker_candidate = Path("/app/data") / Path(EXCEL_PATH).name
    if docker_candidate.exists():
        return str(docker_candidate)
    return str(path)


# ────────────────────────────────────────────────────────────────────
# Shared: read & classify
# ────────────────────────────────────────────────────────────────────

@op(
    description="Read Excel catalog and classify sources by phase",
    out={"catalog_summary": Out(dict)},
)
def read_and_classify_catalog(context: OpExecutionContext) -> dict:
    from src.catalog.excel_reader import read_catalog
    from src.catalog.phase_classifier import classify_all

    excel_path = _resolve_excel_path()
    entries = read_catalog(excel_path)
    grouped = classify_all(entries)

    summary = {
        "total": len(entries),
        "phases": {str(phase): len(items) for phase, items in grouped.items()},
        "excel_path": excel_path,
    }

    context.log.info(f"Catalog: {len(entries)} entries, phases: {summary['phases']}")

    context.log_event(
        AssetMaterialization(
            asset_key="catalog_classification",
            metadata={
                "total_entries": MetadataValue.int(len(entries)),
                "phase_distribution": MetadataValue.json(summary["phases"]),
            },
        )
    )
    return summary


# ────────────────────────────────────────────────────────────────────
# Per-phase ops
# ────────────────────────────────────────────────────────────────────

def _run_phase_op(context: OpExecutionContext, excel_path: str, phase: int) -> dict:
    """Shared logic for running a single phase via batch_orchestrator."""
    from src.catalog.batch_orchestrator import run_batch_pipeline

    context.log.info(f"Phase {phase}: starting processing")

    result = run_batch_pipeline(
        excel_path=excel_path,
        phases=[phase],
        resume=True,
    )

    phase_key = f"phase_{phase}"
    phase_result = result.get(phase_key, {})
    context.log.info(f"Phase {phase} result: {phase_result}")

    context.log_event(
        AssetMaterialization(
            asset_key=f"catalog_phase{phase}",
            metadata={
                "processed": MetadataValue.int(phase_result.get("processed", 0)),
                "failed": MetadataValue.int(phase_result.get("failed", 0)),
                "skipped": MetadataValue.int(phase_result.get("skipped", 0)),
            },
        )
    )

    # Record processing run in PostgreSQL if available
    try:
        from src.database.source_store import SourceStore
        store = SourceStore()
        run_id = store.record_processing_run(
            source_id=f"catalog_phase_{phase}",
            job_name=context.job_name,
            dagster_run_id=context.run_id,
            phase=phase,
            trigger_type="dagster",
        )
        status = "completed" if phase_result.get("failed", 0) == 0 else "partial"
        store.complete_processing_run(
            run_id=run_id,
            status=status,
            chunks_processed=phase_result.get("processed", 0),
        )
    except Exception:
        pass  # DB not available — that's fine

    return result


@op(
    description="Process all catalog entries as metadata-only (Phase 0)",
    ins={"catalog_summary": In(dict)},
    out={"phase0_result": Out(dict)},
)
def catalog_phase0_op(context: OpExecutionContext, catalog_summary: dict) -> dict:
    excel_path = catalog_summary["excel_path"]
    context.log.info(f"Phase 0: processing metadata for {catalog_summary['total']} entries")
    return _run_phase_op(context, excel_path, 0)


@op(
    description="Process direct-download sources (Phase 1)",
    ins={"prev_result": In(dict)},
    out={"phase1_result": Out(dict)},
)
def catalog_phase1_op(context: OpExecutionContext, prev_result: dict) -> dict:
    excel_path = prev_result.get("summary", {}).get("excel_path") or _resolve_excel_path()
    return _run_phase_op(context, excel_path, 1)


@op(
    description="Process registration-required sources (Phase 2)",
    ins={"prev_result": In(dict)},
    out={"phase2_result": Out(dict)},
)
def catalog_phase2_op(context: OpExecutionContext, prev_result: dict) -> dict:
    excel_path = prev_result.get("summary", {}).get("excel_path") or _resolve_excel_path()
    return _run_phase_op(context, excel_path, 2)


@op(
    description="Process API portal sources (Phase 3: CDS, ESGF)",
    ins={"prev_result": In(dict)},
)
def catalog_phase3_op(context: OpExecutionContext, prev_result: dict):
    excel_path = prev_result.get("summary", {}).get("excel_path") or _resolve_excel_path()
    _run_phase_op(context, excel_path, 3)


# ────────────────────────────────────────────────────────────────────
# Jobs
# ────────────────────────────────────────────────────────────────────

@job(description="Full catalog ETL: classify → phase0 → phase1 → phase2 → phase3")
def catalog_full_etl_job():
    summary = read_and_classify_catalog()
    p0 = catalog_phase0_op(summary)
    p1 = catalog_phase1_op(p0)
    p2 = catalog_phase2_op(p1)
    catalog_phase3_op(p2)


@job(description="Batch process D1.1.xlsx: classify → phase0 (metadata) → phase1 (downloads)")
def batch_catalog_etl_job():
    summary = read_and_classify_catalog()
    p0 = catalog_phase0_op(summary)
    catalog_phase1_op(p0)


@job(description="Quick metadata-only processing of the Excel catalog (Phase 0)")
def catalog_metadata_only_job():
    summary = read_and_classify_catalog()
    catalog_phase0_op(summary)
