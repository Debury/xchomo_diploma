"""
Dagster job definitions for batch catalog processing.

Provides a job that reads the D1.1.xlsx catalog and processes entries
through the metadata embedding pipeline (Phase 0) and optionally
through download/embed phases (1-3).
"""

import os
from pathlib import Path

from dagster import job, op, In, Out, Nothing, OpExecutionContext, AssetMaterialization, MetadataValue


EXCEL_PATH = os.getenv("CATALOG_EXCEL_PATH", "Kopie souboru D1.1.xlsx")


@op(
    description="Read Excel catalog and classify sources by phase",
    out={"catalog_summary": Out(dict)},
)
def read_and_classify_catalog(context: OpExecutionContext) -> dict:
    """Read the Excel catalog and produce phase classification."""
    from src.catalog.excel_reader import read_catalog
    from src.catalog.phase_classifier import classify_all

    excel_path = Path(EXCEL_PATH)
    if not excel_path.exists():
        # Try relative to project root
        project_root = Path(__file__).resolve().parents[1]
        excel_path = project_root / EXCEL_PATH

    entries = read_catalog(str(excel_path))
    grouped = classify_all(entries)

    summary = {
        "total": len(entries),
        "phases": {str(phase): len(items) for phase, items in grouped.items()},
        "excel_path": str(excel_path),
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


@op(
    description="Process all catalog entries as metadata-only (Phase 0)",
    ins={"catalog_summary": In(dict)},
    out={"phase0_result": Out(dict)},
)
def run_phase0_metadata(context: OpExecutionContext, catalog_summary: dict) -> dict:
    """Embed catalog metadata into Qdrant (no data download)."""
    from src.catalog.batch_orchestrator import run_batch_pipeline

    excel_path = catalog_summary["excel_path"]
    context.log.info(f"Phase 0: processing metadata for {catalog_summary['total']} entries")

    result = run_batch_pipeline(
        excel_path=excel_path,
        phases=[0],
        resume=True,
    )

    context.log.info(f"Phase 0 result: {result.get('summary', {})}")

    context.log_event(
        AssetMaterialization(
            asset_key="catalog_phase0",
            metadata={
                "processed": MetadataValue.int(result.get("phase_0", {}).get("processed", 0)),
                "failed": MetadataValue.int(result.get("phase_0", {}).get("failed", 0)),
            },
        )
    )

    return result


@op(
    description="Process direct-download sources (Phase 1)",
    ins={"phase0_result": In(dict)},
)
def run_phase1_downloads(context: OpExecutionContext, phase0_result: dict):
    """Download and process open-access data sources."""
    from src.catalog.batch_orchestrator import run_batch_pipeline

    excel_path = phase0_result.get("summary", {}).get("excel_path") or str(
        Path(__file__).resolve().parents[1] / EXCEL_PATH
    )

    context.log.info("Phase 1: processing direct-download sources")

    result = run_batch_pipeline(
        excel_path=excel_path,
        phases=[1],
        resume=True,
    )

    context.log.info(f"Phase 1 result: {result.get('summary', {})}")

    context.log_event(
        AssetMaterialization(
            asset_key="catalog_phase1",
            metadata={
                "processed": MetadataValue.int(result.get("phase_1", {}).get("processed", 0)),
                "failed": MetadataValue.int(result.get("phase_1", {}).get("failed", 0)),
            },
        )
    )


@job(description="Batch process D1.1.xlsx climate catalog: classify → embed metadata → download data")
def batch_catalog_etl_job():
    """Full catalog ETL pipeline: Phase 0 (metadata) → Phase 1 (downloads)."""
    summary = read_and_classify_catalog()
    phase0 = run_phase0_metadata(summary)
    run_phase1_downloads(phase0)


@job(description="Quick metadata-only processing of the Excel catalog (Phase 0)")
def catalog_metadata_only_job():
    """Phase 0 only: embed Excel metadata into Qdrant."""
    summary = read_and_classify_catalog()
    run_phase0_metadata(summary)
