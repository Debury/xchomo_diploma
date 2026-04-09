"""Dagster repository exposing jobs, schedules, and sensors to Dagit."""

import sys
from pathlib import Path

# When Dagster loads this module inside the container, ensure /app is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.exists() and str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from dagster import Definitions

from dagster_project.dynamic_jobs import dynamic_source_etl_job
from dagster_project.catalog_jobs import (
    batch_catalog_etl_job,
    catalog_metadata_only_job,
    catalog_full_etl_job,
)
from dagster_project.source_jobs import single_source_etl_job
from dagster_project.schedules import (
    source_schedule_sensor,
    data_freshness_sensor,
    weekly_catalog_refresh,
)
from dagster_project.resources import (
    ConfigLoaderResource,
    LoggerResource,
    DataPathResource,
    DatabaseResource,
)


# ====================================================================================
# REPOSITORY DEFINITION
# ====================================================================================

climate_etl_repository = Definitions(
    jobs=[
        dynamic_source_etl_job,       # Legacy: process all active sources
        batch_catalog_etl_job,        # Catalog: Phase 0 + Phase 1
        catalog_metadata_only_job,    # Catalog: Phase 0 only
        catalog_full_etl_job,         # Catalog: Phase 0 → 1 → 2 → 3
        single_source_etl_job,        # Single source ETL (used by sensors)
    ],
    schedules=[
        weekly_catalog_refresh,
    ],
    sensors=[
        source_schedule_sensor,
        data_freshness_sensor,
    ],
    resources={
        "config_loader": ConfigLoaderResource(config_path="config/pipeline_config.yaml"),
        "logger": LoggerResource(log_file="logs/dagster_pipeline.log", log_level="INFO"),
        "data_paths": DataPathResource(
            raw_data_dir="data/raw",
            processed_data_dir="data/processed",
            embeddings_dir="qdrant_db",
        ),
        "database": DatabaseResource(),
    },
)


__all__ = ["climate_etl_repository"]
