"""Dagster repository exposing jobs, schedules, and sensors to Dagit."""

import sys
from pathlib import Path

# When Dagster loads this module inside the container, ensure /app is on sys.path
# so that the top-level `src` package can be imported.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if PROJECT_ROOT.exists() and str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Some interactive environments only mount /app/src, so fall back to that if needed.
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.append(str(SRC_PATH))

from dagster import Definitions

from dagster_project.jobs import (
    daily_etl_job,
    embedding_job,
    complete_pipeline_job,
    validation_job
)
from dagster_project.dynamic_jobs import (
    dynamic_source_etl_job
)
from dagster_project.schedules import (
    daily_etl_schedule,
    daily_embedding_schedule,
    weekly_complete_schedule,
    new_processed_data_sensor,
    data_quality_sensor,
    config_change_sensor
)
from dagster_project.resources import (
    ConfigLoaderResource,
    LoggerResource,
    DataPathResource
)


# ====================================================================================
# REPOSITORY DEFINITION
# ====================================================================================

climate_etl_repository = Definitions(
    jobs=[
        daily_etl_job,
        embedding_job,
        complete_pipeline_job,
        validation_job,
        dynamic_source_etl_job  # Phase 5: Dynamic source-driven ETL
    ],
    schedules=[
        daily_etl_schedule,
        daily_embedding_schedule,
        weekly_complete_schedule
    ],
    sensors=[
        new_processed_data_sensor,
        data_quality_sensor,
        config_change_sensor
    ],
    resources={
        "config_loader": ConfigLoaderResource(config_path="config/pipeline_config.yaml"),
        "logger": LoggerResource(log_file="logs/dagster_pipeline.log", log_level="INFO"),
        "data_paths": DataPathResource(
            raw_data_dir="data/raw",
            processed_data_dir="data/processed",
            embeddings_dir="chroma_db"
        )
    }
)


# Export for easy import
__all__ = ["climate_etl_repository"]
