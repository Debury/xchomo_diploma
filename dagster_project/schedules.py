"""
Dagster Schedules and Sensors for Climate ETL Pipeline

NOTE: Legacy schedules/sensors are commented out as they depend on legacy jobs.
These are placeholder implementations to keep repository.py working.
"""

from dagster import (
    schedule,
    ScheduleEvaluationContext,
    RunRequest,
    sensor,
    SensorEvaluationContext,
    DefaultScheduleStatus,
    DefaultSensorStatus
)
from dagster_project.dynamic_jobs import dynamic_source_etl_job


# ====================================================================================
# SCHEDULES - PLACEHOLDERS
# ====================================================================================

@schedule(
    job=dynamic_source_etl_job,
    cron_schedule="0 2 * * *",
    default_status=DefaultScheduleStatus.STOPPED,
    description="Placeholder - not actively used"
)
def daily_etl_schedule(context: ScheduleEvaluationContext):
    """Placeholder schedule - STOPPED by default."""
    return None


@schedule(
    job=dynamic_source_etl_job,
    cron_schedule="0 3 * * *",
    default_status=DefaultScheduleStatus.STOPPED,
    description="Placeholder - not actively used"
)
def daily_embedding_schedule(context: ScheduleEvaluationContext):
    """Placeholder schedule - STOPPED by default."""
    return None


@schedule(
    job=dynamic_source_etl_job,
    cron_schedule="0 1 * * 0",
    default_status=DefaultScheduleStatus.STOPPED,
    description="Placeholder - not actively used"
)
def weekly_complete_schedule(context: ScheduleEvaluationContext):
    """Placeholder schedule - STOPPED by default."""
    return None


# ====================================================================================
# SENSORS - PLACEHOLDERS
# ====================================================================================

@sensor(
    job=dynamic_source_etl_job,
    default_status=DefaultSensorStatus.STOPPED,
    description="Placeholder - not actively used",
    minimum_interval_seconds=300
)
def new_processed_data_sensor(context: SensorEvaluationContext):
    """Placeholder sensor - STOPPED by default."""
    return


@sensor(
    job=dynamic_source_etl_job,
    default_status=DefaultSensorStatus.STOPPED,
    description="Placeholder - not actively used",
    minimum_interval_seconds=600
)
def data_quality_sensor(context: SensorEvaluationContext):
    """Placeholder sensor - STOPPED by default."""
    return


@sensor(
    job=dynamic_source_etl_job,
    default_status=DefaultSensorStatus.STOPPED,
    description="Placeholder - not actively used",
    minimum_interval_seconds=3600
)
def config_change_sensor(context: SensorEvaluationContext):
    """Placeholder sensor - STOPPED by default."""
    return

