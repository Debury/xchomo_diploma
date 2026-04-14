"""
Dagster Schedules and Sensors for Climate ETL Pipeline.

Real implementations replacing the placeholder stubs:
- source_schedule_sensor: polls source_schedules table, triggers per-source ETL
- data_freshness_sensor: detects stale sources (>30 days), triggers refresh
- weekly_catalog_refresh: real schedule for Phase 0 metadata refresh
"""

import logging
from datetime import datetime, timedelta

from dagster import (
    schedule,
    ScheduleEvaluationContext,
    RunRequest,
    sensor,
    SensorEvaluationContext,
    DefaultScheduleStatus,
    DefaultSensorStatus,
    SkipReason,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# SENSORS
# ────────────────────────────────────────────────────────────────────

@sensor(
    job_name="single_source_etl_job",
    default_status=DefaultSensorStatus.RUNNING,
    description="Polls source_schedules table every 60s, triggers ETL for due sources",
    minimum_interval_seconds=60,
)
def source_schedule_sensor(context: SensorEvaluationContext):
    """Check for sources and datasets due for scheduled processing."""
    try:
        from src.database.source_store import SourceStore
        store = SourceStore()
        triggered = 0

        # 1. Per-source schedules
        due_schedules = store.get_due_schedules()
        for sched in due_schedules:
            source_id = sched["source_id"]
            context.log.info(f"Triggering scheduled ETL for source {source_id}")
            yield RunRequest(
                run_key=f"schedule_{source_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                tags={
                    "source_id": source_id,
                    "trigger_type": "schedule",
                    "cron": sched.get("cron_expression", ""),
                },
            )
            store.advance_schedule(source_id)
            triggered += 1

        # 2. Dataset-level schedules — trigger ALL sources under the dataset
        due_dataset_scheds = store.get_due_dataset_schedules()
        for ds_sched in due_dataset_scheds:
            dataset_name = ds_sched["dataset_name"]
            source_ids = store.get_sources_for_dataset(dataset_name)
            context.log.info(
                f"Dataset schedule '{ds_sched['name']}' triggered for "
                f"{dataset_name}: {len(source_ids)} sources"
            )
            for source_id in source_ids:
                yield RunRequest(
                    run_key=f"ds_sched_{source_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
                    tags={
                        "source_id": source_id,
                        "trigger_type": "dataset_schedule",
                        "dataset_schedule": ds_sched["name"],
                        "dataset_name": dataset_name,
                    },
                )
                triggered += 1
            store.advance_dataset_schedule(ds_sched["id"])

        if triggered == 0:
            return SkipReason("No sources or datasets due for scheduled processing")

    except ImportError:
        return SkipReason("PostgreSQL store not available")
    except Exception as e:
        context.log.error(f"Schedule sensor error: {e}")
        return SkipReason(f"Error: {e}")


@sensor(
    job_name="single_source_etl_job",
    default_status=DefaultSensorStatus.RUNNING,
    description="Detects stale sources (>30 days since last update) hourly",
    minimum_interval_seconds=3600,
)
def data_freshness_sensor(context: SensorEvaluationContext):
    """Check for stale sources that haven't been updated in 30+ days."""
    try:
        from src.database.connection import get_db_session
        from src.database.models import Source

        stale_threshold = datetime.utcnow() - timedelta(days=30)
        triggered = 0

        with get_db_session() as session:
            stale_sources = (
                session.query(Source)
                .filter(
                    Source.is_active == True,
                    Source.processing_status == "completed",
                    Source.last_processed_at < stale_threshold,
                )
                .all()
            )

            if not stale_sources:
                return SkipReason("No stale sources found")

            for source in stale_sources:
                context.log.info(
                    f"Source {source.source_id} is stale "
                    f"(last processed: {source.last_processed_at})"
                )
                yield RunRequest(
                    run_key=f"freshness_{source.source_id}_{datetime.utcnow().strftime('%Y%m%d')}",
                    tags={
                        "source_id": source.source_id,
                        "trigger_type": "freshness",
                    },
                )
                triggered += 1

        context.log.info(f"Freshness sensor triggered {triggered} runs")

    except ImportError:
        return SkipReason("PostgreSQL store not available")
    except Exception as e:
        context.log.error(f"Freshness sensor error: {e}")
        return SkipReason(f"Error: {e}")


# ────────────────────────────────────────────────────────────────────
# SCHEDULES
# ────────────────────────────────────────────────────────────────────

@schedule(
    job_name="catalog_metadata_only_job",
    cron_schedule="0 3 * * 0",  # Sunday 3am
    default_status=DefaultScheduleStatus.RUNNING,
    description="Weekly catalog Phase 0 metadata refresh (Sundays 3am)",
)
def weekly_catalog_refresh(context: ScheduleEvaluationContext):
    """Weekly refresh of catalog metadata embeddings."""
    return RunRequest(
        tags={"trigger_type": "schedule", "schedule": "weekly_catalog_refresh"},
    )
