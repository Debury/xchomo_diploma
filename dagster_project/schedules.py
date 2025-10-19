"""
Dagster Schedules and Sensors for Climate ETL Pipeline

Defines automated execution triggers for jobs.
"""

from dagster import (
    schedule,
    ScheduleEvaluationContext,
    RunRequest,
    sensor,
    SensorEvaluationContext,
    RunConfig,
    DefaultScheduleStatus,
    DefaultSensorStatus
)
from datetime import datetime
from pathlib import Path
import json

from dagster_project.jobs import (
    daily_etl_job,
    embedding_job,
    complete_pipeline_job,
    validation_job
)


# ====================================================================================
# SCHEDULES
# ====================================================================================

@schedule(
    job=daily_etl_job,
    cron_schedule="0 2 * * *",  # Run at 2 AM every day
    default_status=DefaultScheduleStatus.STOPPED,  # Start in stopped state for safety
    description="Daily schedule for ETL pipeline execution"
)
def daily_etl_schedule(context: ScheduleEvaluationContext):
    """
    Daily schedule that triggers the ETL job at 2 AM.
    
    Downloads data for the previous day and processes it.
    
    Args:
        context: Schedule evaluation context
    
    Returns:
        RunRequest with appropriate configuration
    """
    # Calculate date for data to download (yesterday)
    from datetime import datetime, timedelta
    yesterday = datetime.now() - timedelta(days=1)
    
    run_config = RunConfig(
        ops={
            "download_era5_data": {
                "config": {
                    "variables": ["2m_temperature", "total_precipitation"],
                    "year": yesterday.year,
                    "month": yesterday.month,
                    "days": [yesterday.day]
                }
            },
            "transform_data": {
                "config": {
                    "convert_temperature": True,
                    "rename_dimensions": True,
                    "normalize": False,
                    "aggregate_to_daily": False
                }
            }
        }
    )
    
    return RunRequest(
        run_key=f"daily_etl_{yesterday.strftime('%Y%m%d')}",
        run_config=run_config,
        tags={
            "schedule": "daily",
            "date": yesterday.strftime('%Y-%m-%d')
        }
    )


@schedule(
    job=embedding_job,
    cron_schedule="0 4 * * *",  # Run at 4 AM every day (after ETL)
    default_status=DefaultScheduleStatus.STOPPED,
    description="Daily schedule for embedding generation"
)
def daily_embedding_schedule(context: ScheduleEvaluationContext):
    """
    Daily schedule that triggers embedding generation at 4 AM.
    
    Runs after the ETL job to process newly downloaded data.
    
    Args:
        context: Schedule evaluation context
    
    Returns:
        RunRequest with appropriate configuration
    """
    run_config = RunConfig(
        ops={
            "generate_embeddings_standalone": {
                "config": {
                    "batch_size": 64,
                    "model_name": "all-MiniLM-L6-v2",
                    "process_directory": True
                }
            }
        }
    )
    
    return RunRequest(
        run_key=f"daily_embeddings_{datetime.now().strftime('%Y%m%d')}",
        run_config=run_config,
        tags={
            "schedule": "daily_embeddings",
            "date": datetime.now().strftime('%Y-%m-%d')
        }
    )


@schedule(
    job=complete_pipeline_job,
    cron_schedule="0 3 * * 0",  # Run at 3 AM every Sunday
    default_status=DefaultScheduleStatus.STOPPED,
    description="Weekly schedule for complete pipeline execution"
)
def weekly_complete_schedule(context: ScheduleEvaluationContext):
    """
    Weekly schedule that triggers the complete pipeline every Sunday.
    
    Runs full pipeline including all phases for comprehensive processing.
    
    Args:
        context: Schedule evaluation context
    
    Returns:
        RunRequest with appropriate configuration
    """
    from datetime import datetime, timedelta
    
    # Process previous week's data
    today = datetime.now()
    week_start = today - timedelta(days=7)
    
    run_config = RunConfig(
        ops={
            "download_era5_data": {
                "config": {
                    "variables": ["2m_temperature", "total_precipitation", "surface_pressure"],
                    "year": week_start.year,
                    "month": week_start.month
                }
            },
            "transform_data": {
                "config": {
                    "convert_temperature": True,
                    "rename_dimensions": True,
                    "normalize": True,
                    "aggregate_to_daily": True
                }
            },
            "generate_embeddings": {
                "config": {
                    "batch_size": 128,
                    "model_name": "all-MiniLM-L6-v2",
                    "process_directory": True
                }
            }
        }
    )
    
    return RunRequest(
        run_key=f"weekly_complete_{today.strftime('%Y_W%W')}",
        run_config=run_config,
        tags={
            "schedule": "weekly",
            "week": today.strftime('%Y-W%W')
        }
    )


# ====================================================================================
# SENSORS
# ====================================================================================

@sensor(
    job=embedding_job,
    default_status=DefaultSensorStatus.STOPPED,
    description="Sensor that triggers embedding job when new processed data is available",
    minimum_interval_seconds=300  # Check every 5 minutes
)
def new_processed_data_sensor(context: SensorEvaluationContext):
    """
    Sensor that monitors for new processed data files and triggers embedding generation.
    
    This sensor watches the processed data directory and triggers the embedding job
    when new files are detected.
    
    Args:
        context: Sensor evaluation context
    
    Yields:
        RunRequest when new data is detected
    """
    processed_dir = Path("data/processed")
    
    if not processed_dir.exists():
        context.log.info("Processed data directory does not exist yet")
        return
    
    # Get cursor (last check timestamp)
    cursor_dict = json.loads(context.cursor) if context.cursor else {}
    last_check = cursor_dict.get("last_check_timestamp", 0)
    
    # Find new files since last check
    new_files = []
    current_timestamp = datetime.now().timestamp()
    
    for file_path in processed_dir.glob("*.nc"):
        file_mtime = file_path.stat().st_mtime
        if file_mtime > last_check:
            new_files.append(str(file_path))
    
    if new_files:
        context.log.info(f"Detected {len(new_files)} new processed files")
        
        run_config = RunConfig(
            ops={
                "generate_embeddings_standalone": {
                    "config": {
                        "batch_size": 64,
                        "model_name": "all-MiniLM-L6-v2",
                        "process_directory": True
                    }
                }
            }
        )
        
        # Update cursor
        new_cursor = json.dumps({"last_check_timestamp": current_timestamp})
        
        yield RunRequest(
            run_key=f"sensor_embedding_{int(current_timestamp)}",
            run_config=run_config,
            tags={
                "trigger": "sensor",
                "new_files": str(len(new_files))
            }
        )
        
        context.update_cursor(new_cursor)
    else:
        # Update cursor even if no new files
        new_cursor = json.dumps({"last_check_timestamp": current_timestamp})
        context.update_cursor(new_cursor)


@sensor(
    job=validation_job,
    default_status=DefaultSensorStatus.STOPPED,
    description="Sensor that validates data quality on schedule",
    minimum_interval_seconds=3600  # Check every hour
)
def data_quality_sensor(context: SensorEvaluationContext):
    """
    Sensor that periodically checks data quality.
    
    This sensor runs validation checks on existing data to ensure quality
    and can trigger alerts if issues are detected.
    
    Args:
        context: Sensor evaluation context
    
    Yields:
        RunRequest when validation should be performed
    """
    raw_dir = Path("data/raw")
    
    if not raw_dir.exists():
        context.log.info("Raw data directory does not exist yet")
        return
    
    # Check if there are files to validate
    nc_files = list(raw_dir.glob("*.nc"))
    
    if not nc_files:
        context.log.info("No NetCDF files found for validation")
        return
    
    # Get cursor to track last validation
    cursor_dict = json.loads(context.cursor) if context.cursor else {}
    last_validation = cursor_dict.get("last_validation_timestamp", 0)
    current_timestamp = datetime.now().timestamp()
    
    # Run validation every 6 hours
    if current_timestamp - last_validation > 21600:  # 6 hours in seconds
        context.log.info(f"Running data quality validation on {len(nc_files)} files")
        
        # Update cursor
        new_cursor = json.dumps({"last_validation_timestamp": current_timestamp})
        
        yield RunRequest(
            run_key=f"validation_{int(current_timestamp)}",
            run_config=RunConfig(),
            tags={
                "trigger": "quality_check",
                "num_files": str(len(nc_files))
            }
        )
        
        context.update_cursor(new_cursor)
    else:
        context.log.debug("Skipping validation, not enough time elapsed since last check")


@sensor(
    job=daily_etl_job,
    default_status=DefaultSensorStatus.STOPPED,
    description="Sensor that monitors configuration changes and triggers pipeline",
    minimum_interval_seconds=600  # Check every 10 minutes
)
def config_change_sensor(context: SensorEvaluationContext):
    """
    Sensor that monitors configuration file changes.
    
    When pipeline_config.yaml is modified, this sensor can trigger
    a pipeline run to apply new configurations.
    
    Args:
        context: Sensor evaluation context
    
    Yields:
        RunRequest when configuration changes are detected
    """
    config_file = Path("config/pipeline_config.yaml")
    
    if not config_file.exists():
        context.log.warning("Configuration file not found")
        return
    
    # Get cursor (last known config modification time)
    cursor_dict = json.loads(context.cursor) if context.cursor else {}
    last_mtime = cursor_dict.get("config_mtime", 0)
    
    current_mtime = config_file.stat().st_mtime
    
    if current_mtime > last_mtime and last_mtime > 0:  # Skip on first run
        context.log.info("Configuration file changed, triggering pipeline run")
        
        # Update cursor
        new_cursor = json.dumps({"config_mtime": current_mtime})
        
        yield RunRequest(
            run_key=f"config_change_{int(current_mtime)}",
            run_config=RunConfig(
                ops={
                    "transform_data": {
                        "config": {
                            "convert_temperature": True,
                            "rename_dimensions": True,
                            "normalize": False,
                            "aggregate_to_daily": False
                        }
                    }
                }
            ),
            tags={
                "trigger": "config_change",
                "config_mtime": str(current_mtime)
            }
        )
        
        context.update_cursor(new_cursor)
    else:
        # Update cursor on first run
        if last_mtime == 0:
            new_cursor = json.dumps({"config_mtime": current_mtime})
            context.update_cursor(new_cursor)

