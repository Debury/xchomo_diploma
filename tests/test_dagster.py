"""
Tests for Dagster Components

Tests for ops, resources, schedules, and sensors using Dagster test utilities.
"""

import pytest
from datetime import datetime

from dagster import build_op_context, DagsterInstance, build_schedule_context, build_sensor_context
from dagster_project.resources import ConfigLoaderResource, LoggerResource, DataPathResource
from dagster_project.schedules import daily_etl_schedule, new_processed_data_sensor


# ====================================================================================
# RESOURCE TESTS
# ====================================================================================

class TestResources:
    """Test Dagster resources"""

    def test_config_loader_resource_initialization(self):
        resource = ConfigLoaderResource(config_path="config/pipeline_config.yaml")
        assert resource.config_path == "config/pipeline_config.yaml"

    def test_logger_resource_initialization(self):
        resource = LoggerResource(log_level="INFO", log_file="logs/test.log")
        assert resource.log_level == "INFO"
        assert resource.log_file == "logs/test.log"

    def test_data_path_resource_initialization(self):
        resource = DataPathResource(
            raw_data_dir="data/raw",
            processed_data_dir="data/processed",
            embeddings_dir="chroma_db"
        )
        assert resource.raw_data_dir == "data/raw"
        assert resource.processed_data_dir == "data/processed"


# ====================================================================================
# SCHEDULE TESTS
# ====================================================================================

class TestSchedules:
    """Test Dagster schedules"""

    def test_daily_etl_schedule_evaluation(self):
        instance = DagsterInstance.ephemeral()
        context = build_schedule_context(instance=instance)

        result = daily_etl_schedule(context)

        assert result is not None
        assert result.run_key is not None
        assert "daily_etl_" in result.run_key
        assert result.tags is not None
        assert "date" in result.tags

    def test_daily_etl_schedule_cron(self):
        assert daily_etl_schedule.cron_schedule == "0 2 * * *"


# ====================================================================================
# SENSOR TESTS
# ====================================================================================

class TestSensors:
    """Test Dagster sensors"""

    def test_new_processed_data_sensor_no_files(self):
        instance = DagsterInstance.ephemeral()
        context = build_sensor_context(instance=instance)

        results = list(new_processed_data_sensor(context))
        assert len(results) == 0

    def test_new_processed_data_sensor_structure(self):
        assert new_processed_data_sensor is not None
        assert new_processed_data_sensor.minimum_interval_seconds == 300


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
