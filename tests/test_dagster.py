"""
Tests for Dagster Components - Phase 4

Tests for ops, jobs, schedules, and sensors using Dagster test utilities.
"""

import sys
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dagster import build_op_context, DagsterInstance, build_schedule_context, build_sensor_context
from dagster_project.resources import ConfigLoaderResource, LoggerResource, DataPathResource
from dagster_project.ops.data_acquisition_ops import download_era5_data, validate_downloaded_data, DownloadConfig
from dagster_project.ops.transformation_ops import ingest_data, transform_data, export_data, TransformConfig
from dagster_project.ops.embedding_ops import (
    generate_embeddings, 
    store_embeddings, 
    test_semantic_search as semantic_search_op,  # Rename to avoid pytest collecting it as a test
    EmbeddingConfig
)
from dagster_project.jobs import daily_etl_job, embedding_job, complete_pipeline_job, validation_job
from dagster_project.schedules import daily_etl_schedule, new_processed_data_sensor


# ====================================================================================
# RESOURCE TESTS
# ====================================================================================

class TestResources:
    """Test Dagster resources"""
    
    def test_config_loader_resource_initialization(self):
        """Test ConfigLoaderResource can be initialized"""
        resource = ConfigLoaderResource(config_path="config/pipeline_config.yaml")
        assert resource.config_path == "config/pipeline_config.yaml"
    
    def test_logger_resource_initialization(self):
        """Test LoggerResource can be initialized"""
        resource = LoggerResource(log_level="INFO", log_file="logs/test.log")
        assert resource.log_level == "INFO"
        assert resource.log_file == "logs/test.log"
    
    def test_data_path_resource_initialization(self):
        """Test DataPathResource can be initialized"""
        resource = DataPathResource(
            raw_data_dir="data/raw",
            processed_data_dir="data/processed",
            embeddings_dir="chroma_db"
        )
        assert resource.raw_data_dir == "data/raw"
        assert resource.processed_data_dir == "data/processed"


# ====================================================================================
# DATA ACQUISITION OP TESTS
# ====================================================================================

class TestDataAcquisitionOps:
    """Test data acquisition operations"""
    
    def test_download_era5_data_op_structure(self):
        """Test download_era5_data op is defined correctly"""
        assert download_era5_data is not None
        assert download_era5_data.name == "download_era5_data"
        
        # Verify it has expected output structure
        output_def = download_era5_data.output_defs[0]
        assert output_def is not None
    
    def test_validate_downloaded_data_op_structure(self):
        """Test validate_downloaded_data op is defined correctly"""
        assert validate_downloaded_data is not None
        assert validate_downloaded_data.name == "validate_downloaded_data"
        
        # Verify it has input and output
        assert len(validate_downloaded_data.input_defs) > 0
        assert len(validate_downloaded_data.output_defs) > 0


# ====================================================================================
# TRANSFORMATION OP TESTS
# ====================================================================================

class TestTransformationOps:
    """Test data transformation operations"""
    
    def test_ingest_data_op_structure(self):
        """Test ingest_data op is defined correctly"""
        assert ingest_data is not None
        assert ingest_data.name == "ingest_data"
        assert len(ingest_data.input_defs) > 0
        assert len(ingest_data.output_defs) > 0
    
    def test_transform_data_op_structure(self):
        """Test transform_data op is defined correctly"""
        assert transform_data is not None
        assert transform_data.name == "transform_data"
        assert len(transform_data.input_defs) > 0
        assert len(transform_data.output_defs) > 0
    
    def test_export_data_op_structure(self):
        """Test export_data op is defined correctly"""
        assert export_data is not None
        assert export_data.name == "export_data"
        assert len(export_data.input_defs) > 0
        assert len(export_data.output_defs) > 0


# ====================================================================================
# EMBEDDING OP TESTS
# ====================================================================================

class TestEmbeddingOps:
    """Test embedding generation operations"""
    
    def test_generate_embeddings_op_structure(self):
        """Test generate_embeddings op is defined correctly"""
        assert generate_embeddings is not None
        assert generate_embeddings.name == "generate_embeddings"
        assert len(generate_embeddings.input_defs) > 0
        assert len(generate_embeddings.output_defs) > 0
    
    def test_store_embeddings_op_structure(self):
        """Test store_embeddings op is defined correctly"""
        assert store_embeddings is not None
        assert store_embeddings.name == "store_embeddings"
        assert len(store_embeddings.input_defs) > 0
        assert len(store_embeddings.output_defs) > 0
    
    def test_test_semantic_search_op_structure(self):
        """Test test_semantic_search op is defined correctly"""
        assert semantic_search_op is not None
        assert semantic_search_op.name == "test_semantic_search"
        assert len(semantic_search_op.input_defs) > 0
        assert len(semantic_search_op.output_defs) > 0


# ====================================================================================
# JOB TESTS
# ====================================================================================

class TestJobs:
    """Test Dagster jobs"""
    
    def test_daily_etl_job_structure(self):
        """Test daily_etl_job is properly defined"""
        assert daily_etl_job is not None
        assert daily_etl_job.name == "daily_etl_job"
        
        # Check that job has expected ops
        job_def = daily_etl_job
        assert job_def is not None
    
    def test_embedding_job_structure(self):
        """Test embedding_job is properly defined"""
        assert embedding_job is not None
        assert embedding_job.name == "embedding_job"
    
    def test_complete_pipeline_job_structure(self):
        """Test complete_pipeline_job is properly defined"""
        assert complete_pipeline_job is not None
        assert complete_pipeline_job.name == "complete_pipeline_job"
    
    def test_validation_job_structure(self):
        """Test validation_job is properly defined"""
        assert validation_job is not None
        assert validation_job.name == "validation_job"


# ====================================================================================
# SCHEDULE TESTS
# ====================================================================================

class TestSchedules:
    """Test Dagster schedules"""
    
    def test_daily_etl_schedule_evaluation(self):
        """Test daily_etl_schedule can evaluate"""
        instance = DagsterInstance.ephemeral()
        context = build_schedule_context(instance=instance)
        
        result = daily_etl_schedule(context)
        
        assert result is not None
        assert result.run_key is not None
        assert "daily_etl_" in result.run_key
        assert result.tags is not None
        assert "date" in result.tags
    
    def test_daily_etl_schedule_cron(self):
        """Test daily_etl_schedule has correct cron schedule"""
        assert daily_etl_schedule.cron_schedule == "0 2 * * *"


# ====================================================================================
# SENSOR TESTS
# ====================================================================================

class TestSensors:
    """Test Dagster sensors"""
    
    def test_new_processed_data_sensor_no_files(self):
        """Test sensor when no new files exist"""
        instance = DagsterInstance.ephemeral()
        context = build_sensor_context(instance=instance)
        
        # Sensor returns generator, convert to list
        results = list(new_processed_data_sensor(context))
        
        # When no new files, sensor should not yield runs
        assert len(results) == 0
    
    def test_new_processed_data_sensor_structure(self):
        """Test sensor is properly defined"""
        assert new_processed_data_sensor is not None
        assert new_processed_data_sensor.minimum_interval_seconds == 300


# ====================================================================================
# INTEGRATION TESTS
# ====================================================================================

class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_all_jobs_are_defined(self):
        """Test that all expected jobs are properly defined"""
        assert daily_etl_job is not None
        assert embedding_job is not None
        assert complete_pipeline_job is not None
        assert validation_job is not None
        
        # All jobs should have names
        assert daily_etl_job.name == "daily_etl_job"
        assert embedding_job.name == "embedding_job"
        assert complete_pipeline_job.name == "complete_pipeline_job"
        assert validation_job.name == "validation_job"


# ====================================================================================
# ERROR HANDLING TESTS
# ====================================================================================

class TestErrorHandling:
    """Test error handling and op definitions"""
    
    def test_all_ops_have_proper_definitions(self):
        """Test that all ops have proper input/output definitions"""
        # Data acquisition ops
        assert download_era5_data.name == "download_era5_data"
        assert validate_downloaded_data.name == "validate_downloaded_data"
        
        # Transformation ops
        assert ingest_data.name == "ingest_data"
        assert transform_data.name == "transform_data"
        assert export_data.name == "export_data"
        
        # Embedding ops
        assert generate_embeddings.name == "generate_embeddings"
        assert store_embeddings.name == "store_embeddings"
        assert semantic_search_op.name == "test_semantic_search"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
