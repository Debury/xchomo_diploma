"""
Dagster Jobs for Climate ETL Pipeline

Defines complete workflows by composing ops into jobs.
"""

from dagster import job

from dagster_project.resources import ConfigLoaderResource, LoggerResource, DataPathResource
from dagster_project.ops.data_acquisition_ops import download_era5_data, validate_downloaded_data
from dagster_project.ops.transformation_ops import ingest_data, transform_data, export_data
from dagster_project.ops.embedding_ops import (
    generate_embeddings,
    store_embeddings,
    test_semantic_search
)


# ====================================================================================
# DAILY ETL JOB
# ====================================================================================

@job(
    description="Complete daily ETL workflow: download → transform → export",
    resource_defs={
        "config_loader": ConfigLoaderResource(config_path="config/pipeline_config.yaml"),
        "logger": LoggerResource(log_file="logs/dagster_etl.log", log_level="INFO"),
        "data_paths": DataPathResource(
            raw_data_dir="data/raw",
            processed_data_dir="data/processed",
            embeddings_dir="chroma_db"
        )
    },
    tags={
        "pipeline": "etl",
        "frequency": "daily",
        "phase": "1-2"
    }
)
def daily_etl_job():
    """
    Daily ETL job that downloads, transforms, and exports climate data.
    
    Workflow:
    1. Download ERA5 data from CDS API
    2. Validate downloaded files
    3. Ingest data into xarray datasets
    4. Apply transformations (unit conversion, dimension renaming, etc.)
    5. Export to multiple formats (NetCDF, Parquet, CSV)
    
    This job runs on a daily schedule to keep climate data up-to-date.
    """
    # Phase 1: Data Acquisition
    download_result = download_era5_data()
    validation_result = validate_downloaded_data(download_result)
    
    # Phase 2: Data Transformation
    ingestion_result = ingest_data(validation_result)
    transformation_result = transform_data(ingestion_result)
    export_data(transformation_result)


# ====================================================================================
# EMBEDDING GENERATION JOB
# ====================================================================================

@job(
    description="Generate embeddings from processed data and store in vector database",
    resource_defs={
        "config_loader": ConfigLoaderResource(config_path="config/pipeline_config.yaml"),
        "logger": LoggerResource(log_file="logs/dagster_embeddings.log", log_level="INFO"),
        "data_paths": DataPathResource(
            raw_data_dir="data/raw",
            processed_data_dir="data/processed",
            embeddings_dir="chroma_db"
        )
    },
    tags={
        "pipeline": "embeddings",
        "frequency": "on-demand",
        "phase": "3"
    }
)
def embedding_job():
    """
    Embedding generation job for semantic search capabilities.
    
    Workflow:
    1. Generate embeddings from processed climate datasets
    2. Store embeddings in ChromaDB vector database
    3. Test semantic search functionality
    
    This job can be triggered on-demand or after new data is processed.
    Note: This job assumes processed data already exists in data/processed/
    """
    # Phase 3: Embedding Generation
    # This job starts directly with embedding generation
    # It expects processed data files to exist in data/processed/
    embedding_result = generate_embeddings()
    storage_result = store_embeddings(embedding_result)
    test_semantic_search(storage_result)


# ====================================================================================
# COMPLETE PIPELINE JOB
# ====================================================================================

@job(
    description="Complete end-to-end pipeline: download → transform → embed",
    resource_defs={
        "config_loader": ConfigLoaderResource(config_path="config/pipeline_config.yaml"),
        "logger": LoggerResource(log_file="logs/dagster_complete.log", log_level="INFO"),
        "data_paths": DataPathResource(
            raw_data_dir="data/raw",
            processed_data_dir="data/processed",
            embeddings_dir="chroma_db"
        )
    },
    tags={
        "pipeline": "complete",
        "frequency": "weekly",
        "phase": "1-2-3"
    }
)
def complete_pipeline_job():
    """
    Complete end-to-end pipeline combining all phases.
    
    Workflow:
    1. Download ERA5 data from CDS API
    2. Validate downloaded files
    3. Ingest data into xarray datasets
    4. Apply transformations
    5. Export to multiple formats
    6. Generate embeddings
    7. Store in vector database
    8. Test semantic search
    
    This job provides a complete data pipeline from raw data to searchable embeddings.
    """
    # Phase 1: Data Acquisition
    download_result = download_era5_data()
    validation_result = validate_downloaded_data(download_result)
    
    # Phase 2: Data Transformation
    ingestion_result = ingest_data(validation_result)
    transformation_result = transform_data(ingestion_result)
    export_result = export_data(transformation_result)
    
    # Phase 3: Embedding Generation
    embedding_result = generate_embeddings(export_result)
    storage_result = store_embeddings(embedding_result)
    test_semantic_search(storage_result)


# ====================================================================================
# VALIDATION JOB
# ====================================================================================

@job(
    description="Validation-only job to check data quality",
    resource_defs={
        "config_loader": ConfigLoaderResource(config_path="config/pipeline_config.yaml"),
        "logger": LoggerResource(log_file="logs/dagster_validation.log", log_level="INFO"),
        "data_paths": DataPathResource(
            raw_data_dir="data/raw",
            processed_data_dir="data/processed",
            embeddings_dir="chroma_db"
        )
    },
    tags={
        "pipeline": "validation",
        "frequency": "on-demand"
    }
)
def validation_job():
    """
    Data validation job to check quality without processing.
    
    This job validates existing downloaded files in data/raw/
    without running the full pipeline.
    """
    # Download just to get the list of files, then validate them
    download_result = download_era5_data()
    validate_downloaded_data(download_result)
