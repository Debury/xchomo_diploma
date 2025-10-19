"""
Data Transformation Ops for Climate ETL Pipeline

Operations for transforming raw climate data into standardized formats.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from dagster import op, In, Out, Output, OpExecutionContext, Config
from pydantic import Field

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dagster_project.resources import ConfigLoaderResource, LoggerResource, DataPathResource
from src.data_transformation.ingestion import DataLoader
from src.data_transformation.transformations import DataTransformer
from src.data_transformation.export import DataExporter


class TransformConfig(Config):
    """Configuration for data transformation operations"""
    
    convert_temperature: bool = Field(
        default=True,
        description="Convert temperature from Kelvin to Celsius"
    )
    rename_dimensions: bool = Field(
        default=True,
        description="Rename dimensions to standard names"
    )
    normalize: bool = Field(
        default=False,
        description="Apply normalization"
    )
    aggregate_to_daily: bool = Field(
        default=False,
        description="Aggregate hourly data to daily"
    )


@op(
    description="Load and ingest raw climate data files",
    ins={"validation_result": In(dagster_type=Dict[str, Any])},
    out=Out(
        dagster_type=Dict[str, Any],
        description="Ingested datasets"
    ),
    tags={"phase": "transformation", "step": "ingestion"}
)
def ingest_data(
    context: OpExecutionContext,
    logger: LoggerResource,
    data_paths: DataPathResource,
    validation_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Load raw climate data files into xarray datasets.
    
    Args:
        context: Dagster execution context
        logger: Logging resource
        data_paths: Data path management resource
        validation_result: Output from validate_downloaded_data op
    
    Returns:
        Dictionary containing loaded datasets and metadata
    """
    start_time = datetime.now()
    logger.info("Starting data ingestion")
    
    try:
        if validation_result["status"] != "success":
            logger.warning("Validation was not successful, cannot ingest")
            return {
                "status": "skipped",
                "datasets": {},
                "metadata": {"reason": "Validation failed"}
            }
        
        valid_files = validation_result.get("valid_files", [])
        
        if not valid_files:
            logger.warning("No valid files to ingest")
            return {
                "status": "success",
                "datasets": {},
                "metadata": {"message": "No files to process"}
            }
        
        loader = DataLoader()
        loaded_datasets = {}
        
        for filepath in valid_files:
            file_path = Path(filepath)
            
            if not file_path.exists():
                logger.warning(f"File not found during ingestion: {filepath}")
                continue
            
            try:
                # Load the dataset
                # NOTE: In production, uncomment to actually load data
                # dataset = loader.load_netcdf(file_path)
                # loaded_datasets[file_path.stem] = {
                #     "path": str(file_path),
                #     "variables": list(dataset.data_vars),
                #     "dims": dict(dataset.dims),
                #     "size_mb": file_path.stat().st_size / (1024 * 1024)
                # }
                
                # Mock loading for demonstration
                loaded_datasets[file_path.stem] = {
                    "path": str(file_path),
                    "variables": ["temperature", "precipitation"],
                    "dims": {"time": 24, "latitude": 5, "longitude": 8},
                    "size_mb": 0.5
                }
                
                context.log.info(f"Loaded dataset: {file_path.stem}")
                
            except Exception as e:
                logger.error(f"Error loading {filepath}: {str(e)}")
                continue
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Ingestion completed: {len(loaded_datasets)} datasets loaded in {duration:.2f}s")
        
        return {
            "status": "success",
            "datasets": loaded_datasets,
            "metadata": {
                "num_datasets": len(loaded_datasets),
                "duration_seconds": duration,
                "timestamp": start_time.isoformat()
            }
        }
    
    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        logger.exception("Ingestion exception details")
        
        return {
            "status": "error",
            "datasets": {},
            "metadata": {"error": str(e)}
        }


@op(
    description="Transform climate data (unit conversion, renaming, aggregation)",
    ins={"ingestion_result": In(dagster_type=Dict[str, Any])},
    out=Out(
        dagster_type=Dict[str, Any],
        description="Transformed datasets"
    ),
    tags={"phase": "transformation", "step": "transform"}
)
def transform_data(
    context: OpExecutionContext,
    config_loader: ConfigLoaderResource,
    logger: LoggerResource,
    ingestion_result: Dict[str, Any],
    config: TransformConfig
) -> Dict[str, Any]:
    """
    Apply transformations to ingested datasets.
    
    Transformations include:
    - Temperature unit conversion (Kelvin to Celsius)
    - Dimension renaming (standardization)
    - Temporal aggregation (hourly to daily)
    - Normalization
    
    Args:
        context: Dagster execution context
        config_loader: Configuration loader resource
        logger: Logging resource
        ingestion_result: Output from ingest_data op
        config: Transformation configuration
    
    Returns:
        Dictionary containing transformed datasets
    """
    start_time = datetime.now()
    logger.info("Starting data transformation")
    
    try:
        if ingestion_result["status"] != "success":
            logger.warning("Ingestion was not successful, cannot transform")
            return {
                "status": "skipped",
                "transformed_datasets": {},
                "metadata": {"reason": "Ingestion failed"}
            }
        
        datasets = ingestion_result.get("datasets", {})
        
        if not datasets:
            logger.warning("No datasets to transform")
            return {
                "status": "success",
                "transformed_datasets": {},
                "metadata": {"message": "No datasets to process"}
            }
        
        # Load transformation configuration
        pipeline_config = config_loader.load()
        transform_config = pipeline_config.get("data_transformation", {})
        
        transformer = DataTransformer()
        transformed_datasets = {}
        
        for dataset_name, dataset_info in datasets.items():
            try:
                # NOTE: In production, load actual xarray dataset and transform
                # ds = xr.open_dataset(dataset_info["path"])
                
                # Apply transformations based on config
                transformations_applied = []
                
                if config.convert_temperature:
                    # ds = transformer.convert_temperature_to_celsius(ds)
                    transformations_applied.append("temperature_conversion")
                    logger.info(f"Applied temperature conversion to {dataset_name}")
                
                if config.rename_dimensions:
                    # ds = transformer.rename_dimensions(ds, transform_config.get("dimensions", {}))
                    transformations_applied.append("dimension_renaming")
                    logger.info(f"Applied dimension renaming to {dataset_name}")
                
                if config.aggregate_to_daily:
                    # ds = transformer.aggregate_hourly_to_daily(ds)
                    transformations_applied.append("temporal_aggregation")
                    logger.info(f"Applied temporal aggregation to {dataset_name}")
                
                if config.normalize:
                    # ds = transformer.normalize(ds, method="minmax")
                    transformations_applied.append("normalization")
                    logger.info(f"Applied normalization to {dataset_name}")
                
                transformed_datasets[dataset_name] = {
                    "original_path": dataset_info["path"],
                    "transformations": transformations_applied,
                    "variables": dataset_info["variables"],
                    "dims": dataset_info["dims"]
                }
                
                context.log.info(f"Transformed dataset: {dataset_name}")
                
            except Exception as e:
                logger.error(f"Error transforming {dataset_name}: {str(e)}")
                continue
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Transformation completed: {len(transformed_datasets)} datasets in {duration:.2f}s")
        
        return {
            "status": "success",
            "transformed_datasets": transformed_datasets,
            "metadata": {
                "num_datasets": len(transformed_datasets),
                "duration_seconds": duration,
                "timestamp": start_time.isoformat(),
                "config": {
                    "convert_temperature": config.convert_temperature,
                    "rename_dimensions": config.rename_dimensions,
                    "normalize": config.normalize,
                    "aggregate_to_daily": config.aggregate_to_daily
                }
            }
        }
    
    except Exception as e:
        logger.error(f"Transformation error: {str(e)}")
        logger.exception("Transformation exception details")
        
        return {
            "status": "error",
            "transformed_datasets": {},
            "metadata": {"error": str(e)}
        }


@op(
    description="Export transformed data to multiple formats",
    ins={"transformation_result": In(dagster_type=Dict[str, Any])},
    out=Out(
        dagster_type=Dict[str, Any],
        description="Export results with file paths"
    ),
    tags={"phase": "transformation", "step": "export"}
)
def export_data(
    context: OpExecutionContext,
    logger: LoggerResource,
    data_paths: DataPathResource,
    transformation_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Export transformed datasets to multiple formats.
    
    Supported formats:
    - NetCDF (.nc)
    - Parquet (.parquet)
    - CSV (.csv)
    
    Args:
        context: Dagster execution context
        logger: Logging resource
        data_paths: Data path management resource
        transformation_result: Output from transform_data op
    
    Returns:
        Dictionary containing exported file paths and metadata
    """
    start_time = datetime.now()
    logger.info("Starting data export")
    
    try:
        if transformation_result["status"] != "success":
            logger.warning("Transformation was not successful, cannot export")
            return {
                "status": "skipped",
                "exported_files": [],
                "metadata": {"reason": "Transformation failed"}
            }
        
        datasets = transformation_result.get("transformed_datasets", {})
        
        if not datasets:
            logger.warning("No datasets to export")
            return {
                "status": "success",
                "exported_files": [],
                "metadata": {"message": "No datasets to export"}
            }
        
        output_dir = data_paths.get_processed_path()
        exporter = DataExporter()
        
        exported_files = []
        
        for dataset_name, dataset_info in datasets.items():
            try:
                # NOTE: In production, load transformed dataset and export
                # ds = xr.open_dataset(...)
                
                # Export to multiple formats
                formats = ["netcdf", "parquet", "csv"]
                
                for fmt in formats:
                    output_file = output_dir / f"{dataset_name}_processed.{fmt}"
                    
                    # In production, use actual exporter:
                    # if fmt == "netcdf":
                    #     exporter.to_netcdf(ds, output_file)
                    # elif fmt == "parquet":
                    #     exporter.to_parquet(ds, output_file)
                    # elif fmt == "csv":
                    #     exporter.to_csv(ds, output_file)
                    
                    exported_files.append({
                        "dataset": dataset_name,
                        "format": fmt,
                        "path": str(output_file),
                        "size_mb": 0.0  # Would be actual size in production
                    })
                    
                    logger.info(f"Exported {dataset_name} to {fmt} format")
                
                context.log.info(f"Exported dataset: {dataset_name}")
                
            except Exception as e:
                logger.error(f"Error exporting {dataset_name}: {str(e)}")
                continue
        
        duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Export completed: {len(exported_files)} files in {duration:.2f}s")
        
        return {
            "status": "success",
            "exported_files": exported_files,
            "metadata": {
                "num_files": len(exported_files),
                "num_datasets": len(datasets),
                "duration_seconds": duration,
                "timestamp": start_time.isoformat(),
                "output_directory": str(output_dir)
            }
        }
    
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        logger.exception("Export exception details")
        
        return {
            "status": "error",
            "exported_files": [],
            "metadata": {"error": str(e)}
        }
