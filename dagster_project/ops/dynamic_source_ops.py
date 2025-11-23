"""
Dynamic Source Operations for Phase 5

Operations that process individual climate data sources dynamically.
Each source is loaded from the database and processed independently.
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import mimetypes

from dagster import op, In, Out, Output, OpExecutionContext, Config, DynamicOut, DynamicOutput
from pydantic import Field

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dagster_project.resources import ConfigLoaderResource, LoggerResource, DataPathResource


def detect_format_from_url(url: str) -> str:
    """Infer the format from the URL/extension before downloading."""

    url_lower = url.lower()
    extension_map = {
        ".nc": "netcdf",
        ".nc4": "netcdf",
        ".grib": "grib",
        ".grib2": "grib",
        ".h5": "hdf5",
        ".hdf": "hdf5",
        ".json": "json",
        ".geojson": "json",
        ".csv": "csv",
        ".tsv": "tsv",
        ".txt": "txt",
        ".tif": "geotiff",
        ".tiff": "geotiff",
        ".cog": "geotiff",
    ".asc": "ascii_grid",
        ".shp": "shapefile",
        ".zip": "zip",
    }

    for ext, fmt in extension_map.items():
        if url_lower.endswith(ext):
            return fmt

    # NOAA GFS forecast files often end with .f000, .f003, etc. but are GRIB payloads
    if "." in url_lower:
        suffix = url_lower.rsplit(".", 1)[-1]
        if suffix.startswith("f") and suffix[1:].isdigit():
            return "grib"

    if "grib" in url_lower:
        return "grib"
    if "netcdf" in url_lower:
        return "netcdf"
    if "csv" in url_lower:
        return "csv"
    if "geojson" in url_lower:
        return "json"

    mime_type, _ = mimetypes.guess_type(url)
    if mime_type:
        mime_map = {
            'application/netcdf': 'netcdf',
            'application/x-netcdf': 'netcdf',
            'text/csv': 'csv',
            'application/json': 'json',
            'application/geotiff': 'geotiff',
        }
        if mime_type in mime_map:
            return mime_map[mime_type]

    return 'netcdf'


class SourceIngestConfig(Config):
    """Configuration for ingesting a single source"""
    source_id: str = Field(..., description="Unique source identifier")
    url: str = Field(..., description="URL or path to data source")
    format: Optional[str] = Field(None, description="Data format (auto-detected if None)")
    variables: Optional[list[str]] = Field(None, description="Variables to extract (None = all)")
    time_range: Optional[Dict[str, str]] = Field(None, description="Time range to process")
    spatial_bbox: Optional[list[float]] = Field(None, description="Spatial bounding box")


@op(
    description="Download/fetch data from a single source",
    out=Out(
        dagster_type=Dict[str, Any],
        description="Downloaded data info for this source"
    ),
    tags={"phase": "5", "type": "dynamic"}
)
def fetch_source_data(
    context: OpExecutionContext,
    logger: LoggerResource,
    data_paths: DataPathResource,
    config: SourceIngestConfig
) -> Dict[str, Any]:
    """
    Fetch data from a single climate data source.
    
    This op downloads or loads data from URL/path and saves it to raw data directory.
    Supports multiple formats: NetCDF, GRIB, CSV, Parquet, Zarr, HDF5.
    
    Args:
        context: Dagster execution context
        logger: Logging resource
        data_paths: Data path management
        config: Source configuration
        
    Returns:
        Dictionary with:
            - source_id: Source identifier
            - status: "success" or "error"
            - files: List of downloaded file paths
            - format: Detected data format
            - metadata: Additional info
    """
    start_time = datetime.now()
    source_id = config.source_id
    url = config.url
    
    logger.info(f"[{source_id}] Fetching data from: {url}")
    
    try:
        # Auto-detect format if not specified
        format = config.format or detect_format_from_url(url)
        logger.info(f"[{source_id}] Detected format: {format}")
        
        # Get output directory
        output_dir = data_paths.get_raw_path()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{source_id}_{timestamp}.{format.replace('netcdf', 'nc')}"
        filepath = output_dir / filename
        
        # REAL DOWNLOAD - using requests/xarray
        import requests
        
        logger.info(f"[{source_id}] Downloading to: {filepath}")
        
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = filepath.stat().st_size
        logger.info(f"[{source_id}] Downloaded {file_size / 1024 / 1024:.2f} MB")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = {
            "source_id": source_id,
            "status": "success",
            "files": [str(filepath)],
            "format": format,
            "url": url,
            "file_size_mb": file_size / 1024 / 1024,
            "duration_seconds": duration,
            "metadata": {
                "downloaded_at": datetime.now().isoformat(),
                "variables": config.variables,
                "time_range": config.time_range,
                "spatial_bbox": config.spatial_bbox
            }
        }
        
        context.log.info(f"[{source_id}] Fetch completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"[{source_id}] Fetch failed: {e}")
        
        return {
            "source_id": source_id,
            "status": "error",
            "files": [],
            "format": format if 'format' in locals() else "unknown",
            "url": url,
            "error": str(e),
            "metadata": {}
        }


@op(
    description="Process downloaded source data",
    ins={"fetch_result": In(dagster_type=Dict[str, Any])},
    out=Out(
        dagster_type=Dict[str, Any],
        description="Processed data info"
    ),
    tags={"phase": "5", "type": "dynamic"}
)
def process_source_data(
    context: OpExecutionContext,
    logger: LoggerResource,
    data_paths: DataPathResource,
    fetch_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Process downloaded source data (load, transform, save).
    
    Args:
        context: Dagster execution context
        logger: Logging resource
        data_paths: Data path management
        fetch_result: Result from fetch_source_data op
        
    Returns:
        Dictionary with processing results
    """
    source_id = fetch_result.get("source_id")
    
    if fetch_result.get("status") == "error":
        logger.warning(f"[{source_id}] Skipping processing due to fetch error")
        return {
            "source_id": source_id,
            "status": "skipped",
            "reason": "fetch_failed"
        }
    
    logger.info(f"[{source_id}] Processing data...")
    
    try:
        files = fetch_result.get("files", [])
        if not files:
            raise ValueError("No files to process")
        
        input_file = Path(files[0])
        format = fetch_result.get("format")
        
        # Load data based on format
        if format == "netcdf":
            import xarray as xr
            ds = xr.open_dataset(input_file)
            
            logger.info(f"[{source_id}] Loaded NetCDF with variables: {list(ds.data_vars)}")
            
            # Apply basic transformations
            # Example: Convert Kelvin to Celsius if temperature variable exists
            for var in ds.data_vars:
                if 'temperature' in var.lower() or var in ['t2m', 'air', 'tas']:
                    if ds[var].attrs.get('units') == 'K':
                        logger.info(f"[{source_id}] Converting {var} from Kelvin to Celsius")
                        ds[var] = ds[var] - 273.15
                        ds[var].attrs['units'] = 'degC'
            
            # Save processed data
            output_dir = data_paths.get_processed_path()
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"{source_id}_processed.nc"
            ds.to_netcdf(output_file)
            
            logger.info(f"[{source_id}] Saved processed data to: {output_file}")
            
            result = {
                "source_id": source_id,
                "status": "success",
                "input_file": str(input_file),
                "output_file": str(output_file),
                "format": format,
                "variables": list(ds.data_vars),
                "dimensions": dict(ds.dims),
                "metadata": fetch_result.get("metadata", {})
            }
            
            ds.close()
            
        elif format == "csv":
            import pandas as pd
            df = pd.read_csv(input_file)
            
            logger.info(f"[{source_id}] Loaded CSV with {len(df)} rows, {len(df.columns)} columns")
            
            # Save as Parquet for better performance
            output_dir = data_paths.get_processed_path()
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = output_dir / f"{source_id}_processed.parquet"
            df.to_parquet(output_file)
            
            result = {
                "source_id": source_id,
                "status": "success",
                "input_file": str(input_file),
                "output_file": str(output_file),
                "format": "parquet",
                "rows": len(df),
                "columns": list(df.columns),
                "metadata": fetch_result.get("metadata", {})
            }
            
        else:
            logger.warning(f"[{source_id}] Format '{format}' not yet supported for processing")
            result = {
                "source_id": source_id,
                "status": "unsupported_format",
                "format": format,
                "input_file": str(input_file)
            }
        
        context.log.info(f"[{source_id}] Processing completed")
        return result
        
    except Exception as e:
        logger.error(f"[{source_id}] Processing failed: {e}")
        
        return {
            "source_id": source_id,
            "status": "error",
            "error": str(e),
            "input_file": str(input_file) if 'input_file' in locals() else None
        }


@op(
    description="Load active sources from database",
    out=DynamicOut(
        dagster_type=Dict[str, Any],
        description="Individual source configurations"
    ),
    tags={"phase": "5", "type": "dynamic"}
)
def load_active_sources(
    context: OpExecutionContext,
    logger: LoggerResource
) -> list:
    """
    Load all active sources from the database.
    
    Yields one dynamic output per source for parallel processing.
    
    Args:
        context: Dagster execution context
        logger: Logging resource
        
    Yields:
        DynamicOutput for each active source
    """
    from src.sources import get_source_store
    
    logger.info("Loading active sources from database...")
    
    try:
        store = get_source_store()
        sources = store.get_all_sources(active_only=True)
        
        logger.info(f"Found {len(sources)} active source(s)")
        
        for source in sources:
            source_config = {
                "source_id": source.source_id,
                "url": source.url,
                "format": source.format,
                "variables": source.variables,
                "time_range": source.time_range,
                "spatial_bbox": source.spatial_bbox
            }
            
            logger.info(f"Yielding source: {source.source_id}")
            
            # Yield dynamic output for this source
            yield DynamicOutput(
                value=source_config,
                mapping_key=source.source_id.replace("-", "_")  # Dagster-safe key
            )
            
    except Exception as e:
        logger.error(f"Failed to load sources: {e}")
        # Yield empty if error
        context.log.error(f"Source loading failed: {e}")
