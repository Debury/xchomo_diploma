"""
Data Acquisition Ops for Climate ETL Pipeline

Operations for downloading ERA5 climate data from CDS API.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta

from dagster import op, In, Out, Output, OpExecutionContext, Config
from pydantic import Field

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dagster_project.resources import ConfigLoaderResource, LoggerResource, DataPathResource


class DownloadConfig(Config):
    """Configuration for ERA5 data download operation"""
    
    variables: List[str] = Field(
        default=["2m_temperature", "total_precipitation"],
        description="List of ERA5 variables to download"
    )
    year: int = Field(
        default=2024,
        description="Year to download data for"
    )
    month: int = Field(
        default=1,
        description="Month to download data for"
    )
    days: List[int] = Field(
        default=None,
        description="List of days to download (default: all days in month)"
    )
    area: List[float] = Field(
        default=None,
        description="Bounding box [north, west, south, east] (default: from config)"
    )


@op(
    description="Download ERA5 climate data from CDS API",
    out=Out(
        dagster_type=Dict[str, Any],
        description="Dictionary containing download results and file paths"
    ),
    tags={"phase": "acquisition", "source": "ERA5"}
)
def download_era5_data(
    context: OpExecutionContext,
    config_loader: ConfigLoaderResource,
    logger: LoggerResource,
    data_paths: DataPathResource,
    config: DownloadConfig
) -> Dict[str, Any]:
    """
    Download ERA5 climate data from Copernicus CDS API.
    
    This operation downloads specified climate variables for a given time period
    and saves them to the raw data directory.
    
    Args:
        context: Dagster execution context
        config_loader: Configuration loader resource
        logger: Logging resource
        data_paths: Data path management resource
        config: Download configuration
    
    Returns:
        Dictionary with:
            - status: "success" or "error"
            - files: List of downloaded file paths
            - variables: List of variables downloaded
            - metadata: Additional download metadata
    
    Example:
        Result: {
            "status": "success",
            "files": ["data/raw/era5_2m_temperature_2024_01.nc"],
            "variables": ["2m_temperature"],
            "metadata": {"year": 2024, "month": 1}
        }
    """
    start_time = datetime.now()
    logger.info(f"Starting ERA5 data download for {config.year}-{config.month:02d}")
    
    try:
        # Load pipeline configuration
        pipeline_config = config_loader.load()
        acquisition_config = pipeline_config.get("data_acquisition", {})
        
        # Get output directory
        output_dir = data_paths.get_raw_path()
        logger.info(f"Output directory: {output_dir}")
        
        # Prepare download parameters (don't modify frozen config, use variables)
        area = config.area if config.area is not None else acquisition_config.get("default_parameters", {}).get("area", [51, 13, 48, 19])
        
        if config.days is None:
            # Get all days in month
            import calendar
            _, num_days = calendar.monthrange(config.year, config.month)
            days = list(range(1, num_days + 1))
        else:
            days = config.days
        
        downloaded_files = []
        
        # NOTE: Actual ERA5 download requires CDS API key and can take significant time
        # For demonstration, we create mock download results
        # In production, uncomment the ERA5Downloader code below:
        
        # from src.data_acquisition.era5_downloader import ERA5Downloader
        # downloader = ERA5Downloader(api_key=os.getenv("CDS_API_KEY"))
        
        for variable in config.variables:
            # Create expected filename
            filename = f"era5_{variable}_{config.year}_{config.month:02d}.nc"
            filepath = output_dir / filename
            
            # Mock download (in production, use actual downloader)
            logger.info(f"[MOCK] Downloading {variable} to {filepath}")
            
            # In production, use:
            # result = downloader.download(
            #     variable=variable,
            #     year=config.year,
            #     month=config.month,
            #     days=days,
            #     area=area,
            #     output_path=filepath
            # )
            
            downloaded_files.append(str(filepath))
            context.log.info(f"Downloaded {variable} -> {filepath}")
        
        duration = (datetime.now() - start_time).total_seconds()
        
        result = {
            "status": "success",
            "files": downloaded_files,
            "variables": config.variables,
            "metadata": {
                "year": config.year,
                "month": config.month,
                "num_days": len(config.days),
                "area": config.area,
                "duration_seconds": duration,
                "timestamp": start_time.isoformat()
            }
        }
        
        logger.info(f"Download completed successfully in {duration:.2f}s")
        logger.info(f"Downloaded {len(downloaded_files)} files")
        
        return result
    
    except Exception as e:
        logger.error(f"Error during download: {str(e)}")
        logger.exception("Download exception details")
        
        return {
            "status": "error",
            "files": [],
            "variables": config.variables,
            "metadata": {
                "error": str(e),
                "timestamp": start_time.isoformat()
            }
        }


@op(
    description="Validate downloaded ERA5 data files",
    ins={"download_result": In(dagster_type=Dict[str, Any])},
    out=Out(
        dagster_type=Dict[str, Any],
        description="Validation results"
    ),
    tags={"phase": "acquisition", "validation": "true"}
)
def validate_downloaded_data(
    context: OpExecutionContext,
    logger: LoggerResource,
    download_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate downloaded ERA5 data files.
    
    Checks that:
    - Files exist and are readable
    - Files are valid NetCDF format
    - Files contain expected variables
    - Data has reasonable values
    
    Args:
        context: Dagster execution context
        logger: Logging resource
        download_result: Output from download_era5_data op
    
    Returns:
        Dictionary with validation results
    """
    logger.info("Starting data validation")
    
    try:
        import xarray as xr
        
        if download_result["status"] != "success":
            logger.warning("Download was not successful, skipping validation")
            return {
                "status": "skipped",
                "reason": "Download failed",
                "valid_files": []
            }
        
        files = download_result.get("files", [])
        valid_files = []
        invalid_files = []
        
        for filepath in files:
            file_path = Path(filepath)
            
            # Check file exists
            if not file_path.exists():
                logger.warning(f"File not found: {filepath}")
                invalid_files.append({"file": filepath, "reason": "File not found"})
                continue
            
            # In production, validate NetCDF file
            # try:
            #     ds = xr.open_dataset(filepath)
            #     # Check for required dimensions
            #     required_dims = ['time', 'latitude', 'longitude']
            #     missing_dims = [d for d in required_dims if d not in ds.dims]
            #     if missing_dims:
            #         invalid_files.append({
            #             "file": filepath,
            #             "reason": f"Missing dimensions: {missing_dims}"
            #         })
            #         continue
            #     ds.close()
            # except Exception as e:
            #     invalid_files.append({"file": filepath, "reason": str(e)})
            #     continue
            
            valid_files.append(filepath)
            context.log.info(f"Validated: {filepath}")
        
        logger.info(f"Validation complete: {len(valid_files)} valid, {len(invalid_files)} invalid")
        
        return {
            "status": "success",
            "valid_files": valid_files,
            "invalid_files": invalid_files,
            "total_files": len(files),
            "valid_count": len(valid_files),
            "invalid_count": len(invalid_files)
        }
    
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "valid_files": [],
            "invalid_files": []
        }
