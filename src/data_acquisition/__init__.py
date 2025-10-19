"""
Data Acquisition Module
Handles downloading and fetching climate data from various sources.
"""

__version__ = "2.0.0"

# Lazy imports to avoid requiring .cdsapirc during module import
def get_era5_downloader():
    """Lazy import ERA5Downloader to avoid cdsapi initialization."""
    from .era5_downloader import ERA5Downloader
    return ERA5Downloader

def get_download_era5_data():
    """Lazy import download_era5_data function."""
    from .era5_downloader import download_era5_data
    return download_era5_data

from .visualizer import visualize_netcdf, create_temperature_map

__all__ = [
    "get_era5_downloader",
    "get_download_era5_data",
    "visualize_netcdf",
    "create_temperature_map",
]
