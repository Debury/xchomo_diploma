from pathlib import Path
from typing import Dict, List, Union
from urllib.parse import urlparse

SUPPORTED_EXTENSIONS = {
    ".nc": "netcdf", ".nc4": "netcdf", ".grib": "grib", ".grb2": "grib",
    ".h5": "hdf5", ".tif": "geotiff", ".tiff": "geotiff",
    ".asc": "ascii", ".csv": "csv", ".zarr": "zarr", ".zip": "zip",
}

def detect_format_from_url(url: str) -> str:
    try:
        path_obj = Path(urlparse(url).path)
        return SUPPORTED_EXTENSIONS.get(path_obj.suffix.lower(), "unknown")
    except Exception:
        return "unknown"