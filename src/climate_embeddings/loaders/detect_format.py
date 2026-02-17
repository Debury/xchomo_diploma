from pathlib import Path
from typing import Dict, List, Union
from urllib.parse import urlparse

SUPPORTED_EXTENSIONS = {
    ".nc": "netcdf", ".nc4": "netcdf", ".grib": "grib", ".grib2": "grib", ".grb2": "grib",
    ".h5": "hdf5", ".hdf": "hdf5", ".he5": "hdf5",
    ".tif": "geotiff", ".tiff": "geotiff",
    ".asc": "ascii", ".csv": "csv", ".tsv": "csv",
    ".zarr": "zarr", ".zip": "zip", ".gz": "gz",
    ".tar": "tar", ".txt": "csv",
}

# Compound extensions that must be checked before single-suffix lookup
COMPOUND_EXTENSIONS = {
    ".nc.gz": "gz",
    ".tar.gz": "gz",
}


def detect_format_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        # Use only the path component (strips query params and fragments)
        path_lower = parsed.path.lower()

        # Check compound extensions first
        for compound_ext, fmt in COMPOUND_EXTENSIONS.items():
            if path_lower.endswith(compound_ext):
                return fmt

        path_obj = Path(parsed.path)
        return SUPPORTED_EXTENSIONS.get(path_obj.suffix.lower(), "unknown")
    except Exception:
        return "unknown"
