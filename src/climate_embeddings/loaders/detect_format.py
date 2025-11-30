"""
Format detection for climate data files.
Handles local files, directories (Zarr), and URL-based inference.
"""

from pathlib import Path
from typing import Dict, List, Union
from urllib.parse import urlparse

SUPPORTED_EXTENSIONS: Dict[str, str] = {
    ".nc": "netcdf",
    ".nc4": "netcdf",
    ".cdf": "netcdf",
    ".grib": "grib",
    ".grb": "grib",
    ".grb2": "grib",
    ".h5": "hdf5",
    ".hdf5": "hdf5",
    ".tif": "geotiff",
    ".tiff": "geotiff",
    ".asc": "ascii",
    ".txt": "csv",
    ".csv": "csv",
    ".zarr": "zarr",
    ".zip": "zip",
}


def detect_format(file_path: Union[str, Path]) -> str:
    """
    Detect format of a local climate data file or directory.

    Args:
        file_path: Path to file or directory

    Returns:
        Format string: netcdf, grib, hdf5, geotiff, ascii, csv, zarr, zip
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # 1. Check if directory (Zarr)
    if file_path.is_dir():
        if (file_path / ".zarray").exists() or (file_path / ".zgroup").exists() or file_path.suffix == ".zarr":
            return "zarr"
        # If it's a directory but not zarr, we can't process it yet
        raise ValueError(f"Unknown directory format: {file_path}")

    # 2. Check extension
    suffix = file_path.suffix.lower()
    fmt = SUPPORTED_EXTENSIONS.get(suffix)

    if fmt:
        return fmt

    # 3. Fallback: You could add Magic Byte detection here if extensions fail
    # For now, we rely on extensions.
    raise ValueError(f"Unsupported format extension: {suffix}")


def detect_format_from_url(url: str) -> str:
    """
    Attempt to infer format from a URL string before downloading.
    
    Args:
        url: The source URL (e.g., https://example.com/data.nc)
        
    Returns:
        Format string or 'unknown'
    """
    try:
        parsed = urlparse(url)
        # Extract path to ignore query parameters (e.g., ?download=true)
        path_obj = Path(parsed.path)
        suffix = path_obj.suffix.lower()
        return SUPPORTED_EXTENSIONS.get(suffix, "unknown")
    except Exception:
        return "unknown"


def list_supported_formats() -> List[str]:
    """Return list of supported format names."""
    return list(set(SUPPORTED_EXTENSIONS.values()))