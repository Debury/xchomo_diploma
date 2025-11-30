"""
Format detection for climate data files.
"""

from pathlib import Path
from typing import Dict, List

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


def detect_format(file_path: str | Path) -> str:
    """
    Detect format of climate data file.
    
    Args:
        file_path: Path to file
        
    Returns:
        Format string: netcdf, grib, hdf5, geotiff, ascii, csv, zarr, zip
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Check if directory (Zarr)
    if file_path.is_dir():
        if (file_path / ".zarray").exists() or (file_path / ".zgroup").exists():
            return "zarr"
        raise ValueError(f"Unknown directory format: {file_path}")
    
    # Check extension
    suffix = file_path.suffix.lower()
    fmt = SUPPORTED_EXTENSIONS.get(suffix)
    
    if fmt:
        return fmt
    
    raise ValueError(f"Unsupported format: {suffix}")


def list_supported_formats() -> List[str]:
    """Return list of supported format names."""
    return list(set(SUPPORTED_EXTENSIONS.values()))
