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
    # Unstructured documents → processed by rag-mendelu GeneralEtl
    ".pdf": "pdf", ".docx": "pdf", ".pptx": "pdf", ".md": "pdf",
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
        suffix = path_obj.suffix.lower()
        if suffix in SUPPORTED_EXTENSIONS:
            return SUPPORTED_EXTENSIONS[suffix]

        # Zenodo / OSF / Dataverse style API URLs put the real filename
        # one segment up: `…/files/30days.zip/content` or
        # `…/data/foo.nc/download`. Walk back through the path segments
        # and use the first one whose suffix we recognise.
        for segment in reversed(path_lower.strip("/").split("/")):
            seg_suffix = Path(segment).suffix.lower()
            if seg_suffix in SUPPORTED_EXTENSIONS:
                return SUPPORTED_EXTENSIONS[seg_suffix]

        return "unknown"
    except Exception:
        return "unknown"
