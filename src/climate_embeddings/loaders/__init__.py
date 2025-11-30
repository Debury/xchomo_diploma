"""Loaders for various climate data formats."""

from .detect_format import detect_format, list_supported_formats
from .raster_pipeline import (
    load_raster_auto,
    raster_to_embeddings,
    load_from_zip,
    save_embeddings,
)

__all__ = [
    "detect_format",
    "list_supported_formats",
    "load_raster_auto",
    "raster_to_embeddings",
    "load_from_zip",
    "save_embeddings",
]
