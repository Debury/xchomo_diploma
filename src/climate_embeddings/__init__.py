# src/climate_embeddings/__init__.py

from .loaders import (
    load_raster_auto, 
    raster_to_embeddings, 
    detect_format,
    detect_format_from_url
)

__all__ = [
    "load_raster_auto", 
    "raster_to_embeddings", 
    "detect_format",
    "detect_format_from_url"
]