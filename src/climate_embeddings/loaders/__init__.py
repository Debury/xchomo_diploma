# src/climate_embeddings/loaders/__init__.py

from .raster_pipeline import load_raster_auto, raster_to_embeddings, load_data_source
from .detect_format import detect_format, detect_format_from_url

__all__ = [
    "load_raster_auto",
    "raster_to_embeddings",
    "load_data_source",
    "detect_format",
    "detect_format_from_url",
]