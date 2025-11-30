"""
Climate Embeddings Package

A comprehensive system for:
- Loading climate rasters in multiple formats (NetCDF, GRIB, GeoTIFF, HDF5, CSV, ZIP)
- Generating embeddings with powerful text models (BGE, GTE)
- Building vector indices for similarity search
- RAG (Retrieval-Augmented Generation) for climate Q&A
"""

__version__ = "1.0.0"

from climate_embeddings.loaders import load_raster_auto, raster_to_embeddings, detect_format
from climate_embeddings.embeddings import get_text_embedder, TextEmbedder
from climate_embeddings.index import VectorIndex, SearchResult
from climate_embeddings.rag import RAGPipeline, build_index_from_embeddings

__all__ = [
    "load_raster_auto",
    "raster_to_embeddings",
    "detect_format",
    "get_text_embedder",
    "TextEmbedder",
    "VectorIndex",
    "SearchResult",
    "RAGPipeline",
    "build_index_from_embeddings",
]
