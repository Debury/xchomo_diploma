"""
Climate Data RAG Pipeline - Source Package
Organized climate_embeddings package with loaders, embeddings, index, and RAG.
"""

__version__ = "5.0.0"
__author__ = "Climate Research Team"

# Import main package
from . import climate_embeddings

# Re-export key components for backward compatibility
from .climate_embeddings import (
    load_raster_auto,
    raster_to_embeddings,
    get_text_embedder,
    VectorIndex,
    RAGPipeline,
)

# Keep old modules for now (transitional)
from . import llm
from . import sources
from . import utils

__all__ = [
    "climate_embeddings",
    "load_raster_auto",
    "raster_to_embeddings",
    "get_text_embedder",
    "VectorIndex",
    "RAGPipeline",
    "llm",
    "sources",
    "utils",
]

