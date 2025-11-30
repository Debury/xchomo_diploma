"""
Embedding utilities - now uses climate_embeddings package.

Keep only Qdrant-specific modules here (database, generator, search).
All raster processing and text embeddings moved to climate_embeddings/.
"""

from .generator import EmbeddingGenerator
from .database import VectorDatabase
from .search import SemanticSearcher, semantic_search

# Re-export from climate_embeddings for convenience
from src.climate_embeddings.embeddings import TextEmbedder, get_text_embedder
from src.climate_embeddings.index import VectorIndex, SearchResult
from src.climate_embeddings.loaders import load_raster_auto, raster_to_embeddings, load_from_zip

__all__ = [
    # Qdrant-specific (kept here)
    "EmbeddingGenerator",
    "VectorDatabase",
    "SemanticSearcher",
    "semantic_search",
    # Re-exported from climate_embeddings
    "TextEmbedder",
    "get_text_embedder",
    "VectorIndex",
    "SearchResult",
    "load_raster_auto",
    "raster_to_embeddings",
    "load_from_zip",
]
