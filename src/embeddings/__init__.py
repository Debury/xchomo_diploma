"""Embedding utilities bridging the climate pipeline with the vector store."""

from .generator import EmbeddingGenerator
from .database import VectorDatabase
from .search import SemanticSearcher, semantic_search
from .pipeline import EmbeddingPipeline
from .metadata import MetadataExtractor
from .text_generator import TextGenerator

__all__ = [
    "EmbeddingGenerator",
    "VectorDatabase",
    "SemanticSearcher",
    "semantic_search",
    "EmbeddingPipeline",
    "MetadataExtractor",
    "TextGenerator",
]
