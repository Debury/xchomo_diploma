"""RAG (Retrieval-Augmented Generation) pipeline."""

from climate_embeddings.rag.rag_pipeline import (
    RAGPipeline,
    RAGContext,
    build_index_from_embeddings,
)

__all__ = [
    "RAGPipeline",
    "RAGContext",
    "build_index_from_embeddings",
]
