"""Semantic search helpers built on top of the embedding generator and vector DB."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .database import VectorDatabase
from .generator import EmbeddingGenerator


class SemanticSearcher:
    """Convenience wrapper combining embedding generation with vector queries."""

    def __init__(
        self,
        database: Optional[VectorDatabase] = None,
        generator: Optional[EmbeddingGenerator] = None,
    ) -> None:
        self.database = database or VectorDatabase()
        self.generator = generator or EmbeddingGenerator()

    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for the documents that best match ``query``."""
        if not query or not query.strip():
            raise ValueError("Query text cannot be empty.")

        embedding = self.generator.generate_single_embedding(query)
        raw = self.database.query(query_embeddings=embedding, k=k)
        return self._format_results(raw)

    def search_batch(self, queries: Sequence[str], k: int = 5) -> List[List[Dict[str, Any]]]:
        if not queries:
            raise ValueError("At least one query is required.")
        return [self.search(query, k=k) for query in queries]

    # ------------------------------------------------------------------
    def _format_results(self, raw: Dict[str, List[List[Any]]]) -> List[Dict[str, Any]]:
        return self._format_single_result(raw, 0)

    def _format_single_result(self, raw: Dict[str, List[List[Any]]], batch_index: int) -> List[Dict[str, Any]]:
        ids_batch = raw.get("ids", [[]])
        docs_batch = raw.get("documents", [[]])
        metadata_batch = raw.get("metadatas", [[]])
        scores_batch = raw.get("scores") or raw.get("distances") or [[]]
        distances_batch = raw.get("distances") or raw.get("scores") or [[]]

        ids = ids_batch[batch_index] if batch_index < len(ids_batch) else []
        docs = docs_batch[batch_index] if batch_index < len(docs_batch) else []
        metadatas = metadata_batch[batch_index] if batch_index < len(metadata_batch) else []
        scores = scores_batch[batch_index] if batch_index < len(scores_batch) else []
        distances = distances_batch[batch_index] if batch_index < len(distances_batch) else []

        results: List[Dict[str, Any]] = []
        for idx, text, metadata, score, distance in zip(ids, docs, metadatas, scores, distances):
            metadata = metadata or {}
            if isinstance(score, (int, float)) and isinstance(distance, (int, float)):
                similarity = float(score) if raw.get("scores") else 1 - float(distance)
                dist_value = float(distance) if raw.get("distances") else 1 - float(score)
            else:
                similarity = 0.0
                dist_value = 1.0
            text_value = text or metadata.get("text", "")
            results.append(
                {
                    "id": str(idx),
                    "text": text_value,
                    "document": text_value,
                    "metadata": metadata,
                    "similarity": similarity,
                    "distance": dist_value,
                }
            )
        return results


def semantic_search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """Stateless helper mirroring the older Chroma convenience function."""
    return SemanticSearcher().search(query=query, k=k)
