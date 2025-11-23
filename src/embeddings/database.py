"""Qdrant-backed (with graceful in-memory fallback) vector store wrapper."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency when using pure in-memory mode
    from qdrant_client import QdrantClient, models
except ImportError:  # pragma: no cover
    QdrantClient = None  # type: ignore
    models = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class _MemoryPoint:
    """Container for the lightweight fallback backend."""

    id: str
    vector: np.ndarray
    text: str
    metadata: Dict[str, Any]


class _CollectionProxy:
    """Expose Chroma-like ``collection`` helpers for compatibility."""

    def __init__(self, database: "VectorDatabase") -> None:
        self._database = database

    def count(self) -> int:
        return self._database.count()

    def get(self, limit: Optional[int] = None, include: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        return self._database.get(limit=limit, include=include)


class VectorDatabase:
    """High-level helper for interacting with the vector database.

    The class prefers Qdrant but gracefully falls back to an in-memory storage so
    that unit tests and notebooks can run without a running Qdrant instance.
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        host: Optional[str] = None,
        rest_port: Optional[int] = None,
        grpc_port: Optional[int] = None,
        vector_size: Optional[int] = None,
        distance: str = "COSINE",
        prefer_grpc: Optional[bool] = None,
        auto_connect: bool = True,
    ) -> None:
        self.collection_name = collection_name or os.getenv("QDRANT_COLLECTION", "climate_data")
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.rest_port = int(rest_port or os.getenv("QDRANT_REST_PORT", 6333))
        self.grpc_port = int(grpc_port or os.getenv("QDRANT_GRPC_PORT", 6334))
        self.vector_size = vector_size or int(os.getenv("VECTOR_DB_VECTOR_SIZE", 384))
        self.distance = distance.upper()
        env_prefer_grpc = os.getenv("QDRANT_PREFER_GRPC")
        if prefer_grpc is None and env_prefer_grpc is not None:
            prefer_grpc = env_prefer_grpc.lower() == "true"
        self.prefer_grpc = True if prefer_grpc is None else prefer_grpc
        self.memory_only = os.getenv("VECTOR_DB_MEMORY_ONLY", "false").lower() == "true"

        self.client: Optional[QdrantClient] = None
        self._backend: str = "memory"
        self._memory_store: List[_MemoryPoint] = []
        self.collection = _CollectionProxy(self)

        if auto_connect and not self.memory_only:
            self._initialize()

    # ------------------------------------------------------------------
    def _initialize(self) -> None:
        try:
            if QdrantClient is None:
                raise RuntimeError("qdrant-client package is not installed")
            logger.info(
                "Connecting to Qdrant at %s (REST %s, gRPC %s)"
                " for collection '%s'",
                self.host,
                self.rest_port,
                self.grpc_port,
                self.collection_name,
            )
            self.client = QdrantClient(
                host=self.host,
                port=self.rest_port,
                grpc_port=self.grpc_port if self.prefer_grpc else None,
            )

            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=getattr(models.Distance, self.distance, models.Distance.COSINE),
                    ),
                    optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
                )
            self._backend = "qdrant"
            logger.info("Qdrant vector store ready (collection '%s').", self.collection_name)
        except Exception as exc:  # pragma: no cover - only hit when qdrant unavailable
            logger.warning(
                "Falling back to in-memory vector store because Qdrant is unavailable: %s",
                exc,
            )
            self.client = None
            self._backend = "memory"

    # ------------------------------------------------------------------
    # Basic operations
    # ------------------------------------------------------------------
    def count(self) -> int:
        if self._backend == "qdrant" and self.client:
            return self.client.count(self.collection_name, exact=True).count
        return len(self._memory_store)

    def add_embeddings(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray | Sequence[Sequence[float]],
        documents: Optional[Sequence[str]] = None,
        metadatas: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
    ) -> None:
        if not ids:
            logger.warning("No embeddings to add; skipping vector store upsert.")
            return
        documents = documents or ["" for _ in ids]
        metadatas = metadatas or [{} for _ in ids]

        self._validate_lengths(ids, embeddings, documents, metadatas)

        embeddings_array = np.asarray(embeddings, dtype=np.float32)

        def _normalize_point_id(value: Any) -> str | int:
            if isinstance(value, (int, np.integer)):
                return int(value)
            text_value = str(value)
            try:
                return str(uuid.UUID(text_value))
            except ValueError:
                return str(uuid.uuid5(uuid.NAMESPACE_URL, text_value))

        if self._backend == "qdrant" and self.client:
            points = []
            for idx, embedding, text, metadata in zip(ids, embeddings_array, documents, metadatas):
                payload = dict(metadata or {})
                payload.setdefault("text", text)
                point_id = _normalize_point_id(idx)
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=list(map(float, embedding)),
                        payload=payload,
                    )
                )
            self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
            return

        # memory backend
        for idx, embedding, text, metadata in zip(ids, embeddings_array, documents, metadatas):
            vector = np.asarray(embedding, dtype=np.float32)
            self._memory_store.append(
                _MemoryPoint(id=str(_normalize_point_id(idx)), vector=vector, text=text, metadata=metadata or {})
            )

    def get(self, limit: Optional[int] = None, include: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        limit = limit or 100
        include = include or ["embeddings", "metadatas", "documents", "ids"]

        if self._backend == "qdrant" and self.client:
            with_vectors = "embeddings" in include
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=with_vectors,
            )
            return self._points_to_chroma_like(points, include)

        # memory backend
        return self._points_to_chroma_like(self._memory_store[:limit], include)

    def delete_embeddings_by_source(self, source_id: str) -> int:
        """Remove all embeddings whose payload contains the given source id."""

        if not source_id:
            return 0

        if self._backend == "qdrant" and self.client and models:
            condition = models.FieldCondition(
                key="source_id",
                match=models.MatchValue(value=source_id),
            )
            source_filter = models.Filter(must=[condition])
            deleted_count = 0

            try:  # query count before deleting so we can report how many were removed
                deleted_count = int(
                    self.client.count(
                        collection_name=self.collection_name,
                        exact=True,
                        query_filter=source_filter,
                    ).count
                )
            except Exception:  # pragma: no cover - informative only
                deleted_count = 0

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=source_filter),
                wait=True,
            )
            return deleted_count

        # memory backend cleanup
        before = len(self._memory_store)
        self._memory_store = [
            point
            for point in self._memory_store
            if (point.metadata or {}).get("source_id") != source_id
        ]
        return before - len(self._memory_store)

    def clear_collection(self) -> int:
        """Delete every embedding in the collection and return how many were removed."""

        removed = self.count()

        if self._backend == "qdrant" and self.client and models:
            vector_params = models.VectorParams(
                size=self.vector_size,
                distance=getattr(models.Distance, self.distance, models.Distance.COSINE),
            )
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=vector_params,
                optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
            )
            return removed

        self._memory_store.clear()
        return removed

    def query(
        self,
        query_embeddings: Sequence[float] | Sequence[Sequence[float]],
        k: int = 5,
    ) -> Dict[str, Any]:
        query_array = np.asarray(query_embeddings, dtype=np.float32)
        if query_array.ndim == 1:
            query_array = query_array.reshape(1, -1)

        if self._backend == "qdrant" and self.client:
            combined: Optional[Dict[str, List[List[Any]]]] = None
            for vector in query_array:
                result = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vector.tolist(),
                    limit=k,
                    with_vectors=False,
                    with_payload=True,
                )
                formatted = self._format_query_result(result.points)
                if combined is None:
                    combined = formatted
                else:
                    for key, value in formatted.items():
                        combined.setdefault(key, []).extend(value)
            return combined or self._empty_query_payload()

        # memory backend: cosine similarity
        if not self._memory_store:
            return self._empty_query_payload()

        vectors = np.stack([point.vector for point in self._memory_store])
        query_vec = query_array
        sims = query_vec @ vectors.T
        norms = np.linalg.norm(query_vec, axis=1, keepdims=True) * np.linalg.norm(vectors, axis=1)
        similarities = sims / np.clip(norms, 1e-12, None)

        top_indices = np.argsort(-similarities, axis=1)[:, :k]
        ids = []
        documents = []
        metadatas = []
        distances = []
        for row, indices in zip(similarities, top_indices):
            ids.append([self._memory_store[i].id for i in indices])
            documents.append([self._memory_store[i].text for i in indices])
            metadatas.append([self._memory_store[i].metadata for i in indices])
            distances.append([1 - float(row[i]) for i in indices])
        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
            "scores": [[1 - d for d in distance_row] for distance_row in distances],
        }

    # ------------------------------------------------------------------
    def _format_query_result(self, points: Sequence[models.ScoredPoint]) -> Dict[str, Any]:
        ids_batch: List[List[str]] = []
        documents_batch: List[List[str]] = []
        metadatas_batch: List[List[Dict[str, Any]]] = []
        distances_batch: List[List[float]] = []
        scores_batch: List[List[float]] = []

        scored_points = list(points)
        if not scored_points:
            return self._empty_query_payload()

        ids_batch.append([str(point.id) for point in scored_points])
        documents_batch.append([(point.payload or {}).get("text", "") for point in scored_points])
        metadatas_batch.append([point.payload or {} for point in scored_points])
        scores_batch.append([float(point.score) for point in scored_points])
        distances_batch.append([1 - float(point.score) for point in scored_points])

        return {
            "ids": ids_batch,
            "documents": documents_batch,
            "metadatas": metadatas_batch,
            "distances": distances_batch,
            "scores": scores_batch,
        }

    @staticmethod
    def _empty_query_payload() -> Dict[str, List[List[Any]]]:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "scores": [[]],
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _validate_lengths(
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]] | np.ndarray,
        documents: Sequence[str],
        metadatas: Sequence[Optional[Dict[str, Any]]],
    ) -> None:
        lengths = {"ids": len(ids), "embeddings": len(embeddings), "documents": len(documents), "metadatas": len(metadatas)}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Mismatched input lengths: {lengths}")

    # ------------------------------------------------------------------
    def _points_to_chroma_like(
        self,
        points: Iterable[Any],
        include: Sequence[str],
    ) -> Dict[str, Any]:
        ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []
        documents: List[str] = []

        for point in points:
            payload = getattr(point, "payload", None)
            vector = getattr(point, "vector", None)
            point_id = getattr(point, "id", None)
            # memory points store attributes directly
            if payload is None and isinstance(point, _MemoryPoint):
                payload = point.metadata
            if vector is None and isinstance(point, _MemoryPoint):
                vector = point.vector
            if point_id is None and isinstance(point, _MemoryPoint):
                point_id = point.id

            ids.append(str(point_id))
            if "embeddings" in include and vector is not None:
                embeddings.append(list(map(float, vector)))
            if "metadatas" in include:
                metadatas.append(dict(payload or {}))
            if "documents" in include:
                documents.append((payload or {}).get("text") if payload else getattr(point, "text", ""))

        return {
            "ids": ids,
            "embeddings": embeddings if "embeddings" in include else None,
            "metadatas": metadatas if "metadatas" in include else None,
            "documents": documents if "documents" in include else None,
        }