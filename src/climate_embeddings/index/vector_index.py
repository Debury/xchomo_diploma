"""
Simple in-memory vector index with similarity search and metadata filtering.

Supports:
- Cosine similarity (default), dot product, Euclidean distance
- Metadata filtering (exact match, range queries)
- Batch addition and search
- Persistence to disk (pickle/npz)

Note: For production with millions of vectors, consider FAISS or Qdrant.
This implementation uses NumPy for simplicity and works well up to ~100k vectors.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result with score, embedding, and metadata."""
    
    index: int  # Position in the index
    score: float  # Similarity score
    embedding: np.ndarray
    metadata: Dict[str, Any]


class VectorIndex:
    """
    In-memory vector index with similarity search.
    
    Features:
    - Multiple similarity metrics (cosine, dot, euclidean)
    - Metadata filtering
    - Batch operations
    - Disk persistence
    """
    
    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
    ):
        """
        Initialize vector index.
        
        Args:
            dim: Embedding dimension
            metric: Similarity metric ("cosine", "dot", "euclidean")
        """
        self.dim = dim
        self.metric = metric
        
        # Storage
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        
        # Similarity function
        self._similarity_fn = self._get_similarity_function(metric)
    
    def _get_similarity_function(self, metric: str) -> Callable:
        """Get similarity function for the specified metric."""
        if metric == "cosine":
            return self._cosine_similarity
        elif metric == "dot":
            return self._dot_product
        elif metric == "euclidean":
            return self._euclidean_distance
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'cosine', 'dot', or 'euclidean'")
    
    @staticmethod
    def _cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and vectors."""
        # Assumes vectors are already L2-normalized
        return np.dot(vectors, query)
    
    @staticmethod
    def _dot_product(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute dot product between query and vectors."""
        return np.dot(vectors, query)
    
    @staticmethod
    def _euclidean_distance(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute negative Euclidean distance (higher is better)."""
        # Negative distance so higher scores are better
        return -np.linalg.norm(vectors - query, axis=1)
    
    def add(
        self,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add a single embedding with metadata.
        
        Args:
            embedding: Numpy array of shape (dim,)
            metadata: Optional metadata dictionary
            
        Returns:
            Index position of added embedding
        """
        if embedding.shape != (self.dim,):
            raise ValueError(f"Expected embedding shape ({self.dim},), got {embedding.shape}")
        
        self.embeddings.append(embedding.astype(np.float32))
        self.metadata.append(metadata or {})
        
        return len(self.embeddings) - 1
    
    def add_batch(
        self,
        embeddings: np.ndarray,
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ) -> List[int]:
        """
        Add multiple embeddings at once.
        
        Args:
            embeddings: Numpy array of shape (n, dim)
            metadata_list: Optional list of metadata dicts (length n)
            
        Returns:
            List of index positions
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dim:
            raise ValueError(f"Expected embeddings shape (n, {self.dim}), got {embeddings.shape}")
        
        n = len(embeddings)
        if metadata_list is None:
            metadata_list = [{} for _ in range(n)]
        elif len(metadata_list) != n:
            raise ValueError(f"metadata_list length ({len(metadata_list)}) != embeddings count ({n})")
        
        start_idx = len(self.embeddings)
        self.embeddings.extend(embeddings.astype(np.float32))
        self.metadata.extend(metadata_list)
        
        return list(range(start_idx, start_idx + n))
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Search for top-k most similar embeddings.
        
        Args:
            query_embedding: Query vector of shape (dim,)
            k: Number of results to return
            filters: Optional metadata filters (exact match or range queries)
                     Examples:
                     - {"source_id": "gistemp"}  # exact match
                     - {"year": {"$gte": 2000, "$lte": 2020}}  # range
            
        Returns:
            List of SearchResult objects, sorted by score (descending)
        """
        if not self.embeddings:
            return []
        
        if query_embedding.shape != (self.dim,):
            raise ValueError(f"Expected query shape ({self.dim},), got {query_embedding.shape}")
        
        # Apply filters first to get candidate indices
        candidate_indices = self._apply_filters(filters) if filters else list(range(len(self.embeddings)))
        
        if not candidate_indices:
            return []
        
        # Get candidate embeddings
        candidate_embeddings = np.array([self.embeddings[i] for i in candidate_indices])
        
        # Compute similarities
        scores = self._similarity_fn(query_embedding, candidate_embeddings)
        
        # Get top-k indices (in candidate space)
        top_k = min(k, len(candidate_indices))
        top_indices_in_candidates = np.argsort(scores)[::-1][:top_k]
        
        # Map back to original indices
        results = []
        for candidate_idx in top_indices_in_candidates:
            original_idx = candidate_indices[candidate_idx]
            results.append(SearchResult(
                index=original_idx,
                score=float(scores[candidate_idx]),
                embedding=self.embeddings[original_idx],
                metadata=self.metadata[original_idx],
            ))
        
        return results
    
    def _apply_filters(self, filters: Dict[str, Any]) -> List[int]:
        """
        Apply metadata filters to get candidate indices.
        
        Supports:
        - Exact match: {"key": "value"}
        - Range queries: {"key": {"$gte": min_val, "$lte": max_val}}
        - Multiple filters (AND logic)
        """
        candidate_indices = list(range(len(self.embeddings)))
        
        for key, condition in filters.items():
            candidate_indices = [
                i for i in candidate_indices
                if self._matches_condition(self.metadata[i], key, condition)
            ]
        
        return candidate_indices
    
    @staticmethod
    def _matches_condition(metadata: Dict[str, Any], key: str, condition: Any) -> bool:
        """Check if metadata matches a filter condition."""
        if key not in metadata:
            return False
        
        value = metadata[key]
        
        # Exact match
        if not isinstance(condition, dict):
            return value == condition
        
        # Range queries
        if "$gte" in condition and value < condition["$gte"]:
            return False
        if "$lte" in condition and value > condition["$lte"]:
            return False
        if "$gt" in condition and value <= condition["$gt"]:
            return False
        if "$lt" in condition and value < condition["$lt"]:
            return False
        if "$ne" in condition and value == condition["$ne"]:
            return False
        
        return True
    
    def __len__(self) -> int:
        """Return number of embeddings in index."""
        return len(self.embeddings)
    
    def save(self, path: str | Path) -> None:
        """
        Save index to disk.
        
        Args:
            path: Path to save file (.pkl for pickle, .npz for numpy)
        """
        path = Path(path)
        
        if path.suffix == ".npz":
            # Save as NumPy archive
            embeddings_array = np.array(self.embeddings)
            np.savez_compressed(
                path,
                embeddings=embeddings_array,
                metadata=pickle.dumps(self.metadata),
                dim=self.dim,
                metric=self.metric,
            )
        else:
            # Save as pickle
            with open(path, "wb") as f:
                pickle.dump({
                    "embeddings": self.embeddings,
                    "metadata": self.metadata,
                    "dim": self.dim,
                    "metric": self.metric,
                }, f)
        
        logger.info(f"Saved vector index with {len(self)} embeddings to {path}")
    
    @classmethod
    def load(cls, path: str | Path) -> VectorIndex:
        """
        Load index from disk.
        
        Args:
            path: Path to saved index file
            
        Returns:
            VectorIndex instance
        """
        path = Path(path)
        
        if path.suffix == ".npz":
            # Load from NumPy archive
            data = np.load(path, allow_pickle=True)
            index = cls(dim=int(data["dim"]), metric=str(data["metric"]))
            index.embeddings = list(data["embeddings"])
            index.metadata = pickle.loads(data["metadata"].tobytes())
        else:
            # Load from pickle
            with open(path, "rb") as f:
                data = pickle.load(f)
            index = cls(dim=data["dim"], metric=data["metric"])
            index.embeddings = data["embeddings"]
            index.metadata = data["metadata"]
        
        logger.info(f"Loaded vector index with {len(index)} embeddings from {path}")
        return index


# Example usage
if __name__ == "__main__":
    print("Testing VectorIndex...")
    
    # Create index
    index = VectorIndex(dim=384, metric="cosine")
    
    # Add some vectors
    vec1 = np.random.randn(384).astype(np.float32)
    vec1 /= np.linalg.norm(vec1)  # Normalize for cosine
    
    vec2 = np.random.randn(384).astype(np.float32)
    vec2 /= np.linalg.norm(vec2)
    
    vec3 = np.random.randn(384).astype(np.float32)
    vec3 /= np.linalg.norm(vec3)
    
    index.add(vec1, {"source": "gistemp", "year": 2020})
    index.add(vec2, {"source": "era5", "year": 2021})
    index.add(vec3, {"source": "gistemp", "year": 2022})
    
    print(f"Index size: {len(index)}")
    
    # Search
    query = np.random.randn(384).astype(np.float32)
    query /= np.linalg.norm(query)
    
    results = index.search(query, k=2)
    print(f"\nTop-2 results:")
    for r in results:
        print(f"  Index: {r.index}, Score: {r.score:.4f}, Metadata: {r.metadata}")
    
    # Search with filter
    results_filtered = index.search(query, k=10, filters={"source": "gistemp"})
    print(f"\nFiltered results (source=gistemp):")
    for r in results_filtered:
        print(f"  Index: {r.index}, Score: {r.score:.4f}, Metadata: {r.metadata}")
