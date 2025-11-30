"""
Powerful text embedding models for RAG queries.

Supports modern, high-quality embedding models:
- BAAI/bge-large-en-v1.5 (1024-dim, SOTA for retrieval)
- Alibaba-NLP/gte-large (1024-dim alternative)
- sentence-transformers/all-mpnet-base-v2 (768-dim fallback)

Uses sentence-transformers for easy loading and inference.
All embeddings are L2-normalized for cosine similarity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "sentence-transformers is required for text embeddings. "
        "Install with: pip install sentence-transformers"
    )

logger = logging.getLogger(__name__)

# Model registry
AVAILABLE_MODELS = {
    "bge-large": "BAAI/bge-large-en-v1.5",  # 1024-dim, best for retrieval
    "gte-large": "Alibaba-NLP/gte-large",  # 1024-dim alternative
    "mpnet": "sentence-transformers/all-mpnet-base-v2",  # 768-dim fallback
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",  # 384-dim fast
}

DEFAULT_MODEL = "bge-large"

# Global cache for loaded models
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


class TextEmbedder:
    """
    Text embedding model wrapper with caching and normalization.
    
    Features:
    - Automatic model downloading and caching
    - L2 normalization for cosine similarity
    - Batch encoding support
    - Device auto-detection (CUDA if available)
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        """
        Initialize text embedder.
        
        Args:
            model_name: Model identifier (e.g., "bge-large", "gte-large") or HuggingFace model path
            device: Device to use ("cuda", "cpu", or None for auto-detection)
            normalize: Whether to L2-normalize embeddings (recommended for cosine similarity)
        """
        # Resolve model name
        self.model_path = AVAILABLE_MODELS.get(model_name, model_name)
        self.normalize = normalize
        
        # Load or retrieve cached model
        cache_key = f"{self.model_path}_{device}"
        if cache_key in _MODEL_CACHE:
            self.model = _MODEL_CACHE[cache_key]
            logger.info(f"Using cached model: {self.model_path}")
        else:
            logger.info(f"Loading text embedding model: {self.model_path}")
            self.model = SentenceTransformer(self.model_path, device=device)
            _MODEL_CACHE[cache_key] = self.model
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension of the model."""
        return self.model.get_sentence_embedding_dimension()
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Encode text(s) to embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress_bar: Whether to show progress bar
            
        Returns:
            Numpy array of shape (n, embedding_dim) or (embedding_dim,) for single text
        """
        # Handle single string
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        # Encode
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        
        # Return single vector for single input
        if single_input:
            return embeddings[0]
        
        return embeddings
    
    def __call__(self, text: str) -> np.ndarray:
        """Convenience method for encoding single text."""
        return self.encode(text)


def get_text_embedder(
    model_name: str = DEFAULT_MODEL,
    device: Optional[str] = None,
    normalize: bool = True,
) -> Callable[[str], np.ndarray]:
    """
    Get a text embedding function (factory pattern).
    
    Args:
        model_name: Model identifier or HuggingFace path
        device: Device to use (None for auto)
        normalize: Whether to L2-normalize
        
    Returns:
        Function that takes a string and returns numpy embedding vector
        
    Example:
        >>> embedder = get_text_embedder("bge-large")
        >>> query_vec = embedder("What is the temperature trend?")
        >>> query_vec.shape
        (1024,)
    """
    text_embedder = TextEmbedder(model_name=model_name, device=device, normalize=normalize)
    return text_embedder


def list_available_models() -> dict[str, str]:
    """Return dictionary of available model shortcuts and their paths."""
    return AVAILABLE_MODELS.copy()


def get_embedding_dim(model_name: str = DEFAULT_MODEL) -> int:
    """
    Get embedding dimension for a model without loading it fully.
    
    Args:
        model_name: Model identifier
        
    Returns:
        Embedding dimension
    """
    # Known dimensions
    known_dims = {
        "bge-large": 1024,
        "gte-large": 1024,
        "mpnet": 768,
        "minilm": 384,
    }
    
    if model_name in known_dims:
        return known_dims[model_name]
    
    # Load model to check dimension
    embedder = TextEmbedder(model_name)
    return embedder.embedding_dim


# Example usage and testing
if __name__ == "__main__":
    # Test text embedder
    print("Testing TextEmbedder...")
    
    embedder = get_text_embedder("minilm")  # Fast model for testing
    
    # Single text
    query = "What is the global temperature anomaly in 2023?"
    vec = embedder(query)
    print(f"Query: {query}")
    print(f"Embedding shape: {vec.shape}")
    print(f"Embedding norm: {np.linalg.norm(vec):.4f}")  # Should be ~1.0 if normalized
    
    # Batch encoding
    embedder_obj = TextEmbedder("minilm")
    texts = [
        "Temperature anomaly analysis",
        "Precipitation trends over Europe",
        "Sea level rise projections",
    ]
    vecs = embedder_obj.encode(texts)
    print(f"\nBatch encoding:")
    print(f"Input: {len(texts)} texts")
    print(f"Output shape: {vecs.shape}")
    
    # Cosine similarity
    from numpy.linalg import norm
    cos_sim = np.dot(vecs[0], vecs[1]) / (norm(vecs[0]) * norm(vecs[1]))
    print(f"Cosine similarity between first two texts: {cos_sim:.4f}")
