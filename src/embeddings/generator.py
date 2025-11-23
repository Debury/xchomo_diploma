"""Sentence-transformer based embedding generation utilities."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Thin wrapper around ``SentenceTransformer`` with handy helpers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name, device=device)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def generate_embeddings(self, texts: Iterable[str]) -> np.ndarray:
        """Encode a batch of texts into embeddings."""

        texts_list = list(texts)
        if not texts_list:
            raise ValueError("No texts provided for embedding generation")

        embeddings = self.model.encode(
            texts_list,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embeddings.astype(np.float32)

    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Encode a single string and return a 1-D numpy vector."""

        if not text:
            raise ValueError("Text cannot be empty")
        return self.generate_embeddings([text])[0]

    # ------------------------------------------------------------------
    def get_embedding_dimension(self) -> int:
        """Return the dimensionality of the loaded model."""

        return self.model.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    @property
    def vector_size(self) -> int:
        return self.get_embedding_dimension()
