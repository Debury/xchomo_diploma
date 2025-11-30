# climate_embeddings/embeddings/text_models.py
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        print(f"Loading Text Model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        # BGE instructions: usually need a prompt for queries, but vanilla works for metadata
        self.instruction = "Represent this sentence for searching relevant passages: "

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        # BGE expects instruction for queries
        inputs = [self.instruction + q for q in queries]
        embeddings = self.model.encode(inputs, normalize_embeddings=True)
        return embeddings

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)