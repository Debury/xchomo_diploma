# climate_embeddings/rag/vector_index.py
import numpy as np
import pickle
from typing import List, Dict, Any

class SimpleVectorIndex:
    """
    Stores embeddings and performs Cosine Similarity search.
    For RAG: We index the 'Text Description' of the chunk.
    """
    def __init__(self):
        self.vectors = None
        self.metadata = []
        self.raster_stats = [] # Store the numeric stats here

    def add(self, text_vector: np.ndarray, meta: Dict, stats_vector: np.ndarray):
        if self.vectors is None:
            self.vectors = text_vector.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, text_vector])
        
        self.metadata.append(meta)
        self.raster_stats.append(stats_vector)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'vectors': self.vectors, 
                'metadata': self.metadata,
                'raster_stats': self.raster_stats
            }, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectors = data['vectors']
            self.metadata = data['metadata']
            self.raster_stats = data['raster_stats']

    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Dict]:
        """Returns top K matches with metadata and stats."""
        if self.vectors is None:
            return []
            
        # Cosine Similarity
        scores = np.dot(self.vectors, query_vector.T).flatten()
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "score": float(scores[idx]),
                "metadata": self.metadata[idx],
                "stats": self.raster_stats[idx].tolist()
            })
        return results