import pickle
import numpy as np
from typing import Dict, List, Any

class VectorIndex:
    def __init__(self):
        self.text_vectors = None
        self.stats_vectors = []
        self.metadata = []

    def add(self, text_vector, meta, stats_vector=None):
        if self.text_vectors is None:
            self.text_vectors = text_vector.reshape(1, -1)
        else:
            self.text_vectors = np.vstack([self.text_vectors, text_vector])
        self.metadata.append(meta)
        self.stats_vectors.append(stats_vector if stats_vector is not None else [])

    def search(self, query_vec, k=5):
        if self.text_vectors is None: return []
        
        # Cosine Sim
        norm_q = np.linalg.norm(query_vec)
        scores = np.dot(self.text_vectors, query_vec / norm_q)
        top_k = np.argsort(scores)[::-1][:k]
        
        return [{
            "score": float(scores[i]),
            "metadata": self.metadata[i],
            "stats": self.stats_vectors[i]
        } for i in top_k]

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.__dict__.update(pickle.load(f))