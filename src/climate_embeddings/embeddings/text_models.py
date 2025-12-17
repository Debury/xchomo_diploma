from typing import List, Optional
import logging
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("pip install sentence-transformers")

logger = logging.getLogger(__name__)
_CACHE = {}

class TextEmbedder:
    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        self.name = model_name
        if model_name in _CACHE:
            self.model = _CACHE[model_name]
        else:
            logger.info(f"Loading {model_name}...")
            self.model = SentenceTransformer(model_name)
            _CACHE[model_name] = self.model
        
        self.instruction = "Represent this sentence for searching relevant passages: " if "bge" in model_name else ""

    def embed_queries(self, queries: List[str]):
        inputs = [self.instruction + q for q in queries]
        return self.model.encode(inputs, normalize_embeddings=True, convert_to_numpy=True)

    def embed_documents(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)