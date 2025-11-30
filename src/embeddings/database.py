import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

class VectorDatabase:
    def __init__(self):
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_REST_PORT", 6333))
        self.client = QdrantClient(host=host, port=port)
        self.collection = "climate_rag"
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
            )

    def add_embeddings(self, ids, embeddings, metadatas, documents):
        points = []
        for i, (uid, vec, meta, doc) in enumerate(zip(ids, embeddings, metadatas, documents)):
            meta["text_content"] = doc
            points.append({"id": hash(uid) % 10**18, "vector": vec, "payload": meta}) # Simple ID hash
        
        # Batch upload
        self.client.upsert(collection_name=self.collection, points=points)