import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, ScoredPoint

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self):
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_REST_PORT", 6333))
        
        # Initialize client
        try:
            self.client = QdrantClient(host=host, port=port)
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.client = None
            
        self.collection = "climate_rag"
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client: return
        try:
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
                )
        except Exception as e:
            logger.warning(f"Could not check/create collection: {e}")

    def add_embeddings(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict], documents: List[str]):
        if not self.client: return
        
        points = []
        for i, (uid, vec, meta, doc) in enumerate(zip(ids, embeddings, metadatas, documents)):
            # Ensure payload is robust
            payload = meta.copy()
            payload["text_content"] = doc
            
            # Qdrant requires integer or UUID ids for best performance, 
            # but string ids are supported if formatted as UUIDs. 
            # We'll use a deterministic hash for simplicity here.
            point_id = hash(uid) % (2**63 - 1) 
            
            points.append({
                "id": point_id, 
                "vector": vec, 
                "payload": payload
            })
        
        try:
            self.client.upsert(collection_name=self.collection, points=points)
            logger.info(f"Upserted {len(points)} points to {self.collection}")
        except Exception as e:
            logger.error(f"Failed to upsert embeddings: {e}")
            raise e

    def search(self, query_vector: List[float], limit: int = 5) -> List[ScoredPoint]:
        """
        Encapsulated search method.
        """
        if not self.client: return []
        
        try:
            # Standard search
            return self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=limit
            )
        except AttributeError:
            # Fallback if 'search' is missing (very rare edge case)
            # Try newer query API or verify client type
            logger.warning("Client missing 'search', attempting 'search_batch'")
            results = self.client.search_batch(
                collection_name=self.collection,
                requests=[{"vector": query_vector, "limit": limit}]
            )
            return results[0] if results else []
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []