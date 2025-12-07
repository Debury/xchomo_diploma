import os
import logging
import requests
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, ScoredPoint

logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize VectorDatabase with optional configuration.
        
        Args:
            config: Optional configuration dict. If provided, reads vector_size
                   from config['vector_db']['qdrant']['vector_size'].
                   Defaults to 1024 (BAAI/bge-large-en-v1.5 dimension).
        """
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = int(os.getenv("QDRANT_REST_PORT", 6333))
        self.base_url = f"http://{self.host}:{self.port}"
        
        # Get vector size and collection name from config or defaults
        if config and "vector_db" in config:
            qdrant_config = config.get("vector_db", {}).get("qdrant", {})
            self.vector_size = qdrant_config.get("vector_size", 1024)
            self.collection_name = qdrant_config.get("collection_name", "climate_data")
            logger.info(f"Using config: collection='{self.collection_name}', vector_size={self.vector_size}")
        else:
            self.vector_size = 1024  # Default for BAAI/bge-large-en-v1.5
            self.collection_name = os.getenv("QDRANT_COLLECTION", "climate_data")
            logger.info(f"Using defaults: collection='{self.collection_name}', vector_size={self.vector_size}")
        
        # Initialize client
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
            logger.info(f"Qdrant Client initialized: {self.client}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant client: {e}")
            self.client = None
            
        self.collection = self.collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the collection exists with correct vector size."""
        # Try via client first
        if self.client:
            try:
                if self.client.collection_exists(self.collection):
                    # Check if existing collection has correct vector size
                    collection_info = self.client.get_collection(self.collection)
                    existing_size = collection_info.config.params.vectors.size
                    
                    if existing_size != self.vector_size:
                        logger.warning(
                            f"Collection '{self.collection}' exists with vector size {existing_size}, "
                            f"but expected {self.vector_size}. Deleting and recreating..."
                        )
                        # Delete existing collection
                        self.client.delete_collection(self.collection)
                        # Create new collection with correct size
                        self.client.create_collection(
                            collection_name=self.collection,
                            vectors_config=VectorParams(
                                size=self.vector_size,
                                distance=Distance.COSINE
                            )
                        )
                        logger.info(f"Recreated collection '{self.collection}' with vector size {self.vector_size}")
                    else:
                        logger.info(f"Collection '{self.collection}' exists with correct vector size {self.vector_size}")
                else:
                    # Create new collection
                    self.client.create_collection(
                        collection_name=self.collection,
                        vectors_config=VectorParams(
                            size=self.vector_size,
                            distance=Distance.COSINE
                        )
                    )
                    logger.info(f"Created collection '{self.collection}' with vector size {self.vector_size}")
                return
            except Exception as e:
                logger.warning(f"Client collection check failed ({e}), trying REST...")

        # Fallback: Create via REST if client fails
        try:
            url = f"{self.base_url}/collections/{self.collection}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                # Collection exists, check vector size
                collection_info = resp.json().get("result", {})
                config = collection_info.get("config", {})
                params = config.get("params", {})
                vectors = params.get("vectors", {})
                existing_size = vectors.get("size") if isinstance(vectors, dict) else None
                
                if existing_size and existing_size != self.vector_size:
                    logger.warning(
                        f"Collection '{self.collection}' exists with vector size {existing_size}, "
                        f"but expected {self.vector_size}. Deleting and recreating..."
                    )
                    # Delete existing collection
                    requests.delete(url, timeout=5)
                    # Create new collection
                    payload = {
                        "vectors": {
                            "size": self.vector_size,
                            "distance": "Cosine"
                        }
                    }
                    requests.put(url, json=payload, timeout=5)
                    logger.info(f"Recreated collection '{self.collection}' via REST with vector size {self.vector_size}")
                else:
                    logger.info(f"Collection '{self.collection}' exists with correct vector size")
            else:
                # Create new collection
                payload = {
                    "vectors": {
                        "size": self.vector_size,
                        "distance": "Cosine"
                    }
                }
                requests.put(url, json=payload, timeout=5)
                logger.info(f"Created collection '{self.collection}' via REST with vector size {self.vector_size}")
        except Exception as e:
            logger.error(f"REST collection creation failed: {e}")

    def add_embeddings(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict], documents: List[str]):
        if not self.client:
            logger.error("No client available for upsert")
            return
        
        points = []
        for i, (uid, vec, meta, doc) in enumerate(zip(ids, embeddings, metadatas, documents)):
            payload = meta.copy()
            payload["text_content"] = doc
            
            # Deterministic integer ID
            point_id = hash(uid) % (2**63 - 1)
            
            points.append({
                "id": point_id, 
                "vector": vec, 
                "payload": payload
            })
        
        try:
            self.client.upsert(collection_name=self.collection, points=points)
            logger.info(f"Upserted {len(points)} points")
        except Exception as e:
            logger.error(f"Upsert failed: {e}")
            raise e

    def search(self, query_vector: List[float], limit: int = 5) -> List[Any]:
        """
        Search using Client, falling back to direct REST API if client methods are missing.
        """
        # Strategy 1: Python Client
        if self.client:
            try:
                # Try standard search
                return self.client.search(
                    collection_name=self.collection,
                    query_vector=query_vector,
                    limit=limit
                )
            except (AttributeError, TypeError, Exception) as e:
                logger.warning(f"Client search failed ({e}). Falling back to REST API.")

        # Strategy 2: Direct REST API (Fail-safe)
        return self._search_via_rest(query_vector, limit)

    def _search_via_rest(self, query_vector: List[float], limit: int) -> List[Any]:
        """Manual search via HTTP requests."""
        url = f"{self.base_url}/collections/{self.collection}/points/search"
        payload = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True
        }
        
        try:
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            result = response.json().get("result", [])
            
            # Convert dict result to object-like structure for compatibility
            # The API expects objects with .score and .payload attributes
            class ScoredResult:
                def __init__(self, item):
                    self.id = item.get("id")
                    self.score = item.get("score")
                    self.payload = item.get("payload")
            
            return [ScoredResult(item) for item in result]
            
        except Exception as e:
            logger.error(f"REST search failed: {e}")
            return []