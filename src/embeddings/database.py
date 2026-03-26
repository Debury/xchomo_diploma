import math
import os
import logging
import requests
import uuid
import warnings
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, ScoredPoint, PointStruct

logger = logging.getLogger(__name__)

# Suppress Qdrant version compatibility warnings globally
warnings.filterwarnings("ignore", message=".*version.*incompatible.*", category=UserWarning)

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
        
        # Initialize client (disable version check to avoid warnings with minor version differences)
        try:
            import warnings
            # Suppress version compatibility warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*version.*incompatible.*")
                grpc_port = int(os.getenv("QDRANT_GRPC_PORT", 6334))
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    grpc_port=grpc_port,
                    prefer_grpc=True,
                    timeout=120
                )
            logger.info(f"Qdrant Client initialized: {self.host}:{self.port}")
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

    def disable_indexing(self):
        """Disable HNSW indexing for bulk ingestion (Qdrant best practice).

        Sets indexing_threshold very high so Qdrant skips incremental
        HNSW graph rebuilds during upsert.  Call enable_indexing() after
        the bulk load to trigger a single optimized index build.
        """
        if not self.client:
            logger.warning("No client available — cannot disable indexing")
            return
        try:
            from qdrant_client.http.models import OptimizersConfigDiff
            self.client.update_collection(
                collection_name=self.collection,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=100_000_000,
                ),
            )
            logger.info(f"Disabled HNSW indexing on '{self.collection}' for bulk load")
        except Exception as e:
            logger.warning(f"Failed to disable indexing: {e}")

    def enable_indexing(self):
        """Re-enable HNSW indexing after bulk ingestion.

        Restores the default indexing_threshold (20000) so Qdrant
        triggers a single optimized HNSW build for all ingested points.
        """
        if not self.client:
            logger.warning("No client available — cannot enable indexing")
            return
        try:
            from qdrant_client.http.models import OptimizersConfigDiff
            self.client.update_collection(
                collection_name=self.collection,
                optimizer_config=OptimizersConfigDiff(
                    indexing_threshold=20_000,
                ),
            )
            logger.info(f"Re-enabled HNSW indexing on '{self.collection}' (threshold=20000)")
        except Exception as e:
            logger.warning(f"Failed to enable indexing: {e}")

    def add_embeddings(self, ids: List[str], embeddings: List[List[float]], metadatas: List[Dict], documents: List[str] = None):
        """
        Add embeddings to vector database with structured metadata.
        
        Args:
            ids: List of unique identifiers
            embeddings: List of embedding vectors
            metadatas: List of structured metadata dictionaries (normalized schema)
            documents: Optional list of text documents (deprecated - text is generated dynamically)
        
        Note: The 'documents' parameter is kept for backward compatibility but is not stored.
        Text descriptions are generated dynamically from metadata when needed.
        """
        if not self.client:
            logger.error("No client available for upsert")
            return
        
        points = []
        for i, (uid, vec, meta) in enumerate(zip(ids, embeddings, metadatas)):
            # Use metadata directly (already normalized and structured)
            # Sanitize NaN/Infinity values that break Qdrant JSON serialization
            payload = {
                k: v for k, v in meta.items()
                if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
            }
            
            # DO NOT store text_content - it's generated dynamically from metadata
            # This keeps the DB clean and filterable
            
            # Deterministic UUID from string ID (gRPC requires PointStruct)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(uid)))

            points.append(PointStruct(
                id=point_id,
                vector=vec,
                payload=payload,
            ))
        
        # Batch upserts — 2000 points per call (gRPC binary protobuf handles large batches)
        BATCH_SIZE = 2000
        MAX_RETRIES = 3
        import time

        for batch_start in range(0, len(points), BATCH_SIZE):
            batch = points[batch_start:batch_start + BATCH_SIZE]
            for attempt in range(MAX_RETRIES):
                try:
                    self.client.upsert(collection_name=self.collection, points=batch)
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        wait = 2 ** attempt
                        logger.warning(f"Upsert batch {batch_start // BATCH_SIZE} failed (attempt {attempt + 1}), retrying in {wait}s: {e}")
                        time.sleep(wait)
                    else:
                        logger.error(f"Upsert batch failed after {MAX_RETRIES} attempts: {e}")
                        raise e

        logger.info(f"Upserted {len(points)} points in {(len(points) - 1) // BATCH_SIZE + 1} batches")

    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        query_filter=None,
        exact: bool = False,
    ) -> List[Any]:
        """
        Search using Client, falling back to direct REST API if client methods are missing.
        Supports optional filtering by metadata fields.

        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results to return
            filter_dict: Optional filter dictionary. Examples:
                - {"source_id": "NOAA_GSOM"}  # Filter by source
                - {"variable": "TMAX"}  # Filter by variable
                - {"source_id": "NOAA_GSOM", "variable": "TMIN"}  # Multiple filters (AND)
            query_filter: Optional pre-built Qdrant Filter object (takes precedence
                over filter_dict). Supports Range conditions for spatial filtering.

        Returns:
            List of objects with .score and .payload attributes (ScoredPoint-like).
        """
        # Strategy 1: Try Python Client (various API versions)
        if self.client:
            # Try v1.11.x API first (search method)
            if hasattr(self.client, 'search'):
                try:
                    from qdrant_client.models import SearchParams
                    search_kwargs = {
                        "collection_name": self.collection,
                        "query_vector": query_vector,
                        "limit": limit,
                        "with_payload": True,
                        "search_params": SearchParams(
                            hnsw_ef=256,
                            exact=exact,
                        ),
                    }
                    # Use pre-built filter if provided, else build from dict
                    if query_filter is not None:
                        search_kwargs["query_filter"] = query_filter
                    elif filter_dict:
                        from qdrant_client.models import Filter, FieldCondition, MatchValue
                        conditions = []
                        for key, value in filter_dict.items():
                            conditions.append(
                                FieldCondition(key=key, match=MatchValue(value=value))
                            )
                        if conditions:
                            search_kwargs["query_filter"] = Filter(must=conditions)

                    results = self.client.search(**search_kwargs)
                    if results:
                        return results
                except Exception as e:
                    logger.debug(f"Client.search() failed ({e}), trying REST fallback")
            # Try newer API (query_points) if search doesn't exist
            elif hasattr(self.client, 'query_points'):
                try:
                    from qdrant_client.models import Filter, FieldCondition, MatchValue
                    query_kwargs = {
                        "collection_name": self.collection,
                        "query": query_vector,
                        "limit": limit,
                        "with_payload": True
                    }
                    # Use pre-built filter if provided, else build from dict
                    if query_filter is not None:
                        query_kwargs["query_filter"] = query_filter
                    elif filter_dict:
                        conditions = []
                        for key, value in filter_dict.items():
                            conditions.append(
                                FieldCondition(key=key, match=MatchValue(value=value))
                            )
                        if conditions:
                            query_kwargs["query_filter"] = Filter(must=conditions)

                    results = self.client.query_points(**query_kwargs)
                    if results and hasattr(results, 'points'):
                        return results.points
                except Exception as e:
                    logger.debug(f"Client.query_points() failed ({e}), trying REST fallback")
            else:
                logger.debug("No search method available on client, using REST API")

        # Strategy 2: Direct REST API (Fail-safe - always works)
        return self._search_via_rest(query_vector, limit, filter_dict)

    def search_and_rerank(
        self,
        query_text: str,
        query_vector: List[float],
        limit: int = 5,
        candidates: int = 40,
        filter_dict: Optional[Dict[str, Any]] = None,
        query_filter=None,
        reranker=None,
    ) -> List[Any]:
        """Two-stage retrieval: over-retrieve with bi-encoder, rerank with cross-encoder.

        Args:
            query_text: Original query string (used by the cross-encoder).
            query_vector: Pre-computed query embedding for the first-stage search.
            limit: Final number of results to return after reranking.
            candidates: Number of candidates to retrieve in the first stage.
            filter_dict: Optional metadata filter dict for the first-stage search.
            query_filter: Optional pre-built Qdrant Filter object.
            reranker: A Reranker instance with a .rerank(query, passages, top_k) method.

        Returns:
            Reranked list of ScoredPoint-like objects (top ``limit``).
        """
        if reranker is None:
            logger.warning("search_and_rerank called without reranker, falling back to search()")
            return self.search(query_vector, limit=limit, filter_dict=filter_dict, query_filter=query_filter)

        # Stage 1: over-retrieve
        raw_results = self.search(
            query_vector=query_vector,
            limit=candidates,
            filter_dict=filter_dict,
            query_filter=query_filter,
        )

        if not raw_results:
            return []

        # Generate text passages for the cross-encoder
        from src.climate_embeddings.schema import generate_human_readable_text

        passages = []
        for hit in raw_results:
            payload = hit.payload if hasattr(hit, "payload") else (hit.get("payload", {}) if isinstance(hit, dict) else {})
            passages.append(generate_human_readable_text(payload))

        # Stage 2: cross-encoder rerank
        ranked = reranker.rerank(query_text, passages, top_k=limit)

        reranked_results = []
        for entry in ranked:
            idx = entry["index"]
            hit = raw_results[idx]
            # Overwrite the score with the cross-encoder score
            if hasattr(hit, "score"):
                hit.score = entry["score"]
            elif isinstance(hit, dict):
                hit["score"] = entry["score"]
            reranked_results.append(hit)

        return reranked_results

    def search_grouped(
        self,
        query_vector: List[float],
        group_by: str = "dataset_name",
        group_limit: int = 10,
        group_size: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None,
        query_filter=None,
    ) -> List[Any]:
        """Search with client-side grouping by a payload field.

        Over-retrieves then groups results to guarantee diversity:
        at most ``group_size`` results per unique value of ``group_by``,
        across up to ``group_limit`` groups.

        Returns a flat list of ScoredPoint-like objects.
        """
        # Use exact search via REST to guarantee we find all relevant groups
        # (HNSW approximate search misses small clusters in large collections)
        over_retrieve = max(group_limit * group_size * 30, 500)
        raw = self._search_exact_rest(query_vector, over_retrieve, filter_dict)

        if not raw:
            return []

        # Client-side grouping: pick top group_size hits per group_by value
        groups: Dict[str, List[Any]] = {}
        for hit in raw:
            payload = hit.payload if hasattr(hit, "payload") else (
                hit.get("payload", {}) if isinstance(hit, dict) else {}
            )
            key = payload.get(group_by, "unknown")
            if key not in groups:
                groups[key] = []
            if len(groups[key]) < group_size:
                groups[key].append(hit)

        # Sort groups by best score, take top group_limit
        sorted_groups = sorted(
            groups.items(),
            key=lambda g: max(
                (getattr(h, "score", 0) if hasattr(h, "score") else h.get("score", 0))
                for h in g[1]
            ),
            reverse=True,
        )[:group_limit]

        flat: List[Any] = []
        for _, hits in sorted_groups:
            flat.extend(hits)

        return flat

    def _search_exact_rest(
        self,
        query_vector: List[float],
        limit: int,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Exact (brute-force) search via REST API. Slower but guarantees
        finding all nearest neighbors, including small clusters that HNSW misses."""
        url = f"{self.base_url}/collections/{self.collection}/points/search"
        payload = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True,
            "params": {"exact": True},
        }
        if filter_dict:
            must_conditions = [
                {"key": k, "match": {"value": v}} for k, v in filter_dict.items()
            ]
            if must_conditions:
                payload["filter"] = {"must": must_conditions}
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json().get("result", [])
            class ScoredResult:
                def __init__(self, item):
                    self.id = item.get("id")
                    self.score = float(item.get("score", 0.0))
                    self.payload = item.get("payload", {}) if isinstance(item.get("payload"), dict) else {}
                    self.metadata = self.payload
            return [ScoredResult(item) for item in result]
        except Exception as e:
            logger.warning(f"Exact REST search failed ({e}), falling back to HNSW search")
            return self.search(query_vector, limit=limit, filter_dict=filter_dict)

    def _search_via_rest(
        self, 
        query_vector: List[float], 
        limit: int,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Manual search via HTTP requests. Returns ScoredPoint-like objects.
        Supports filtering via Qdrant REST API filter format.
        """
        url = f"{self.base_url}/collections/{self.collection}/points/search"
        payload = {
            "vector": query_vector,
            "limit": limit,
            "with_payload": True
        }
        
        # Add filter if provided (Qdrant REST API format)
        if filter_dict:
            must_conditions = []
            for key, value in filter_dict.items():
                must_conditions.append({
                    "key": key,
                    "match": {"value": value}
                })
            if must_conditions:
                payload["filter"] = {
                    "must": must_conditions
                }
        
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json().get("result", [])
            
            # Convert dict result to ScoredPoint-like structure for compatibility
            # The API expects objects with .score and .payload attributes
            class ScoredResult:
                """ScoredPoint-like object for compatibility with client API."""
                def __init__(self, item):
                    self.id = item.get("id")
                    self.score = float(item.get("score", 0.0))
                    # Ensure payload is a dict
                    payload_data = item.get("payload", {})
                    self.payload = payload_data if isinstance(payload_data, dict) else {}
                    # For compatibility with code expecting metadata attribute
                    self.metadata = self.payload
            
            return [ScoredResult(item) for item in result]
            
        except Exception as e:
            logger.error(f"REST search failed: {e}")
            return []