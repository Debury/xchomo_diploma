"""Quick helper to inspect embeddings stored in Qdrant."""

from __future__ import annotations

import os
from typing import Any, Dict

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct


def get_qdrant_client() -> QdrantClient:
    host = os.getenv("QDRANT_HOST", "localhost")
    rest_port = int(os.getenv("QDRANT_REST_PORT", "6333"))
    grpc_port = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
    prefer_grpc = os.getenv("QDRANT_PREFER_GRPC", "true").lower() == "true"

    if prefer_grpc:
        return QdrantClient(host=host, grpc_port=grpc_port, port=rest_port)
    return QdrantClient(host=host, port=rest_port)


def main() -> None:
    client = get_qdrant_client()
    collection_name = os.getenv("QDRANT_COLLECTION", "climate_data")

    print("\n" + "=" * 60)
    print("Qdrant Embeddings Status")
    print("=" * 60)

    if not client.collection_exists(collection_name=collection_name):
        print(f"⚠️  Collection '{collection_name}' does not exist")
        print("=" * 60 + "\n")
        return

    count = client.count(collection_name=collection_name, exact=True).count
    print(f"Total embeddings: {count}")

    if count == 0:
        print("\n⚠️  No embeddings found in collection")
        print("=" * 60 + "\n")
        return

    points, _ = client.scroll(
        collection_name=collection_name,
        limit=5,
        with_payload=True,
        with_vectors=False,
    )

    def fmt_point(point: PointStruct) -> Dict[str, Any]:
        payload = point.payload or {}
        return {
            "id": point.id,
            "source": payload.get("source_id") or payload.get("source"),
            "variable": payload.get("variable"),
        }

    print("\nSample Points:")
    for point in points:
        info = fmt_point(point)
        print(f"  - ID: {info['id']} | Source: {info['source']} | Variable: {info['variable']}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
