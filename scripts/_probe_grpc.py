import os, sys, socket
sys.path.insert(0, "/app")

print("=== /etc/hosts ===")
with open("/etc/hosts") as f:
    print(f.read())

print("=== DNS lookup for 'qdrant' ===")
try:
    print(socket.getaddrinfo("qdrant", 6334))
except Exception as e:
    print("getaddrinfo failed:", e)

print("=== simulate VectorDatabase gRPC connect ===")
from qdrant_client import QdrantClient
import warnings
warnings.filterwarnings("ignore")
c = QdrantClient(host="qdrant", port=6333, grpc_port=6334, prefer_grpc=True, timeout=10)
print("client created")
try:
    info = c.get_collection("climate_data")
    print("collection count:", info.points_count)
except Exception as e:
    print("get_collection failed:", repr(e)[:200])

print("=== raw gRPC call ===")
try:
    count = c.count(collection_name="climate_data", exact=False).count
    print("exact count:", count)
except Exception as e:
    print("count failed:", repr(e)[:200])
