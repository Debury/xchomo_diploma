"""Quick script to check embeddings in ChromaDB"""
from src.embeddings.database import VectorDatabase

db = VectorDatabase()
count = db.collection.count()

print(f"\n{'='*60}")
print(f"ChromaDB Embeddings Status")
print(f"{'='*60}")
print(f"Total embeddings: {count}")

if count > 0:
    sample = db.collection.get(limit=5)
    print(f"\nSample IDs:")
    for id_val in sample['ids'][:5]:
        print(f"  - {id_val}")
    
    if sample['metadatas']:
        print(f"\nSample Metadata:")
        for meta in sample['metadatas'][:3]:
            print(f"  - Source: {meta.get('source_id')}, Variable: {meta.get('variable')}")
    
    print(f"\nEmbedding dimension: {len(sample['embeddings'][0]) if sample['embeddings'] else 'N/A'}")
else:
    print("\n⚠️  No embeddings found in database")

print(f"{'='*60}\n")
