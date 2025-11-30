#!/bin/bash
# Test script for verifying all format support on external server
# Usage: bash test_formats_server.sh

# Don't exit on error - we want to count all failures
set +e

echo "🧪 Climate Embeddings Format Testing Suite"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

# Function to run test
run_test() {
    local test_name="$1"
    local command="$2"
    
    echo -e "${BLUE}Testing: ${test_name}${NC}"
    if eval "$command"; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((FAILED++))
    fi
    echo ""
}

# Check Docker services
echo "📋 Step 1: Checking Docker services..."
echo "--------------------------------------"
run_test "Docker Compose" "docker compose ps"
run_test "Qdrant health" "curl -s http://localhost:6333/health"
run_test "Ollama health" "curl -s http://localhost:11434/api/tags"
run_test "API health" "curl -s http://localhost:8000/health"

# Test imports
echo ""
echo "📦 Step 2: Testing Python imports..."
echo "--------------------------------------"
run_test "Import climate_embeddings" "docker compose exec -T web-api python -c 'import src.climate_embeddings; print(\"OK\")'"
run_test "Import loaders" "docker compose exec -T web-api python -c 'from src.climate_embeddings.loaders import load_raster_auto; print(\"OK\")'"
run_test "Import embeddings" "docker compose exec -T web-api python -c 'from src.climate_embeddings.embeddings import get_text_embedder; print(\"OK\")'"
run_test "Import RAG" "docker compose exec -T web-api python -c 'from src.climate_embeddings.rag import RAGPipeline; print(\"OK\")'"

# Create test data directory
echo ""
echo "📁 Step 3: Preparing test data..."
echo "--------------------------------------"
mkdir -p data/test_formats
cd data/test_formats

# Test NetCDF format
echo ""
echo "🗂️  Step 4: Testing NetCDF format..."
echo "--------------------------------------"
cat > test_netcdf.py << 'EOF'
import numpy as np
import xarray as xr
from pathlib import Path

# Create NetCDF
ds = xr.Dataset({
    "temperature": (["time", "lat", "lon"], np.random.randn(10, 5, 5) * 10 + 15),
    "precipitation": (["time", "lat", "lon"], np.random.rand(10, 5, 5) * 50)
}, coords={
    "time": range(10),
    "lat": np.linspace(45, 50, 5),
    "lon": np.linspace(10, 15, 5)
})
ds.to_netcdf("test.nc")
print("✓ Created test.nc")
EOF

docker compose exec -T web-api python data/test_formats/test_netcdf.py
run_test "Load NetCDF" "docker compose exec -T web-api python -c '
from src.climate_embeddings.loaders import load_raster_auto
result = load_raster_auto(\"data/test_formats/test.nc\")
print(f\"Loaded {len(result.embeddings)} embeddings\")
assert len(result.embeddings) > 0
'"

# Test GeoTIFF format
echo ""
echo "🗂️  Step 5: Testing GeoTIFF format..."
echo "--------------------------------------"
cat > test_geotiff.py << 'EOF'
import numpy as np
import rasterio
from rasterio.transform import from_origin

# Create GeoTIFF
data = np.random.randn(100, 100) * 10 + 15
transform = from_origin(10.0, 50.0, 0.1, 0.1)

with rasterio.open(
    "test.tif", "w",
    driver="GTiff",
    height=100, width=100,
    count=1, dtype=data.dtype,
    crs="EPSG:4326",
    transform=transform
) as dst:
    dst.write(data, 1)
print("✓ Created test.tif")
EOF

docker compose exec -T web-api python data/test_formats/test_geotiff.py
run_test "Load GeoTIFF" "docker compose exec -T web-api python -c '
from src.climate_embeddings.loaders import load_raster_auto
result = load_raster_auto(\"data/test_formats/test.tif\")
print(f\"Loaded {len(result.embeddings)} embeddings\")
assert len(result.embeddings) > 0
'"

# Test CSV format
echo ""
echo "🗂️  Step 6: Testing CSV format..."
echo "--------------------------------------"
cat > test.csv << 'EOF'
latitude,longitude,temperature,precipitation,timestamp
48.5,17.2,15.3,2.5,2024-01-01
48.6,17.3,16.1,1.8,2024-01-02
48.7,17.4,14.9,3.2,2024-01-03
48.8,17.5,15.7,2.1,2024-01-04
48.9,17.6,16.5,1.5,2024-01-05
EOF

run_test "Load CSV" "docker compose exec -T web-api python -c '
from src.climate_embeddings.loaders import load_raster_auto
result = load_raster_auto(\"data/test_formats/test.csv\")
print(f\"Loaded {len(result.embeddings)} embeddings\")
assert len(result.embeddings) > 0
'"

# Test ZIP format
echo ""
echo "🗂️  Step 7: Testing ZIP archive..."
echo "--------------------------------------"
run_test "Create ZIP with NetCDF" "cd data/test_formats && zip -q test_archive.zip test.nc test.tif"
run_test "Load ZIP" "docker compose exec -T web-api python -c '
from src.climate_embeddings.loaders.raster_pipeline import load_from_zip
results = load_from_zip(\"data/test_formats/test_archive.zip\")
print(f\"Loaded {len(results)} files from ZIP\")
assert len(results) > 0
'"

# Test Text Embeddings
echo ""
echo "📝 Step 8: Testing text embeddings..."
echo "--------------------------------------"
run_test "BGE embeddings" "docker compose exec -T web-api python -c '
from src.climate_embeddings.embeddings import get_text_embedder
embedder = get_text_embedder(\"bge-large\")
emb = embedder.encode(\"temperature data from climate model\")
print(f\"Embedding shape: {emb.shape}\")
assert emb.shape[0] == 1024  # BGE dimension
'"

run_test "GTE embeddings" "docker compose exec -T web-api python -c '
from src.climate_embeddings.embeddings import get_text_embedder
embedder = get_text_embedder(\"gte-large\")
emb = embedder.encode(\"precipitation data\")
print(f\"Embedding shape: {emb.shape}\")
assert emb.shape[0] == 1024  # GTE dimension
'"

# Test Vector Index
echo ""
echo "🔍 Step 9: Testing vector index..."
echo "--------------------------------------"
run_test "Vector index operations" "docker compose exec -T web-api python -c '
import numpy as np
from src.climate_embeddings.index import VectorIndex

# Create index
index = VectorIndex(dimension=128, metric=\"cosine\")

# Add vectors
vectors = np.random.randn(10, 128).astype(np.float32)
metadata = [{\"id\": i, \"type\": \"test\"} for i in range(10)]
index.add_batch(vectors, metadata)

# Search
query = np.random.randn(128).astype(np.float32)
results = index.search(query, k=3)

print(f\"Added {len(vectors)} vectors, found {len(results)} results\")
assert len(results) == 3
'"

# Test RAG Pipeline
echo ""
echo "🤖 Step 10: Testing RAG pipeline..."
echo "--------------------------------------"
run_test "RAG pipeline setup" "docker compose exec -T web-api python -c '
import numpy as np
from src.climate_embeddings.rag import RAGPipeline
from src.climate_embeddings.index import VectorIndex

# Create index with data
index = VectorIndex(dimension=1024, metric=\"cosine\")
vectors = np.random.randn(5, 1024).astype(np.float32)
metadata = [
    {\"text\": \"Temperature increased by 2°C in 2023\"},
    {\"text\": \"Precipitation decreased in summer months\"},
    {\"text\": \"Sea level rose by 3mm\"},
    {\"text\": \"Arctic ice melting accelerated\"},
    {\"text\": \"CO2 levels reached 420 ppm\"}
]
index.add_batch(vectors, metadata)

# Create RAG pipeline
rag = RAGPipeline(
    index=index,
    embedder_name=\"bge-large\",
    llm_model=\"llama3.2:1b\",
    llm_base_url=\"http://ollama:11434\"
)

print(\"✓ RAG pipeline initialized\")
'"

# Test Qdrant integration
echo ""
echo "💾 Step 11: Testing Qdrant integration..."
echo "--------------------------------------"
run_test "Qdrant VectorDatabase" "docker compose exec -T web-api python -c '
from src.embeddings.database import VectorDatabase

db = VectorDatabase(collection_name=\"test_collection\")
print(f\"✓ Connected to Qdrant, collection: {db.collection_name}\")
'"

run_test "Store embeddings in Qdrant" "docker compose exec -T web-api python -c '
import numpy as np
from src.embeddings.database import VectorDatabase
from src.embeddings.generator import EmbeddingGenerator

# Generate embeddings
generator = EmbeddingGenerator()
texts = [\"temperature data\", \"precipitation data\", \"wind speed data\"]
embeddings = generator.generate_embeddings(texts)

# Store in Qdrant
db = VectorDatabase(collection_name=\"test_embeddings\")
metadata = [{\"text\": t, \"type\": \"climate\"} for t in texts]
db.add_embeddings(embeddings, metadata)

print(f\"✓ Stored {len(embeddings)} embeddings in Qdrant\")
'"

# Test API endpoints
echo ""
echo "🌐 Step 12: Testing API endpoints..."
echo "--------------------------------------"
run_test "List sources" "curl -s http://localhost:8000/sources | python -c 'import sys, json; data=json.load(sys.stdin); print(f\"Found {len(data)} sources\")'"
run_test "List jobs" "curl -s http://localhost:8000/jobs | python -c 'import sys, json; data=json.load(sys.stdin); print(f\"Found {len(data)} jobs\")'"
run_test "Health check" "curl -s http://localhost:8000/health | grep -q '\"status\":\"healthy\"'"

# Test Dagster integration
echo ""
echo "⚙️  Step 13: Testing Dagster integration..."
echo "--------------------------------------"
run_test "Dagster workspace" "docker compose exec -T dagit python -c '
from dagster import DagsterInstance
from dagster_project.repository import climate_repository

repo = climate_repository()
print(f\"✓ Repository loaded: {repo.name}\")
print(f\"  Jobs: {[j.name for j in repo.get_all_jobs()]}\")
'"

# Summary
echo ""
echo "=========================================="
echo "📊 Test Summary"
echo "=========================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! 🎉${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
