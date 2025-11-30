#!/bin/bash
set +e  # Continue on errors to see all results

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

function run_test() {
    local test_name="$1"
    local command="$2"
    echo -n "Testing: $test_name "
    
    if output=$(eval "$command" 2>&1); then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED++))
        [[ -n "$output" ]] && echo "$output"
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((FAILED++))
        echo "$output"
    fi
    echo ""
}

echo "=========================================="
echo "🌍 Climate Data API - Format Tests"
echo "=========================================="
echo ""

# API base URL
API_URL="http://localhost:8000"

# Step 1: Health check
echo "📋 Step 1: API Health Check..."
echo "--------------------------------------"
run_test "API health" "curl -sf $API_URL/health | jq -r '.status'"

# Step 2: Create test data files IN CONTAINER
echo ""
echo "📁 Step 2: Creating test data files..."
echo "--------------------------------------"

# Create NetCDF test file
run_test "Create NetCDF sample" "docker compose exec -T web-api python -c '
import numpy as np
import xarray as xr
from pathlib import Path

Path(\"data/raw\").mkdir(parents=True, exist_ok=True)

ds = xr.Dataset({
    \"temperature\": ([\"time\", \"lat\", \"lon\"], np.random.randn(10, 5, 5) * 10 + 15),
    \"precipitation\": ([\"time\", \"lat\", \"lon\"], np.random.rand(10, 5, 5) * 50)
}, coords={
    \"time\": range(10),
    \"lat\": np.linspace(45, 50, 5),
    \"lon\": np.linspace(10, 15, 5)
})
ds.to_netcdf(\"data/raw/test_climate.nc\")
print(\"✓ Created test_climate.nc\")
'"

# Create GeoTIFF test file
run_test "Create GeoTIFF sample" "docker compose exec -T web-api python -c '
import numpy as np
import rasterio
from rasterio.transform import from_origin
from pathlib import Path

Path(\"data/raw\").mkdir(parents=True, exist_ok=True)

data = np.random.randn(100, 100) * 10 + 15
transform = from_origin(10.0, 50.0, 0.1, 0.1)

with rasterio.open(
    \"data/raw/test_raster.tif\", \"w\",
    driver=\"GTiff\",
    height=100, width=100,
    count=1, dtype=data.dtype,
    crs=\"EPSG:4326\",
    transform=transform
) as dst:
    dst.write(data, 1)
print(\"✓ Created test_raster.tif\")
'"

# Create CSV test file
run_test "Create CSV sample" "docker compose exec -T web-api bash -c '
cat > data/raw/test_stations.csv << EOF
latitude,longitude,temperature,precipitation,timestamp
48.5,15.2,18.5,2.3,2024-01-01
49.0,16.0,17.8,3.1,2024-01-01
47.5,14.5,19.2,1.8,2024-01-01
EOF
echo \"✓ Created test_stations.csv\"
'"

# Step 3: Register sources via API
echo ""
echo "🔗 Step 3: Registering sources via API..."
echo "--------------------------------------"

# Clear existing test sources first
curl -sf -X DELETE "$API_URL/sources/test_netcdf_source" 2>/dev/null || true
curl -sf -X DELETE "$API_URL/sources/test_geotiff_source" 2>/dev/null || true
curl -sf -X DELETE "$API_URL/sources/test_csv_source" 2>/dev/null || true

# Register NetCDF source
run_test "Register NetCDF source" "curl -sf -X POST '$API_URL/sources' \
  -H 'Content-Type: application/json' \
  -d '{
    \"source_id\": \"test_netcdf_source\",
    \"url\": \"file:///app/data/raw/test_climate.nc\",
    \"format\": \"netcdf\",
    \"description\": \"Test NetCDF climate data\",
    \"tags\": [\"test\", \"netcdf\"],
    \"active\": true
  }' | jq -r '.source_id'"

# Register GeoTIFF source
run_test "Register GeoTIFF source" "curl -sf -X POST '$API_URL/sources' \
  -H 'Content-Type: application/json' \
  -d '{
    \"source_id\": \"test_geotiff_source\",
    \"url\": \"file:///app/data/raw/test_raster.tif\",
    \"format\": \"geotiff\",
    \"description\": \"Test GeoTIFF raster data\",
    \"tags\": [\"test\", \"geotiff\"],
    \"active\": true
  }' | jq -r '.source_id'"

# Register CSV source
run_test "Register CSV source" "curl -sf -X POST '$API_URL/sources' \
  -H 'Content-Type: application/json' \
  -d '{
    \"source_id\": \"test_csv_source\",
    \"url\": \"file:///app/data/raw/test_stations.csv\",
    \"format\": \"csv\",
    \"description\": \"Test CSV station data\",
    \"tags\": [\"test\", \"csv\"],
    \"active\": true
  }' | jq -r '.source_id'"

# Step 4: Verify sources registered
echo ""
echo "📜 Step 4: Verifying sources in database..."
echo "--------------------------------------"
run_test "List all sources" "curl -sf '$API_URL/sources?active_only=false' | jq -r '.[].source_id' | grep -E 'test_(netcdf|geotiff|csv)_source'"

# Step 5: Test loading formats directly
echo ""
echo "🗂️  Step 5: Testing direct format loading..."
echo "--------------------------------------"

run_test "Load NetCDF directly" "docker compose exec -T web-api python -c '
from src.climate_embeddings.loaders import load_raster_auto

result = load_raster_auto(\"data/raw/test_climate.nc\")
print(f\"✓ Loaded NetCDF: {result.dataset.dims if hasattr(result, \\\"dataset\\\") else \\\"OK\\\"}\")
'"

run_test "Load GeoTIFF directly" "docker compose exec -T web-api python -c '
from src.climate_embeddings.loaders import load_raster_auto

result = load_raster_auto(\"data/raw/test_raster.tif\")
print(f\"✓ Loaded GeoTIFF: {result.dataset.dims if hasattr(result, \\\"dataset\\\") else \\\"OK\\\"}\")
'"

run_test "Load CSV directly" "docker compose exec -T web-api python -c '
from src.climate_embeddings.loaders import load_raster_auto

result = load_raster_auto(\"data/raw/test_stations.csv\")
print(f\"✓ Loaded CSV: OK\")
'"

# Step 6: Test embeddings generation
echo ""
echo "🧮 Step 6: Testing embeddings from formats..."
echo "--------------------------------------"

run_test "Generate embeddings from NetCDF" "docker compose exec -T web-api python -c '
from src.climate_embeddings.loaders import raster_to_embeddings
from src.climate_embeddings.embeddings import get_text_embedder

embedder = get_text_embedder(\"bge-large\", device=\"cpu\")
result = raster_to_embeddings(
    \"data/raw/test_climate.nc\",
    embedder=embedder,
    chunk_size=5
)
print(f\"✓ Generated embeddings: {len(list(result.chunk_iterator))} chunks\")
'"

# Step 7: Summary
echo ""
echo "=========================================="
echo "📊 Test Summary"
echo "=========================================="
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
