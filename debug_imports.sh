#!/bin/bash
# Debug script to check Python imports in container

echo "🔍 Debugging Python imports in container"
echo "========================================"
echo ""

echo "1. Python sys.path:"
docker compose exec -T web-api python -c "import sys; print('\n'.join(sys.path))"

echo ""
echo "2. Directory structure /app/src:"
docker compose exec -T web-api ls -la /app/src/

echo ""
echo "3. climate_embeddings exists?"
docker compose exec -T web-api ls -la /app/src/climate_embeddings/ 2>/dev/null || echo "❌ NOT FOUND"

echo ""
echo "4. Installed packages:"
docker compose exec -T web-api pip list | grep -i climate

echo ""
echo "5. Try importing src.climate_embeddings:"
docker compose exec -T web-api python -c "import src.climate_embeddings; print('✓ OK')" 2>&1

echo ""
echo "6. Try importing climate_embeddings directly:"
docker compose exec -T web-api python -c "import climate_embeddings; print('✓ OK')" 2>&1

echo ""
echo "7. Check if package installed in editable mode:"
docker compose exec -T web-api pip show climate-etl-pipeline

echo ""
echo "8. Check __init__.py files:"
docker compose exec -T web-api find /app/src/climate_embeddings -name "__init__.py"
