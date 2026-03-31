#!/bin/bash
# =============================================================================
# Setup script: Restore full project state from exported data
# =============================================================================
#
# Prerequisites:
#   1. Git repo cloned: git clone <repo_url> && cd xchomo_diploma
#   2. Docker + Docker Compose installed
#   3. Two exported files in data/ directory:
#      - data/climate_data.snapshot  (~8.6 GB, Qdrant collection snapshot)
#      - data/climate_app_dump.sql   (~60 KB, PostgreSQL database dump)
#   4. .env file in project root (copy from this machine or .env.example)
#
# Usage:
#   chmod +x scripts/setup_from_export.sh
#   ./scripts/setup_from_export.sh
#
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[+]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[x]${NC} $1"; exit 1; }

# ---------------------------------------------------------------------------
# 1. Preflight checks
# ---------------------------------------------------------------------------
log "Checking prerequisites..."

[[ -f "docker-compose.yml" ]] || error "Run this script from the project root (where docker-compose.yml is)"
[[ -f ".env" ]]               || error ".env file missing — copy it from the other machine"
[[ -f "data/climate_data.snapshot" ]] || error "data/climate_data.snapshot not found — copy the Qdrant export here"
[[ -f "data/climate_app_dump.sql" ]]  || error "data/climate_app_dump.sql not found — copy the PostgreSQL dump here"

SNAPSHOT_SIZE=$(stat -c%s "data/climate_data.snapshot" 2>/dev/null || stat -f%z "data/climate_data.snapshot" 2>/dev/null)
if [[ "$SNAPSHOT_SIZE" -lt 1000000000 ]]; then
    warn "Snapshot is only $(( SNAPSHOT_SIZE / 1024 / 1024 ))MB — expected ~8.6GB. Is it complete?"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] || exit 1
fi

# ---------------------------------------------------------------------------
# 2. Start core services (Qdrant + PostgreSQL)
# ---------------------------------------------------------------------------
log "Starting Qdrant and PostgreSQL..."
docker compose up -d qdrant dagster-postgres

log "Waiting for Qdrant to be ready..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
        break
    fi
    sleep 2
done
curl -sf http://localhost:6333/healthz > /dev/null || error "Qdrant not healthy after 60s"
log "Qdrant is ready."

log "Waiting for PostgreSQL to be ready..."
for i in $(seq 1 30); do
    if docker compose exec -T dagster-postgres pg_isready -U dagster > /dev/null 2>&1; then
        break
    fi
    sleep 2
done
docker compose exec -T dagster-postgres pg_isready -U dagster > /dev/null || error "PostgreSQL not ready after 60s"
log "PostgreSQL is ready."

# ---------------------------------------------------------------------------
# 3. Restore Qdrant collection from snapshot
# ---------------------------------------------------------------------------
log "Copying snapshot into Qdrant container..."
docker cp "data/climate_data.snapshot" climate-qdrant:/qdrant/climate_data.snapshot

log "Restoring Qdrant collection 'climate_data' from snapshot (this takes 1-3 minutes)..."
RESTORE_RESULT=$(curl -sf -X PUT "http://localhost:6333/collections/climate_data/snapshots/recover" \
    -H "Content-Type: application/json" \
    -d '{"location": "file:///qdrant/climate_data.snapshot"}' \
    --max-time 600 2>&1) || error "Qdrant restore failed: $RESTORE_RESULT"

log "Verifying Qdrant collection..."
POINTS=$(curl -sf http://localhost:6333/collections/climate_data | python3 -c "import sys,json; print(json.load(sys.stdin)['result']['points_count'])" 2>/dev/null || echo "0")
if [[ "$POINTS" -gt 1000000 ]]; then
    log "Qdrant OK: ${POINTS} points restored."
else
    error "Qdrant verification failed: only ${POINTS} points (expected ~1.5M)"
fi

log "Cleaning up snapshot from container..."
docker exec climate-qdrant rm -f /qdrant/climate_data.snapshot

# ---------------------------------------------------------------------------
# 4. Restore PostgreSQL climate_app database
# ---------------------------------------------------------------------------
log "Creating climate_app database (if not exists)..."
docker compose exec -T dagster-postgres psql -U dagster -c "SELECT 1 FROM pg_database WHERE datname='climate_app'" | grep -q 1 || \
    docker compose exec -T dagster-postgres psql -U dagster -c "CREATE DATABASE climate_app;"

log "Restoring PostgreSQL data..."
docker compose exec -T dagster-postgres psql -U dagster -d climate_app < data/climate_app_dump.sql > /dev/null 2>&1

SOURCES=$(docker compose exec -T dagster-postgres psql -U dagster -d climate_app -tAc "SELECT COUNT(*) FROM sources;" 2>/dev/null || echo "0")
log "PostgreSQL OK: ${SOURCES} sources restored."

# ---------------------------------------------------------------------------
# 5. Start all services
# ---------------------------------------------------------------------------
log "Building and starting all services..."
docker compose build
docker compose up -d

log "Waiting for web-api to be healthy..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        break
    fi
    sleep 3
done

if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    log "Web API is healthy!"
else
    warn "Web API not responding yet — check logs with: docker compose logs -f web-api"
fi

# ---------------------------------------------------------------------------
# 6. Verify RAG pipeline works
# ---------------------------------------------------------------------------
log "Testing RAG query..."
RAG_RESULT=$(curl -sf -X POST http://localhost:8000/rag/query \
    -H "Content-Type: application/json" \
    -d '{"question": "What temperature data is available?", "top_k": 3}' \
    --max-time 30 2>/dev/null)

if echo "$RAG_RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('answer','')[:80])" 2>/dev/null | grep -qi "temperature\|data\|ERA5"; then
    log "RAG pipeline working!"
else
    warn "RAG query returned unexpected result — check manually"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN} Setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "  Web UI:     http://localhost:8000/app"
echo "  API:        http://localhost:8000"
echo "  Dagster:    http://localhost:3000"
echo "  Qdrant:     http://localhost:6333/dashboard"
echo ""
echo "  Qdrant:     ${POINTS} points"
echo "  PostgreSQL: ${SOURCES} sources"
echo ""
echo "  Run eval:   docker compose exec web-api python3 tests/run_rag_evaluation.py"
echo "  Run eval V2: docker compose exec web-api python3 tests/run_rag_evaluation_v2.py"
echo ""
