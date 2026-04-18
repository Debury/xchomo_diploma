#!/usr/bin/env bash
# Create a Qdrant snapshot of the `climate_data` collection.
# Usage:
#   ./scripts/qdrant/snapshot.sh                 # default collection + host
#   COLLECTION=climate_data ./scripts/qdrant/snapshot.sh
#   QDRANT_URL=http://localhost:6333 ./scripts/qdrant/snapshot.sh
#
# Snapshots live inside the `qdrant-data` Docker volume at
#   /qdrant/snapshots/<collection>/
# You can copy the .snapshot file to the host with:
#   docker cp climate-qdrant:/qdrant/snapshots/climate_data/<file> ./backups/

set -euo pipefail

COLLECTION="${COLLECTION:-climate_data}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

echo "Creating snapshot for collection '$COLLECTION' at $QDRANT_URL ..."
curl -fsS -X POST "$QDRANT_URL/collections/$COLLECTION/snapshots" \
  | python -m json.tool
