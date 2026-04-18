#!/usr/bin/env bash
# Restore a Qdrant collection from a snapshot file.
# Usage:
#   ./scripts/qdrant/restore.sh <snapshot-file-path-in-container>
#
# To restore from a backup file that lives on the host, first copy it into
# the Qdrant container:
#   docker cp ./backups/climate_data-XXXX.snapshot \
#     climate-qdrant:/qdrant/snapshots/climate_data/
# Then call this script with the in-container path:
#   ./scripts/qdrant/restore.sh /qdrant/snapshots/climate_data/climate_data-XXXX.snapshot
#
# WARNING: restore overwrites the current collection.

set -euo pipefail

COLLECTION="${COLLECTION:-climate_data}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

if [ $# -lt 1 ]; then
  echo "usage: $0 <snapshot-path-in-container>" >&2
  exit 1
fi

SNAPSHOT="$1"

read -r -p "About to RESTORE $SNAPSHOT into '$COLLECTION'. This will OVERWRITE the current collection. Continue? [y/N] " confirm
case "$confirm" in
  y|Y|yes|YES) ;;
  *) echo "Aborted." ; exit 1 ;;
esac

echo "Restoring …"
curl -fsS -X PUT "$QDRANT_URL/collections/$COLLECTION/snapshots/recover" \
  -H 'Content-Type: application/json' \
  -d "{\"location\": \"file://$SNAPSHOT\"}" \
  | python -m json.tool
