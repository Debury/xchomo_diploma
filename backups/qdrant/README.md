# Qdrant snapshots

`climate_data-886361186156628-2026-04-18-11-29-12.snapshot` (~11 GB) is the
full snapshot of the `climate_data` collection taken on 2026-04-18, just
before the defense. It contains ~1.53M vector chunks across 72 datasets.

## Restore from this host-side backup

1. Make sure Qdrant is running: `docker compose up -d qdrant`
2. Copy the snapshot back into the container:
   ```bash
   docker cp backups/qdrant/climate_data-886361186156628-2026-04-18-11-29-12.snapshot \
     climate-qdrant:/qdrant/snapshots/climate_data/
   ```
3. Trigger the restore:
   ```bash
   ./scripts/qdrant/restore.sh /qdrant/snapshots/climate_data/climate_data-886361186156628-2026-04-18-11-29-12.snapshot
   ```

The restore overwrites the collection — use with care.
