"""
Migration script: shelve + app_settings.json → PostgreSQL.

Usage:
    python -m src.database.migrate_shelve

Reads all entries from data/sources_db (shelve) and credentials from
data/app_settings.json, then inserts them into the climate_app PostgreSQL
database.  Backs up the old files after successful migration.
"""

import json
import logging
import os
import shelve
import shutil
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHELVE_PATH = os.getenv("SOURCE_DB_PATH", str(PROJECT_ROOT / "data" / "sources_db"))
SETTINGS_PATH = PROJECT_ROOT / "data" / "app_settings.json"
BACKUP_DIR = PROJECT_ROOT / "data" / "backup_pre_postgres"


def migrate():
    """Run the full migration."""
    from src.database.connection import init_db, get_db_session
    from src.database.models import Source, SourceCredential

    # 1. Initialize tables
    logger.info("Creating database tables...")
    init_db()

    # 2. Read shelve entries
    sources = []
    try:
        with shelve.open(SHELVE_PATH, flag="r") as db:
            for key in db:
                obj = db[key]
                sources.append(obj)
        logger.info(f"Read {len(sources)} sources from shelve at {SHELVE_PATH}")
    except Exception as e:
        logger.warning(f"Could not read shelve ({e}). Proceeding with empty source list.")

    # 3. Read app_settings.json credentials
    settings = {}
    if SETTINGS_PATH.exists():
        try:
            settings = json.loads(SETTINGS_PATH.read_text())
            logger.info(f"Read settings from {SETTINGS_PATH}")
        except Exception as e:
            logger.warning(f"Could not read settings ({e}).")

    # 4. Insert into PostgreSQL
    with get_db_session() as session:
        inserted = 0
        skipped = 0
        for obj in sources:
            data = obj.to_dict() if hasattr(obj, "to_dict") else obj.__dict__

            # Check if already exists
            existing = session.query(Source).filter(Source.source_id == data["source_id"]).first()
            if existing:
                logger.info(f"  Skipping {data['source_id']} (already exists)")
                skipped += 1
                continue

            # Map fields
            model_fields = {c.key for c in Source.__table__.columns}
            clean = {}
            for k, v in data.items():
                if k in model_fields and k != "id":
                    # Convert string timestamps to datetime
                    if k in ("created_at", "updated_at") and isinstance(v, str):
                        try:
                            v = datetime.fromisoformat(v)
                        except (ValueError, TypeError):
                            v = datetime.utcnow()
                    clean[k] = v

            source = Source(**clean)
            session.add(source)
            inserted += 1
            logger.info(f"  Inserted source: {data['source_id']}")

        # Insert credentials
        source_creds = settings.get("source_credentials", {})
        cred_count = 0
        for source_id, cred_info in source_creds.items():
            auth_method = cred_info.get("auth_method")
            credentials = cred_info.get("credentials", {})
            portal = cred_info.get("portal")

            for ckey, cval in credentials.items():
                existing = (
                    session.query(SourceCredential)
                    .filter(
                        SourceCredential.source_id == source_id,
                        SourceCredential.credential_key == ckey,
                    )
                    .first()
                )
                if not existing:
                    session.add(SourceCredential(
                        source_id=source_id,
                        credential_key=ckey,
                        credential_value=str(cval),
                    ))
                    cred_count += 1

        # Insert global credentials
        global_creds = settings.get("credentials", {})
        for ckey, cval in global_creds.items():
            if cval:
                existing = (
                    session.query(SourceCredential)
                    .filter(
                        SourceCredential.source_id == None,
                        SourceCredential.credential_key == ckey,
                    )
                    .first()
                )
                if not existing:
                    session.add(SourceCredential(
                        source_id=None,
                        credential_key=ckey,
                        credential_value=str(cval),
                    ))
                    cred_count += 1

        logger.info(f"Migration complete: {inserted} sources inserted, {skipped} skipped, {cred_count} credentials")

    # 5. Backup old files
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    for ext in (".db", ".dir", ".bak", ".dat"):
        src = Path(SHELVE_PATH + ext)
        if src.exists():
            shutil.copy2(src, BACKUP_DIR / src.name)
            logger.info(f"  Backed up {src.name}")

    if SETTINGS_PATH.exists():
        shutil.copy2(SETTINGS_PATH, BACKUP_DIR / SETTINGS_PATH.name)
        logger.info(f"  Backed up {SETTINGS_PATH.name}")

    logger.info(f"Backups saved to {BACKUP_DIR}")
    logger.info("Migration finished successfully!")


if __name__ == "__main__":
    migrate()
