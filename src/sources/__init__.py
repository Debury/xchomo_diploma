"""
Source store — backward-compatible shim.

Imports from the PostgreSQL-backed store when the database is available,
falls back to the legacy shelve-based store otherwise.

Also provides source lifecycle utilities (connection testing, URL analysis)
via the connection_tester submodule.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# DTO used everywhere in the codebase (ops, pipelines, API responses)
@dataclass
class ClimateDataSource:
    """Internal representation of a data source."""
    source_id: str
    url: str
    is_active: bool = True
    format: Optional[str] = None
    variables: Optional[List[str]] = None
    spatial_bbox: Optional[List[float]] = None
    time_range: Optional[Dict[str, str]] = None
    transformations: Optional[List[str]] = None
    aggregation_method: Optional[str] = "mean"
    output_resolution: Optional[float] = None
    embedding_model: Optional[str] = "BAAI/bge-large-en-v1.5"
    chunk_size: Optional[int] = 512
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    custom_metadata: Optional[Dict[str, str]] = None

    # Portal / auth (needed by dagster_project.source_jobs to route to the
    # right portal adapter — previously reachable only via app_settings.json).
    portal: Optional[str] = None
    auth_method: Optional[str] = None

    # User-supplied taxonomy (hazard/region/sector/spatial). The backend
    # previously accepted these via the API but dropped them before persistence.
    hazard_type: Optional[str] = None
    region_country: Optional[str] = None
    spatial_coverage: Optional[str] = None
    impact_sector: Optional[str] = None

    # Tracking fields
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    processing_status: str = "pending"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


def _is_postgres_available() -> bool:
    """Check if the PostgreSQL database is available."""
    url = os.getenv("APP_DATABASE_URL", "")
    if not url:
        return False
    try:
        from src.database.connection import get_engine
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(conn.dialect.do_ping(conn) if hasattr(conn.dialect, 'do_ping') else None)
        return True
    except Exception:
        return False


# Determine backend at import time
_USE_POSTGRES = bool(os.getenv("APP_DATABASE_URL", ""))

if _USE_POSTGRES:
    try:
        from src.database.source_store import SourceStore
        from src.database.connection import init_db
        # Initialize tables on first import
        try:
            init_db()
        except Exception as e:
            logger.warning(f"Failed to initialize database tables: {e}")
        logger.info("Using PostgreSQL-backed SourceStore")
    except ImportError as e:
        logger.warning(f"PostgreSQL store unavailable ({e}), falling back to shelve")
        _USE_POSTGRES = False


if not _USE_POSTGRES:
    # Legacy shelve-based store
    import shelve

    # __file__ is src/sources/__init__.py, so parent.parent.parent -> project root
    _DEFAULT_DB = str(Path(__file__).resolve().parent.parent.parent / "data" / "sources_db")
    DB_PATH = os.getenv("SOURCE_DB_PATH", _DEFAULT_DB)

    class SourceStore:
        """Persistent key-value store for sources using Python's shelve module."""

        def __init__(self, db_path: str = None):
            self.db_path = db_path or DB_PATH
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        def _get_db(self):
            return shelve.open(self.db_path, writeback=True)

        def create_source(self, data: Dict[str, Any]) -> ClimateDataSource:
            valid_keys = ClimateDataSource.__dataclass_fields__.keys()
            clean_data = {k: v for k, v in data.items() if k in valid_keys}
            obj = ClimateDataSource(**clean_data)
            with self._get_db() as db:
                db[obj.source_id] = obj
            return obj

        def get_source(self, source_id: str) -> Optional[ClimateDataSource]:
            with self._get_db() as db:
                return db.get(source_id)

        def get_all_sources(self, active_only: bool = True) -> List[ClimateDataSource]:
            with self._get_db() as db:
                if active_only:
                    return [s for s in db.values() if s.is_active]
                return list(db.values())

        def update_source(self, source_id: str, updates: Dict[str, Any]) -> Optional[ClimateDataSource]:
            with self._get_db() as db:
                if source_id not in db:
                    return None
                source = db[source_id]
                for key, value in updates.items():
                    if hasattr(source, key):
                        setattr(source, key, value)
                db[source_id] = source
                return source

        def update_processing_status(self, source_id: str, status: str, error_message: str = None):
            with self._get_db() as db:
                if source_id in db:
                    source = db[source_id]
                    source.processing_status = status
                    if error_message:
                        source.error_message = error_message
                    db[source_id] = source
                    logger.info(f"[SourceStore-shelve] Updated {source_id} -> {status}")
                    return True
            return False

        def reset_orphaned_processing(self, max_age_minutes: int = 30):
            """Startup reconciliation: blanket-reset anything in 'processing'.

            The shelve backend doesn't track updated_at on status transitions,
            so we can't use age to distinguish live from orphaned. Called only
            at web-api boot, when no run can legitimately be in flight yet.
            """
            reset_ids: List[str] = []
            with self._get_db() as db:
                for key in list(db.keys()):
                    source = db[key]
                    if getattr(source, "processing_status", None) == "processing":
                        source.processing_status = "failed"
                        source.error_message = (
                            "Processing interrupted (orphaned from a previous run). "
                            "Click Reprocess to retry."
                        )
                        db[key] = source
                        reset_ids.append(key)
            if reset_ids:
                logger.warning(
                    f"[SourceStore-shelve] Reset {len(reset_ids)} orphaned 'processing' sources: "
                    f"{reset_ids}"
                )
            return reset_ids

        def delete_source(self, source_id: str) -> bool:
            with self._get_db() as db:
                if source_id in db:
                    source = db[source_id]
                    source.is_active = False
                    db[source_id] = source
                    return True
            return False

        def hard_delete_source(self, source_id: str) -> bool:
            with self._get_db() as db:
                if source_id in db:
                    del db[source_id]
                    return True
            return False

        def get_sources_by_tags(self, tags: List[str]) -> List[ClimateDataSource]:
            with self._get_db() as db:
                results = []
                for source in db.values():
                    if source.tags and any(t in source.tags for t in tags):
                        results.append(source)
                return results


def get_source_store(db_url: str = None) -> SourceStore:
    """Get a SourceStore instance."""
    return SourceStore()
