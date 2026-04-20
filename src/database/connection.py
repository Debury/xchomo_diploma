"""
Database connection and session management for the climate_app database.

Uses SQLAlchemy with PostgreSQL connection pooling.
"""

import logging
import os
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session

from src.database.models import Base

logger = logging.getLogger(__name__)

# Database URL: uses the same PostgreSQL container as Dagster, different database
DATABASE_URL = os.getenv(
    "APP_DATABASE_URL",
    "postgresql://dagster:dagster@dagster-postgres:5432/climate_app",
)

_engine = None
_SessionLocal = None


def get_engine():
    """Get or create the SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )
    return _engine


def get_session_factory():
    """Get or create the session factory (singleton)."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _SessionLocal


def get_db():
    """FastAPI dependency: yields a database session, auto-closes on completion."""
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Session:
    """Context manager for non-FastAPI usage (Dagster ops, scripts, etc.)."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _source_lock_key(source_id: str) -> int:
    """Stable 64-bit signed int for ``pg_try_advisory_lock(bigint)``.

    blake2b is collision-resistant; over a 64-bit space the birthday bound is
    ~2³² (~4 B) distinct source_ids before a single expected collision — so
    for this project the risk is effectively zero.
    """
    import hashlib

    digest = hashlib.blake2b(source_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


@contextmanager
def acquire_source_lock(source_id: str):
    """Postgres session-level advisory lock keyed on a 64-bit hash of ``source_id``.

    Both the FastAPI trigger path (web-api container) and the Dagster sensor
    (dagster-daemon container) share the same ``climate_app`` database, so a
    lock taken here is mutually exclusive across containers — closes the
    API-vs-sensor race that otherwise allowed two runs to launch for the same
    ``source_id`` before either appeared in Dagster's active-run list.

    Non-blocking: yields ``True`` if acquired, ``False`` if another caller
    already holds it. Release happens on context exit (or implicitly when the
    connection closes, since the lock is session-scoped).
    """
    key = _source_lock_key(source_id)
    engine = get_engine()
    conn = engine.connect()
    acquired = False
    try:
        acquired = bool(
            conn.execute(
                text("SELECT pg_try_advisory_lock(:key)"),
                {"key": key},
            ).scalar()
        )
        yield acquired
    finally:
        if acquired:
            try:
                conn.execute(
                    text("SELECT pg_advisory_unlock(:key)"),
                    {"key": key},
                )
            except Exception as unlock_err:
                logger.warning(f"pg_advisory_unlock failed for {source_id}: {unlock_err}")
        conn.close()


def init_db():
    """Create all tables if they don't exist, then apply idempotent ALTER TABLE
    migrations for columns that were added after the initial schema shipped.

    We're not using Alembic for this project — the schema changes infrequently
    and `create_all` handles fresh installs. For production-adjacent deploys
    (the diploma droplet) we still need to add missing columns to existing
    tables without data loss, hence the inline migration below.
    """
    from sqlalchemy import inspect, text

    engine = get_engine()
    Base.metadata.create_all(bind=engine)

    # Columns added after the table was first shipped. `key` is the Source
    # attribute name; `ddl` is the raw type for ALTER TABLE ADD COLUMN.
    sources_migrations = [
        ("hazard_type", "VARCHAR(100)"),
        ("region_country", "VARCHAR(255)"),
        ("spatial_coverage", "TEXT"),
        ("impact_sector", "VARCHAR(255)"),
        ("dataset_name", "VARCHAR(255)"),
    ]

    inspector = inspect(engine)
    if inspector.has_table("sources"):
        existing = {c["name"] for c in inspector.get_columns("sources")}
        missing = [(name, ddl) for name, ddl in sources_migrations if name not in existing]
        if missing:
            with engine.begin() as conn:
                for name, ddl in missing:
                    conn.execute(text(f"ALTER TABLE sources ADD COLUMN {name} {ddl}"))
