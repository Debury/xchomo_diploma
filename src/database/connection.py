"""
Database connection and session management for the climate_app database.

Uses SQLAlchemy with PostgreSQL connection pooling.
"""

import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.database.models import Base

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
    ]

    inspector = inspect(engine)
    if inspector.has_table("sources"):
        existing = {c["name"] for c in inspector.get_columns("sources")}
        missing = [(name, ddl) for name, ddl in sources_migrations if name not in existing]
        if missing:
            with engine.begin() as conn:
                for name, ddl in missing:
                    conn.execute(text(f"ALTER TABLE sources ADD COLUMN {name} {ddl}"))
