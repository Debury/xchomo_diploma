"""
Database module for PostgreSQL-backed source storage.

Replaces the shelve-based SourceStore with SQLAlchemy + PostgreSQL.
"""

from src.database.connection import get_db, init_db, get_engine
from src.database.models import Source, SourceCredential, SourceSchedule, ProcessingRun

__all__ = [
    "get_db",
    "init_db",
    "get_engine",
    "Source",
    "SourceCredential",
    "SourceSchedule",
    "ProcessingRun",
]
