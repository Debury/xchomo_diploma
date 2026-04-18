"""Per-run ingestion context.

Holds the current ETL run identifier so downstream helpers (portal adapters,
chunk assembly) can tag every Qdrant payload with `ingestion_run_id`. That tag
enables atomic versioned sweeps: a re-ingestion writes new chunks tagged with
the current run id, then deletes everything else for the same source. Queries
during the run keep seeing the previous version; failed runs leave the
previous version intact.

Set via :func:`set_ingestion_run_id` at the top of a Dagster op. Read via
:func:`get_ingestion_run_id` from any callee in the same task.
"""

from __future__ import annotations

import contextvars
from typing import Optional

_current_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "ingestion_run_id", default=None
)


def set_ingestion_run_id(run_id: Optional[str]) -> None:
    _current_run_id.set(run_id)


def get_ingestion_run_id() -> Optional[str]:
    return _current_run_id.get()
