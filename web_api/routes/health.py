"""Health check endpoint."""

import asyncio
import logging
from datetime import datetime
from fastapi import APIRouter
from sqlalchemy import text

from web_api.models import HealthResponse
from web_api.dependencies import execute_graphql_query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


def _check_database_sync() -> bool:
    """Synchronous `SELECT 1` against the climate_app Postgres.

    Kept synchronous because SQLAlchemy's default engine uses psycopg2.
    The async wrapper below offloads it to a thread.
    """
    try:
        from src.database.connection import get_engine

        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as err:
        logger.warning(f"/health: database unreachable: {err}")
        return False


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness + substrate check.

    Returns `status="degraded"` when any substrate dependency (Dagster or
    the application Postgres) is unreachable. This lets a monitoring
    dashboard or reverse proxy route around the web-api instead of
    treating it as fully healthy, and surfaces silent DB outages that
    would otherwise only show up as 500s on every protected endpoint.
    The `dagster_available` / `database_available` flags stay on the
    response for callers that want finer-grained signals.
    """
    dagster_up = False
    try:
        await execute_graphql_query("{ __typename }")
        dagster_up = True
    except Exception:
        pass

    loop = asyncio.get_event_loop()
    db_up = await loop.run_in_executor(None, _check_database_sync)

    return HealthResponse(
        status="healthy" if (dagster_up and db_up) else "degraded",
        dagster_available=dagster_up,
        database_available=db_up,
        timestamp=datetime.now().isoformat(),
    )
