"""Health check endpoint."""

from datetime import datetime
from fastapi import APIRouter

from web_api.models import HealthResponse
from web_api.dependencies import execute_graphql_query

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness + substrate check.

    Returns `status="degraded"` if a substrate dependency (currently just
    Dagster) is unreachable, so a monitoring dashboard / reverse proxy can
    route around the web-api instead of treating it as fully healthy. The
    underlying `dagster_available` flag stays so callers that just want
    liveness can still read it.
    """
    dagster_up = False
    try:
        await execute_graphql_query("{ __typename }")
        dagster_up = True
    except Exception:
        pass
    return HealthResponse(
        status="healthy" if dagster_up else "degraded",
        dagster_available=dagster_up,
        timestamp=datetime.now().isoformat(),
    )
