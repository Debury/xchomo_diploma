"""Health check endpoint."""

from datetime import datetime
from fastapi import APIRouter

from web_api.models import HealthResponse
from web_api.dependencies import execute_graphql_query

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    dagster_up = False
    try:
        await execute_graphql_query("{ __typename }")
        dagster_up = True
    except Exception:
        pass
    return HealthResponse(
        status="healthy",
        dagster_available=dagster_up,
        timestamp=datetime.now().isoformat(),
    )
