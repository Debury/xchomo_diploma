"""Dagster schedule management endpoints."""

import logging

from fastapi import APIRouter, HTTPException

from web_api.dependencies import execute_graphql_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/schedules", tags=["schedules"])


@router.get("/")
async def list_schedules():
    """List all Dagster schedules with their status."""
    try:
        query = """
        query {
            schedulesOrError {
                ... on Schedules {
                    results {
                        name
                        cronSchedule
                        scheduleState {
                            status
                        }
                        pipelineName
                        futureTicks(limit: 1) {
                            results {
                                timestamp
                            }
                        }
                    }
                }
                ... on PythonError { message }
            }
        }
        """
        data = await execute_graphql_query(query)
        schedules_data = data.get("schedulesOrError", {}).get("results", [])

        result = []
        for s in schedules_data:
            next_ticks = s.get("futureTicks", {}).get("results", [])
            next_run = next_ticks[0]["timestamp"] if next_ticks else None
            result.append(
                {
                    "name": s["name"],
                    "cron_schedule": s["cronSchedule"],
                    "status": s.get("scheduleState", {}).get("status", "UNKNOWN"),
                    "job_name": s.get("pipelineName"),
                    "next_run": next_run,
                }
            )
        return result
    except Exception as e:
        logger.warning(f"Failed to list schedules: {e}")
        return []


@router.post("/{schedule_name}/toggle")
async def toggle_schedule(schedule_name: str, enable: bool = True):
    """Enable or disable a Dagster schedule."""
    try:
        action = "startSchedule" if enable else "stopRunningSchedule"
        mutation = f"""
        mutation {{
            {action}(scheduleSelector: {{
                scheduleName: "{schedule_name}"
            }}) {{
                __typename
                ... on ScheduleStateResult {{ scheduleState {{ status }} }}
                ... on PythonError {{ message }}
            }}
        }}
        """
        data = await execute_graphql_query(mutation)
        result = data.get(action, {})

        if result.get("__typename") == "PythonError":
            raise HTTPException(500, result.get("message", "Unknown error"))

        return {
            "name": schedule_name,
            "status": result.get("scheduleState", {}).get("status", "UNKNOWN"),
            "message": f"Schedule {'enabled' if enable else 'disabled'}",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ────────────────────────────────────────────────────────────────────
# Dataset-level schedules
# ────────────────────────────────────────────────────────────────────

@router.get("/datasets")
async def list_dataset_schedules():
    """List all dataset-level schedules."""
    try:
        from src.database.source_store import SourceStore
        store = SourceStore()
        return store.list_dataset_schedules()
    except Exception as e:
        logger.warning(f"Failed to list dataset schedules: {e}")
        return []


@router.post("/datasets")
async def create_dataset_schedule(data: dict):
    """Create or update a dataset-level schedule."""
    name = data.get("name")
    dataset_name = data.get("dataset_name")
    cron_expression = data.get("cron_expression")

    if not name or not dataset_name or not cron_expression:
        raise HTTPException(400, "name, dataset_name, and cron_expression are required")

    try:
        from src.database.source_store import SourceStore
        store = SourceStore()
        result = store.create_dataset_schedule(
            name=name,
            dataset_name=dataset_name,
            cron_expression=cron_expression,
            is_enabled=data.get("is_enabled", True),
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/datasets/{schedule_id}")
async def delete_dataset_schedule(schedule_id: int):
    """Delete a dataset-level schedule."""
    try:
        from src.database.source_store import SourceStore
        store = SourceStore()
        if store.delete_dataset_schedule(schedule_id):
            return {"status": "deleted"}
        raise HTTPException(404, "Schedule not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))
