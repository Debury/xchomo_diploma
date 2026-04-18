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
# Dataset-level schedules were removed in favor of per-source scheduling
# (see /sources/{source_id}/schedule in web_api.routes.sources).
# ────────────────────────────────────────────────────────────────────
