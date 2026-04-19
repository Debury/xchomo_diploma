"""Shared dependencies and helpers for the Climate ETL Pipeline API."""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# Dagster GraphQL configuration
DAGSTER_HOST = os.getenv("DAGSTER_HOST", "localhost")
DAGSTER_PORT = os.getenv("DAGSTER_PORT", "3000")
DAGSTER_GRAPHQL_URL = f"http://{DAGSTER_HOST}:{DAGSTER_PORT}/graphql"


async def execute_graphql_query(query: str, variables: Optional[Dict] = None) -> Dict:
    """Execute a GraphQL query against Dagster.

    Always returns a dict. Dagster returns ``{"data": null}`` when a mutation
    targets an unknown entity, so callers can't assume the payload is truthy.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                DAGSTER_GRAPHQL_URL,
                json={"query": query, "variables": variables or {}},
                timeout=30.0,
            )
            payload = response.json().get("data")
            return payload if isinstance(payload, dict) else {}
        except Exception as e:
            logger.error(f"Dagster GraphQL error: {e}")
            raise HTTPException(status_code=503, detail="Dagster unavailable")


async def get_active_runs_for_source(source_id: str) -> List[Dict[str, Any]]:
    """Return currently-active Dagster runs tagged with ``source_id=<source_id>``.

    "Active" means not yet terminal — STARTED / STARTING / QUEUED. Used by
    ``trigger_source_etl`` to refuse a second concurrent run for the same source
    (prevents the "two clicks on Reprocess → race on the same Qdrant source_id"
    bug). Returns ``[]`` on GraphQL error so the caller can fall through rather
    than 503 on a transient Dagster hiccup.
    """
    query = """
    query ActiveSourceRuns($tagValue: String!) {
        runsOrError(
            filter: {
                statuses: [STARTED, STARTING, QUEUED]
                tags: [{ key: "source_id", value: $tagValue }]
            }
            limit: 5
        ) {
            __typename
            ... on Runs { results { runId status jobName } }
            ... on PythonError { message }
        }
    }
    """
    try:
        data = await execute_graphql_query(query, {"tagValue": source_id})
    except HTTPException:
        return []
    runs_or_error = data.get("runsOrError") or {}
    if runs_or_error.get("__typename") != "Runs":
        return []
    return runs_or_error.get("results") or []


async def launch_dagster_run(job_name: str, run_config: Dict, tags: Dict = None):
    """Launch a Dagster job run via GraphQL."""
    repo_data = await execute_graphql_query("""
    query { repositoriesOrError { __typename ... on RepositoryConnection { nodes { name location { name } } } ... on PythonError { message } } }
    """)
    repos = repo_data.get("repositoriesOrError") or {}
    if repos.get("__typename") == "PythonError":
        raise HTTPException(status_code=502, detail=f"Dagster error: {repos.get('message', 'unknown')}")
    nodes = repos.get("nodes") or []
    if not nodes:
        raise HTTPException(status_code=500, detail="No repositories found")

    repo_name = nodes[0]["name"]
    repo_loc = nodes[0]["location"]["name"]

    mutation = """
    mutation LaunchRun($selector: JobOrPipelineSelector!, $config: RunConfigData!, $tags: [ExecutionTag!]) {
        launchRun(
            executionParams: {
                selector: $selector
                runConfigData: $config
                executionMetadata: { tags: $tags }
            }
        ) {
            __typename
            ... on LaunchRunSuccess { run { runId status } }
            ... on PythonError { message }
            ... on RunConfigValidationInvalid { errors { message } }
        }
    }
    """

    variables = {
        "selector": {
            "repositoryLocationName": repo_loc,
            "repositoryName": repo_name,
            "jobName": job_name,
        },
        "config": run_config or {},
        "tags": [{"key": k, "value": v} for k, v in (tags or {}).items()],
    }

    result = await execute_graphql_query(mutation, variables)
    launch_res = result.get("launchRun", {})

    if launch_res.get("__typename") == "LaunchRunSuccess":
        return {
            "runId": launch_res["run"]["runId"],
            "status": launch_res["run"]["status"],
            "jobName": job_name,
        }

    err_msg = launch_res.get("message") or str(launch_res.get("errors"))
    raise HTTPException(status_code=500, detail=f"Job launch failed: {err_msg}")


def get_qdrant_client():
    """Create a Qdrant client from environment variables."""
    from qdrant_client import QdrantClient

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_REST_PORT", 6333))
    grpc_port = int(os.getenv("QDRANT_GRPC_PORT", 6334))
    return QdrantClient(host=host, port=port, grpc_port=grpc_port, prefer_grpc=True, timeout=10)


def get_collection_name() -> str:
    """Get Qdrant collection name from pipeline config."""
    from src.utils.config_loader import ConfigLoader

    config_loader = ConfigLoader("config/pipeline_config.yaml")
    pipeline_config = config_loader.load()
    qdrant_config = pipeline_config.get("vector_db", {}).get("qdrant", {})
    return qdrant_config.get("collection_name", "climate_data")


def get_vector_database():
    """Get a VectorDatabase instance configured from pipeline config."""
    from src.embeddings.database import VectorDatabase
    from src.utils.config_loader import ConfigLoader

    config_loader = ConfigLoader("config/pipeline_config.yaml")
    pipeline_config = config_loader.load()
    return VectorDatabase(config=pipeline_config)


def summarize_hits(results: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Summarize search hits into a brief summary and list of references."""
    if not results:
        return "No context found.", []
    refs = []
    seen = set()
    for h in results:
        m = h.get("metadata", {})
        ref = f"{m.get('source_id', '?')}:{m.get('variable', '?')}"
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return f"Found {len(results)} relevant data points.", refs


def clear_rag_cache() -> None:
    """Clear the RAG endpoint's cached collection info."""
    try:
        import web_api.rag_endpoint as rag_module

        rag_module._CACHED_INFO = {}
        rag_module._CACHED_INFO_TS = 0
        logger.info("Cleared RAG info cache")
    except Exception as e:
        logger.warning(f"Could not clear RAG cache: {e}")
