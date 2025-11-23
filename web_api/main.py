"""
FastAPI Web Service for Climate ETL Pipeline

Provides REST API endpoints for interacting with Dagster pipelines.
Endpoints allow listing jobs, triggering runs, and checking status.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ====================================================================================
# PYDANTIC MODELS
# ====================================================================================

class JobInfo(BaseModel):
    """Information about a Dagster job"""
    name: str = Field(..., description="Job name")
    description: Optional[str] = Field(None, description="Job description")
    tags: Dict[str, str] = Field(default_factory=dict, description="Job tags")
    ops: List[str] = Field(default_factory=list, description="List of op names in job")


class RunRequest(BaseModel):
    """Request to trigger a job run"""
    run_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional run configuration"
    )
    tags: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional tags for the run"
    )


class RunStatus(BaseModel):
    """Status information for a pipeline run"""
    run_id: str = Field(..., description="Unique run identifier")
    status: str = Field(..., description="Run status (QUEUED, STARTED, SUCCESS, FAILURE, CANCELED)")
    job_name: str = Field(..., description="Name of the job")
    start_time: Optional[str] = Field(None, description="Run start timestamp")
    end_time: Optional[str] = Field(None, description="Run end timestamp")
    duration: Optional[float] = Field(None, description="Run duration in seconds")
    tags: Dict[str, str] = Field(default_factory=dict, description="Run tags")


class RunResponse(BaseModel):
    """Response after triggering a run"""
    run_id: str = Field(..., description="Unique run identifier")
    job_name: str = Field(..., description="Name of the job")
    status: str = Field(..., description="Initial run status")
    message: str = Field(..., description="Response message")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    dagster_available: bool = Field(..., description="Whether Dagster is reachable")
    timestamp: str = Field(..., description="Current timestamp")


# ====================================================================================
# SOURCE MANAGEMENT MODELS (Phase 5)
# ====================================================================================

class SourceCreate(BaseModel):
    """Request to create a new data source"""
    source_id: str = Field(..., description="Unique identifier for the source")
    url: str = Field(..., description="URL or path to data source")
    format: Optional[str] = Field(
        None, 
        description="Data format (netcdf, grib, csv, parquet, zarr, hdf5). If not provided, will be auto-detected from URL."
    )
    variables: Optional[List[str]] = Field(
        None, 
        description="List of variables to extract. If None, all available variables will be processed."
    )
    time_range: Optional[Dict[str, str]] = Field(None, description="Time range {start, end}")
    spatial_bbox: Optional[List[float]] = Field(
        None, 
        description="Geographic bounding box: [south, west, north, east] in degrees. Example: [48.0, 13.0, 51.0, 19.0] for Slovakia region. If None, entire available area will be used."
    )
    is_active: bool = Field(
        True, 
        description="Whether this source should be actively processed by the ETL pipeline. Set to False to temporarily disable processing without deleting the source."
    )
    transformations: Optional[List[str]] = Field(None, description="Transformations to apply")
    aggregation_method: Optional[str] = Field("mean", description="Aggregation method")
    output_resolution: Optional[float] = Field(None, description="Output spatial resolution")
    embedding_model: Optional[str] = Field("all-MiniLM-L6-v2", description="Embedding model")
    chunk_size: Optional[int] = Field(512, description="Embedding chunk size")
    description: Optional[str] = Field(None, description="Source description")
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")
    
    class Config:
        json_schema_extra = {
            "example": {
                "source_id": "xarray_air_temp",
                "url": "https://github.com/pydata/xarray-data/raw/master/air_temperature.nc",
                "description": "Sample air temperature data from xarray tutorial (format auto-detected)",
                "is_active": True,
                "tags": ["xarray", "air_temp", "tutorial"]
            }
        }


class SourceUpdate(BaseModel):
    """Request to update an existing source"""
    url: Optional[str] = None
    variables: Optional[List[str]] = None
    time_range: Optional[Dict[str, str]] = None
    spatial_bbox: Optional[List[float]] = None
    transformations: Optional[List[str]] = None
    aggregation_method: Optional[str] = None
    output_resolution: Optional[float] = None
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    is_active: Optional[bool] = None


class EmbeddingResponse(BaseModel):
    """Response for a single embedding"""
    id: str = Field(..., description="Unique identifier for the embedding")
    variable: str = Field(..., description="Climate variable name (e.g., 't2m', 'tp')")
    timestamp: Optional[str] = Field(None, description="Timestamp of the data point")
    location: Optional[Dict[str, float]] = Field(None, description="Geographic location (lat, lon)")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    embedding_preview: Optional[List[float]] = Field(None, description="First 10 dimensions of embedding vector")


class EmbeddingStatsResponse(BaseModel):
    """Statistics about the vector database"""
    total_embeddings: int = Field(..., description="Total number of embeddings in the database")
    variables: List[str] = Field(..., description="List of unique variables")
    date_range: Optional[Dict[str, str]] = Field(None, description="Earliest and latest timestamps")
    sources: List[str] = Field(..., description="List of source IDs")
    collection_name: str = Field(..., description="Qdrant collection name")


class EmbeddingSearchResult(BaseModel):
    """Search result with similarity score"""
    id: str
    variable: str
    timestamp: Optional[str]
    location: Optional[Dict[str, float]]
    metadata: Dict[str, Any]
    similarity_score: float = Field(..., description="Cosine similarity score (0-1)")


class RAGChunk(BaseModel):
    source_id: str
    variable: Optional[str]
    similarity: float
    text: str
    metadata: Dict[str, Any]


class RAGChatRequest(BaseModel):
    question: str = Field(..., description="Natural language climate question", min_length=3)
    top_k: int = Field(3, ge=1, le=10, description="How many chunks to retrieve")


class RAGChatResponse(BaseModel):
    question: str
    answer: str
    references: List[str]
    chunks: List[RAGChunk]


class SourceResponse(BaseModel):
    """Response with source information"""
    id: int
    source_id: str
    url: str
    format: str
    variables: Optional[List[str]]  # Changed to Optional
    time_range: Optional[Dict[str, str]]
    spatial_bbox: Optional[List[float]]
    transformations: Optional[List[str]]
    aggregation_method: Optional[str]  # Changed to Optional
    output_resolution: Optional[float]
    embedding_model: Optional[str]  # Changed to Optional
    chunk_size: Optional[int]  # Changed to Optional
    collection_name: Optional[str]
    description: Optional[str]
    tags: Optional[List[str]]
    is_active: bool
    last_processed: Optional[str]
    processing_status: str
    error_message: Optional[str]
    created_at: str
    updated_at: str
    created_by: Optional[str]


# ====================================================================================
# FASTAPI APPLICATION
# ====================================================================================

app = FastAPI(
    title="Climate ETL Pipeline API",
    description="REST API for managing climate data ETL workflows via Dagster",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

FRONTEND_DIR = Path(__file__).parent / "frontend"

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/ui/static", StaticFiles(directory=FRONTEND_DIR, html=False), name="rag-ui-static")

# Configuration
DAGSTER_HOST = os.getenv("DAGSTER_HOST", "localhost")
DAGSTER_PORT = os.getenv("DAGSTER_PORT", "3000")
DAGSTER_GRAPHQL_URL = f"http://{DAGSTER_HOST}:{DAGSTER_PORT}/graphql"


# ====================================================================================
# HELPER FUNCTIONS
# ====================================================================================

async def execute_graphql_query(query: str, variables: Optional[Dict] = None) -> Dict:
    """
    Execute a GraphQL query against Dagster's GraphQL API.
    
    Args:
        query: GraphQL query string
        variables: Optional query variables
    
    Returns:
        Query result dictionary
    
    Raises:
        HTTPException: If query fails
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                DAGSTER_GRAPHQL_URL,
                json={"query": query, "variables": variables or {}},
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            
            if "errors" in data:
                raise HTTPException(
                    status_code=500,
                    detail=f"GraphQL errors: {data['errors']}"
                )
            
            return data.get("data", {})
        
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot connect to Dagster at {DAGSTER_GRAPHQL_URL}: {str(e)}"
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Dagster request failed: {str(e)}"
            )


def summarize_hits(results: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """Build a human-readable answer and reference list from semantic hits."""
    if not results:
        return "No supporting context found.", []

    sentences: List[str] = []
    references: List[str] = []
    for hit in results:
        metadata = hit.get("metadata") or {}
        variable = metadata.get("variable", "variable")
        source_id = metadata.get("source_id", "source")
        unit = metadata.get("unit") or ""
        stat_mean = metadata.get("stat_mean")
        temporal = metadata.get("temporal_extent") or {}
        latest = temporal.get("end") or metadata.get("timestamp") or "n/a"
        if isinstance(stat_mean, (int, float)):
            sentences.append(
                f"{variable} from {source_id} averages {stat_mean:.2f}{unit} (latest {latest})."
            )
        else:
            sentences.append(
                f"{variable} from {source_id} covers {metadata.get('long_name', 'the dataset')} (latest {latest})."
            )
        references.append(f"{source_id}:{variable}")

    ordered_refs: List[str] = []
    seen = set()
    for ref in references:
        if ref not in seen:
            seen.add(ref)
            ordered_refs.append(ref)
    return " ".join(sentences), ordered_refs


# ====================================================================================
# API ENDPOINTS
# ====================================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "name": "Climate ETL Pipeline API",
        "version": "1.0.0",
        "description": "REST API for managing climate data workflows",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/ui", response_class=FileResponse)
async def serve_frontend():
    """Serve the lightweight RAG dashboard frontend."""
    if not FRONTEND_DIR.exists():
        raise HTTPException(status_code=404, detail="Frontend assets not found. Run git pull or rebuild the image.")
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify service and Dagster connectivity.
    """
    dagster_available = False
    
    try:
        # Simple query to check Dagster availability
        query = """
        {
            __typename
        }
        """
        await execute_graphql_query(query)
        dagster_available = True
    except Exception:
        pass
    
    return HealthResponse(
        status="healthy" if dagster_available else "degraded",
        dagster_available=dagster_available,
        timestamp=datetime.now().isoformat()
    )


@app.get("/jobs", response_model=List[JobInfo])
async def list_jobs():
    """
    List all available Dagster jobs in the repository.
    
    Returns:
        List of job information
    """
    query = """
    {
        repositoryOrError(repositorySelector: {
            repositoryLocationName: "__repository__",
            repositoryName: "__repository__"
        }) {
            ... on Repository {
                pipelines {
                    name
                    description
                    tags {
                        key
                        value
                    }
                    solidHandles {
                        solid {
                            name
                        }
                    }
                }
            }
        }
    }
    """
    
    try:
        data = await execute_graphql_query(query)
        
        # For demonstration, return mock jobs if GraphQL fails
        # In production, this would parse actual GraphQL response
        mock_jobs = [
            JobInfo(
                name="daily_etl_job",
                description="Complete daily ETL workflow: download → transform → export",
                tags={"pipeline": "etl", "frequency": "daily", "phase": "1-2"},
                ops=["download_era5_data", "validate_downloaded_data", "ingest_data", 
                     "transform_data", "export_data"]
            ),
            JobInfo(
                name="embedding_job",
                description="Generate embeddings from processed data and store in vector database",
                tags={"pipeline": "embeddings", "frequency": "on-demand", "phase": "3"},
                ops=["generate_embeddings_standalone", "store_embeddings", "test_semantic_search"]
            ),
            JobInfo(
                name="complete_pipeline_job",
                description="Complete end-to-end pipeline: download → transform → embed",
                tags={"pipeline": "complete", "frequency": "weekly", "phase": "1-2-3"},
                ops=["download_era5_data", "validate_downloaded_data", "ingest_data",
                     "transform_data", "export_data", "generate_embeddings",
                     "store_embeddings", "test_semantic_search"]
            ),
            JobInfo(
                name="validation_job",
                description="Validation-only job to check data quality",
                tags={"pipeline": "validation", "frequency": "on-demand"},
                ops=["validate_existing_data"]
            )
        ]
        
        return mock_jobs
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing jobs: {str(e)}")


@app.post("/jobs/{job_name}/run", response_model=RunResponse)
async def trigger_job_run(
    job_name: str = PathParam(..., description="Name of the job to run"),
    run_request: RunRequest = None
):
    """
    Trigger a new run of the specified job.
    
    Args:
        job_name: Name of the job to execute
        run_request: Optional run configuration and tags
    
    Returns:
        Run response with run ID and status
    """
    # Validate job name
    valid_jobs = ["daily_etl_job", "embedding_job", "complete_pipeline_job", "validation_job"]
    if job_name not in valid_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_name}' not found. Available jobs: {valid_jobs}"
        )
    
    # GraphQL mutation to launch run
    mutation = """
    mutation LaunchRun($executionParams: ExecutionParams!) {
        launchRun(executionParams: $executionParams) {
            __typename
            ... on LaunchRunSuccess {
                run {
                    runId
                    status
                    pipelineName
                }
            }
            ... on PythonError {
                message
            }
        }
    }
    """
    
    try:
        # For demonstration, return mock response
        # In production, execute actual GraphQL mutation
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_name}"
        
        return RunResponse(
            run_id=run_id,
            job_name=job_name,
            status="QUEUED",
            message=f"Successfully triggered run for job '{job_name}'"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error triggering job run: {str(e)}"
        )


@app.get("/runs/{run_id}/status", response_model=RunStatus)
async def get_run_status(
    run_id: str = PathParam(..., description="Run ID to check status for")
):
    """
    Get the status of a specific pipeline run.
    
    Args:
        run_id: Unique run identifier
    
    Returns:
        Run status information
    """
    query = """
    query GetRunStatus($runId: ID!) {
        runOrError(runId: $runId) {
            __typename
            ... on Run {
                runId
                status
                pipelineName
                startTime
                endTime
                stats {
                    ... on RunStatsSnapshot {
                        startTime
                        endTime
                    }
                }
                tags {
                    key
                    value
                }
            }
            ... on RunNotFoundError {
                message
            }
        }
    }
    """
    
    try:
        # For demonstration, return mock status
        # In production, execute actual GraphQL query
        
        # Parse job name from run_id (mock)
        job_name = run_id.split("_")[-1] if "_" in run_id else "unknown_job"
        
        return RunStatus(
            run_id=run_id,
            status="SUCCESS",
            job_name=job_name,
            start_time=datetime.now().isoformat(),
            end_time=None,
            duration=None,
            tags={"api_triggered": "true"}
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching run status: {str(e)}"
        )


@app.get("/runs", response_model=List[RunStatus])
async def list_recent_runs(
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of runs to return")
):
    """
    List recent pipeline runs.
    
    Args:
        limit: Maximum number of runs to return (1-100)
    
    Returns:
        List of recent run statuses
    """
    query = """
    query ListRuns($limit: Int!) {
        runsOrError {
            __typename
            ... on Runs {
                results(limit: $limit) {
                    runId
                    status
                    pipelineName
                    startTime
                    endTime
                    tags {
                        key
                        value
                    }
                }
            }
        }
    }
    """
    
    try:
        # For demonstration, return mock runs
        mock_runs = [
            RunStatus(
                run_id=f"run_20250119_140000_daily_etl_job",
                status="SUCCESS",
                job_name="daily_etl_job",
                start_time=datetime.now().isoformat(),
                end_time=datetime.now().isoformat(),
                duration=125.5,
                tags={"schedule": "daily", "date": "2025-01-19"}
            ),
            RunStatus(
                run_id=f"run_20250119_160000_embedding_job",
                status="SUCCESS",
                job_name="embedding_job",
                start_time=datetime.now().isoformat(),
                end_time=None,
                duration=None,
                tags={"schedule": "daily_embeddings"}
            )
        ]
        
        return mock_runs[:limit]
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing runs: {str(e)}"
        )


# ====================================================================================
# SOURCE MANAGEMENT ENDPOINTS (Phase 5)
# ====================================================================================

@app.post("/sources", response_model=SourceResponse, status_code=201)
async def create_source(source: SourceCreate):
    """
    Create a new climate data source.
    
    The source will be stored in the database and can be processed by Dagster jobs.
    """
    try:
        from src.sources import get_source_store
        
        store = get_source_store()
        
        # Check if source_id already exists
        existing = store.get_source(source.source_id)
        if existing:
            raise HTTPException(
                status_code=400,
                detail=f"Source with ID '{source.source_id}' already exists"
            )
        
        # Auto-detect format if not provided
        if not source.format:
            from dagster_project.ops.dynamic_source_ops import detect_format_from_url
            source.format = detect_format_from_url(source.url)
        
        # Create source
        source_data = source.dict()
        created_source = store.create_source(source_data)
        
        return created_source.to_dict()
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating source: {str(e)}"
        )


@app.get("/sources", response_model=List[SourceResponse])
async def list_sources(
    active_only: bool = Query(default=True, description="Filter by active sources only"),
    tags: Optional[str] = Query(default=None, description="Comma-separated tags to filter by")
):
    """
    List all climate data sources.
    
    Args:
        active_only: If true, only return active sources
        tags: Optional comma-separated list of tags to filter by
    
    Returns:
        List of sources
    """
    try:
        from src.sources import get_source_store
        
        store = get_source_store()
        
        if tags:
            tag_list = [t.strip() for t in tags.split(",")]
            sources = store.get_sources_by_tags(tag_list)
        else:
            sources = store.get_all_sources(active_only=active_only)
        
        return [source.to_dict() for source in sources]
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing sources: {str(e)}"
        )


@app.get("/sources/{source_id}", response_model=SourceResponse)
async def get_source(source_id: str = PathParam(..., description="Source ID")):
    """
    Get a specific climate data source by ID.
    
    Args:
        source_id: Unique identifier of the source
    
    Returns:
        Source information
    """
    try:
        from src.sources import get_source_store
        
        store = get_source_store()
        source = store.get_source(source_id)
        
        if not source:
            raise HTTPException(
                status_code=404,
                detail=f"Source with ID '{source_id}' not found"
            )
        
        return source.to_dict()
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching source: {str(e)}"
        )


@app.put("/sources/{source_id}", response_model=SourceResponse)
async def update_source(
    source_id: str = PathParam(..., description="Source ID"),
    updates: SourceUpdate = ...
):
    """
    Update an existing climate data source.
    
    Args:
        source_id: Unique identifier of the source
        updates: Fields to update
    
    Returns:
        Updated source information
    """
    try:
        from src.sources import get_source_store
        
        store = get_source_store()
        
        # Check if source exists
        existing = store.get_source(source_id)
        if not existing:
            raise HTTPException(
                status_code=404,
                detail=f"Source with ID '{source_id}' not found"
            )
        
        # Update source
        update_data = updates.dict(exclude_unset=True)
        updated_source = store.update_source(source_id, update_data)
        
        if not updated_source:
            raise HTTPException(
                status_code=500,
                detail="Failed to update source"
            )
        
        return updated_source.to_dict()
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating source: {str(e)}"
        )


@app.delete("/sources/{source_id}", status_code=204)
async def delete_source(
    source_id: str = PathParam(..., description="Source ID"),
    hard_delete: bool = Query(default=False, description="Permanently delete (true) or soft delete (false)")
):
    """
    Delete a climate data source.
    
    Args:
        source_id: Unique identifier of the source
        hard_delete: If true, permanently delete. If false, soft delete (set is_active=False)
    
    Returns:
        No content on success
    """
    try:
        from src.sources import get_source_store
        
        store = get_source_store()
        
        if hard_delete:
            success = store.hard_delete_source(source_id)
        else:
            success = store.delete_source(source_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Source with ID '{source_id}' not found"
            )
        
        return None
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting source: {str(e)}"
        )


@app.post("/sources/{source_id}/trigger", response_model=RunResponse)
async def trigger_source_etl(
    source_id: str = PathParam(..., description="Source ID"),
    job_name: str = Query(
        default="dynamic_source_etl_job",
        description="Job to run: dynamic_source_etl_job (processes ALL active sources)"
    ),
    run_config: Optional[Dict[str, Any]] = None
):
    """
    Trigger ETL job for sources.
    
    Args:
        source_id: Source identifier (used to verify source exists and is active)
        job_name: Which job to run (default: dynamic_source_etl_job)
        run_config: Optional additional run configuration
    
    Returns:
        Run information
    
    Available jobs:
    - dynamic_source_etl_job: Processes ALL active sources (recommended)
    - complete_pipeline_job: Legacy full pipeline
    - daily_etl_job: Legacy download + transform only
    
    Note: dynamic_source_etl_job processes ALL active sources in the database,
    not just the specified source_id. Use is_active flag to control which sources run.
    """
    try:
        from src.sources import get_source_store
        
        store = get_source_store()
        source = store.get_source(source_id)
        
        if not source:
            raise HTTPException(
                status_code=404,
                detail=f"Source with ID '{source_id}' not found"
            )
        
        if not source.is_active:
            raise HTTPException(
                status_code=400,
                detail=f"Source '{source_id}' is not active"
            )
        
        # Prepare run config based on job type
        if job_name == "dynamic_source_etl_job":
            # Dynamic job doesn't need config - it loads sources from DB automatically
            job_config = {}
        else:
            # Legacy jobs need explicit config
            job_config = {
                "ops": {
                    "download_era5_data": {
                        "config": {
                            "variables": source.variables if source.variables else ["2m_temperature"],
                            "year": 2025,
                            "month": 1,
                            "days": None,
                            "area": source.spatial_bbox if source.spatial_bbox else None
                        }
                    }
                }
            }
        
        # Merge with any additional config
        if run_config:
            job_config.update(run_config)
        
        # Note: job_name is now passed as parameter (default: dynamic_source_etl_job)
        
        # First, query to get repository information
        repo_query = """
        query GetRepositories {
            repositoriesOrError {
                __typename
                ... on RepositoryConnection {
                    nodes {
                        name
                        location {
                            name
                        }
                    }
                }
            }
        }
        """
        
        # Get repository info
        async with httpx.AsyncClient() as client:
            repo_response = await client.post(
                DAGSTER_GRAPHQL_URL,
                json={"query": repo_query},
                timeout=10.0
            )
            
            if repo_response.status_code != 200:
                raise HTTPException(
                    status_code=repo_response.status_code,
                    detail=f"Cannot get repository info: {repo_response.text}"
                )
            
            repo_data = repo_response.json()
            
            if "errors" in repo_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"GraphQL error getting repos: {repo_data['errors']}"
                )
            
            # Extract repository information
            repos = repo_data.get("data", {}).get("repositoriesOrError", {}).get("nodes", [])
            if not repos:
                raise HTTPException(
                    status_code=404,
                    detail="No Dagster repositories found"
                )
            
            # Use first repository
            repo_name = repos[0]["name"]
            repo_location = repos[0]["location"]["name"]
        
        # Now launch the run with correct repository info
        mutation = """
        mutation LaunchRun($repositoryLocationName: String!, $repositoryName: String!, $jobName: String!, $runConfigData: RunConfigData!) {
            launchRun(
                executionParams: {
                    selector: {
                        repositoryLocationName: $repositoryLocationName
                        repositoryName: $repositoryName
                        jobName: $jobName
                    }
                    runConfigData: $runConfigData
                }
            ) {
                __typename
                ... on LaunchRunSuccess {
                    run {
                        runId
                        status
                        jobName
                    }
                }
                ... on PythonError {
                    message
                    stack
                }
                ... on RunConfigValidationInvalid {
                    errors {
                        message
                    }
                }
            }
        }
        """
        
        variables = {
            "repositoryLocationName": repo_location,
            "repositoryName": repo_name,
            "jobName": job_name,
            "runConfigData": job_config
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                DAGSTER_GRAPHQL_URL,
                json={
                    "query": mutation,
                    "variables": variables
                },
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Dagster API error: {response.text}"
                )
            
            data = response.json()
            
            if "errors" in data:
                raise HTTPException(
                    status_code=500,
                    detail=f"GraphQL error: {data['errors']}"
                )
            
            launch_result = data["data"]["launchRun"]
            
            if launch_result["__typename"] == "LaunchRunSuccess":
                run = launch_result["run"]
                
                # Update source processing status
                store.update_processing_status(source_id, "processing")
                
                return RunResponse(
                    run_id=run["runId"],
                    job_name=run["jobName"],
                    status=run["status"],
                    message=f"ETL job triggered successfully for source '{source_id}'"
                )
            elif launch_result["__typename"] == "PythonError":
                raise HTTPException(
                    status_code=500,
                    detail=f"Job launch error: {launch_result['message']}"
                )
            elif launch_result["__typename"] == "RunConfigValidationInvalid":
                errors = [e["message"] for e in launch_result["errors"]]
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid run config: {'; '.join(errors)}"
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Unexpected result type: {launch_result['__typename']}"
                )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error triggering ETL: {str(e)}"
        )


# ====================================================================================
# EMBEDDINGS ENDPOINTS - Vector Database Observability
# ====================================================================================

@app.get("/embeddings/stats", response_model=EmbeddingStatsResponse)
async def get_embeddings_stats():
    """
    Get statistics about the vector database.
    
    Returns comprehensive information about embeddings stored in Qdrant:
    - Total number of embeddings
    - Unique variables present
    - Date range covered
    - Source identifiers
    
    This endpoint is useful for:
    - Verifying that embeddings are being generated correctly
    - Monitoring the RAG knowledge base growth
    - Debugging missing or incomplete data
    """
    try:
        from src.embeddings.database import VectorDatabase
        
        db = VectorDatabase()
        
        # Get all embeddings to analyze
        collection = db.collection
        results = collection.get()
        
        if not results['ids']:
            return EmbeddingStatsResponse(
                total_embeddings=0,
                variables=[],
                date_range=None,
                sources=[],
                collection_name=db.collection_name
            )
        
        # Extract statistics from metadata
        variables = set()
        sources = set()
        timestamps = []
        
        for metadata in results.get('metadatas', []):
            if metadata:
                if 'variable' in metadata:
                    variables.add(metadata['variable'])
                if 'source_id' in metadata:
                    sources.add(metadata['source_id'])
                if 'timestamp' in metadata:
                    timestamps.append(metadata['timestamp'])
        
        # Calculate date range
        date_range = None
        if timestamps:
            timestamps.sort()
            date_range = {
                "earliest": timestamps[0],
                "latest": timestamps[-1]
            }
        
        return EmbeddingStatsResponse(
            total_embeddings=len(results['ids']),
            variables=sorted(list(variables)),
            date_range=date_range,
            sources=sorted(list(sources)),
            collection_name=db.collection_name
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving embeddings statistics: {str(e)}"
        )


@app.get("/embeddings/sample", response_model=List[EmbeddingResponse])
async def get_sample_embeddings(
    limit: int = Query(default=10, ge=1, le=100, description="Number of samples to return")
):
    """
    Get a sample of embeddings from the vector database.
    
    Args:
        limit: Number of sample embeddings to return (1-100)
    
    Returns:
        List of embedding samples with metadata and preview of embedding vector
    
    Use this endpoint to:
    - Verify embeddings are being stored correctly
    - Inspect metadata structure
    - Debug data quality issues
    - Preview embedding dimensions
    """
    try:
        from src.embeddings.database import VectorDatabase
        
        db = VectorDatabase()
        collection = db.collection
        
        # Get sample embeddings
        results = collection.get(limit=limit, include=['embeddings', 'metadatas'])
        
        if not results['ids']:
            return []
        
        samples = []
        for i, embedding_id in enumerate(results['ids']):
            metadata = results.get('metadatas', [{}])[i] or {}
            embedding = results.get('embeddings', [[]])[i] or []
            
            samples.append(EmbeddingResponse(
                id=embedding_id,
                variable=metadata.get('variable', 'unknown'),
                timestamp=metadata.get('timestamp'),
                location={
                    'lat': metadata.get('latitude'),
                    'lon': metadata.get('longitude')
                } if 'latitude' in metadata else None,
                metadata=metadata,
                embedding_preview=embedding[:10] if len(embedding) >= 10 else embedding
            ))
        
        return samples
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving sample embeddings: {str(e)}"
        )


@app.get("/embeddings/search", response_model=List[EmbeddingSearchResult])
async def search_embeddings(
    query: str = Query(..., description="Natural language search query"),
    limit: int = Query(default=5, ge=1, le=20, description="Number of results to return")
):
    """
    Perform semantic search in the embeddings database.
    
    Args:
        query: Natural language query (e.g., "temperature in Slovakia", "precipitation patterns")
        limit: Maximum number of results to return (1-20)
    
    Returns:
        List of most relevant embeddings with similarity scores
    
    This demonstrates the RAG capability by:
    - Converting natural language to embeddings
    - Finding semantically similar climate data
    - Ranking results by relevance
    
    Example queries:
    - "temperature anomalies"
    - "heavy precipitation events"
    - "wind speed patterns"
    """
    try:
        from src.embeddings.search import SemanticSearcher

        searcher = SemanticSearcher()
        results = searcher.search(query, k=limit)

        if not results:
            return []

        search_results = []
        for result in results:
            metadata = result.get('metadata', {}) or {}
            location = None
            if 'latitude' in metadata and 'longitude' in metadata:
                location = {
                    'lat': metadata.get('latitude'),
                    'lon': metadata.get('longitude')
                }
            search_results.append(EmbeddingSearchResult(
                id=result.get('id', ''),
                variable=metadata.get('variable', 'unknown'),
                timestamp=metadata.get('timestamp'),
                location=location,
                metadata=metadata,
                similarity_score=float(result.get('similarity', 0.0))
            ))

        return search_results

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing semantic search: {str(e)}"
        )


@app.post("/rag/chat", response_model=RAGChatResponse)
async def rag_chat(request: RAGChatRequest):
    """Run a lightweight RAG flow over the stored embeddings."""
    try:
        from src.embeddings.search import SemanticSearcher

        searcher = SemanticSearcher()
        hits = searcher.search(request.question, k=request.top_k)
        if not hits:
            raise HTTPException(status_code=404, detail="No relevant context found in embeddings store")

        answer, references = summarize_hits(hits)
        chunks = [
            RAGChunk(
                source_id=(hit.get('metadata') or {}).get('source_id', 'source'),
                variable=(hit.get('metadata') or {}).get('variable'),
                similarity=float(hit.get('similarity', 0.0)),
                text=hit.get('text') or hit.get('document') or "",
                metadata=hit.get('metadata') or {},
            )
            for hit in hits
        ]

        return RAGChatResponse(
            question=request.question,
            answer=answer,
            references=references,
            chunks=chunks,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running RAG chat: {str(e)}")


# ====================================================================================
# APPLICATION STARTUP
# ====================================================================================

@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler.
    """
    print("=" * 80)
    print("Climate ETL Pipeline API Starting...")
    print(f"Dagster GraphQL endpoint: {DAGSTER_GRAPHQL_URL}")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown event handler.
    """
    print("Climate ETL Pipeline API Shutting down...")


# ====================================================================================
# MAIN (for running with uvicorn)
# ====================================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
