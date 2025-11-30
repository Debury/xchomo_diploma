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
# SOURCE MANAGEMENT MODELS
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
        description="Geographic bounding box: [south, west, north, east]. If None, entire available area will be used."
    )
    is_active: bool = Field(
        True, 
        description="Whether this source should be actively processed by the ETL pipeline."
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
                "description": "Sample air temperature data",
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
    variable: str = Field(..., description="Climate variable name")
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
    use_llm: bool = Field(True, description="Use LLM for answer generation (requires Ollama)")
    temperature: float = Field(0.3, ge=0.0, le=1.0, description="LLM temperature for response generation")


class RAGChatResponse(BaseModel):
    question: str
    answer: str
    references: List[str]
    chunks: List[RAGChunk]
    llm_used: bool = Field(False, description="Whether LLM was used for answer generation")


class SourceResponse(BaseModel):
    """Response with source information"""
    id: int
    source_id: str
    url: str
    format: str
    variables: Optional[List[str]]
    time_range: Optional[Dict[str, str]]
    spatial_bbox: Optional[List[float]]
    transformations: Optional[List[str]]
    aggregation_method: Optional[str]
    output_resolution: Optional[float]
    embedding_model: Optional[str]
    chunk_size: Optional[int]
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

_sample_dir_env = os.getenv("SAMPLE_DATA_DIR")
if _sample_dir_env:
    SAMPLE_DATA_DIR = Path(_sample_dir_env)
else:
    SAMPLE_DATA_DIR = Path(__file__).parent.parent / "data" / "raw" # Pointing to local raw data
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

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
    """Execute a GraphQL query against Dagster."""
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

async def get_dagster_repo_info():
    """Fetch repository location and name from Dagster."""
    repo_query = """
    query GetRepositories {
        repositoriesOrError {
            __typename
            ... on RepositoryConnection {
                nodes {
                    name
                    location { name }
                }
            }
        }
    }
    """
    data = await execute_graphql_query(repo_query)
    repos = data.get("repositoriesOrError", {}).get("nodes", [])
    if not repos:
        raise HTTPException(status_code=404, detail="No Dagster repositories found")
    # Return first repo found
    return repos[0]["location"]["name"], repos[0]["name"]

async def launch_dagster_run(job_name: str, run_config: Dict, tags: Dict = None):
    """Launch a run using the first available repository."""
    repo_loc, repo_name = await get_dagster_repo_info()
    
    mutation = """
    mutation LaunchRun($repositoryLocationName: String!, $repositoryName: String!, $jobName: String!, $runConfigData: RunConfigData!, $tags: [ExecutionTag!]) {
        launchRun(
            executionParams: {
                selector: {
                    repositoryLocationName: $repositoryLocationName
                    repositoryName: $repositoryName
                    jobName: $jobName
                }
                runConfigData: $runConfigData
                executionMetadata: { tags: $tags }
            }
        ) {
            __typename
            ... on LaunchRunSuccess {
                run { runId status jobName }
            }
            ... on PythonError { message }
            ... on RunConfigValidationInvalid {
                errors { message }
            }
        }
    }
    """
    
    graphql_tags = [{"key": k, "value": v} for k, v in (tags or {}).items()]

    variables = {
        "repositoryLocationName": repo_loc,
        "repositoryName": repo_name,
        "jobName": job_name,
        "runConfigData": run_config or {},
        "tags": graphql_tags
    }
    
    result = await execute_graphql_query(mutation, variables)
    launch_result = result.get("launchRun", {})
    
    if launch_result.get("__typename") == "LaunchRunSuccess":
        return launch_result["run"]
    elif launch_result.get("__typename") == "PythonError":
        raise Exception(f"Python Error: {launch_result.get('message')}")
    elif launch_result.get("__typename") == "RunConfigValidationInvalid":
        errs = "; ".join([e["message"] for e in launch_result.get("errors", [])])
        raise Exception(f"Config Invalid: {errs}")
    else:
        raise Exception(f"Unexpected error: {launch_result}")

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
    return {
        "name": "Climate ETL Pipeline API",
        "version": "1.0.0",
        "description": "REST API for managing climate data workflows",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/ui", response_class=FileResponse)
async def serve_frontend():
    if not FRONTEND_DIR.exists():
        raise HTTPException(status_code=404, detail="Frontend assets not found.")
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    dagster_available = False
    try:
        await execute_graphql_query("{ __typename }")
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
    """List all available Dagster jobs."""
    try:
        # Fetch dynamically
        repo_loc, repo_name = await get_dagster_repo_info()
        
        query = """
        query GetPipelines($selector: RepositorySelector!) {
            repositoryOrError(repositorySelector: $selector) {
                ... on Repository {
                    pipelines {
                        name
                        description
                        tags { key value }
                    }
                }
            }
        }
        """
        variables = {
            "selector": {
                "repositoryLocationName": repo_loc,
                "repositoryName": repo_name
            }
        }
        
        data = await execute_graphql_query(query, variables)
        pipelines = data.get("repositoryOrError", {}).get("pipelines", [])
        
        jobs = []
        for p in pipelines:
            tags_dict = {t["key"]: t["value"] for t in p.get("tags", [])}
            jobs.append(JobInfo(
                name=p["name"],
                description=p.get("description"),
                tags=tags_dict,
                ops=[] # Fetching ops requires deeper query, keeping simpler for list
            ))
            
        return jobs
    
    except Exception as e:
        # Fallback to mock list if Dagster is down
        return [
            JobInfo(name="dynamic_source_etl_job", description="Process all active sources", tags={"type": "dynamic"}),
            JobInfo(name="daily_etl_job", description="Legacy ETL", tags={"type": "legacy"})
        ]

@app.post("/jobs/{job_name}/run", response_model=RunResponse)
async def trigger_job_run(
    job_name: str = PathParam(..., description="Name of the job to run"),
    run_request: RunRequest = None
):
    valid_jobs = ["daily_etl_job", "embedding_job", "complete_pipeline_job", "validation_job", "dynamic_source_etl_job"]
    if job_name not in valid_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_name}' not found.")
    
    config = run_request.run_config if run_request else {}
    tags = run_request.tags if run_request else {}
    
    try:
        run_data = await launch_dagster_run(job_name, config, tags)
        return RunResponse(
            run_id=run_data["runId"],
            job_name=run_data["jobName"],
            status=run_data["status"],
            message=f"Successfully triggered run for job '{job_name}'"
        )
    except Exception as e:
        status = 400 if "Config Invalid" in str(e) else 500
        raise HTTPException(status_code=status, detail=str(e))

@app.get("/runs/{run_id}/status", response_model=RunStatus)
async def get_run_status(run_id: str):
    query = """
    query GetRunStatus($runId: ID!) {
        runOrError(runId: $runId) {
            __typename
            ... on Run {
                runId status pipelineName startTime endTime
                tags { key value }
            }
            ... on RunNotFoundError { message }
        }
    }
    """
    try:
        data = await execute_graphql_query(query, {"runId": run_id})
        run_data = data.get("runOrError", {})
        
        if run_data.get("__typename") == "RunNotFoundError":
            raise HTTPException(status_code=404, detail=run_data["message"])
        
        if run_data.get("__typename") != "Run":
            raise HTTPException(status_code=500, detail="Unknown error fetching status")
            
        tags_dict = {t["key"]: t["value"] for t in run_data.get("tags", [])}
        
        return RunStatus(
            run_id=run_data["runId"],
            status=run_data["status"],
            job_name=run_data["pipelineName"],
            start_time=str(run_data.get("startTime")) if run_data.get("startTime") else None,
            end_time=str(run_data.get("endTime")) if run_data.get("endTime") else None,
            tags=tags_dict
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/runs", response_model=List[RunStatus])
async def list_recent_runs(limit: int = 10):
    query = """
    query ListRuns($limit: Int!) {
        runsOrError {
            ... on Runs {
                results(limit: $limit) {
                    runId status pipelineName startTime endTime
                    tags { key value }
                }
            }
        }
    }
    """
    try:
        data = await execute_graphql_query(query, {"limit": limit})
        runs_list = data.get("runsOrError", {}).get("results", [])
        
        results = []
        for r in runs_list:
            tags_dict = {t["key"]: t["value"] for t in r.get("tags", [])}
            results.append(RunStatus(
                run_id=r["runId"],
                status=r["status"],
                job_name=r["pipelineName"],
                start_time=str(r.get("startTime")),
                end_time=str(r.get("endTime")),
                tags=tags_dict
            ))
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ====================================================================================
# SOURCE MANAGEMENT ENDPOINTS
# ====================================================================================

@app.post("/sources", response_model=SourceResponse, status_code=201)
async def create_source(source: SourceCreate):
    try:
        from src.sources import get_source_store
        store = get_source_store()
        
        if store.get_source(source.source_id):
            raise HTTPException(status_code=400, detail=f"Source '{source.source_id}' already exists")
        
        if not source.format:
            from dagster_project.ops.dynamic_source_ops import detect_format_from_url
            source.format = detect_format_from_url(source.url)
        
        created_source = store.create_source(source.dict())
        return created_source.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources", response_model=List[SourceResponse])
async def list_sources(active_only: bool = True, tags: Optional[str] = None):
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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources/{source_id}", response_model=SourceResponse)
async def get_source(source_id: str):
    try:
        from src.sources import get_source_store
        store = get_source_store()
        source = store.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        return source.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/sources/{source_id}", response_model=SourceResponse)
async def update_source(source_id: str, updates: SourceUpdate):
    try:
        from src.sources import get_source_store
        store = get_source_store()
        
        if not store.get_source(source_id):
            raise HTTPException(status_code=404, detail="Source not found")
            
        update_data = updates.dict(exclude_unset=True)
        updated_source = store.update_source(source_id, update_data)
        return updated_source.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sources/{source_id}", status_code=204)
async def delete_source(source_id: str, hard_delete: bool = False):
    try:
        from src.sources import get_source_store
        from src.embeddings.database import VectorDatabase
        
        store = get_source_store()
        if not store.get_source(source_id):
            raise HTTPException(status_code=404, detail="Source not found")
        
        if hard_delete:
            store.hard_delete_source(source_id)
        else:
            store.delete_source(source_id)
            
        # Clean up vector DB
        try:
            db = VectorDatabase()
            db.delete_embeddings_by_source(source_id)
        except Exception:
            pass # Non-fatal if DB cleanup fails
            
        return None
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sources/{source_id}/trigger", response_model=RunResponse)
async def trigger_source_etl(
    source_id: str = PathParam(..., description="Source ID"),
    job_name: str = Query("dynamic_source_etl_job"),
    run_config: Optional[Dict[str, Any]] = None
):
    try:
        from src.sources import get_source_store
        store = get_source_store()
        source = store.get_source(source_id)
        
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        if not source.is_active:
            raise HTTPException(status_code=400, detail="Source is not active")
            
        # Config Logic
        job_config = {}
        if job_name != "dynamic_source_etl_job":
             job_config = {
                "ops": {
                    "download_era5_data": {
                        "config": {
                            "variables": source.variables or ["2m_temperature"],
                            "year": 2025,
                            "month": 1,
                            "area": source.spatial_bbox
                        }
                    }
                }
            }
        
        if run_config:
            job_config.update(run_config)
            
        run_data = await launch_dagster_run(
            job_name,
            job_config,
            tags={"source_id": source_id, "triggered_by": "api"}
        )
        
        store.update_processing_status(source_id, "processing")
        
        return RunResponse(
            run_id=run_data["runId"],
            job_name=run_data["jobName"],
            status=run_data["status"],
            message=f"Triggered {job_name} for source {source_id}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ====================================================================================
# EMBEDDINGS & RAG
# ====================================================================================

@app.get("/embeddings/stats", response_model=EmbeddingStatsResponse)
async def get_embeddings_stats():
    try:
        from src.embeddings.database import VectorDatabase
        db = VectorDatabase()
        collection = db.collection
        results = collection.get()
        
        count = len(results['ids']) if results['ids'] else 0
        metas = results.get('metadatas', []) or []
        
        variables = set()
        sources = set()
        timestamps = []
        
        for m in metas:
            if m:
                if 'variable' in m: variables.add(m['variable'])
                if 'source_id' in m: sources.add(m['source_id'])
                if 'timestamp' in m: timestamps.append(m['timestamp'])
                
        date_range = None
        if timestamps:
            date_range = {"earliest": min(timestamps), "latest": max(timestamps)}
            
        return EmbeddingStatsResponse(
            total_embeddings=count,
            variables=sorted(list(variables)),
            date_range=date_range,
            sources=sorted(list(sources)),
            collection_name=db.collection_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embeddings/clear")
async def clear_embeddings(confirm: bool = False):
    if not confirm:
        raise HTTPException(status_code=400, detail="Set confirm=true")
    try:
        from src.embeddings.database import VectorDatabase
        db = VectorDatabase()
        removed = db.clear_collection()
        return {"removed": removed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/embeddings/sample", response_model=List[EmbeddingResponse])
async def get_sample_embeddings(limit: int = 10):
    try:
        from src.embeddings.database import VectorDatabase
        db = VectorDatabase()
        results = db.collection.get(limit=limit, include=['embeddings', 'metadatas'])
        
        samples = []
        ids = results.get('ids', [])
        for i, eid in enumerate(ids):
            meta = results['metadatas'][i]
            emb = results['embeddings'][i]
            samples.append(EmbeddingResponse(
                id=eid,
                variable=meta.get('variable', 'unknown'),
                timestamp=meta.get('timestamp'),
                location={'lat': meta.get('latitude'), 'lon': meta.get('longitude')} if 'latitude' in meta else None,
                metadata=meta,
                embedding_preview=emb[:10]
            ))
        return samples
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/embeddings/search", response_model=List[EmbeddingSearchResult])
async def search_embeddings(query: str, limit: int = 5):
    try:
        from src.embeddings.search import SemanticSearcher
        searcher = SemanticSearcher()
        results = searcher.search(query, k=limit)
        
        out = []
        for r in results:
            m = r.get('metadata', {})
            out.append(EmbeddingSearchResult(
                id=r.get('id', ''),
                variable=m.get('variable', 'unknown'),
                timestamp=m.get('timestamp'),
                location={'lat': m.get('latitude'), 'lon': m.get('longitude')} if 'latitude' in m else None,
                metadata=m,
                similarity_score=r.get('similarity', 0.0)
            ))
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/chat", response_model=RAGChatResponse)
async def rag_chat(request: RAGChatRequest):
    try:
        from src.embeddings.search import SemanticSearcher
        searcher = SemanticSearcher()
        hits = searcher.search(request.question, k=request.top_k)
        
        if not hits:
            raise HTTPException(status_code=404, detail="No relevant context found.")
            
        llm_used = False
        answer = ""
        
        if request.use_llm:
            try:
                from src.llm import OllamaClient
                llm = OllamaClient()
                if await llm.check_health():
                    answer = await llm.generate_rag_answer(
                        question=request.question,
                        context_chunks=hits,
                        temperature=request.temperature
                    )
                    llm_used = True
            except Exception as e:
                print(f"LLM failure: {e}")
                
        if not llm_used:
            answer, _ = summarize_hits(hits)
            answer += " (Template-based fallback)"
            
        # Build references
        chunks = []
        refs = []
        seen = set()
        
        for h in hits:
            m = h.get('metadata', {})
            ref_str = f"{m.get('source_id')}:{m.get('variable')}"
            if ref_str not in seen:
                seen.add(ref_str)
                refs.append(ref_str)
            
            chunks.append(RAGChunk(
                source_id=m.get('source_id', ''),
                variable=m.get('variable'),
                similarity=h.get('similarity', 0.0),
                text=h.get('text', ''),
                metadata=m
            ))
            
        return RAGChatResponse(
            question=request.question,
            answer=answer,
            references=refs,
            chunks=chunks,
            llm_used=llm_used
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ====================================================================================
# SAMPLE DATA & STARTUP
# ====================================================================================

@app.get("/samples/{file_name:path}")
async def download_sample_data(file_name: str):
    """Serve files from data/raw for testing."""
    if ".." in file_name: raise HTTPException(400)
    file_path = SAMPLE_DATA_DIR / file_name
    if not file_path.exists():
        raise HTTPException(404, detail="File not found")
    return FileResponse(file_path)

@app.on_event("startup")
async def startup_event():
    print(f"Climate API Starting. Dagster at {DAGSTER_GRAPHQL_URL}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)