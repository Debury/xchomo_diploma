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
    name: str
    description: Optional[str] = None
    tags: Dict[str, str] = {}
    ops: List[str] = []

class RunRequest(BaseModel):
    run_config: Optional[Dict[str, Any]] = None
    tags: Optional[Dict[str, str]] = None

class RunStatus(BaseModel):
    run_id: str
    status: str
    job_name: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration: Optional[float] = None
    tags: Dict[str, str] = {}

class RunResponse(BaseModel):
    run_id: str
    job_name: str
    status: str
    message: str

class HealthResponse(BaseModel):
    status: str
    dagster_available: bool
    timestamp: str

class SourceCreate(BaseModel):
    source_id: str
    url: str
    format: Optional[str] = None
    variables: Optional[List[str]] = None
    spatial_bbox: Optional[List[float]] = None
    time_range: Optional[Dict[str, str]] = None
    is_active: bool = True
    embedding_model: Optional[str] = "all-MiniLM-L6-v2"
    description: Optional[str] = None
    tags: Optional[List[str]] = None

class SourceUpdate(BaseModel):
    url: Optional[str] = None
    is_active: Optional[bool] = None

class EmbeddingResponse(BaseModel):
    id: str
    variable: str
    timestamp: Optional[str] = None
    location: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any]
    embedding_preview: Optional[List[float]] = None

class EmbeddingStatsResponse(BaseModel):
    total_embeddings: int
    variables: List[str]
    date_range: Optional[Dict[str, str]]
    sources: List[str]
    collection_name: str

class EmbeddingSearchResult(BaseModel):
    id: str
    variable: str
    timestamp: Optional[str]
    location: Optional[Dict[str, float]]
    metadata: Dict[str, Any]
    similarity_score: float

class RAGChunk(BaseModel):
    source_id: str
    variable: Optional[str]
    similarity: float
    text: str
    metadata: Dict[str, Any]

class RAGChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = None  # If None, will be auto-determined based on query type
    use_llm: bool = True
    temperature: float = 0.3
    # Optional filters for narrowing search results
    source_id: Optional[str] = None  # Filter by source (e.g., "NOAA_GSOM")
    variable: Optional[str] = None  # Filter by variable (e.g., "TMAX", "TMIN")

class RAGChatResponse(BaseModel):
    question: str
    answer: str
    references: List[str]
    chunks: List[RAGChunk]
    llm_used: bool

class SourceResponse(BaseModel):
    source_id: str
    url: str
    format: Optional[str] = None
    variables: Optional[List[str]] = None
    is_active: bool
    processing_status: str
    error_message: Optional[str] = None
    # Optional fields for compatibility
    id: Optional[int] = None
    collection_name: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    created_by: Optional[str] = None
    last_processed: Optional[str] = None


# ====================================================================================
# FASTAPI APPLICATION
# ====================================================================================

app = FastAPI(
    title="Climate ETL Pipeline API",
    version="1.0.0"
)

FRONTEND_DIR = Path(__file__).parent / "frontend"
SAMPLE_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/ui/static", StaticFiles(directory=FRONTEND_DIR, html=False), name="rag-ui-static")

DAGSTER_HOST = os.getenv("DAGSTER_HOST", "localhost")
DAGSTER_PORT = os.getenv("DAGSTER_PORT", "3000")
DAGSTER_GRAPHQL_URL = f"http://{DAGSTER_HOST}:{DAGSTER_PORT}/graphql"


# ====================================================================================
# HELPER FUNCTIONS
# ====================================================================================

async def execute_graphql_query(query: str, variables: Optional[Dict] = None) -> Dict:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                DAGSTER_GRAPHQL_URL,
                json={"query": query, "variables": variables or {}},
                timeout=30.0
            )
            return response.json().get("data", {})
        except Exception as e:
            print(f"Dagster Error: {e}")
            raise HTTPException(status_code=503, detail="Dagster unavailable")

async def launch_dagster_run(job_name: str, run_config: Dict, tags: Dict = None):
    # Fetch repo info
    repo_data = await execute_graphql_query("""
    query { repositoriesOrError { ... on RepositoryConnection { nodes { name location { name } } } } }
    """)
    nodes = repo_data.get("repositoriesOrError", {}).get("nodes", [])
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
            "jobName": job_name
        },
        "config": run_config or {},
        "tags": [{"key": k, "value": v} for k, v in (tags or {}).items()]
    }
    
    result = await execute_graphql_query(mutation, variables)
    launch_res = result.get("launchRun", {})
    
    if launch_res.get("__typename") == "LaunchRunSuccess":
        return {"runId": launch_res["run"]["runId"], "status": launch_res["run"]["status"], "jobName": job_name}
    
    err_msg = launch_res.get("message") or str(launch_res.get("errors"))
    raise HTTPException(status_code=500, detail=f"Job launch failed: {err_msg}")

def summarize_hits(results: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    if not results: return "No context found.", []
    refs = []
    seen = set()
    for h in results:
        m = h.get('metadata', {})
        ref = f"{m.get('source_id', '?')}:{m.get('variable', '?')}"
        if ref not in seen:
            seen.add(ref)
            refs.append(ref)
    return f"Found {len(results)} relevant data points.", refs

# ====================================================================================
# ENDPOINTS
# ====================================================================================

@app.get("/")
async def root():
    return {"status": "ok", "docs": "/docs"}

@app.get("/ui", response_class=FileResponse)
async def serve_frontend():
    if not FRONTEND_DIR.exists():
        raise HTTPException(404, "Frontend not found")
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    dagster_up = False
    try:
        await execute_graphql_query("{ __typename }")
        dagster_up = True
    except: pass
    return HealthResponse(status="healthy", dagster_available=dagster_up, timestamp=datetime.now().isoformat())

# --- EMBEDDINGS (QDRANT) ---

@app.get("/embeddings/stats", response_model=EmbeddingStatsResponse)
async def get_embeddings_stats():
    """Get stats from Qdrant."""
    try:
        from src.embeddings.database import VectorDatabase
        from src.utils.config_loader import ConfigLoader
        # Load config for proper vector size and collection name
        config_loader = ConfigLoader("config/pipeline_config.yaml")
        pipeline_config = config_loader.load()
        db = VectorDatabase(config=pipeline_config)
        
        # FIX: Check if collection exists first
        collections = db.client.get_collections().collections
        exists = any(c.name == db.collection for c in collections)
        
        if not exists:
            return EmbeddingStatsResponse(
                total_embeddings=0, variables=[], date_range=None, sources=[], collection_name=db.collection
            )

        # FIX: Use Qdrant Client API properly
        count_res = db.client.count(collection_name=db.collection)
        
        # Scroll to get metadata samples
        points, _ = db.client.scroll(
            collection_name=db.collection,
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        
        variables = set()
        sources = set()
        timestamps = []
        
        for point in points:
            payload = point.payload or {}
            if 'variable' in payload: variables.add(payload['variable'])
            if 'source_id' in payload: sources.add(payload['source_id'])
            if 'timestamp' in payload: timestamps.append(payload['timestamp'])
            
        date_range = None
        if timestamps:
            date_range = {"earliest": min(timestamps), "latest": max(timestamps)}
            
        return EmbeddingStatsResponse(
            total_embeddings=count_res.count,
            variables=sorted(list(variables)),
            date_range=date_range,
            sources=sorted(list(sources)),
            collection_name=db.collection
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/embeddings/sample", response_model=List[EmbeddingResponse])
async def get_sample_embeddings(limit: int = 10):
    try:
        from src.embeddings.database import VectorDatabase
        from src.utils.config_loader import ConfigLoader
        # Load config for proper vector size and collection name
        config_loader = ConfigLoader("config/pipeline_config.yaml")
        pipeline_config = config_loader.load()
        db = VectorDatabase(config=pipeline_config)
        
        # FIX: Use Qdrant scroll
        points, _ = db.client.scroll(
            collection_name=db.collection,
            limit=limit,
            with_payload=True,
            with_vectors=True
        )
        
        samples = []
        for point in points:
            meta = point.payload or {}
            samples.append(EmbeddingResponse(
                id=str(point.id),
                variable=meta.get('variable', 'unknown'),
                timestamp=meta.get('timestamp'),
                location=None, # Extract lat/lon if available in meta
                metadata=meta,
                embedding_preview=point.vector[:10] if point.vector else []
            ))
        return samples
    except Exception as e:
        # Return empty list if collection doesn't exist yet
        return []

@app.post("/embeddings/clear")
async def clear_embeddings(confirm: bool = False):
    if not confirm: raise HTTPException(400, "Set confirm=true")
    from src.embeddings.database import VectorDatabase
    from src.utils.config_loader import ConfigLoader
    # Load config for proper collection name
    config_loader = ConfigLoader("config/pipeline_config.yaml")
    pipeline_config = config_loader.load()
    db = VectorDatabase(config=pipeline_config)
    db.client.delete_collection(db.collection)
    return {"status": "cleared", "collection": db.collection}

# --- SOURCES ---

@app.post("/sources", response_model=SourceResponse, status_code=201)
async def create_source(source: SourceCreate):
    try:
        from src.sources import get_source_store
        store = get_source_store()
        
        if not source.format:
            from src.climate_embeddings.loaders.detect_format import detect_format_from_url
            source.format = detect_format_from_url(source.url)
            
        new_source = store.create_source(source.dict())
        return new_source.to_dict()
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/sources", response_model=List[SourceResponse])
async def list_sources(active_only: bool = True):
    from src.sources import get_source_store
    return [s.to_dict() for s in get_source_store().get_all_sources(active_only)]

@app.post("/sources/{source_id}/trigger", response_model=RunResponse)
async def trigger_source_etl(source_id: str, job_name: str = "dynamic_source_etl_job"):
    from src.sources import get_source_store
    store = get_source_store()
    source = store.get_source(source_id)
    
    if not source: raise HTTPException(404, "Source not found")
    
    # Launch logic
    run = await launch_dagster_run(job_name, {}, tags={"source_id": source_id})
    store.update_processing_status(source_id, "processing")
    
    return RunResponse(
        run_id=run["runId"],
        job_name=run["jobName"],
        status=run["status"],
        message="Job triggered"
    )

@app.delete("/sources/{source_id}", status_code=204)
async def delete_source(source_id: str):
    from src.sources import get_source_store
    get_source_store().hard_delete_source(source_id)
    return None

# --- RAG ---

# ... (rest of file remains unchanged) ...

# --- RAG ---

@app.post("/rag/chat", response_model=RAGChatResponse)
async def rag_chat(request: RAGChatRequest):
    try:
        from src.climate_embeddings.embeddings.text_models import TextEmbedder
        from src.llm.ollama_client import OllamaClient
        from src.embeddings.database import VectorDatabase
        from src.utils.config_loader import ConfigLoader
        
        # Load config for VectorDatabase initialization
        config_loader = ConfigLoader(config_path="config/pipeline_config.yaml")
        pipeline_config = config_loader.load()
        
        db = VectorDatabase(config=pipeline_config)
        embedder = TextEmbedder()
        
        # 1. Detect query type for intelligent retrieval and dynamic top_k
        question_lower = request.question.lower()
        is_temp_range_query = any(phrase in question_lower for phrase in [
            "temperature range", "temp range", "range of temperature",
            "min and max temperature", "maximum and minimum temperature",
            "temperature min max", "temp min max"
        ])
        
        # Dynamic top_k based on query type (no hard limits - Qdrant handles large limits well)
        # For temperature range queries, we need more results to ensure TMAX+TMIN
        # For simple queries, smaller top_k is sufficient
        if request.top_k is None:
            if is_temp_range_query:
                top_k = 20  # Higher for multi-variable queries
            elif any(word in question_lower for word in ["compare", "difference", "versus", "vs", "both"]):
                top_k = 15  # Medium for comparison queries
            else:
                top_k = 10  # Default for simple queries
        else:
            top_k = request.top_k
        
        # 2. Build filter dict if filters provided
        filter_dict = {}
        if request.source_id:
            filter_dict["source_id"] = request.source_id
        if request.variable:
            filter_dict["variable"] = request.variable
        
        # 3. HYBRID SEARCH STRATEGY for temperature range queries
        # Best practice: Use query expansion + multiple targeted searches
        search_result = []
        
        if is_temp_range_query and not request.variable:
            # Strategy: Do 3 separate searches and merge results
            # 1. General semantic search (broader context)
            # 2. Targeted search for TMAX
            # 3. Targeted search for TMIN
            
            # Search 1: General semantic search (get broader context)
            general_query_vec = embedder.embed_queries([request.question])[0]
            general_results = db.search(
                query_vector=general_query_vec.tolist(),
                limit=top_k,
                filter_dict=filter_dict if filter_dict else None
            )
            search_result.extend(general_results)
            
            # Search 2: Targeted search for TMAX (query expansion)
            tmax_query = f"{request.question} maximum temperature TMAX"
            tmax_query_vec = embedder.embed_queries([tmax_query])[0]
            tmax_filter = filter_dict.copy()
            tmax_filter["variable"] = "TMAX"
            tmax_results = db.search(
                query_vector=tmax_query_vec.tolist(),
                limit=min(8, top_k),  # Get top 8 TMAX results (more to ensure we find it)
                filter_dict=tmax_filter
            )
            search_result.extend(tmax_results)
            
            # Search 3: Targeted search for TMIN (query expansion)
            tmin_query = f"{request.question} minimum temperature TMIN"
            tmin_query_vec = embedder.embed_queries([tmin_query])[0]
            tmin_filter = filter_dict.copy()
            tmin_filter["variable"] = "TMIN"
            tmin_results = db.search(
                query_vector=tmin_query_vec.tolist(),
                limit=min(8, top_k),  # Get top 8 TMIN results (more to ensure we find it)
                filter_dict=tmin_filter
            )
            search_result.extend(tmin_results)
            
            # Deduplicate by variable + time + location (same chunk shouldn't appear twice)
            seen_chunks = set()
            deduplicated_results = []
            for hit in search_result:
                meta = hit.payload if hasattr(hit, 'payload') else (hit if isinstance(hit, dict) else {})
                var = meta.get('variable', '') if isinstance(meta, dict) else ''
                time = meta.get('time_start', '') if isinstance(meta, dict) else ''
                source = meta.get('source_id', '') if isinstance(meta, dict) else ''
                chunk_key = f"{source}:{var}:{time}"
                
                if chunk_key not in seen_chunks:
                    seen_chunks.add(chunk_key)
                    deduplicated_results.append(hit)
            
            # Re-sort by score (best matches first)
            search_result = sorted(
                deduplicated_results,
                key=lambda x: getattr(x, 'score', 0.0) if hasattr(x, 'score') else (x.get('score', 0.0) if isinstance(x, dict) else 0.0),
                reverse=True
            )[:top_k]
            
        else:
            # Standard semantic search for non-temperature-range queries
            query_vec = embedder.embed_queries([request.question])[0]
            search_result = db.search(
                query_vector=query_vec.tolist(),
                limit=top_k,
                filter_dict=filter_dict if filter_dict else None
            )
        
        # 3. Build Context
        context_chunks = []
        chunks_model = []
        refs = set()
        
        for hit in search_result:
            # Handle both ScoredPoint (client) and ScoredResult (REST fallback)
            # Both should have .payload and .score attributes
            if hasattr(hit, 'payload'):
                meta = hit.payload if isinstance(hit.payload, dict) else {}
            else:
                # Fallback: try to get from dict or default
                meta = getattr(hit, 'payload', {}) if hasattr(hit, 'payload') else (hit if isinstance(hit, dict) else {})
            
            score = getattr(hit, 'score', 0.0) if hasattr(hit, 'score') else (hit.get('score', 0.0) if isinstance(hit, dict) else 0.0)
            text = meta.get('text_content', '') if isinstance(meta, dict) else str(meta)
            
            context_chunks.append({"metadata": meta, "score": score})
            
            # Safely get variables
            s_id = meta.get('source_id', 'unknown')
            var = meta.get('variable', 'unknown')
            
            ref = f"{s_id}:{var}"
            refs.add(ref)
            
            chunks_model.append(RAGChunk(
                source_id=s_id,
                variable=var,
                similarity=score,
                text=text[:200] + "...",
                metadata=meta
            ))

        # 4. LLM Answer
        llm_used = False
        answer = ""
        
        if request.use_llm:
            try:
                client = OllamaClient()
                if client.check_health():
                    answer = client.generate_rag_answer(request.question, context_chunks, request.temperature)
                    llm_used = True
            except Exception as e:
                print(f"LLM Error: {e}")
        
        if not answer:
            if not context_chunks:
                answer = "No relevant climate data found."
            else:
                answer = f"Found {len(context_chunks)} relevant records. (LLM disabled or unavailable)"
            
        return RAGChatResponse(
            question=request.question,
            answer=answer,
            references=list(refs),
            chunks=chunks_model,
            llm_used=llm_used
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)