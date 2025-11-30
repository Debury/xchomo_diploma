"""
FastAPI Web Service for Climate ETL Pipeline
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
# PYDANTIC MODELS (Simplified for brevity)
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
    top_k: int = 3
    use_llm: bool = True
    temperature: float = 0.3

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
    id: Optional[int] = None
    collection_name: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    created_by: Optional[str] = None
    last_processed: Optional[str] = None


# ====================================================================================
# FASTAPI APPLICATION
# ====================================================================================

app = FastAPI(title="Climate ETL Pipeline API", version="1.0.0")

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

# ====================================================================================
# ENDPOINTS
# ====================================================================================

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
    """Get stats from Qdrant, using climate dates instead of ingest dates."""
    try:
        from src.embeddings.database import VectorDatabase
        db = VectorDatabase()
        
        # Check collection existence safely
        if db.client:
            try:
                collections = db.client.get_collections().collections
                if not any(c.name == db.collection for c in collections):
                    raise Exception("No collection")
                count_res = db.client.count(collection_name=db.collection)
                count = count_res.count
            except:
                return EmbeddingStatsResponse(
                    total_embeddings=0, variables=[], date_range=None, sources=[], collection_name=db.collection
                )
        else:
             # Fallback if client init failed
             return EmbeddingStatsResponse(
                total_embeddings=0, variables=[], date_range=None, sources=[], collection_name="offline"
            )

        # Scroll to get sample metadata
        points, _ = db.client.scroll(
            collection_name=db.collection,
            limit=1000, # Increased limit to see more dates
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
            
            # --- FIX: DATE RANGE ---
            # Prefer 'time_start' (Actual Climate Data) over 'timestamp' (System Time)
            t_val = payload.get('time_start')
            if not t_val: 
                t_val = payload.get('timestamp')
            
            if t_val:
                # Clean up format (e.g. remove T00:00...) for sorting
                timestamps.append(str(t_val).split('T')[0])
            
        date_range = None
        if timestamps:
            timestamps.sort()
            date_range = {"earliest": timestamps[0], "latest": timestamps[-1]}
            
        return EmbeddingStatsResponse(
            total_embeddings=count,
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
        db = VectorDatabase()
        if not db.client: return []
        
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
                timestamp=meta.get('time_start') or meta.get('timestamp'),
                location=None,
                metadata=meta,
                embedding_preview=point.vector[:10] if point.vector else []
            ))
        return samples
    except:
        return []

@app.post("/embeddings/clear")
async def clear_embeddings(confirm: bool = False):
    if not confirm: raise HTTPException(400, "Set confirm=true")
    from src.embeddings.database import VectorDatabase
    db = VectorDatabase()
    if db.client: db.client.delete_collection(db.collection)
    return {"status": "cleared"}

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
    
    run = await launch_dagster_run(job_name, {}, tags={"source_id": source_id})
    store.update_processing_status(source_id, "processing")
    
    return RunResponse(run_id=run["runId"], job_name=run["jobName"], status=run["status"], message="Job triggered")

@app.delete("/sources/{source_id}", status_code=204)
async def delete_source(source_id: str):
    from src.sources import get_source_store
    get_source_store().hard_delete_source(source_id)
    return None

# --- RAG ---

@app.post("/rag/chat", response_model=RAGChatResponse)
async def rag_chat(request: RAGChatRequest):
    try:
        from src.climate_embeddings.embeddings.text_models import TextEmbedder
        from src.llm.ollama_client import OllamaClient
        from src.embeddings.database import VectorDatabase
        
        db = VectorDatabase()
        
        # --- FEATURE: HYBRID SEARCH FOR "MAX/MIN" ---
        query_lower = request.question.lower()
        forced_chunks = []
        is_hybrid = False
        
        # If user asks for extremes, perform a SORT instead of just semantic search
        if "max" in query_lower or "hottest" in query_lower or "highest" in query_lower:
            # Fetch ALL points (up to 2000) and sort by stat_max in Python
            # (Qdrant payload sorting is more complex to setup dynamically, this is safer for <10k items)
            points, _ = db.client.scroll(
                collection_name=db.collection,
                limit=2000, 
                with_payload=True,
                with_vectors=False
            )
            # Sort: items with higher stat_max first
            sorted_points = sorted(
                points, 
                key=lambda p: p.payload.get('stat_max', -9999), 
                reverse=True
            )
            # Take top 5 actual hottest chunks
            for point in sorted_points[:5]:
                forced_chunks.append({
                    "metadata": point.payload, 
                    "score": 1.0  # Artificial high score
                })
            is_hybrid = True
            
        elif "min" in query_lower or "coldest" in query_lower or "lowest" in query_lower:
            points, _ = db.client.scroll(
                collection_name=db.collection,
                limit=2000, 
                with_payload=True,
                with_vectors=False
            )
            # Sort: items with lower stat_min first
            sorted_points = sorted(
                points, 
                key=lambda p: p.payload.get('stat_min', 9999), 
                reverse=False
            )
            # Take top 5 actual coldest chunks
            for point in sorted_points[:5]:
                forced_chunks.append({
                    "metadata": point.payload, 
                    "score": 1.0
                })
            is_hybrid = True

        # --- STANDARD VECTOR SEARCH ---
        embedder = TextEmbedder()
        query_vec = embedder.embed_queries([request.question])[0]
        
        # Use our safe wrapper from database.py (or fallback logic)
        search_result = db.search(
            query_vector=query_vec.tolist(),
            limit=request.top_k
        )
        
        # --- COMBINE RESULTS ---
        # If we found "forced" chunks (extreme values), prioritize them
        if is_hybrid:
            # Prepend the sorted chunks to the vector results
            # (In a real system, you'd dedup, but here it ensures the LLM sees the extremes)
            context_chunks = forced_chunks
        else:
            context_chunks = []
            for hit in search_result:
                # Handle both ScoredPoint (client) and dict (rest fallback)
                meta = hit.payload if hasattr(hit, 'payload') else hit.get('payload')
                score = hit.score if hasattr(hit, 'score') else hit.get('score')
                context_chunks.append({"metadata": meta, "score": score})

        # --- BUILD RESPONSE ---
        chunks_model = []
        refs = set()
        
        for c in context_chunks:
            meta = c["metadata"]
            text = meta.get('text_content', str(meta))
            
            ref = f"{meta.get('source_id')}:{meta.get('variable')}"
            refs.add(ref)
            
            chunks_model.append(RAGChunk(
                source_id=meta.get('source_id', 'unknown'),
                variable=meta.get('variable'),
                similarity=c["score"],
                text=text[:200] + "...",
                metadata=meta
            ))

        llm_used = False
        answer = ""
        
        if request.use_llm:
            try:
                client = OllamaClient()
                if client.check_health():
                    answer = client.generate_rag_answer(request.question, context_chunks, request.temperature)
                    llm_used = True
            except: pass
        
        if not answer:
            answer = f"Found {len(context_chunks)} relevant records."
            
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