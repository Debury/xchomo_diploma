"""
FastAPI Web Service for Climate ETL Pipeline

Provides REST API endpoints for interacting with Dagster pipelines.
Endpoints allow listing jobs, triggering runs, and checking status.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import httpx

logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ====================================================================================
# PERSISTENT SETTINGS
# ====================================================================================

SETTINGS_PATH = Path(__file__).parent.parent / "data" / "app_settings.json"

# Credential keys that map to environment variables
CREDENTIAL_KEYS = {
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "groq_api_key": "GROQ_API_KEY",
    "cds_api_key": "CDS_API_KEY",
    "nasa_earthdata_token": "NASA_EARTHDATA_TOKEN",
    "cmems_username": "CMEMS_USERNAME",
    "cmems_password": "CMEMS_PASSWORD",
}


def _load_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_settings(data: dict):
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, indent=2))


# Load persisted settings on startup
_persisted = _load_settings()
_runtime_settings = {k: v for k, v in _persisted.get("llm", {}).items()}

# Restore credentials into os.environ
for cred_key, env_var in CREDENTIAL_KEYS.items():
    value = _persisted.get("credentials", {}).get(cred_key, "")
    if value:
        os.environ[env_var] = value

# Restore LLM env vars
if "model" in _runtime_settings:
    os.environ["LLM_MODEL"] = str(_runtime_settings["model"])
if "temperature" in _runtime_settings:
    os.environ["LLM_TEMPERATURE"] = str(_runtime_settings["temperature"])


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
    auth_method: Optional[str] = None  # none, api_key, bearer_token, basic
    auth_credentials: Optional[Dict[str, str]] = None
    portal: Optional[str] = None  # CDS, NASA, MARINE, ESGF, NOAA

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
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    auth_method: Optional[str] = None
    portal: Optional[str] = None
    # Optional fields for compatibility
    id: Optional[int] = None
    collection_name: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    created_by: Optional[str] = None
    last_processed: Optional[str] = None


class AuthRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    message: str
    username: Optional[str] = None


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

# Import optimized RAG endpoint
from web_api.rag_endpoint import rag_query, simple_search, get_collection_info, RAGRequest, RAGResponse

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
    """Root endpoint - redirects to Vue app"""
    # Serve Vue app at root instead of JSON
    if VUE_DIST_DIR.exists():
        index_file = VUE_DIST_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
    # Fallback to JSON if Vue app not built
    return {"status": "ok", "docs": "/docs", "ui": "/app"}

@app.get("/ui", response_class=FileResponse)
async def serve_frontend():
    if not FRONTEND_DIR.exists():
        raise HTTPException(404, "Frontend not found")
    # Serve legacy UI for backwards compatibility
    legacy_file = FRONTEND_DIR / "legacy-index.html"
    if legacy_file.exists():
        return FileResponse(legacy_file)
    return FileResponse(FRONTEND_DIR / "index.html")

@app.get("/chat", response_class=FileResponse)
async def serve_chat():
    # Serve the new SPA chat interface
    spa_file = FRONTEND_DIR / "spa.html"
    if not spa_file.exists():
        raise HTTPException(404, "Chat UI not found")
    return FileResponse(spa_file)

# Vue SPA dist directory
VUE_DIST_DIR = FRONTEND_DIR / "dist"

# Debug endpoint to check frontend paths
@app.get("/app/debug")
async def debug_frontend():
    """Debug endpoint to check frontend file paths"""
    import os
    return {
        "FRONTEND_DIR": str(FRONTEND_DIR),
        "FRONTEND_DIR_exists": FRONTEND_DIR.exists(),
        "VUE_DIST_DIR": str(VUE_DIST_DIR),
        "VUE_DIST_DIR_exists": VUE_DIST_DIR.exists(),
        "frontend_contents": os.listdir(FRONTEND_DIR) if FRONTEND_DIR.exists() else [],
        "dist_contents": os.listdir(VUE_DIST_DIR) if VUE_DIST_DIR.exists() else [],
        "cwd": os.getcwd()
    }

# Serve Vue SPA static assets (JS, CSS, etc.)
@app.get("/app/assets/{file_path:path}")
async def serve_vue_assets(file_path: str):
    """Serve Vue built assets"""
    asset_file = VUE_DIST_DIR / "assets" / file_path
    if asset_file.exists() and asset_file.is_file():
        return FileResponse(asset_file)
    raise HTTPException(404, f"Asset not found: {file_path}")

# Serve Vue SPA - catch-all for client-side routing
@app.get("/app")
@app.get("/app/{path:path}")
async def serve_vue_app(path: str = ""):
    """Serve Vue SPA - all routes return index.html for client-side routing"""
    if VUE_DIST_DIR.exists():
        # Production: serve built Vue app
        index_file = VUE_DIST_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
    raise HTTPException(404, "Vue app not built. Run 'npm run build' in frontend/")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    dagster_up = False
    try:
        await execute_graphql_query("{ __typename }")
        dagster_up = True
    except: pass
    return HealthResponse(status="healthy", dagster_available=dagster_up, timestamp=datetime.now().isoformat())


# --- AUTHENTICATION ---

import hashlib
import secrets

# Simple token storage (in production, use Redis or database)
_valid_tokens: Dict[str, str] = {}

@app.post("/auth/login", response_model=AuthResponse)
async def auth_login(request: AuthRequest):
    """
    Simple authentication endpoint.
    Credentials are read from AUTH_USERNAME and AUTH_PASSWORD environment variables.
    """
    expected_username = os.getenv("AUTH_USERNAME", "admin")
    expected_password = os.getenv("AUTH_PASSWORD", "climate2024")
    
    if request.username == expected_username and request.password == expected_password:
        # Generate a simple token
        token = secrets.token_urlsafe(32)
        _valid_tokens[token] = request.username
        return AuthResponse(
            success=True,
            token=token,
            message="Login successful",
            username=request.username
        )
    
    return AuthResponse(
        success=False,
        message="Invalid username or password"
    )

@app.post("/auth/logout")
async def auth_logout(token: str = Query(...)):
    """Invalidate a token"""
    if token in _valid_tokens:
        del _valid_tokens[token]
        return {"success": True, "message": "Logged out"}
    return {"success": False, "message": "Invalid token"}

@app.get("/auth/verify")
async def auth_verify(token: str = Query(...)):
    """Verify if a token is valid"""
    if token in _valid_tokens:
        return {"valid": True, "username": _valid_tokens[token]}
    return {"valid": False}


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
async def clear_embeddings(confirm: bool = False, delete_sources: bool = False):
    """
    Clear all embeddings from Qdrant. Optionally also delete all sources.
    
    Args:
        confirm: Must be True to proceed
        delete_sources: If True, also delete all sources
    """
    if not confirm:
        raise HTTPException(400, "Set confirm=true to clear embeddings")
    
    from src.utils.config_loader import ConfigLoader
    
    try:
        # Load config for proper collection name
        config_loader = ConfigLoader("config/pipeline_config.yaml")
        pipeline_config = config_loader.load()
        
        # Get collection name from config (don't use VectorDatabase to avoid auto-recreation)
        qdrant_config = pipeline_config.get("vector_db", {}).get("qdrant", {})
        collection_name = qdrant_config.get("collection_name", "climate_data")
        
        # Initialize client directly (don't use VectorDatabase to avoid _ensure_collection)
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_REST_PORT", 6333))
        from qdrant_client import QdrantClient
        client = QdrantClient(host=host, port=port, prefer_grpc=False, timeout=10)
        
        logger.info(f"Attempting to delete collection: {collection_name}")
        
        # Check if collection exists
        try:
            collections = client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            
            if exists:
                # Get count before deletion for verification
                try:
                    count_res = client.count(collection_name=collection_name)
                    points_before = count_res.count if hasattr(count_res, 'count') else 0
                    logger.info(f"Collection {collection_name} has {points_before} points")
                except Exception as e:
                    logger.warning(f"Could not count points: {e}")
                    points_before = 0
                
                # Delete the collection
                client.delete_collection(collection_name)
                logger.info(f"Deleted collection: {collection_name} (had {points_before} points)")
                
                # Verify deletion - wait a moment and check again
                import time
                time.sleep(0.5)  # Brief wait for deletion to complete
                
                collections_after = client.get_collections().collections
                still_exists = any(c.name == collection_name for c in collections_after)
                
                if still_exists:
                    logger.error(f"Collection {collection_name} still exists after deletion attempt!")
                    raise HTTPException(500, f"Failed to delete collection {collection_name}")
                else:
                    logger.info(f"Successfully verified collection {collection_name} was deleted")
            else:
                logger.warning(f"Collection {collection_name} does not exist")
                points_before = 0
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting collection: {e}", exc_info=True)
            raise HTTPException(500, f"Failed to delete collection: {str(e)}")
        
        # Clear RAG info cache after deletion
        try:
            import web_api.rag_endpoint as rag_module
            import time
            rag_module._CACHED_INFO = {}
            rag_module._CACHED_INFO_TS = 0
            logger.info("Cleared RAG info cache")
        except Exception as e:
            logger.warning(f"Could not clear cache: {e}")
        
        # Delete sources if requested
        sources_deleted = 0
        if delete_sources:
            from src.sources import get_source_store
            store = get_source_store()
            all_sources = store.get_all_sources(active_only=False)
            for source in all_sources:
                if store.hard_delete_source(source.source_id):
                    sources_deleted += 1
            logger.info(f"Deleted {sources_deleted} sources")
        
        return {
            "status": "cleared",
            "collection": collection_name,
            "points_deleted": points_before if exists else 0,
            "sources_deleted": sources_deleted if delete_sources else None,
            "message": f"Collection {collection_name} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in clear_embeddings: {e}", exc_info=True)
        raise HTTPException(500, f"Unexpected error: {str(e)}")

# --- SOURCES ---

@app.post("/sources", response_model=SourceResponse, status_code=201)
async def create_source(source: SourceCreate):
    try:
        from src.sources import get_source_store
        store = get_source_store()

        if not source.format:
            from src.climate_embeddings.loaders.detect_format import detect_format_from_url
            source.format = detect_format_from_url(source.url)

        # Extract auth fields before creating source (don't store credentials in source config)
        auth_method = source.auth_method
        auth_credentials = source.auth_credentials
        portal = source.portal

        source_data = source.dict()
        # Remove auth_credentials from source data (stored separately for security)
        source_data.pop("auth_credentials", None)

        new_source = store.create_source(source_data)
        source_dict = new_source.to_dict()

        # Store auth credentials in app_settings.json under source_credentials
        if auth_credentials and auth_method and auth_method != "none":
            persisted = _load_settings()
            source_creds = persisted.get("source_credentials", {})
            source_creds[source.source_id] = {
                "auth_method": auth_method,
                "credentials": auth_credentials,
                "portal": portal,
            }
            persisted["source_credentials"] = source_creds
            _save_settings(persisted)

        # Include auth_method and portal in response
        source_dict["auth_method"] = auth_method
        source_dict["portal"] = portal
        return source_dict
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/sources", response_model=List[SourceResponse])
async def list_sources(active_only: bool = True):
    from src.sources import get_source_store
    persisted = _load_settings()
    source_creds = persisted.get("source_credentials", {})
    results = []
    for s in get_source_store().get_all_sources(active_only):
        d = s.to_dict()
        cred_info = source_creds.get(d.get("source_id", ""), {})
        d["auth_method"] = d.get("auth_method") or cred_info.get("auth_method")
        d["portal"] = d.get("portal") or cred_info.get("portal")
        results.append(d)
    return results

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

@app.put("/sources/{source_id}")
async def update_source(source_id: str, updates: dict):
    """Update source metadata (url, format, description, is_active, auth)."""
    from src.sources import get_source_store
    store = get_source_store()
    source = store.get_source(source_id)
    if not source:
        raise HTTPException(404, f"Source '{source_id}' not found")

    # Handle auth fields separately
    auth_method = updates.pop("auth_method", None)
    auth_credentials = updates.pop("auth_credentials", None)
    portal = updates.pop("portal", None)

    allowed_fields = {"url", "format", "description", "is_active", "variables", "tags"}
    filtered = {k: v for k, v in updates.items() if k in allowed_fields}

    # Store/update auth credentials in app_settings.json
    if auth_method is not None or auth_credentials is not None or portal is not None:
        persisted = _load_settings()
        source_creds = persisted.get("source_credentials", {})
        existing = source_creds.get(source_id, {})
        if auth_method is not None:
            existing["auth_method"] = auth_method
        if auth_credentials is not None:
            existing["credentials"] = auth_credentials
        if portal is not None:
            existing["portal"] = portal
        source_creds[source_id] = existing
        persisted["source_credentials"] = source_creds
        _save_settings(persisted)

    if not filtered and auth_method is None and auth_credentials is None and portal is None:
        raise HTTPException(400, "No valid fields to update")

    if filtered:
        try:
            updated = store.update_source(source_id, filtered)
            result = updated.to_dict() if hasattr(updated, 'to_dict') else {"source_id": source_id, "updated": True}
        except AttributeError:
            source_dict = source.to_dict() if hasattr(source, 'to_dict') else source.__dict__
            source_dict.update(filtered)
            store.hard_delete_source(source_id)
            new_source = store.create_source(source_dict)
            result = new_source.to_dict() if hasattr(new_source, 'to_dict') else {"source_id": source_id, "updated": True}
    else:
        result = {"source_id": source_id, "updated": True}

    # Add auth info to response
    persisted = _load_settings()
    cred_info = persisted.get("source_credentials", {}).get(source_id, {})
    result["auth_method"] = cred_info.get("auth_method")
    result["portal"] = cred_info.get("portal")
    return result

@app.delete("/sources/{source_id}", status_code=204)
async def delete_source(source_id: str):
    from src.sources import get_source_store
    get_source_store().hard_delete_source(source_id)
    return None

@app.delete("/sources", status_code=200)
async def delete_all_sources(confirm: bool = False, delete_embeddings: bool = False):
    """
    Delete all sources. Optionally also delete embeddings from Qdrant.
    
    Args:
        confirm: Must be True to proceed
        delete_embeddings: If True, also delete embeddings from Qdrant for these sources
    """
    if not confirm:
        raise HTTPException(400, "Set confirm=true to delete all sources")
    
    from src.sources import get_source_store
    store = get_source_store()
    
    # Get all sources first
    all_sources = store.get_all_sources(active_only=False)
    source_ids = [s.source_id for s in all_sources]
    
    # Delete embeddings if requested
    embeddings_deleted_count = 0
    if delete_embeddings:
        try:
            from src.embeddings.database import VectorDatabase
            from src.utils.config_loader import ConfigLoader
            from qdrant_client import models
            
            config_loader = ConfigLoader("config/pipeline_config.yaml")
            pipeline_config = config_loader.load()
            
            # Initialize client directly without _ensure_collection to avoid auto-recreation
            host = os.getenv("QDRANT_HOST", "localhost")
            port = int(os.getenv("QDRANT_REST_PORT", 6333))
            from qdrant_client import QdrantClient
            client = QdrantClient(host=host, port=port, prefer_grpc=False, timeout=10)
            
            # Get collection name from config
            qdrant_config = pipeline_config.get("vector_db", {}).get("qdrant", {})
            collection_name = qdrant_config.get("collection_name", "climate_data")
            
            # Check if collection exists
            collections = client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)
            
            if exists and source_ids:
                # Delete points by source_id filter for each source
                for source_id in source_ids:
                    try:
                        result = client.delete(
                            collection_name=collection_name,
                            points_selector=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="source_id",
                                        match=models.MatchValue(value=source_id)
                                    )
                                ]
                            )
                        )
                        # Count deleted points if available
                        if hasattr(result, 'operation_id'):
                            embeddings_deleted_count += 1
                        logger.info(f"Deleted embeddings for source: {source_id}")
                    except Exception as e:
                        logger.warning(f"Failed to delete embeddings for {source_id}: {e}")
            elif exists:
                # If no source_ids but collection exists, delete entire collection
                try:
                    count_res = client.count(collection_name=collection_name)
                    points_before = count_res.count if hasattr(count_res, 'count') else 0
                    client.delete_collection(collection_name)
                    embeddings_deleted_count = points_before
                    logger.info(f"Deleted entire collection {collection_name} with {points_before} points")
                except Exception as e:
                    logger.error(f"Failed to delete collection: {e}")
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}", exc_info=True)
    
    # Delete all sources
    deleted_count = 0
    for source_id in source_ids:
        if store.hard_delete_source(source_id):
            deleted_count += 1
    
    # Clear RAG info cache after deletion
    try:
        import web_api.rag_endpoint as rag_module
        rag_module._CACHED_INFO = {}
        rag_module._CACHED_INFO_TS = 0
        logger.info("Cleared RAG info cache")
    except Exception as e:
        logger.warning(f"Could not clear cache: {e}")
    
    return {
        "status": "deleted",
        "sources_deleted": deleted_count,
        "embeddings_deleted": embeddings_deleted_count if delete_embeddings else None,
        "message": f"Deleted {deleted_count} source(s)" + (f" and {embeddings_deleted_count} embedding(s)" if delete_embeddings else "")
    }

@app.delete("/sources/{source_id}/embeddings", status_code=200)
async def delete_source_embeddings(source_id: str, confirm: bool = False):
    """
    Delete all embeddings for a specific source from Qdrant.
    
    Args:
        source_id: Source ID to delete embeddings for
        confirm: Must be True to proceed
    """
    if not confirm:
        raise HTTPException(400, "Set confirm=true to delete embeddings")
    
    from src.sources import get_source_store
    store = get_source_store()
    source = store.get_source(source_id)
    
    if not source:
        raise HTTPException(404, "Source not found")
    
    try:
        from src.utils.config_loader import ConfigLoader
        from qdrant_client import models, QdrantClient
        
        config_loader = ConfigLoader("config/pipeline_config.yaml")
        pipeline_config = config_loader.load()
        
        # Get collection name from config
        qdrant_config = pipeline_config.get("vector_db", {}).get("qdrant", {})
        collection_name = qdrant_config.get("collection_name", "climate_data")
        
        # Initialize client directly
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_REST_PORT", 6333))
        client = QdrantClient(host=host, port=port, prefer_grpc=False, timeout=10)
        
        # Delete points by source_id filter
        client.delete(
            collection_name=collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_id",
                        match=models.MatchValue(value=source_id)
                    )
                ]
            )
        )
        
        logger.info(f"Deleted embeddings for source {source_id} from collection {collection_name}")
        
        # Clear RAG info cache after deletion
        try:
            import web_api.rag_endpoint as rag_module
            rag_module._CACHED_INFO = {}
            rag_module._CACHED_INFO_TS = 0
            logger.info("Cleared RAG info cache")
        except Exception as e:
            logger.warning(f"Could not clear cache: {e}")
        
        return {
            "status": "deleted",
            "source_id": source_id,
            "message": f"Embeddings for source {source_id} deleted from collection {collection_name}"
        }
    except Exception as e:
        logger.error(f"Error deleting embeddings for {source_id}: {e}", exc_info=True)
        raise HTTPException(500, f"Failed to delete embeddings: {str(e)}")

# --- RAG ---

# ... (rest of file remains unchanged) ...

# --- RAG (OPTIMIZED) ---

@app.post("/rag/query", response_model=RAGResponse)
async def rag_query_endpoint(request: RAGRequest):
    """
    Optimized RAG endpoint with timeout handling.
    Fast vector search + optional LLM generation.
    """
    return await rag_query(request)

@app.get("/rag/search")
async def rag_search_only(query: str, top_k: int = 5, source_id: Optional[str] = None, variable: Optional[str] = None):
    """
    Fast vector search without LLM - for testing and debugging.
    """
    filters = {}
    if source_id:
        filters["source_id"] = source_id
    if variable:
        filters["variable"] = variable
    return await simple_search(query, top_k, filters if filters else None)

@app.get("/rag/info")
async def rag_info():
    """
    Get collection info: variables, sources, count. Instant - no embedding/LLM.
    """
    return await get_collection_info()

# --- RAG (LEGACY - for backwards compatibility) ---

@app.post("/rag/chat", response_model=RAGChatResponse)
async def rag_chat_legacy(request: RAGChatRequest):
    try:
        from src.climate_embeddings.embeddings.text_models import TextEmbedder
        from src.embeddings.database import VectorDatabase
        from src.utils.config_loader import ConfigLoader
        
        # Load config for VectorDatabase initialization
        config_loader = ConfigLoader(config_path="config/pipeline_config.yaml")
        pipeline_config = config_loader.load()
        
        db = VectorDatabase(config=pipeline_config)
        embedder = TextEmbedder()
        
        # 1. Detect query type and extract mentioned variables for intelligent retrieval
        question_lower = request.question.lower()
        
        # IMPORTANT: Check if this is a variable list question FIRST
        # Import the detection function from rag_endpoint
        from web_api.rag_endpoint import _is_variable_list_question, _get_variable_list, get_collection_info
        
        # Fast path: variable listing questions - ALWAYS get ALL variables from database
        if _is_variable_list_question(request.question):
            # Get ALL variables from database, not just from search results
            all_variables = _get_variable_list(db, force_refresh=False)
            if not all_variables:
                # Fallback: try to get from collection info
                try:
                    collection_info = await get_collection_info()
                    all_variables = collection_info.get("variables", [])
                except Exception as e:
                    logger.warning(f"Failed to get collection info: {e}")
                    all_variables = []
            
            # Get sources info
            sources_text = ""
            try:
                collection_info = await get_collection_info()
                sources = collection_info.get("sources", [])
                if sources:
                    sources_text = f" from {len(sources)} source(s): {', '.join(sources)}"
            except Exception as e:
                logger.warning(f"Failed to get sources info: {e}")
            
            # Build answer with ALL variables
            answer = f"Available climate variables{sources_text}:\n" + ", ".join(all_variables)
            
            # Still do a search to get some context chunks for the response
            # Build filter dict if filters provided
            filter_dict = {}
            if request.source_id:
                filter_dict["source_id"] = request.source_id
            if request.variable:
                filter_dict["variable"] = request.variable
            
            query_vec = embedder.embed_queries([request.question])[0]
            search_result = db.search(
                query_vector=query_vec.tolist(),
                limit=10,
                filter_dict=filter_dict if filter_dict else None
            )
            
            # Build chunks for response
            chunks_model = []
            refs = set()
            for hit in search_result[:10]:
                if hasattr(hit, 'payload'):
                    meta = hit.payload if isinstance(hit.payload, dict) else {}
                else:
                    meta = getattr(hit, 'payload', {}) if hasattr(hit, 'payload') else (hit if isinstance(hit, dict) else {})
                
                score = getattr(hit, 'score', 0.0) if hasattr(hit, 'score') else (hit.get('score', 0.0) if isinstance(hit, dict) else 0.0)
                text = meta.get('text_content', '') if isinstance(meta, dict) else str(meta)
                
                s_id = meta.get('source_id', 'unknown')
                var = meta.get('variable', 'unknown')
                ref = f"{s_id}:{var}"
                refs.add(ref)
                
                chunks_model.append(RAGChunk(
                    source_id=s_id,
                    variable=var,
                    similarity=score,
                    text=text[:200] + "..." if len(text) > 200 else text,
                    metadata=meta
                ))
            
            logger.info(f"Fast path: Returning {len(all_variables)} variables for variable list question")
            return RAGChatResponse(
                question=request.question,
                answer=answer,
                references=sorted(list(refs)),
                chunks=chunks_model,
                llm_used=False  # Don't use LLM for variable list questions
            )
        
        # Detect temperature range queries
        is_temp_range_query = any(phrase in question_lower for phrase in [
            "temperature range", "temp range", "range of temperature",
            "min and max temperature", "maximum and minimum temperature",
            "temperature min max", "temp min max"
        ])
        
        # Detect multi-variable queries (mentions multiple climate variables)
        # Common variable keywords to detect
        variable_keywords = {
            "temperature": ["temperature", "temp", "tmax", "tmin", "thermal"],
            "precipitation": ["precipitation", "rain", "rainfall", "prcp", "snow"],
            "wind": ["wind", "speed", "velocity", "awnd", "wdf", "wsf"],
            "pressure": ["pressure", "barometric", "msl"],
            "humidity": ["humidity", "moisture", "dewpoint"],
        }
        
        mentioned_variables = []
        for var_type, keywords in variable_keywords.items():
            if any(kw in question_lower for kw in keywords):
                mentioned_variables.append(var_type)
        
        is_multi_variable_query = len(mentioned_variables) > 1 or any(word in question_lower for word in [
            "compare", "difference", "versus", "vs", "both", "and", "also", "additionally"
        ])
        
        # Dynamic top_k based on query complexity
        if request.top_k is None:
            if is_temp_range_query:
                top_k = 25  # Higher for temperature range (need TMAX + TMIN)
            elif is_multi_variable_query:
                top_k = 30  # Higher for multi-variable queries (need all variables)
            elif any(word in question_lower for word in ["compare", "difference", "versus", "vs"]):
                top_k = 20  # Medium for comparison queries
            else:
                top_k = 15  # Default (increased from 10 for better coverage)
        else:
            top_k = request.top_k
        
        # 2. Build filter dict and spatial filter
        filter_dict = {}
        if request.source_id:
            filter_dict["source_id"] = request.source_id
        if request.variable:
            filter_dict["variable"] = request.variable

        # Spatial-aware filtering (Spatial-RAG, arXiv:2502.18470)
        # Extract geographic constraints from the query and apply as Qdrant
        # payload Range filters to ensure spatially correct retrieval.
        from web_api.spatial_filter import extract_spatial_intent, build_qdrant_filter
        spatial_intent = extract_spatial_intent(request.question)
        spatial_qdrant_filter = build_qdrant_filter(spatial_intent, filter_dict if filter_dict else None)
        if spatial_intent.region_name:
            logger.info(f"Spatial filter active: region='{spatial_intent.region_name}'")

        # 3. MULTI-QUERY RETRIEVAL STRATEGY
        # Best practice: Use query expansion + multiple targeted searches for complex queries
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
                query_filter=spatial_qdrant_filter,
            )
            search_result.extend(general_results)

            # Search 2: Targeted search for TMAX (query expansion)
            tmax_query = f"{request.question} maximum temperature TMAX"
            tmax_query_vec = embedder.embed_queries([tmax_query])[0]
            tmax_extra = filter_dict.copy()
            tmax_extra["variable"] = "TMAX"
            tmax_filter = build_qdrant_filter(spatial_intent, tmax_extra)
            tmax_results = db.search(
                query_vector=tmax_query_vec.tolist(),
                limit=min(8, top_k),
                query_filter=tmax_filter,
            )
            search_result.extend(tmax_results)

            # Search 3: Targeted search for TMIN (query expansion)
            tmin_query = f"{request.question} minimum temperature TMIN"
            tmin_query_vec = embedder.embed_queries([tmin_query])[0]
            tmin_extra = filter_dict.copy()
            tmin_extra["variable"] = "TMIN"
            tmin_filter = build_qdrant_filter(spatial_intent, tmin_extra)
            tmin_results = db.search(
                query_vector=tmin_query_vec.tolist(),
                limit=min(8, top_k),
                query_filter=tmin_filter,
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
            
        elif is_multi_variable_query and not request.variable:
            # Multi-variable query strategy: Perform multiple targeted searches
            # 1. General semantic search (broader context)
            # 2. Targeted searches for each mentioned variable type
            
            # Search 1: General semantic search
            general_query_vec = embedder.embed_queries([request.question])[0]
            general_results = db.search(
                query_vector=general_query_vec.tolist(),
                limit=top_k,
                query_filter=spatial_qdrant_filter,
            )
            search_result.extend(general_results)

            # Search 2-N: Targeted searches for each mentioned variable type
            # Use query expansion to improve retrieval for each variable
            for var_type in mentioned_variables:
                expanded_query = f"{request.question} {var_type}"
                expanded_query_vec = embedder.embed_queries([expanded_query])[0]

                var_results = db.search(
                    query_vector=expanded_query_vec.tolist(),
                    limit=min(10, top_k),
                    query_filter=spatial_qdrant_filter,
                )
                search_result.extend(var_results)
            
            # Deduplicate by variable + time + location
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
            # Standard semantic search for simple single-variable queries
            query_vec = embedder.embed_queries([request.question])[0]
            search_result = db.search(
                query_vector=query_vec.tolist(),
                limit=top_k,
                query_filter=spatial_qdrant_filter,
            )
        
        # Fallback: if spatial filter returned too few results, retry without it
        if len(search_result) < 2 and spatial_qdrant_filter is not None:
            logger.warning(
                f"Spatial filter returned only {len(search_result)} results, "
                "retrying without spatial constraints"
            )
            query_vec = embedder.embed_queries([request.question])[0]
            search_result = db.search(
                query_vector=query_vec.tolist(),
                limit=top_k,
                filter_dict=filter_dict if filter_dict else None,
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
                # Use OpenRouter instead of Ollama
                import os
                from src.llm.openrouter_client import OpenRouterClient
                from web_api.prompt_builder import build_rag_prompt, detect_question_type
                from web_api.rag_endpoint import _get_variable_list, get_collection_info
                
                if not os.getenv("OPENROUTER_API_KEY"):
                    raise ValueError("OPENROUTER_API_KEY not set")
                
                client = OpenRouterClient()
                
                # Detect question type
                question_type = detect_question_type(request.question)
                
                # CRITICAL: Always get ALL variables from database for context
                # This ensures LLM knows about ALL available variables, not just those in search results
                all_variables = _get_variable_list(db, force_refresh=False)
                if not all_variables:
                    try:
                        collection_info = await get_collection_info()
                        all_variables = collection_info.get("variables", [])
                    except Exception as e:
                        logger.warning(f"Failed to get collection info: {e}")
                        # Fallback: use variables from chunks
                        all_variables = sorted({c.get("variable") for c in context_chunks if c.get("variable")})
                
                # Get sources info (useful for all question types)
                sources = None
                try:
                    collection_info = await get_collection_info()
                    sources = collection_info.get("sources", [])
                except Exception as e:
                    logger.warning(f"Failed to get collection info: {e}")
                
                # MULTI-PROMPTING: First, select relevant variables (if we have multiple variables)
                selected_variables = None
                if all_variables and len(all_variables) > 3:
                    try:
                        from web_api.prompt_builder import build_variable_selection_prompt
                        import os
                        from src.llm.openrouter_client import OpenRouterClient
                        
                        # Extract variable meanings from chunks
                        var_meanings = {}
                        for chunk in context_chunks:
                            meta = chunk.get("metadata", {})
                            var = meta.get("variable", "")
                            if var:
                                long_name = meta.get("long_name") or meta.get("standard_name")
                                if long_name:
                                    var_meanings[var] = long_name
                        
                        # First prompt: Select relevant variables
                        var_selection_prompt = build_variable_selection_prompt(
                            question=request.question,
                            all_variables=all_variables,
                            var_meanings=var_meanings
                        )
                        
                        # Get variable selection from LLM
                        var_selection_client = OpenRouterClient()
                        var_selection_response = var_selection_client.generate(
                            prompt=var_selection_prompt,
                            temperature=0.1,
                            max_tokens=50,
                        )
                        
                        # Parse selected variables
                        selected_vars_text = var_selection_response.strip()
                        selected_variables = [v.strip() for v in selected_vars_text.split(",") if v.strip() in all_variables]
                        logger.info(f"Selected variables from first prompt: {selected_variables}")
                    except Exception as e:
                        logger.warning(f"Variable selection prompt failed: {e}, continuing without it")
                        selected_variables = None
                
                # Build dynamic prompt based on question type
                prompt, max_tokens = build_rag_prompt(
                    question=request.question,
                    context_chunks=context_chunks,
                    all_variables=all_variables,
                    sources=sources,
                    question_type=question_type,
                    selected_variables=selected_variables
                )
                
                logger.info(f"Question type: {question_type}, Max tokens: {max_tokens}")
                
                answer = client.generate(
                    prompt=prompt, 
                    temperature=request.temperature, 
                    max_tokens=max_tokens
                )
                llm_used = True
            except Exception as e:
                logger.error(f"LLM Error: {e}")
                answer = f"LLM Error: {e}"
        
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


@app.get("/debug/variable/{variable_name}")
async def debug_variable(variable_name: str):
    """Debug endpoint to check if a variable exists in database and get sample data"""
    try:
        from src.utils.config_loader import ConfigLoader
        from src.embeddings.database import VectorDatabase
        from src.climate_embeddings.embeddings.text_models import TextEmbedder
        
        config_loader = ConfigLoader("config/pipeline_config.yaml")
        config = config_loader.load()
        db = VectorDatabase(config=config)
        embedder = TextEmbedder()
        
        # Search for the variable with filter
        query_text = f"{variable_name} climate data"
        query_vec = embedder.embed_queries([query_text])[0]
        
        results_with_filter = db.search(
            query_vector=query_vec.tolist(),
            limit=10,
            filter_dict={"variable": variable_name}
        )
        
        # Also search without filter to see what variables are in top results
        results_without_filter = db.search(
            query_vector=query_vec.tolist(),
            limit=20,
            filter_dict=None
        )
        
        # Extract variables from all results
        vars_in_top_20 = set()
        for hit in results_without_filter:
            if hasattr(hit, 'payload'):
                meta = hit.payload
            else:
                meta = hit.get('payload', {})
            var = meta.get('variable', '')
            if var:
                vars_in_top_20.add(var)
        
        # Format filtered results
        filtered_results = []
        for hit in results_with_filter:
            if hasattr(hit, 'payload'):
                meta = hit.payload
                score = hit.score
            else:
                meta = hit.get('payload', {})
                score = hit.get('score', 0.0)
            filtered_results.append({
                "variable": meta.get('variable'),
                "source_id": meta.get('source_id'),
                "time_start": meta.get('time_start'),
                "time_end": meta.get('time_end'),
                "score": float(score),
                "stats_mean": meta.get('stats_mean'),
            })
        
        return {
            "variable": variable_name,
            "found_with_filter": len(results_with_filter),
            "results": filtered_results,
            "all_variables_in_top_20": sorted(list(vars_in_top_20)),
            "variable_in_top_20": variable_name in vars_in_top_20,
            "message": f"Variable {variable_name} {'found' if len(results_with_filter) > 0 else 'NOT found'} in database with filter"
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }



# ====================================================================================
# CATALOG ENDPOINTS
# ====================================================================================

CATALOG_EXCEL_PATH = os.getenv("CATALOG_EXCEL_PATH", "Kopie souboru D1.1.xlsx")
# Resolve relative to project root
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not Path(CATALOG_EXCEL_PATH).exists():
    CATALOG_EXCEL_PATH = str(_PROJECT_ROOT / CATALOG_EXCEL_PATH)


class CatalogEntryResponse(BaseModel):
    row_index: int
    source_id: str
    hazard: Optional[str] = None
    dataset_name: Optional[str] = None
    data_type: Optional[str] = None
    spatial_coverage: Optional[str] = None
    region_country: Optional[str] = None
    spatial_resolution: Optional[str] = None
    temporal_coverage: Optional[str] = None
    temporal_resolution: Optional[str] = None
    bias_corrected: Optional[str] = None
    access: Optional[str] = None
    link: Optional[str] = None
    impact_sector: Optional[str] = None
    notes: Optional[str] = None
    phase: Optional[int] = None
    processing_status: Optional[str] = None


class CatalogProcessRequest(BaseModel):
    phases: Optional[List[int]] = [0]
    source_ids: Optional[List[str]] = None
    dry_run: bool = False


class CatalogProgressResponse(BaseModel):
    total: int = 0
    processed: int = 0
    failed: int = 0
    skipped: int = 0
    pending: int = 0
    current_phase: Optional[int] = None
    current_source: Optional[str] = None
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    thread_alive: bool = False
    thread_crashed: bool = False
    thread_error: Optional[str] = None


# --- Module-level batch thread tracking ---
import threading as _threading

_batch_thread: Optional[_threading.Thread] = None
_batch_thread_error: Optional[str] = None
_batch_thread_phases: Optional[List[int]] = None


@app.get("/catalog", response_model=List[CatalogEntryResponse])
async def list_catalog():
    """List all 234 catalog entries with phase classification and processing status."""
    try:
        from src.catalog.excel_reader import read_catalog
        from src.catalog.phase_classifier import classify_source
        from src.catalog.batch_orchestrator import BatchProgress

        entries = read_catalog(CATALOG_EXCEL_PATH)
        progress = BatchProgress.load()

        result = []
        for entry in entries:
            phase = classify_source(entry)
            processing_status = progress.get_overall_status(entry.source_id, phase)

            result.append(CatalogEntryResponse(
                row_index=entry.row_index,
                source_id=entry.source_id,
                hazard=entry.hazard,
                dataset_name=entry.dataset_name,
                data_type=entry.data_type,
                spatial_coverage=entry.spatial_coverage,
                region_country=entry.region_country,
                spatial_resolution=entry.spatial_resolution,
                temporal_coverage=entry.temporal_coverage,
                temporal_resolution=entry.temporal_resolution,
                bias_corrected=entry.bias_corrected,
                access=entry.access,
                link=entry.link,
                impact_sector=entry.impact_sector,
                notes=entry.notes,
                phase=phase,
                processing_status=processing_status,
            ))
        return result
    except Exception as e:
        logger.error(f"Failed to list catalog: {e}")
        raise HTTPException(500, str(e))


@app.post("/catalog/process")
async def trigger_catalog_processing(request: CatalogProcessRequest):
    """Trigger batch processing of catalog entries."""
    global _batch_thread, _batch_thread_error, _batch_thread_phases

    try:
        from src.catalog.batch_orchestrator import run_batch_pipeline

        if request.dry_run:
            result = run_batch_pipeline(
                excel_path=CATALOG_EXCEL_PATH,
                phases=request.phases,
                dry_run=True,
            )
            return result

        # Prevent duplicate batch starts
        if _batch_thread is not None and _batch_thread.is_alive():
            raise HTTPException(
                409,
                "Batch processing is already running. "
                "Check /catalog/progress for status.",
            )

        # Reset crash state
        _batch_thread_error = None
        _batch_thread_phases = request.phases

        def _run_in_background():
            global _batch_thread_error
            try:
                run_batch_pipeline(
                    excel_path=CATALOG_EXCEL_PATH,
                    phases=request.phases,
                    dry_run=request.dry_run,
                    resume=True,
                )
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                _batch_thread_error = f"{e}\n{tb}"
                logger.error(f"Background catalog processing failed: {e}\n{tb}")
                from src.utils.logger import setup_logger
                cl = setup_logger("catalog_pipeline", "logs/catalog_pipeline.log", "INFO")
                cl.error(f"BACKGROUND THREAD CRASHED: {e}\n{tb}")

        _batch_thread = _threading.Thread(
            target=_run_in_background, daemon=True, name="catalog-batch"
        )
        _batch_thread.start()

        return {
            "status": "started",
            "phases": request.phases,
            "message": f"Catalog processing started for phases {request.phases}",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/catalog/progress", response_model=CatalogProgressResponse)
async def get_catalog_progress():
    """Get current batch processing progress, including thread health."""
    try:
        from src.catalog.batch_orchestrator import get_progress
        data = get_progress()

        thread_alive = _batch_thread is not None and _batch_thread.is_alive()
        thread_crashed = (
            _batch_thread is not None
            and not _batch_thread.is_alive()
            and _batch_thread_error is not None
        )

        data["thread_alive"] = thread_alive
        data["thread_crashed"] = thread_crashed
        data["thread_error"] = _batch_thread_error if thread_crashed else None

        return data
    except Exception as e:
        logger.error(f"Failed to get progress: {e}")
        return CatalogProgressResponse()


@app.post("/catalog/classify")
async def classify_catalog():
    """Run classifier on all entries, return phase distribution."""
    try:
        from src.catalog.excel_reader import read_catalog
        from src.catalog.phase_classifier import classify_all

        entries = read_catalog(CATALOG_EXCEL_PATH)
        grouped = classify_all(entries)

        return {
            "total": len(entries),
            "phases": {str(phase): len(items) for phase, items in grouped.items()},
            "phase_descriptions": {
                "0": "Metadata-only (all entries)",
                "1": "Direct download, open access",
                "2": "Registration-required",
                "3": "API-based portals (CDS, ESGF)",
                "4": "Manual / contact-required",
            },
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/catalog/retry-failed")
async def retry_failed_catalog():
    """Re-run all failed catalog sources."""
    import threading

    try:
        from src.catalog.batch_orchestrator import retry_failed

        def _retry():
            try:
                retry_failed(excel_path=CATALOG_EXCEL_PATH)
            except Exception as e:
                logger.error(f"Retry failed: {e}")

        thread = threading.Thread(target=_retry, daemon=True)
        thread.start()

        return {"status": "started", "message": "Retrying failed sources in background"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/catalog/auto-restart")
async def auto_restart_catalog():
    """Detect crashed batch, reset interrupted entries, and restart processing.

    - Resets any 'processing' entries back to 'pending'.
    - Auto-detects which phases were running from the interrupted progress state.
    - Returns 409 if a batch is already running.
    """
    global _batch_thread, _batch_thread_error, _batch_thread_phases

    if _batch_thread is not None and _batch_thread.is_alive():
        raise HTTPException(
            409, "Batch thread is still alive — no restart needed."
        )

    try:
        from src.catalog.batch_orchestrator import BatchProgress, run_batch_pipeline

        progress = BatchProgress.load()
        reset_count = progress.mark_interrupted()
        progress.save()

        # Determine which phases to restart
        phases = _batch_thread_phases or [0, 1]
        # Also check progress state for the interrupted phase
        if progress.current_phase is not None and progress.current_phase not in phases:
            phases = sorted(set(phases) | {progress.current_phase})

        _batch_thread_error = None

        def _run_in_background():
            global _batch_thread_error
            try:
                run_batch_pipeline(
                    excel_path=CATALOG_EXCEL_PATH,
                    phases=phases,
                    resume=True,
                )
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                _batch_thread_error = f"{e}\n{tb}"
                logger.error(f"Auto-restart batch failed: {e}\n{tb}")

        _batch_thread = _threading.Thread(
            target=_run_in_background, daemon=True, name="catalog-batch-restart"
        )
        _batch_thread.start()

        return {
            "status": "restarted",
            "entries_reset": reset_count,
            "phases": phases,
            "message": f"Reset {reset_count} interrupted entries and restarted phases {phases}",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/catalog/{row_index}")
async def get_catalog_entry(row_index: int):
    """Get a single catalog entry by row index."""
    try:
        from src.catalog.excel_reader import read_catalog
        from src.catalog.phase_classifier import classify_source
        from src.catalog.batch_orchestrator import BatchProgress

        entries = read_catalog(CATALOG_EXCEL_PATH)
        progress = BatchProgress.load()

        for entry in entries:
            if entry.row_index == row_index:
                phase = classify_source(entry)
                status_info = progress.sources.get(entry.source_id, {})
                return {
                    **entry.to_dict(),
                    "phase": phase,
                    "processing_status": status_info.get("status", "pending"),
                    "processing_error": status_info.get("error"),
                }
        raise HTTPException(404, f"Catalog entry {row_index} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ====================================================================================
# SCHEDULE ENDPOINTS
# ====================================================================================

class ScheduleCreate(BaseModel):
    name: str
    cron_schedule: str
    job_name: str = "batch_catalog_etl_job"
    description: Optional[str] = None
    phases: Optional[List[int]] = [0]


@app.get("/schedules")
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
            result.append({
                "name": s["name"],
                "cron_schedule": s["cronSchedule"],
                "status": s.get("scheduleState", {}).get("status", "UNKNOWN"),
                "job_name": s.get("pipelineName"),
                "next_run": next_run,
            })
        return result
    except Exception as e:
        logger.warning(f"Failed to list schedules: {e}")
        return []


@app.post("/schedules/{schedule_name}/toggle")
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


# ====================================================================================
# LOG ENDPOINTS
# ====================================================================================

@app.get("/logs/etl")
async def get_etl_logs(lines: int = 100):
    """Get the last N lines of the ETL log file."""
    log_paths = [
        _PROJECT_ROOT / "logs" / "catalog_pipeline.log",
        _PROJECT_ROOT / "logs" / "dagster_dynamic_etl.log",
        _PROJECT_ROOT / "logs" / "dagster_pipeline.log",
    ]

    for log_path in log_paths:
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    all_lines = f.readlines()
                tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
                return {
                    "file": str(log_path),
                    "total_lines": len(all_lines),
                    "returned_lines": len(tail),
                    "content": "".join(tail),
                }
            except Exception as e:
                raise HTTPException(500, f"Failed to read log: {e}")

    return {"file": None, "total_lines": 0, "returned_lines": 0, "content": "No log file found"}


# ====================================================================================
# SETTINGS ENDPOINTS
# ====================================================================================

@app.get("/settings/system")
async def get_system_settings():
    """Get current system configuration and status."""
    import shutil

    # Disk usage
    disk = shutil.disk_usage("/")

    return {
        "llm": {
            "providers": {
                "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
                "groq": bool(os.getenv("GROQ_API_KEY")),
                "ollama": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            },
            "model": _runtime_settings.get("model", os.getenv("LLM_MODEL", "google/gemini-2.0-flash-001")),
            "temperature": _runtime_settings.get("temperature", float(os.getenv("LLM_TEMPERATURE", "0.3"))),
            "top_k": _runtime_settings.get("top_k", 5),
            "batch_size": _runtime_settings.get("batch_size", 100),
        },
        "embedding_model": {
            "name": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            "dimensions": 1024,
            "distance": "COSINE",
        },
        "qdrant": {
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": int(os.getenv("QDRANT_REST_PORT", 6333)),
        },
        "disk": {
            "total_gb": round(disk.total / (1024**3), 1),
            "used_gb": round(disk.used / (1024**3), 1),
            "free_gb": round(disk.free / (1024**3), 1),
        },
    }


@app.put("/settings/system")
async def update_system_settings(settings: dict):
    """Update runtime LLM settings (model, temperature, top_k, batch_size)."""
    allowed = {"model", "temperature", "top_k", "batch_size"}
    filtered = {k: v for k, v in settings.items() if k in allowed}
    if not filtered:
        raise HTTPException(400, "No valid fields to update")
    _runtime_settings.update(filtered)
    # Also update environment variables for LLM client
    if "model" in filtered:
        os.environ["LLM_MODEL"] = str(filtered["model"])
    if "temperature" in filtered:
        os.environ["LLM_TEMPERATURE"] = str(filtered["temperature"])
    # Persist to disk
    persisted = _load_settings()
    persisted["llm"] = dict(_runtime_settings)
    _save_settings(persisted)
    return {"updated": True, "settings": _runtime_settings}


@app.get("/settings/credentials")
async def get_credentials():
    """Get all portal credentials with masked values."""
    persisted = _load_settings()
    stored_creds = persisted.get("credentials", {})

    result = {}
    for cred_key, env_var in CREDENTIAL_KEYS.items():
        # Check stored value first, then env var
        value = stored_creds.get(cred_key, "") or os.getenv(env_var, "")
        if value:
            # Mask: show first 4 and last 3 chars for keys > 10 chars, otherwise "****"
            if len(value) > 10:
                masked = value[:4] + "..." + value[-3:]
            else:
                masked = "****"
            result[cred_key] = {"configured": True, "masked": masked}
        else:
            result[cred_key] = {"configured": False, "masked": ""}
    return result


@app.put("/settings/credentials")
async def update_credentials(credentials: dict):
    """Update portal credentials. Saves to disk and updates os.environ."""
    persisted = _load_settings()
    stored_creds = persisted.get("credentials", {})

    updated_keys = []
    for key, value in credentials.items():
        if key not in CREDENTIAL_KEYS:
            continue
        stored_creds[key] = value
        # Update environment variable
        env_var = CREDENTIAL_KEYS[key]
        if value:
            os.environ[env_var] = value
        elif env_var in os.environ:
            del os.environ[env_var]
        updated_keys.append(key)

    if not updated_keys:
        raise HTTPException(400, "No valid credential keys provided")

    persisted["credentials"] = stored_creds
    _save_settings(persisted)
    return {"updated": True, "keys": updated_keys}


@app.get("/admin/qdrant/health")
async def qdrant_health():
    """Get Qdrant collection health including dataset and variable breakdowns."""
    try:
        from qdrant_client import QdrantClient
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_REST_PORT", 6333))
        client = QdrantClient(host=host, port=port, timeout=10)
        collection_name = os.getenv("QDRANT_COLLECTION", "climate_data")

        try:
            info = client.get_collection(collection_name)
            status = info.status.value if hasattr(info.status, 'value') else str(info.status)
            points_count = info.points_count or 0

            # Try to get dataset breakdown via scroll
            datasets = {}
            variables = {}
            try:
                from qdrant_client.models import ScrollRequest
                records, _ = client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False,
                )
                for record in records:
                    payload = record.payload or {}
                    ds = payload.get("dataset_name", payload.get("source", "unknown"))
                    var = payload.get("variable", "unknown")
                    datasets[ds] = datasets.get(ds, 0) + 1
                    variables[var] = variables.get(var, 0) + 1
            except Exception:
                pass

            return {
                "status": status,
                "points_count": points_count,
                "segments_count": info.segments_count if hasattr(info, 'segments_count') else None,
                "datasets": datasets,
                "variables": variables,
                "health": "healthy" if status == "green" else "degraded",
            }
        except Exception:
            return {"status": "no_collection", "health": "empty", "datasets": {}, "variables": {}}
    except Exception as e:
        return {"status": "error", "health": "error", "error": str(e), "datasets": {}, "variables": {}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)