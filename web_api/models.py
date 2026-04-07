"""Pydantic request/response models for the Climate ETL Pipeline API."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


# --- Jobs & Runs ---

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


# --- Sources ---

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
    auth_method: Optional[str] = None
    auth_credentials: Optional[Dict[str, str]] = None
    portal: Optional[str] = None
    schedule_cron: Optional[str] = None
    auto_embed: bool = True
    hazard_type: Optional[str] = None
    region_country: Optional[str] = None
    spatial_coverage: Optional[str] = None
    impact_sector: Optional[str] = None
    keywords: Optional[List[str]] = None
    custom_metadata: Optional[Dict[str, str]] = None

class SourceUpdate(BaseModel):
    url: Optional[str] = None
    is_active: Optional[bool] = None

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
    hazard_type: Optional[str] = None
    region_country: Optional[str] = None
    spatial_coverage: Optional[str] = None
    impact_sector: Optional[str] = None
    keywords: Optional[List[str]] = None
    custom_metadata: Optional[Dict[str, str]] = None
    id: Optional[int] = None
    collection_name: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    created_by: Optional[str] = None
    last_processed: Optional[str] = None

class SourceScheduleRequest(BaseModel):
    cron_expression: str
    is_enabled: bool = True


# --- Embeddings ---

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


# --- RAG ---

class RAGChunk(BaseModel):
    source_id: str
    variable: Optional[str]
    similarity: float
    text: str
    metadata: Dict[str, Any]

class RAGChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    use_llm: bool = True
    temperature: float = 0.3
    source_id: Optional[str] = None
    variable: Optional[str] = None

class RAGChatResponse(BaseModel):
    question: str
    answer: str
    references: List[str]
    chunks: List[RAGChunk]
    llm_used: bool


# --- Auth ---

class AuthRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    message: str
    username: Optional[str] = None


# --- Catalog ---

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
    force_reprocess: bool = False

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


# --- Schedules ---

class ScheduleCreate(BaseModel):
    name: str
    cron_schedule: str
    job_name: str = "batch_catalog_etl_job"
    description: Optional[str] = None
    phases: Optional[List[int]] = [0]
