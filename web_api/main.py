"""
FastAPI Web Service for Climate ETL Pipeline

Provides REST API endpoints for interacting with Dagster pipelines,
managing data sources, and querying climate data via RAG.
"""

import os
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from web_api.config import restore_settings_to_env
from web_api.routes import health, auth, frontend, embeddings, sources, rag, catalog, schedules, admin

logger = logging.getLogger(__name__)

# Restore persisted settings into os.environ on startup
_runtime_settings = restore_settings_to_env()

# Initialize admin module's runtime settings reference
admin.init_runtime_settings(_runtime_settings)

# ====================================================================================
# APPLICATION SETUP
# ====================================================================================

app = FastAPI(title="Climate ETL Pipeline API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for legacy frontend
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/ui/static", StaticFiles(directory=FRONTEND_DIR, html=False), name="rag-ui-static")

# Ensure data directories exist
SAMPLE_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ====================================================================================
# REGISTER ROUTERS
# ====================================================================================

app.include_router(frontend.router)
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(embeddings.router)
app.include_router(sources.router)
app.include_router(rag.router)
app.include_router(catalog.router)
app.include_router(schedules.router)
app.include_router(admin.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_api.main:app", host="0.0.0.0", port=8000, reload=True)
