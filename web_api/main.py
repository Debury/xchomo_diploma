"""
FastAPI Web Service for Climate ETL Pipeline

Provides REST API endpoints for interacting with Dagster pipelines,
managing data sources, and querying climate data via RAG.
"""

import os
import logging
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web_api.config import restore_settings_to_env
from web_api.routes import health, auth, frontend, embeddings, sources, rag, catalog, schedules, admin, qdrant_datasets

logger = logging.getLogger(__name__)

# Restore persisted settings into os.environ on startup
_runtime_settings = restore_settings_to_env()

# Initialize admin module's runtime settings reference
admin.init_runtime_settings(_runtime_settings)

# ====================================================================================
# APPLICATION SETUP
# ====================================================================================

app = FastAPI(title="Climate ETL Pipeline API", version="2.0.0")

# Browsers refuse credentialed requests when the allowed origin is "*", so we
# only enable allow_credentials when CORS_ORIGINS lists one or more explicit
# origins. For local dev we default to same-origin (no CORS needed at all).
_cors_origins_raw = os.getenv("CORS_ORIGINS", "").strip()
if _cors_origins_raw:
    _cors_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
    _cors_use_credentials = "*" not in _cors_origins
else:
    _cors_origins = []  # same-origin only — no CORS headers will be added
    _cors_use_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_use_credentials,
    allow_methods=["*"],
    allow_headers=["*", "Authorization"],
)

# Ensure data directories exist
SAMPLE_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ====================================================================================
# REGISTER ROUTERS
# ====================================================================================
#
# Public routers (no auth required):
#   - frontend  → serves the Vue SPA + static assets + root redirect
#   - health    → /health for uptime monitoring
#   - auth      → login / logout / verify themselves
#
# Protected routers use Depends(auth.require_auth), which enforces a valid
# Bearer token in the Authorization header. If AUTH_PASSWORD is not set, the
# dependency no-ops so local dev without secrets still works.

_protected = [Depends(auth.require_auth)]

app.include_router(frontend.router)
app.include_router(health.router)
app.include_router(auth.router)
app.include_router(embeddings.router, dependencies=_protected)
app.include_router(sources.router, dependencies=_protected)
app.include_router(rag.router, dependencies=_protected)
app.include_router(catalog.router, dependencies=_protected)
app.include_router(schedules.router, dependencies=_protected)
app.include_router(admin.router, dependencies=_protected)
app.include_router(qdrant_datasets.router, dependencies=_protected)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_api.main:app", host="0.0.0.0", port=8000, reload=True)
