"""
FastAPI Web Service for Climate ETL Pipeline

Provides REST API endpoints for interacting with Dagster pipelines,
managing data sources, and querying climate data via RAG.
"""

import os
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from web_api.config import restore_settings_to_env
from web_api.routes import health, auth, frontend, embeddings, sources, rag, catalog, schedules, admin, qdrant_datasets, adapters

logger = logging.getLogger(__name__)

# Restore persisted settings into os.environ on startup
_runtime_settings = restore_settings_to_env()

# Initialize admin module's runtime settings reference
admin.init_runtime_settings(_runtime_settings)

# ====================================================================================
# APPLICATION SETUP
# ====================================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks.

    On boot we reconcile any source stuck in 'processing' that was orphaned by
    a previous container restart or killed Dagster run — otherwise those rows
    sit indefinitely with no way for the user to retry, which was the single
    worst demo failure mode identified in the pre-defense audit.
    """
    try:
        from src.sources import get_source_store

        store = get_source_store()
        if hasattr(store, "reset_orphaned_processing"):
            # 30 min is well above the longest legitimate ETL run we've
            # observed; anything older is safely orphaned.
            reset_ids = store.reset_orphaned_processing(max_age_minutes=30)
            if reset_ids:
                logger.warning(
                    f"Startup: reset {len(reset_ids)} orphaned processing sources."
                )
    except Exception as e:
        # Never block the API boot on a reconciliation failure — log and move on.
        logger.error(f"Startup reconciliation failed: {e}", exc_info=True)

    yield
    # no shutdown work


app = FastAPI(title="Climate ETL Pipeline API", version="2.0.0", lifespan=lifespan)

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
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
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
app.include_router(adapters.router, dependencies=_protected)
app.include_router(qdrant_datasets.router, dependencies=_protected)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_api.main:app", host="0.0.0.0", port=8000, reload=True)
