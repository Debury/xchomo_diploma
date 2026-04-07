"""Frontend serving endpoints (Vue SPA, legacy UI, chat)."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(tags=["frontend"])

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
VUE_DIST_DIR = FRONTEND_DIR / "dist"


@router.get("/")
async def root():
    """Root endpoint - serve Vue app or fallback to JSON."""
    if VUE_DIST_DIR.exists():
        index_file = VUE_DIST_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
    return {"status": "ok", "docs": "/docs", "ui": "/app"}


@router.get("/ui", response_class=FileResponse)
async def serve_frontend():
    """Serve legacy UI."""
    if not FRONTEND_DIR.exists():
        raise HTTPException(404, "Frontend not found")
    legacy_file = FRONTEND_DIR / "legacy-index.html"
    if legacy_file.exists():
        return FileResponse(legacy_file)
    return FileResponse(FRONTEND_DIR / "index.html")


@router.get("/chat", response_class=FileResponse)
async def serve_chat():
    """Serve the SPA chat interface."""
    spa_file = FRONTEND_DIR / "spa.html"
    if not spa_file.exists():
        raise HTTPException(404, "Chat UI not found")
    return FileResponse(spa_file)


@router.get("/app/assets/{file_path:path}")
async def serve_vue_assets(file_path: str):
    """Serve Vue built assets."""
    asset_file = VUE_DIST_DIR / "assets" / file_path
    if asset_file.exists() and asset_file.is_file():
        return FileResponse(asset_file)
    raise HTTPException(404, f"Asset not found: {file_path}")


@router.get("/app")
@router.get("/app/{path:path}")
async def serve_vue_app(path: str = ""):
    """Serve Vue SPA - all routes return index.html for client-side routing."""
    if VUE_DIST_DIR.exists():
        index_file = VUE_DIST_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
    raise HTTPException(404, "Vue app not built. Run 'npm run build' in frontend/")
