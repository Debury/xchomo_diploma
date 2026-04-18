"""Frontend serving endpoints for the Vue SPA."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse

router = APIRouter(tags=["frontend"])

VUE_DIST_DIR = Path(__file__).parent.parent / "frontend" / "dist"
_ASSETS_ROOT = (VUE_DIST_DIR / "assets").resolve()


@router.get("/")
async def root():
    """Redirect root to the Vue SPA."""
    if (VUE_DIST_DIR / "index.html").exists():
        return RedirectResponse(url="/app/", status_code=307)
    return {"status": "ok", "docs": "/docs", "ui": "/app"}


@router.get("/app/assets/{file_path:path}")
async def serve_vue_assets(file_path: str):
    """Serve Vue built assets. Resolves the path and refuses anything that
    escapes the assets directory — without this a caller could request
    `../../../etc/passwd` and FastAPI would happily serve it."""
    try:
        asset_file = (_ASSETS_ROOT / file_path).resolve()
    except (OSError, ValueError):
        raise HTTPException(404, f"Asset not found: {file_path}")
    if not asset_file.is_relative_to(_ASSETS_ROOT):
        raise HTTPException(404, f"Asset not found: {file_path}")
    if asset_file.is_file():
        return FileResponse(asset_file)
    raise HTTPException(404, f"Asset not found: {file_path}")


@router.get("/app")
@router.get("/app/{path:path}")
async def serve_vue_app(path: str = ""):
    """Serve Vue SPA — every client-side route returns index.html."""
    index_file = VUE_DIST_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(404, "Vue app not built. Run 'npm run build' in web_api/frontend/")
