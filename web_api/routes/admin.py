"""Admin endpoints: logs, system settings, credentials, Qdrant health."""

import os
import logging
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from web_api.config import CREDENTIAL_KEYS, load_settings, save_settings
from web_api.dependencies import get_qdrant_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["admin"])

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Runtime settings (restored from disk on startup via config.restore_settings_to_env)
_runtime_settings: dict = {}


def init_runtime_settings(settings: dict) -> None:
    """Initialize runtime settings from config module."""
    _runtime_settings.update(settings)


# --- Logs ---


@router.get("/logs/etl")
async def get_etl_logs(lines: int = Query(100, ge=1, le=10000)):
    """Get the last N lines combined across every ETL log we know about.

    Was previously "first existing wins" — that meant only the catalog batch
    log was visible, even when per-source ETL runs were producing fresh
    output in `dagster_dynamic_etl.log`. Now we merge every present file's
    tail and re-sort by leading ISO timestamp so the panel shows both
    catalog AND per-source events together. Lines without a parseable
    timestamp keep their per-file order at the end.
    """
    log_paths = [
        _PROJECT_ROOT / "logs" / "catalog_pipeline.log",
        _PROJECT_ROOT / "logs" / "dagster_dynamic_etl.log",
        _PROJECT_ROOT / "logs" / "dagster_pipeline.log",
    ]

    # Drop files that haven't been written to in over an hour. The
    # dagster_dynamic_etl.log can carry weeks-old tracebacks (e.g. SLOCLIM
    # 403 from a long-dead schedule) that flood into a merged tail and
    # make the ETL Monitor look stuck — even when the active catalog
    # batch is happily progressing in the foreground.
    import time as _time
    STALE_THRESHOLD_SEC = 60 * 60  # 1 hour
    now = _time.time()
    files_present = [
        p for p in log_paths
        if p.exists() and (now - p.stat().st_mtime) < STALE_THRESHOLD_SEC
    ]
    if not files_present:
        return {"file": None, "total_lines": 0, "returned_lines": 0, "content": "No recent log activity"}

    import re
    # Match a leading ISO-ish timestamp like 2026-04-20 08:33:32 (with optional
    # fractional seconds and TZ). Used as a sort key so merged tails stay
    # chronological. Lines with no match get a sentinel that pushes them last.
    TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?)")

    merged: list[tuple[str, str, str]] = []  # (sort_key, file_label, line)
    total_lines = 0
    files_used: list[str] = []

    for log_path in files_present:
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                all_lines = f.readlines()
        except Exception as e:
            raise HTTPException(500, f"Failed to read {log_path.name}: {e}")
        files_used.append(log_path.name)
        total_lines += len(all_lines)
        # Per-file tail so a giant catalog log can't crowd out a small dynamic log.
        per_file_tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        label = log_path.stem  # e.g. "dagster_dynamic_etl"
        for ln in per_file_tail:
            m = TS_RE.match(ln)
            sort_key = m.group(1).replace("T", " ").replace(",", ".") if m else "~"
            merged.append((sort_key, label, ln))

    merged.sort(key=lambda x: x[0])
    tail = merged[-lines:]
    # Prefix each line with its source file in subtle brackets so user can
    # tell at a glance whether a line came from the catalog batch or a
    # per-source run, without parsing.
    rendered = "".join(f"[{label}] {line.rstrip()}\n" for _, label, line in tail)

    return {
        "file": ", ".join(files_used),
        "total_lines": total_lines,
        "returned_lines": len(tail),
        "content": rendered,
    }


# --- System Settings ---


@router.get("/settings/system")
async def get_system_settings():
    """Get current system configuration and status."""
    import shutil

    disk = shutil.disk_usage("/")

    return {
        "llm": {
            "providers": {
                "openrouter": bool(os.getenv("OPENROUTER_API_KEY")),
            },
            "model": _runtime_settings.get("model", os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.6")),
            "fast_model": os.getenv("OPENROUTER_FAST_MODEL", "anthropic/claude-sonnet-4.6"),
            "temperature": _runtime_settings.get("temperature", 0.1),
            "top_k": _runtime_settings.get("top_k", 10),
            "batch_size": _runtime_settings.get("batch_size", 512),
            # Cross-encoder reranker default. Off unless the operator opts in —
            # adds 2–4s per query for marginal quality gain on the golden set.
            "use_reranker": bool(_runtime_settings.get("use_reranker", False)),
        },
        "embedding_model": {
            "name": os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
            "dimensions": 1024,
            "distance": "COSINE",
            # Per-call batch size for the embedder. Tunable per-GPU because
            # BAAI/bge-large at FP32 needs ~1 GB VRAM per 64 docs at 512
            # tokens. Mobile 4 GB GPUs (RTX 3050) hit unified-memory spill
            # at batch_size > ~64 — embedding effectively runs over PCIe
            # which is 5-10× slower. Defaults match the safe path; the UI
            # exposes recommended sizes per VRAM tier.
            "doc_batch_size": int(_runtime_settings.get("embedding_batch_size",
                                  int(os.getenv("EMBEDDING_BATCH_SIZE", "64")))),
            "query_batch_size": int(_runtime_settings.get("embedding_query_batch_size",
                                    int(os.getenv("EMBEDDING_QUERY_BATCH_SIZE", "32")))),
            "device": os.getenv("EMBEDDING_DEVICE", ""),  # blank = auto-detect
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
        "uploads": {
            # Surfaced so the CreateSource UI can show the real cap instead of
            # hardcoding "500 MB". Parsed loosely — any bad env value falls back
            # to 5000 MB to match the backend's behaviour.
            "max_mb": max(1, int(os.getenv("UPLOAD_MAX_MB", "5000")) if os.getenv("UPLOAD_MAX_MB", "5000").lstrip("-").isdigit() else 5000),
        },
    }


@router.put("/settings/system")
async def update_system_settings(settings: dict):
    """Update runtime LLM + embedder settings."""
    allowed = {
        "model", "temperature", "top_k", "batch_size", "use_reranker",
        "embedding_batch_size", "embedding_query_batch_size",
    }
    filtered = {k: v for k, v in settings.items() if k in allowed}
    if "use_reranker" in filtered:
        filtered["use_reranker"] = bool(filtered["use_reranker"])
    # Clamp embedding batches to a sane range — too small wastes GPU,
    # too large OOMs mobile cards.
    for k in ("embedding_batch_size", "embedding_query_batch_size"):
        if k in filtered:
            try:
                filtered[k] = max(1, min(2048, int(filtered[k])))
            except (TypeError, ValueError):
                filtered.pop(k)
    if not filtered:
        raise HTTPException(400, "No valid fields to update")

    _runtime_settings.update(filtered)
    if "model" in filtered:
        os.environ["OPENROUTER_MODEL"] = str(filtered["model"])
    # Mirror embed batch sizes into env so newly-spawned TextEmbedder
    # instances (per-source ETL, RAG queries) pick them up immediately
    # without needing a process restart.
    if "embedding_batch_size" in filtered:
        os.environ["EMBEDDING_BATCH_SIZE"] = str(filtered["embedding_batch_size"])
    if "embedding_query_batch_size" in filtered:
        os.environ["EMBEDDING_QUERY_BATCH_SIZE"] = str(filtered["embedding_query_batch_size"])

    persisted = load_settings()
    persisted["llm"] = dict(_runtime_settings)
    save_settings(persisted)
    return {"updated": True, "settings": _runtime_settings}


# --- Credentials ---


def _all_credential_keys() -> dict:
    """Built-in CREDENTIAL_KEYS plus any fields registered via the custom
    adapter registry. Merged fresh on each call so a custom adapter added
    through `POST /settings/adapters` is immediately usable here without
    a restart."""
    from web_api.routes.adapters import custom_adapter_credential_keys

    merged = dict(CREDENTIAL_KEYS)
    merged.update(custom_adapter_credential_keys())
    return merged


@router.get("/settings/credentials")
async def get_credentials():
    """Get all portal credentials with masked values."""
    persisted = load_settings()
    stored_creds = persisted.get("credentials", {})

    # Strings matching these patterns are treated as unset placeholders.
    PLACEHOLDER_TOKENS = ("your-", "CHANGE_ME", "REPLACE_ME", "xxx", "<", "TODO")

    result = {}
    for cred_key, env_var in _all_credential_keys().items():
        value = stored_creds.get(cred_key, "") or os.getenv(env_var, "")
        if value and not any(p.lower() in value.lower() for p in PLACEHOLDER_TOKENS):
            if len(value) > 10:
                masked = value[:4] + "..." + value[-3:]
            else:
                masked = "****"
            result[cred_key] = {"configured": True, "masked": masked}
        else:
            result[cred_key] = {"configured": False, "masked": ""}
    return result


@router.get("/settings/credentials/{key}")
async def get_credential_value(key: str):
    """Get the full (unmasked) value of a single credential."""
    keys = _all_credential_keys()
    if key not in keys:
        raise HTTPException(404, f"Unknown credential key: {key}")
    persisted = load_settings()
    stored_creds = persisted.get("credentials", {})
    value = stored_creds.get(key, "") or os.getenv(keys[key], "")
    if not value:
        raise HTTPException(404, f"Credential {key} is not configured")
    return {"key": key, "value": value}


@router.put("/settings/credentials")
async def update_credentials(credentials: dict):
    """Update portal credentials. Saves to disk and updates os.environ."""
    persisted = load_settings()
    stored_creds = persisted.get("credentials", {})
    keys = _all_credential_keys()

    updated_keys = []
    for key, value in credentials.items():
        if key not in keys:
            continue
        stored_creds[key] = value
        env_var = keys[key]
        if value:
            os.environ[env_var] = value
        elif env_var in os.environ:
            del os.environ[env_var]
        updated_keys.append(key)

    if not updated_keys:
        raise HTTPException(400, "No valid credential keys provided")

    persisted["credentials"] = stored_creds
    save_settings(persisted)
    return {"updated": True, "keys": updated_keys}


# --- Qdrant Health ---


@router.get("/admin/qdrant/health")
async def qdrant_health():
    """Get Qdrant collection health including dataset and variable breakdowns."""
    try:
        client = get_qdrant_client()
        collection_name = os.getenv("QDRANT_COLLECTION", "climate_data")

        try:
            info = client.get_collection(collection_name)
            status = info.status.value if hasattr(info.status, "value") else str(info.status)
            points_count = info.points_count or 0

            datasets = {}
            variables = {}
            try:
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
                "segments_count": info.segments_count if hasattr(info, "segments_count") else None,
                "datasets": datasets,
                "variables": variables,
                "health": "healthy" if status == "green" else "degraded",
            }
        except Exception:
            return {"status": "no_collection", "health": "empty", "datasets": {}, "variables": {}}
    except Exception as e:
        return {"status": "error", "health": "error", "error": str(e), "datasets": {}, "variables": {}}


# --- /tmp cleanup ---


@router.get("/admin/tmp/list")
async def list_tmp_files():
    """List leftover ETL temp files in /tmp.

    Catalog batch and per-source ETL paths download into ``/tmp/tmpXXX.nc``
    (or ``.gz`` etc). On a clean run those get unlinked when the entry
    succeeds. Crashes, killed daemons, or partial downloads (STEAD's 1 GB
    server cap, IncompleteRead retries) leave 1-2 GB partials behind.
    Surfaced so the operator can clear them without docker exec.
    """
    import glob
    files = []
    for path in glob.glob("/tmp/tmp*"):
        try:
            st = os.stat(path)
            files.append({
                "path": path,
                "size_bytes": st.st_size,
                "modified_at": st.st_mtime,
            })
        except Exception:
            continue
    files.sort(key=lambda f: -f["size_bytes"])
    total = sum(f["size_bytes"] for f in files)
    return {"files": files, "total_bytes": total, "count": len(files)}


@router.post("/admin/tmp/clean")
async def clean_tmp_files():
    """Delete ETL temp files from /tmp.

    Refuses while a catalog batch is alive — wiping a file that's mid-
    write would corrupt that run. Also keeps anything not matching the
    `tmp*` prefix, which is what tempfile.NamedTemporaryFile uses.
    """
    from web_api.routes.catalog import _batch_thread
    if _batch_thread is not None and _batch_thread.is_alive():
        raise HTTPException(
            409,
            "Catalog batch is currently running — cancel it first or wait for "
            "it to finish before cleaning /tmp",
        )

    import glob
    deleted = 0
    freed_bytes = 0
    errors: list = []
    for path in glob.glob("/tmp/tmp*"):
        try:
            sz = os.stat(path).st_size
            os.unlink(path)
            deleted += 1
            freed_bytes += sz
        except Exception as e:
            errors.append({"path": path, "error": str(e)})
    return {
        "deleted": deleted,
        "freed_bytes": freed_bytes,
        "errors": errors,
    }


# --- Qdrant Snapshots ---
#
# Snapshots are Qdrant's native dump format: a tarball containing the
# collection's vectors + payload + indexing config. Exporting one is the
# simplest way to back up the collection or move it between deployments.
# Restoring REPLACES the live collection — the UI gates this behind a
# confirmation dialog, but the endpoint itself trusts the caller.
#
# Snapshot files live inside the qdrant container at
# /qdrant/snapshots/<collection>/<name>.snapshot. We don't expose qdrant's
# REST port publicly (see CLAUDE.md), so the download/upload paths proxy
# through web-api with a streaming response to keep memory bounded.


def _qdrant_snapshot_base_url() -> str:
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_REST_PORT", 6333))
    collection = os.getenv("QDRANT_COLLECTION", "climate_data")
    return f"http://{host}:{port}/collections/{collection}/snapshots"


def _validate_snapshot_name(name: str) -> None:
    """Reject anything that could escape the snapshot directory."""
    if not name or "/" in name or "\\" in name or ".." in name or name.startswith("."):
        raise HTTPException(400, "Invalid snapshot name")


# In-flight snapshot create state. A single dict because Qdrant only allows
# one create per collection at a time anyway. Lives in process memory — a
# web-api restart clears it (the create itself keeps running inside Qdrant
# regardless, the user just loses the progress indicator until they refresh
# the snapshot list and see the new file appear). The background task
# updates this in-place so any HTTP poll sees the latest state.
_active_create: dict = {}


async def _do_snapshot_create() -> None:
    """Background task: call Qdrant and update ``_active_create`` in place."""
    import httpx
    from datetime import datetime, timezone

    url = _qdrant_snapshot_base_url()
    try:
        async with httpx.AsyncClient(timeout=None) as cli:
            resp = await cli.post(url, params={"wait": "true"})
        if resp.status_code == 200:
            result = (resp.json() or {}).get("result") or {}
            _active_create["status"] = "completed"
            _active_create["snapshot"] = {
                "name": result.get("name"),
                "size": result.get("size"),
                "creation_time": result.get("creation_time"),
                "checksum": result.get("checksum"),
            }
        else:
            _active_create["status"] = "failed"
            _active_create["error"] = f"HTTP {resp.status_code}: {resp.text[:500]}"
    except Exception as e:
        _active_create["status"] = "failed"
        _active_create["error"] = str(e)
    finally:
        _active_create["finished_at"] = datetime.now(timezone.utc).isoformat()


@router.post("/admin/qdrant/snapshot/create")
async def create_qdrant_snapshot():
    """Kick off a snapshot create as a background task.

    Returns immediately with ``started_at`` — the actual Qdrant call (multi-
    minute on a large collection) runs detached so a browser reload doesn't
    cancel it. Poll ``/admin/qdrant/snapshot/active`` for live progress.
    Refuses (409) if a create is already running.
    """
    import asyncio
    from datetime import datetime, timezone

    if _active_create.get("status") == "running":
        return {
            "status": "already_running",
            "started_at": _active_create.get("started_at"),
        }

    collection_name = os.getenv("QDRANT_COLLECTION", "climate_data")
    started_at = datetime.now(timezone.utc).isoformat()
    _active_create.clear()
    _active_create.update({
        "status": "running",
        "started_at": started_at,
        "collection": collection_name,
    })
    asyncio.create_task(_do_snapshot_create())
    return {"status": "started", "started_at": started_at, "collection": collection_name}


@router.get("/admin/qdrant/snapshot/active")
async def get_active_snapshot_create():
    """Return the current/last snapshot-create state.

    Used by the UI to restore its progress bar after a page reload.
    Returns ``{"active": false}`` when nothing has been triggered since the
    web-api booted.
    """
    if not _active_create:
        return {"active": False}
    return {"active": True, **_active_create}


@router.post("/admin/qdrant/snapshot/active/dismiss")
async def dismiss_active_snapshot_state():
    """Clear the cached create state once the UI has acknowledged it.

    Stops a stale ``completed``/``failed`` entry from re-popping on every
    reload after the user has already seen the toast.
    """
    if _active_create.get("status") == "running":
        raise HTTPException(409, "Cannot dismiss a running snapshot create")
    _active_create.clear()
    return {"dismissed": True}


@router.get("/admin/qdrant/snapshot/list")
async def list_qdrant_snapshots():
    """List existing Qdrant snapshots for the active collection."""
    import httpx

    collection_name = os.getenv("QDRANT_COLLECTION", "climate_data")
    url = _qdrant_snapshot_base_url()
    try:
        async with httpx.AsyncClient(timeout=30) as cli:
            resp = await cli.get(url)
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, f"Qdrant: {resp.text[:500]}")
        result = resp.json().get("result") or []
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Snapshot list failed: {e}")
    return {
        "collection": collection_name,
        "snapshots": [
            {
                "name": s.get("name"),
                "size": s.get("size"),
                "creation_time": s.get("creation_time"),
                "checksum": s.get("checksum"),
            }
            for s in result
        ],
    }


@router.delete("/admin/qdrant/snapshot/{snapshot_name}")
async def delete_qdrant_snapshot(snapshot_name: str):
    """Delete a stored snapshot from Qdrant."""
    import httpx

    _validate_snapshot_name(snapshot_name)
    base = _qdrant_snapshot_base_url()
    url = f"{base}/{snapshot_name}"
    try:
        async with httpx.AsyncClient(timeout=30) as cli:
            resp = await cli.delete(url)
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, f"Qdrant: {resp.text[:500]}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Snapshot delete failed: {e}")
    return {"deleted": True, "name": snapshot_name}


@router.get("/admin/qdrant/snapshot/download/{snapshot_name}")
async def download_qdrant_snapshot(snapshot_name: str):
    """Stream a snapshot file from Qdrant to the client.

    Snapshots can be multi-GB so we stream chunk-by-chunk rather than
    buffering the whole file in web-api memory.
    """
    import httpx

    _validate_snapshot_name(snapshot_name)
    base = _qdrant_snapshot_base_url()
    url = f"{base}/{snapshot_name}"

    # We open the upstream response inside the generator so it stays alive
    # for the duration of the stream. AsyncClient + stream() is the recipe
    # FastAPI docs recommend for transparent file proxies.
    async def stream_iterator():
        async with httpx.AsyncClient(timeout=None) as cli:
            async with cli.stream("GET", url) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    detail = body.decode("utf-8", "replace")[:500]
                    raise HTTPException(resp.status_code, f"Qdrant: {detail}")
                async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                    yield chunk

    return StreamingResponse(
        stream_iterator(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{snapshot_name}"'},
    )


@router.post("/admin/qdrant/snapshot/restore")
async def restore_qdrant_snapshot(
    snapshot: UploadFile = File(...),
    priority: str = Query("snapshot", pattern="^(snapshot|replica|no_sync)$"),
):
    """Upload a snapshot file and recover the collection from it.

    DESTRUCTIVE: replaces the contents of the active collection. The UI
    requires explicit confirmation before calling this. ``priority=snapshot``
    means "trust the uploaded data over the current collection" — the right
    setting when restoring a backup.
    """
    import httpx

    base = _qdrant_snapshot_base_url()
    url = f"{base}/upload?priority={priority}"

    try:
        files = {
            "snapshot": (
                snapshot.filename or "snapshot.snapshot",
                snapshot.file,
                snapshot.content_type or "application/octet-stream",
            ),
        }
        # No timeout — large snapshots can take minutes to load on disk.
        async with httpx.AsyncClient(timeout=None) as cli:
            resp = await cli.post(url, files=files)
        if resp.status_code != 200:
            raise HTTPException(resp.status_code, f"Qdrant: {resp.text[:500]}")
        return resp.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Snapshot restore failed: {e}")
