"""Admin endpoints: logs, system settings, credentials, Qdrant health."""

import os
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

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
                "groq": bool(os.getenv("GROQ_API_KEY")),
                "ollama": os.getenv("OLLAMA_URL", os.getenv("OLLAMA_HOST", "http://localhost:11434")),
            },
            "model": _runtime_settings.get("model", os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.6")),
            "fast_model": os.getenv("OPENROUTER_FAST_MODEL", "anthropic/claude-sonnet-4.6"),
            "temperature": _runtime_settings.get("temperature", 0.1),
            "top_k": _runtime_settings.get("top_k", 10),
            "batch_size": _runtime_settings.get("batch_size", 512),
        },
        "embedding_model": {
            "name": os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
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


@router.put("/settings/system")
async def update_system_settings(settings: dict):
    """Update runtime LLM settings (model, temperature, top_k, batch_size)."""
    allowed = {"model", "temperature", "top_k", "batch_size"}
    filtered = {k: v for k, v in settings.items() if k in allowed}
    if not filtered:
        raise HTTPException(400, "No valid fields to update")

    _runtime_settings.update(filtered)
    if "model" in filtered:
        os.environ["OPENROUTER_MODEL"] = str(filtered["model"])

    persisted = load_settings()
    persisted["llm"] = dict(_runtime_settings)
    save_settings(persisted)
    return {"updated": True, "settings": _runtime_settings}


# --- Credentials ---


@router.get("/settings/credentials")
async def get_credentials():
    """Get all portal credentials with masked values."""
    persisted = load_settings()
    stored_creds = persisted.get("credentials", {})

    # Strings matching these patterns are treated as unset placeholders.
    PLACEHOLDER_TOKENS = ("your-", "CHANGE_ME", "REPLACE_ME", "xxx", "<", "TODO")

    result = {}
    for cred_key, env_var in CREDENTIAL_KEYS.items():
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
    if key not in CREDENTIAL_KEYS:
        raise HTTPException(404, f"Unknown credential key: {key}")
    persisted = load_settings()
    stored_creds = persisted.get("credentials", {})
    value = stored_creds.get(key, "") or os.getenv(CREDENTIAL_KEYS[key], "")
    if not value:
        raise HTTPException(404, f"Credential {key} is not configured")
    return {"key": key, "value": value}


@router.put("/settings/credentials")
async def update_credentials(credentials: dict):
    """Update portal credentials. Saves to disk and updates os.environ."""
    persisted = load_settings()
    stored_creds = persisted.get("credentials", {})

    updated_keys = []
    for key, value in credentials.items():
        if key not in CREDENTIAL_KEYS:
            continue
        stored_creds[key] = value
        env_var = CREDENTIAL_KEYS[key]
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
