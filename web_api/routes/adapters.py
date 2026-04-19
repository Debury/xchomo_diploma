"""Custom portal-adapter registry.

Built-in adapters (CDS, NASA, MARINE, ESGF, NOAA, EIDC, NCAR) live in the
frontend's hardcoded list. This module adds the ability for an operator to
register *additional* adapters at runtime — for a new portal whose adapter
isn't yet shipped in code — and collect credential fields for it through
the same Settings UI.

Storage is a list under `custom_adapters` in `data/app_settings.json`. Each
entry is a dict with:
  {
    "id": "<slug>",
    "name": "<display name>",
    "description": "<one-liner>",
    "datasets": "<optional: sample dataset names, free text>",
    "public": false,
    "fields": [
      {"key": "<env-var-style key>", "label": "<display label>", "hint": "<optional>"}
    ]
  }

Credential values for these fields flow through the existing
`/settings/credentials` endpoints (we just extend CREDENTIAL_KEYS with the
custom field keys at request time — see `admin.get_credentials`).
"""

import logging
import re
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from web_api.config import load_settings, save_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/settings", tags=["settings"])

_SLUG = re.compile(r"[^a-z0-9]+")
_KEY = re.compile(r"[^a-z0-9_]+")


class AdapterField(BaseModel):
    key: str = Field(..., min_length=1, max_length=64)
    label: str = Field(..., min_length=1, max_length=80)
    hint: str | None = None


class AdapterCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    description: str = Field("", max_length=200)
    datasets: str = Field("", max_length=200)
    fields: List[AdapterField] = Field(default_factory=list, max_length=8)


def _slugify(text: str) -> str:
    return _SLUG.sub("_", text.lower()).strip("_") or "adapter"


def _clean_key(text: str) -> str:
    return _KEY.sub("_", text.lower()).strip("_") or "field"


def _custom_adapters_list() -> List[Dict[str, Any]]:
    """Return the list stored in app_settings.json, always a list."""
    persisted = load_settings()
    adapters = persisted.get("custom_adapters")
    return adapters if isinstance(adapters, list) else []


def custom_adapter_credential_keys() -> Dict[str, str]:
    """Map of credential_key → env_var for every custom adapter field.

    Used by `admin.get_credentials` / `update_credentials` so the UI can
    persist and mask values entered for custom adapters through the same
    endpoints as the built-in ones. Env var name mirrors the key, upper-cased.
    """
    result: Dict[str, str] = {}
    for adapter in _custom_adapters_list():
        for field in adapter.get("fields", []) or []:
            key = field.get("key")
            if key:
                result[key] = key.upper()
    return result


@router.get("/adapters")
async def list_custom_adapters():
    """Return the current list of custom adapter definitions."""
    return {"adapters": _custom_adapters_list()}


@router.post("/adapters", status_code=201)
async def create_custom_adapter(payload: AdapterCreate):
    """Register a new custom adapter. `id` is derived from `name` — collisions
    are rejected so the operator sees that a duplicate exists."""
    persisted = load_settings()
    adapters = persisted.get("custom_adapters")
    if not isinstance(adapters, list):
        adapters = []

    new_id = _slugify(payload.name)
    if any(a.get("id") == new_id for a in adapters):
        raise HTTPException(409, f"Adapter with id '{new_id}' already exists")

    # Also reject collisions with built-in adapter ids so the UI list doesn't
    # end up with two "CDS" entries.
    BUILTIN_IDS = {"cds", "nasa", "marine", "esgf", "noaa", "eidc", "ncar"}
    if new_id in BUILTIN_IDS:
        raise HTTPException(409, f"'{new_id}' is reserved for a built-in adapter")

    # Normalise field keys so they're safe for use as env vars / JSON keys.
    # Duplicate keys within the adapter are rejected.
    seen_keys = set()
    clean_fields: List[Dict[str, Any]] = []
    for f in payload.fields:
        key = _clean_key(f.key)
        if key in seen_keys:
            raise HTTPException(400, f"Duplicate field key: '{key}'")
        seen_keys.add(key)
        clean_fields.append({"key": key, "label": f.label, "hint": f.hint or ""})

    entry = {
        "id": new_id,
        "name": payload.name,
        "description": payload.description,
        "datasets": payload.datasets,
        "public": False,
        "fields": clean_fields,
    }
    adapters.append(entry)
    persisted["custom_adapters"] = adapters
    save_settings(persisted)
    logger.info(f"Registered custom adapter '{new_id}' with {len(clean_fields)} field(s)")
    return entry


@router.delete("/adapters/{adapter_id}", status_code=204)
async def delete_custom_adapter(adapter_id: str):
    """Remove a custom adapter. Credentials stored for its fields are kept —
    the operator can clear them from the credentials section if desired."""
    persisted = load_settings()
    adapters = persisted.get("custom_adapters")
    if not isinstance(adapters, list):
        raise HTTPException(404, f"Adapter not found: {adapter_id}")

    before = len(adapters)
    adapters = [a for a in adapters if a.get("id") != adapter_id]
    if len(adapters) == before:
        raise HTTPException(404, f"Adapter not found: {adapter_id}")

    persisted["custom_adapters"] = adapters
    save_settings(persisted)
    return None
