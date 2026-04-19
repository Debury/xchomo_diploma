"""Read ``data/app_settings.json`` and inject its credentials into ``os.environ``.

Shared between the FastAPI startup (``web_api/config.py``) and Dagster ops.
Both the ``web-api`` container and the Dagster containers bind-mount
``./data`` at ``/app/data``, so the same JSON file is visible in all of them.
Dagster ops call this at the start of every run to pick up credentials that
were rotated via the Settings UI after the container last started.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

# Default to the in-container path. Overridable via env for tests.
_DEFAULT_PATH = Path(os.getenv("APP_SETTINGS_PATH", "/app/data/app_settings.json"))

# Credential keys stored under ``credentials`` → environment variable name.
CREDENTIAL_KEYS: Dict[str, str] = {
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "cds_api_key": "CDS_API_KEY",
    "nasa_earthdata_user": "NASA_EARTHDATA_USER",
    "nasa_earthdata_password": "NASA_EARTHDATA_PASSWORD",
    "cmems_username": "CMEMS_USERNAME",
    "cmems_password": "CMEMS_PASSWORD",
}


def _resolve_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if _DEFAULT_PATH.exists():
        return _DEFAULT_PATH
    # Fallback for non-container runs (local dev, tests): repo-root/data/…
    return Path(__file__).resolve().parents[2] / "data" / "app_settings.json"


def load_persisted_credentials_into_env(
    path: Path | None = None,
    override: bool = False,
) -> Dict[str, str]:
    """Inject persisted credentials + LLM settings into ``os.environ``.

    Parameters
    ----------
    path:
        Optional explicit path to the settings JSON. Defaults to
        ``/app/data/app_settings.json`` (the container bind-mount path).
    override:
        When True, the persisted value replaces anything already in the
        environment. Default False — an explicit ``.env`` entry wins.

    Returns a dict of the ``llm`` section for callers that want it.
    """
    resolved = _resolve_path(path)
    if not resolved.exists():
        return {}
    try:
        persisted = json.loads(resolved.read_text())
    except (OSError, json.JSONDecodeError) as err:
        logger.warning(f"Could not read {resolved}: {err}")
        return {}

    credentials = persisted.get("credentials", {}) or {}
    for cred_key, env_var in CREDENTIAL_KEYS.items():
        value = credentials.get(cred_key) or persisted.get(cred_key)
        if not value:
            continue
        if override or not os.environ.get(env_var):
            os.environ[env_var] = str(value)

    llm = persisted.get("llm", {}) or {}
    model = llm.get("model")
    if model and (override or not os.environ.get("OPENROUTER_MODEL")):
        os.environ["OPENROUTER_MODEL"] = str(model)

    return {k: v for k, v in llm.items()}
