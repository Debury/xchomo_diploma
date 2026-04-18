"""Persistent settings management for the Climate ETL Pipeline API."""

import os
import json
from pathlib import Path
from typing import Dict

SETTINGS_PATH = Path(__file__).parent.parent / "data" / "app_settings.json"

# Credential keys that map to environment variables
CREDENTIAL_KEYS: Dict[str, str] = {
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "cds_api_key": "CDS_API_KEY",
    "nasa_earthdata_user": "NASA_EARTHDATA_USER",
    "nasa_earthdata_password": "NASA_EARTHDATA_PASSWORD",
    "cmems_username": "CMEMS_USERNAME",
    "cmems_password": "CMEMS_PASSWORD",
}


def load_settings() -> dict:
    """Load persisted settings from disk."""
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_settings(data: dict) -> None:
    """Save settings to disk."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, indent=2))


def restore_settings_to_env() -> dict:
    """Restore persisted credentials and LLM settings into os.environ. Returns runtime_settings."""
    persisted = load_settings()
    runtime_settings = {k: v for k, v in persisted.get("llm", {}).items()}

    # Restore credentials into os.environ
    for cred_key, env_var in CREDENTIAL_KEYS.items():
        value = persisted.get("credentials", {}).get(cred_key, "")
        if value:
            os.environ[env_var] = value

    # Restore LLM env vars
    if "model" in runtime_settings:
        os.environ["OPENROUTER_MODEL"] = str(runtime_settings["model"])

    return runtime_settings
