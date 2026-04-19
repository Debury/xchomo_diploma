"""Persistent settings management for the Climate ETL Pipeline API."""

import json
from pathlib import Path

from src.utils.persisted_creds import (
    CREDENTIAL_KEYS,
    load_persisted_credentials_into_env,
)

SETTINGS_PATH = Path(__file__).parent.parent / "data" / "app_settings.json"

# Re-export for existing callers that expect CREDENTIAL_KEYS here.
__all__ = ["SETTINGS_PATH", "CREDENTIAL_KEYS", "load_settings", "save_settings", "restore_settings_to_env"]


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
    return load_persisted_credentials_into_env(path=SETTINGS_PATH, override=True)
