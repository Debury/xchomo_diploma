"""Persistent settings management for the Climate ETL Pipeline API."""

import json
import os
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
    """Save settings to disk and restrict permissions to the owner only.

    ``app_settings.json`` holds plaintext API keys, so we narrow the file mode
    to 0600 on every write. This is a POSIX semantic; on Windows ``chmod``
    degrades to a read-only toggle, which doesn't hurt.
    """
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, indent=2))
    try:
        os.chmod(SETTINGS_PATH, 0o600)
    except OSError:
        # Not fatal — the write succeeded, only the permission tighten failed.
        pass


def restore_settings_to_env() -> dict:
    """Restore persisted credentials and LLM settings into os.environ. Returns runtime_settings."""
    return load_persisted_credentials_into_env(path=SETTINGS_PATH, override=True)
