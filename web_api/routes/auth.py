"""Authentication endpoints and the bearer-token gate used to protect the API."""

import hmac
import logging
import os
import secrets
import time
from typing import Dict, Optional, Tuple

from fastapi import APIRouter, Header, HTTPException, status

from web_api.models import AuthRequest, AuthResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# Tokens expire 24h after issue by default. A 24h window is comfortable for a
# demo-style deployment — the examiner won't have to re-log-in mid-defense — but
# short enough that a stolen localStorage token becomes useless within a day.
_TOKEN_TTL_SECONDS = int(os.getenv("AUTH_TOKEN_TTL_SECONDS", "86400"))

# Simple in-process token storage: token -> (username, expires_at_epoch).
# Production deployments should swap this for Redis or the dagster-postgres
# `climate_app` database, but for a single-node thesis install it is sufficient.
# Tokens are invalidated on server restart.
_valid_tokens: Dict[str, Tuple[str, float]] = {}


def _extract_bearer_token(authorization: Optional[str]) -> Optional[str]:
    """Parse a ``Bearer <token>`` Authorization header. Returns None for anything else."""
    if not authorization:
        return None
    if not authorization.lower().startswith("bearer "):
        return None
    return authorization.split(" ", 1)[1].strip() or None


def _prune_expired() -> None:
    """Drop tokens whose expires_at has passed. Cheap — _valid_tokens is tiny."""
    now = time.time()
    expired = [tok for tok, (_user, exp) in _valid_tokens.items() if exp <= now]
    for tok in expired:
        _valid_tokens.pop(tok, None)


def require_auth(authorization: Optional[str] = Header(None)) -> str:
    """FastAPI dependency that rejects requests without a valid bearer token.

    Returns the authenticated username on success.

    Skipped entirely when AUTH_PASSWORD is not configured, so local smoke tests
    (e.g. CI that boots the stack without secrets) still work. This matches the
    /auth/login behaviour, which returns 503 in that case so the frontend can
    tell the difference between "misconfigured" and "wrong credentials".
    """
    # If the server has no password configured, do NOT enforce auth — there is
    # nothing to compare against and we would otherwise lock everyone out.
    if not os.getenv("AUTH_PASSWORD", ""):
        return "anonymous"

    _prune_expired()

    token = _extract_bearer_token(authorization)
    if not token or token not in _valid_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    username, _expires_at = _valid_tokens[token]
    return username


@router.post("/login", response_model=AuthResponse)
async def auth_login(request: AuthRequest):
    """Authenticate the single admin user defined by AUTH_USERNAME/AUTH_PASSWORD."""
    expected_username = os.getenv("AUTH_USERNAME", "admin")
    expected_password = os.getenv("AUTH_PASSWORD", "")

    if not expected_password:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured on the server (AUTH_PASSWORD missing).",
        )

    username_ok = hmac.compare_digest(request.username or "", expected_username)
    password_ok = hmac.compare_digest(request.password or "", expected_password)

    if not (username_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    _prune_expired()
    token = secrets.token_urlsafe(32)
    _valid_tokens[token] = (request.username, time.time() + _TOKEN_TTL_SECONDS)
    return AuthResponse(
        success=True,
        token=token,
        message="Login successful",
        username=request.username,
    )


@router.post("/logout")
async def auth_logout(authorization: Optional[str] = Header(None)):
    """Invalidate a token received via the `Authorization: Bearer <token>` header."""
    token = _extract_bearer_token(authorization)
    if token and _valid_tokens.pop(token, None) is not None:
        return {"success": True, "message": "Logged out"}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing token",
    )


@router.get("/verify")
async def auth_verify(authorization: Optional[str] = Header(None)):
    """Verify a token received via the `Authorization: Bearer <token>` header."""
    _prune_expired()
    token = _extract_bearer_token(authorization)
    if token and token in _valid_tokens:
        username, _expires_at = _valid_tokens[token]
        return {"valid": True, "username": username}
    return {"valid": False}
