"""Authentication endpoints and the bearer-token gate used to protect the API."""

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, Header, HTTPException, Request, status

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

# Per-IP login throttling. Each IP keeps a list of recent failure timestamps;
# exceed the threshold within the window and login is refused with 429 until
# the oldest failure ages out. Cleared entirely on server restart.
_LOGIN_WINDOW_SECONDS = int(os.getenv("AUTH_LOGIN_WINDOW_SECONDS", "900"))  # 15 min
_LOGIN_MAX_FAILURES = int(os.getenv("AUTH_LOGIN_MAX_FAILURES", "10"))
_login_failures: Dict[str, List[float]] = {}

_PBKDF2_PREFIX = "pbkdf2_sha256$"


def _hash_password(password: str, iterations: int = 600_000) -> str:
    """Produce a ``pbkdf2_sha256$iters$salt_b64$hash_b64`` string."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return (
        f"{_PBKDF2_PREFIX}{iterations}$"
        f"{base64.b64encode(salt).decode('ascii')}$"
        f"{base64.b64encode(digest).decode('ascii')}"
    )


def _verify_pbkdf2(password: str, encoded: str) -> bool:
    """Constant-time check against a ``pbkdf2_sha256$…`` string."""
    if not encoded.startswith(_PBKDF2_PREFIX):
        return False
    try:
        _, iters_s, salt_b64, hash_b64 = encoded.split("$", 3)
        iters = int(iters_s)
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(hash_b64)
    except (ValueError, TypeError):
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters)
    return hmac.compare_digest(candidate, expected)


def _verify_password(provided: str, expected_plain: str, expected_hash: str) -> bool:
    """Hash wins if configured; otherwise fall back to the plaintext env var."""
    if expected_hash:
        return _verify_pbkdf2(provided, expected_hash)
    if expected_plain:
        return hmac.compare_digest(provided, expected_plain)
    return False


def _auth_configured() -> bool:
    """True iff the server has a password or hash set."""
    return bool(os.getenv("AUTH_PASSWORD_HASH") or os.getenv("AUTH_PASSWORD"))


def _anonymous_allowed() -> bool:
    """Opt-in for unauthenticated access (CI/smoke tests only)."""
    return os.getenv("AUTH_ALLOW_ANONYMOUS", "").lower() in ("1", "true", "yes")


def _client_ip(request: Optional[Request]) -> str:
    """Best-effort client IP; falls back to ``unknown`` when the header chain is empty."""
    if request is None:
        return "unknown"
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",", 1)[0].strip() or "unknown"
    return (request.client.host if request.client else "unknown") or "unknown"


def _prune_login_failures(ip: str) -> List[float]:
    """Drop failure timestamps outside the window; return what's left."""
    cutoff = time.time() - _LOGIN_WINDOW_SECONDS
    kept = [ts for ts in _login_failures.get(ip, []) if ts > cutoff]
    if kept:
        _login_failures[ip] = kept
    else:
        _login_failures.pop(ip, None)
    return kept


def _check_login_rate(ip: str) -> None:
    """Raise 429 if this IP has used up its failure budget."""
    recent = _prune_login_failures(ip)
    if len(recent) >= _LOGIN_MAX_FAILURES:
        oldest = min(recent)
        retry_after = max(1, int(oldest + _LOGIN_WINDOW_SECONDS - time.time()))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )


def _record_login_failure(ip: str) -> None:
    _login_failures.setdefault(ip, []).append(time.time())


def _clear_login_failures(ip: str) -> None:
    _login_failures.pop(ip, None)


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

    Fails closed when no password/hash is configured. Set ``AUTH_ALLOW_ANONYMOUS=1``
    to opt in to unauthenticated access (CI and local smoke tests only).
    """
    if not _auth_configured():
        if _anonymous_allowed():
            return "anonymous"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Authentication is not configured. Set AUTH_PASSWORD (or "
                "AUTH_PASSWORD_HASH), or AUTH_ALLOW_ANONYMOUS=1 to disable auth."
            ),
        )

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
async def auth_login(request: AuthRequest, http_request: Request):
    """Authenticate the single admin user defined by AUTH_USERNAME/AUTH_PASSWORD."""
    expected_username = os.getenv("AUTH_USERNAME", "admin")
    expected_password = os.getenv("AUTH_PASSWORD", "")
    expected_hash = os.getenv("AUTH_PASSWORD_HASH", "")

    if not expected_password and not expected_hash:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Authentication is not configured on the server "
                "(set AUTH_PASSWORD_HASH or AUTH_PASSWORD)."
            ),
        )

    ip = _client_ip(http_request)
    _check_login_rate(ip)

    username_ok = hmac.compare_digest(request.username or "", expected_username)
    password_ok = _verify_password(request.password or "", expected_password, expected_hash)

    if not (username_ok and password_ok):
        _record_login_failure(ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    _clear_login_failures(ip)
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
