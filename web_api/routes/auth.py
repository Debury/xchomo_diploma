"""Authentication endpoints and the bearer-token gate used to protect the API.

Library-backed implementation — no hand-rolled crypto:

* Password hashing: ``pwdlib`` with argon2id (OWASP-recommended as of 2024).
* Tokens: stateless JWT (``PyJWT``) signed with HS256. No server-side token
  table — a token is valid iff its signature verifies and it hasn't expired.
* Scheme: FastAPI's built-in ``OAuth2PasswordBearer`` so Swagger UI wires up
  the lock icon automatically.

What we keep:

* Per-IP login throttling backed by Postgres (``login_failures`` table).
* Startup audit log for mis-configured passwords.
* ``AUTH_ALLOW_ANONYMOUS=1`` opt-in for CI / smoke tests.
"""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from pwdlib import PasswordHash

from src.database.connection import get_db_session
from src.database.models import LoginFailure
from web_api.models import AuthRequest, AuthResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────

_TOKEN_TTL_SECONDS = int(os.getenv("AUTH_TOKEN_TTL_SECONDS", "86400"))  # 24 h
_JWT_ALGORITHM = "HS256"
_LOGIN_WINDOW_SECONDS = int(os.getenv("AUTH_LOGIN_WINDOW_SECONDS", "900"))  # 15 min
_LOGIN_MAX_FAILURES = int(os.getenv("AUTH_LOGIN_MAX_FAILURES", "10"))


def _jwt_secret() -> str:
    """Resolve the JWT signing secret.

    Prefers ``JWT_SECRET_KEY`` from the environment. Falls back to
    ``AUTH_PASSWORD_HASH`` as a deterministic-but-deployment-local secret so
    a forgotten ``JWT_SECRET_KEY`` doesn't silently make every token predictable.
    Raises at verify time if neither is set; boot audit warns earlier.
    """
    secret = os.getenv("JWT_SECRET_KEY") or os.getenv("AUTH_PASSWORD_HASH") or os.getenv("AUTH_PASSWORD")
    if not secret:
        raise RuntimeError(
            "Cannot sign/verify JWTs: set JWT_SECRET_KEY (preferred) or any "
            "AUTH_PASSWORD* value. Generate with `python -c 'import secrets; "
            "print(secrets.token_urlsafe(64))'`."
        )
    return secret


# Single global PasswordHash instance. ``recommended()`` selects argon2id with
# OWASP-2024 parameters; we don't need bcrypt/scrypt fallback in this project.
_password_hash = PasswordHash.recommended()

# Dummy hash used to make username-enumeration timing attacks harder: we always
# run verify() even when the user doesn't exist so the response time is uniform.
_DUMMY_HASH = _password_hash.hash("dummy-do-not-use")

# Swagger-visible bearer scheme. Points at /auth/login so the Authorize button
# in /docs wires up with the real endpoint.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


# ──────────────────────────────────────────────────────────────────────
# Password helpers (library-backed)
# ──────────────────────────────────────────────────────────────────────


def hash_password(password: str) -> str:
    """Return an argon2id encoded hash for ``AUTH_PASSWORD_HASH``."""
    return _password_hash.hash(password)


def _verify_password(provided: str, expected_plain: str, expected_hash: str) -> bool:
    """Hash wins if configured; plain is accepted as a one-time fallback."""
    if expected_hash:
        try:
            return _password_hash.verify(provided, expected_hash)
        except Exception:
            # Malformed hash string — treat as denial, don't fall through
            # silently to the plaintext compare.
            logger.error("AUTH_PASSWORD_HASH is malformed — refusing to verify")
            return False
    if expected_plain:
        # Still avoid timing leaks on equal-length strings.
        return secrets.compare_digest(provided, expected_plain)
    return False


def _auth_configured() -> bool:
    return bool(os.getenv("AUTH_PASSWORD_HASH") or os.getenv("AUTH_PASSWORD"))


def _anonymous_allowed() -> bool:
    return os.getenv("AUTH_ALLOW_ANONYMOUS", "").lower() in ("1", "true", "yes")


def _client_ip(request: Optional[Request]) -> str:
    if request is None:
        return "unknown"
    # Prefer X-Real-IP (Caddy sets this); fall back to X-Forwarded-For's first
    # entry; then the direct socket. Trusting XFF from arbitrary clients is a
    # known weak point — we accept it because Caddy is the only public edge.
    real = request.headers.get("x-real-ip")
    if real:
        return real.strip()
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",", 1)[0].strip() or "unknown"
    return (request.client.host if request.client else "unknown") or "unknown"


# ──────────────────────────────────────────────────────────────────────
# JWT helpers
# ──────────────────────────────────────────────────────────────────────


def _create_access_token(username: str, ttl_seconds: int = _TOKEN_TTL_SECONDS) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": username,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ttl_seconds)).timestamp()),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm=_JWT_ALGORITHM)


def _decode_access_token(token: str) -> Optional[str]:
    """Return the ``sub`` claim if the token is valid, else None."""
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=[_JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
    except RuntimeError:
        # _jwt_secret() raises when the server is not configured. Treat as
        # unauthenticated — require_auth's _auth_configured() branch handles
        # the 503 for the genuine misconfiguration case.
        return None
    sub = payload.get("sub")
    return sub if isinstance(sub, str) else None


# ──────────────────────────────────────────────────────────────────────
# Rate limiting (Postgres-backed)
# ──────────────────────────────────────────────────────────────────────


def _check_login_rate(ip: str) -> None:
    """Raise 429 if this IP has used up its failure budget in the window."""
    now = datetime.utcnow()
    cutoff = now - timedelta(seconds=_LOGIN_WINDOW_SECONDS)
    try:
        with get_db_session() as session:
            session.query(LoginFailure).filter(
                LoginFailure.ip == ip,
                LoginFailure.attempted_at <= cutoff,
            ).delete(synchronize_session=False)

            timestamps = [
                ts for (ts,) in session.query(LoginFailure.attempted_at)
                .filter(LoginFailure.ip == ip)
                .order_by(LoginFailure.attempted_at.asc())
                .all()
            ]
    except Exception as err:  # pragma: no cover - DB outage fallback
        logger.warning(f"Rate-limit lookup failed, allowing request: {err}")
        return

    if len(timestamps) >= _LOGIN_MAX_FAILURES:
        oldest = timestamps[0]
        retry_after = max(
            1, int((oldest + timedelta(seconds=_LOGIN_WINDOW_SECONDS) - now).total_seconds())
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )


def _record_login_failure(ip: str) -> None:
    try:
        with get_db_session() as session:
            session.add(LoginFailure(ip=ip))
    except Exception as err:
        logger.warning(f"Could not record login failure for {ip}: {err}")


def _clear_login_failures(ip: str) -> None:
    try:
        with get_db_session() as session:
            session.query(LoginFailure).filter(LoginFailure.ip == ip).delete(
                synchronize_session=False
            )
    except Exception as err:
        logger.warning(f"Could not clear login failures for {ip}: {err}")


# ──────────────────────────────────────────────────────────────────────
# FastAPI dependency — use on every protected route
# ──────────────────────────────────────────────────────────────────────


def require_auth(token: Optional[str] = Depends(oauth2_scheme)) -> str:
    """FastAPI dependency: reject requests without a valid JWT.

    Returns the authenticated username on success. Fails closed when no password
    is configured unless ``AUTH_ALLOW_ANONYMOUS=1`` is explicitly set (CI only).
    """
    if not _auth_configured():
        if _anonymous_allowed():
            return "anonymous"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Authentication is not configured. Set AUTH_PASSWORD_HASH "
                "(preferred) or AUTH_PASSWORD, or AUTH_ALLOW_ANONYMOUS=1 for CI."
            ),
        )

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    username = _decode_access_token(token)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username


# ──────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────


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

    # Run verify against a dummy hash even on unknown usernames so both
    # success and failure paths spend the same time in argon2 (defends
    # against username enumeration via response timing).
    username_ok = secrets.compare_digest(request.username or "", expected_username)
    if username_ok:
        password_ok = _verify_password(request.password or "", expected_password, expected_hash)
    else:
        _password_hash.verify(request.password or "x", _DUMMY_HASH)
        password_ok = False

    if not (username_ok and password_ok):
        _record_login_failure(ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    _clear_login_failures(ip)
    token = _create_access_token(expected_username)
    return AuthResponse(
        success=True,
        token=token,
        message="Login successful",
        username=expected_username,
    )


@router.post("/logout")
async def auth_logout(token: Optional[str] = Depends(oauth2_scheme)):
    """Client-side logout acknowledgement.

    With stateless JWT there is no server-side session to invalidate. The
    client drops the token and we ack with 200 — or 401 if no token was
    provided (to match the old behaviour so the frontend can treat both
    auth schemes the same).
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No token provided",
        )
    return {"success": True, "message": "Logged out"}


@router.get("/verify")
async def auth_verify(token: Optional[str] = Depends(oauth2_scheme)):
    """Verify a bearer token's signature + expiry. Used by the SPA at boot
    to decide whether a cached token is still good, avoiding the flash of
    protected content on reload."""
    if not token:
        return {"valid": False}
    username = _decode_access_token(token)
    if username:
        return {"valid": True, "username": username}
    return {"valid": False}
