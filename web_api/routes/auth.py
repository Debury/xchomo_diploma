"""Authentication endpoints."""

import os
import secrets
from typing import Dict

from fastapi import APIRouter, Query

from web_api.models import AuthRequest, AuthResponse

router = APIRouter(prefix="/auth", tags=["auth"])

# Simple token storage (in production, use Redis or database)
_valid_tokens: Dict[str, str] = {}


@router.post("/login", response_model=AuthResponse)
async def auth_login(request: AuthRequest):
    """Simple authentication endpoint."""
    expected_username = os.getenv("AUTH_USERNAME", "admin")
    expected_password = os.getenv("AUTH_PASSWORD", "")

    if request.username == expected_username and request.password == expected_password:
        token = secrets.token_urlsafe(32)
        _valid_tokens[token] = request.username
        return AuthResponse(
            success=True,
            token=token,
            message="Login successful",
            username=request.username,
        )

    return AuthResponse(success=False, message="Invalid username or password")


@router.post("/logout")
async def auth_logout(token: str = Query(...)):
    """Invalidate a token."""
    if token in _valid_tokens:
        del _valid_tokens[token]
        return {"success": True, "message": "Logged out"}
    return {"success": False, "message": "Invalid token"}


@router.get("/verify")
async def auth_verify(token: str = Query(...)):
    """Verify if a token is valid."""
    if token in _valid_tokens:
        return {"valid": True, "username": _valid_tokens[token]}
    return {"valid": False}
