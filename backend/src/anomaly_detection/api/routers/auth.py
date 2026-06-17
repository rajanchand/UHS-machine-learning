"""Authentication router for session-based login and logout."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from anomaly_detection.db.models import User
from anomaly_detection.utils.auth import verify_password

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    """Schema for login request."""

    username: str
    password: str


class AuthResponse(BaseModel):
    """Schema for authentication status response."""

    username: str
    status: str


def _get_session(request: Request) -> AsyncSession:
    """Get database session from request state."""
    factory = request.app.state.session_factory
    return cast("AsyncSession", factory())


@router.post("/login", response_model=AuthResponse)
async def login(request: Request, payload: LoginRequest) -> AuthResponse:
    """Authenticate a user and start a session."""
    async with _get_session(request) as session:
        result = await session.execute(
            select(User).where(User.username == payload.username)
        )
        user = result.scalar_one_or_none()

    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password",
        )

    # Start session by saving username in session cookie
    request.session["user"] = user.username
    return AuthResponse(username=user.username, status="authenticated")


@router.post("/logout")
async def logout(request: Request) -> dict[str, str]:
    """Terminate the current user session."""
    request.session.clear()
    return {"status": "logged_out"}


@router.get("/me", response_model=AuthResponse)
async def me(request: Request) -> AuthResponse:
    """Get the current authenticated user details."""
    user = request.session.get("user")
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
        )
    return AuthResponse(username=user, status="authenticated")
