"""Authentication router — JWT login, register, forgot password, refresh, logout."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import select

from anomaly_detection.authentication import (
    create_token_pair,
    get_user_from_token,
    hash_password,
    verify_password,
)
from anomaly_detection.db.models import LoginLog, User, UserRole, UserStatus
from anomaly_detection.schemas.common import (
    ForgotPasswordRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(request: Request, body: LoginRequest) -> TokenResponse:
    """Authenticate user and return JWT token pair."""
    settings = request.app.state.settings
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(User).where(User.username == body.username))
        user = result.scalar_one_or_none()

        # Log the attempt
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "")

        if not user or not verify_password(body.password, user.password_hash):
            # Log failed login
            session.add(
                LoginLog(
                    user_id=user.id if user else None,
                    username=body.username,
                    ip_address=client_ip,
                    user_agent=user_agent,
                    success=False,
                    failure_reason="Invalid credentials",
                )
            )
            await session.commit()
            raise HTTPException(status_code=401, detail="Invalid username or password")

        if user.status != UserStatus.ACTIVE:
            session.add(
                LoginLog(
                    user_id=user.id,
                    username=body.username,
                    ip_address=client_ip,
                    user_agent=user_agent,
                    success=False,
                    failure_reason=f"Account {user.status.value}",
                )
            )
            await session.commit()
            raise HTTPException(status_code=403, detail="Account is not active")

        # Create token pair
        token_pair = create_token_pair(
            user_id=str(user.id),
            username=user.username,
            role=user.role.value,
            secret_key=settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
            expiry_minutes=settings.jwt_expiry_minutes,
            refresh_expiry_days=settings.jwt_refresh_expiry_days,
        )

        # Update last login
        user.last_login = datetime.now(UTC)

        # Log successful login
        session.add(
            LoginLog(
                user_id=user.id,
                username=user.username,
                ip_address=client_ip,
                user_agent=user_agent,
                success=True,
            )
        )
        await session.commit()

        user_resp = UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role.value,
            status=user.status.value,
            avatar_url=user.avatar_url,
            last_login=user.last_login,
            created_at=user.created_at,
        )

        return TokenResponse(
            access_token=token_pair.access_token,
            refresh_token=token_pair.refresh_token,
            token_type="bearer",
            expires_in=token_pair.expires_in,
            user=user_resp,
        )


@router.post("/register")
async def register(request: Request, body: RegisterRequest) -> dict[str, Any]:
    """Register a new user account."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        # Check username exists
        result = await session.execute(select(User).where(User.username == body.username))
        if result.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Username already exists")

        # Check email exists
        result = await session.execute(select(User).where(User.email == body.email))
        if result.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Email already registered")

        # Create user
        user = User(
            username=body.username,
            email=body.email,
            password_hash=hash_password(body.password),
            full_name=body.full_name,
            role=UserRole.ANALYST,
            status=UserStatus.ACTIVE,
        )
        session.add(user)
        await session.commit()

        return {"message": "Registration successful", "username": user.username}


@router.post("/forgot-password")
async def forgot_password(body: ForgotPasswordRequest) -> dict[str, Any]:
    """Send password reset instructions (mock implementation for demo)."""
    # In production, this would send an email with reset link
    return {
        "message": "If this email is registered, password reset instructions have been sent.",
        "email": body.email,
    }


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: Request, body: RefreshRequest) -> TokenResponse:
    """Refresh an access token using a refresh token."""
    settings = request.app.state.settings
    session_factory = request.app.state.session_factory

    user_info = get_user_from_token(
        body.refresh_token, settings.jwt_secret_key, settings.jwt_algorithm
    )
    if not user_info or user_info.get("token_type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    async with session_factory() as session:
        result = await session.execute(select(User).where(User.id == user_info["user_id"]))
        user = result.scalar_one_or_none()
        if not user or user.status != UserStatus.ACTIVE:
            raise HTTPException(status_code=401, detail="User not found or inactive")

        token_pair = create_token_pair(
            user_id=str(user.id),
            username=user.username,
            role=user.role.value,
            secret_key=settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
            expiry_minutes=settings.jwt_expiry_minutes,
            refresh_expiry_days=settings.jwt_refresh_expiry_days,
        )

        return TokenResponse(
            access_token=token_pair.access_token,
            refresh_token=token_pair.refresh_token,
            token_type="bearer",
            expires_in=token_pair.expires_in,
        )


@router.post("/logout")
async def logout() -> dict[str, Any]:
    """Logout — client should discard tokens."""
    return {"message": "Logged out successfully"}
