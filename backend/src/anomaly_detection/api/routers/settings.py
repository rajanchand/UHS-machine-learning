"""Settings router — system configuration, password change, API settings."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from starlette.responses import Response

from anomaly_detection.authentication import hash_password, verify_password
from anomaly_detection.db.models import Setting, User
from anomaly_detection.schemas.common import (
    ChangePasswordRequest,
    SettingResponse,
    SettingUpdateRequest,
)

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


@router.get("")
async def get_settings(request: Request) -> list[dict[str, Any]]:
    """Get all system settings."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(Setting))
        settings = result.scalars().all()

        if not settings:
            # Return default settings
            return [
                {"key": "theme", "value": "light", "description": "UI Theme"},
                {
                    "key": "notification_email",
                    "value": "true",
                    "description": "Email notifications",
                },
                {
                    "key": "notification_browser",
                    "value": "true",
                    "description": "Browser notifications",
                },
                {"key": "auto_refresh", "value": "true", "description": "Auto-refresh dashboard"},
                {
                    "key": "refresh_interval",
                    "value": "30",
                    "description": "Refresh interval (seconds)",
                },
                {"key": "api_url", "value": "http://localhost:8000", "description": "API Base URL"},
                {
                    "key": "max_packet_capture",
                    "value": "10000",
                    "description": "Max packets per capture",
                },
                {
                    "key": "alert_threshold",
                    "value": "0.8",
                    "description": "Alert confidence threshold",
                },
            ]

        return [
            SettingResponse(key=s.key, value=s.value, description=s.description).model_dump()
            for s in settings
        ]


@router.put("")
async def update_settings(request: Request, body: SettingUpdateRequest) -> dict[str, Any]:
    """Update system settings."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        for key, value in body.settings.items():
            result = await session.execute(select(Setting).where(Setting.key == key))
            setting = result.scalar_one_or_none()
            if setting:
                setting.value = value
            else:
                session.add(Setting(key=key, value=value))

        await session.commit()

    return {"message": "Settings updated successfully"}


@router.put("/password", response_model=None)
async def change_password(request: Request, body: ChangePasswordRequest) -> Response | dict[str, Any]:
    """Change user password."""
    session_factory = request.app.state.session_factory

    # Get user from JWT context (simplified for demo)
    user_info = getattr(request.state, "user", None)
    if not user_info:
        return JSONResponse(status_code=401, content={"detail": "Not authenticated"})

    async with session_factory() as session:
        result = await session.execute(
            select(User).where(User.username == user_info.get("username"))
        )
        user = result.scalar_one_or_none()
        if not user:
            return JSONResponse(status_code=404, content={"detail": "User not found"})

        if not verify_password(body.current_password, user.password_hash):
            return JSONResponse(
                status_code=400, content={"detail": "Current password is incorrect"}
            )

        user.password_hash = hash_password(body.new_password)
        await session.commit()

    return {"message": "Password changed successfully"}


@router.get("/api")
async def api_settings(request: Request) -> dict[str, Any]:
    """Get API configuration info."""
    settings = request.app.state.settings
    return {
        "api_url": f"http://{settings.api_host}:{settings.api_port}",
        "api_version": "v1",
        "environment": settings.environment,
        "jwt_algorithm": settings.jwt_algorithm,
        "jwt_expiry_minutes": settings.jwt_expiry_minutes,
        "cors_origins": settings.cors_origins,
        "rate_limit": settings.login_rate_limit,
    }
