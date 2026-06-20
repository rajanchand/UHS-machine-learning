"""Profile router — user profile view and update."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select
from starlette.responses import Response

from anomaly_detection.db.models import AuditLog, User
from anomaly_detection.schemas.common import UserResponse

router = APIRouter(prefix="/api/v1/profile", tags=["profile"])


@router.get("", response_model=None)
async def get_profile(request: Request) -> Response | dict[str, Any]:
    """Get current user profile."""
    session_factory = request.app.state.session_factory
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

        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            role=user.role.value,
            status=user.status.value,
            avatar_url=user.avatar_url,
            last_login=user.last_login,
            created_at=user.created_at,
        ).model_dump()


@router.put("", response_model=None)
async def update_profile(request: Request, body: dict[str, Any]) -> Response | dict[str, Any]:
    """Update user profile."""
    session_factory = request.app.state.session_factory
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

        if "full_name" in body:
            user.full_name = body["full_name"]
        if "email" in body:
            user.email = body["email"]
        if "phone" in body:
            user.phone = body["phone"]

        await session.commit()

    return {"message": "Profile updated successfully"}


@router.get("/activity", response_model=None)
async def user_activity(request: Request) -> Response | list[dict[str, Any]]:
    """Get user's recent activity log."""
    session_factory = request.app.state.session_factory
    user_info = getattr(request.state, "user", None)

    if not user_info:
        return JSONResponse(status_code=401, content={"detail": "Not authenticated"})

    async with session_factory() as session:
        result = await session.execute(
            select(AuditLog)
            .where(AuditLog.user_id == user_info.get("user_id"))
            .order_by(AuditLog.created_at.desc())
            .limit(20)
        )
        logs = result.scalars().all()

        return [
            {
                "action": log_entry.action,
                "resource": log_entry.resource,
                "details": log_entry.details,
                "timestamp": log_entry.created_at.isoformat(),
            }
            for log_entry in logs
        ]
