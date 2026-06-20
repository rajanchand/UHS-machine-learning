"""Users router — CRUD operations, role assignment, activate/deactivate (admin only)."""

from __future__ import annotations

import contextlib
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from starlette.responses import Response

from anomaly_detection.authentication import hash_password
from anomaly_detection.db.models import User, UserRole, UserStatus
from anomaly_detection.schemas.common import UserCreateRequest, UserResponse, UserUpdateRequest

router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.get("")
async def list_users(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
    search: str | None = None,
) -> dict[str, Any]:
    """List all users (admin only)."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        query = select(User)
        count_query = select(func.count(User.id))

        if search:
            search_filter = (
                User.username.ilike(f"%{search}%")
                | User.email.ilike(f"%{search}%")
                | User.full_name.ilike(f"%{search}%")
            )
            query = query.where(search_filter)
            count_query = count_query.where(search_filter)

        total = (await session.execute(count_query)).scalar() or 0
        offset = (page - 1) * per_page
        result = await session.execute(
            query.order_by(User.created_at.desc()).offset(offset).limit(per_page)
        )
        users = result.scalars().all()

        return {
            "items": [
                UserResponse(
                    id=u.id,
                    username=u.username,
                    email=u.email,
                    full_name=u.full_name,
                    role=u.role.value,
                    status=u.status.value,
                    avatar_url=u.avatar_url,
                    last_login=u.last_login,
                    created_at=u.created_at,
                ).model_dump()
                for u in users
            ],
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": max(1, (total + per_page - 1) // per_page),
        }


@router.post("", response_model=None)
async def create_user(request: Request, body: UserCreateRequest) -> Response | dict[str, Any]:
    """Create a new user."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        # Check username exists
        result = await session.execute(select(User).where(User.username == body.username))
        if result.scalar_one_or_none():
            return JSONResponse(status_code=409, content={"detail": "Username already exists"})

        # Check email exists
        result = await session.execute(select(User).where(User.email == body.email))
        if result.scalar_one_or_none():
            return JSONResponse(status_code=409, content={"detail": "Email already registered"})

        try:
            role = UserRole(body.role)
        except ValueError:
            role = UserRole.ANALYST

        user = User(
            username=body.username,
            email=body.email,
            password_hash=hash_password(body.password),
            full_name=body.full_name,
            role=role,
            status=UserStatus.ACTIVE,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

        return {
            "message": "User created successfully",
            "user": UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                full_name=user.full_name,
                role=user.role.value,
                status=user.status.value,
                avatar_url=user.avatar_url,
                last_login=user.last_login,
                created_at=user.created_at,
            ).model_dump(),
        }


@router.put("/{user_id}", response_model=None)
async def update_user(request: Request, user_id: str, body: UserUpdateRequest) -> Response | dict[str, Any]:
    """Update user details."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            return JSONResponse(status_code=404, content={"detail": "User not found"})

        if body.full_name is not None:
            user.full_name = body.full_name
        if body.email is not None:
            user.email = body.email
        if body.phone is not None:
            user.phone = body.phone
        if body.role is not None:
            with contextlib.suppress(ValueError):
                user.role = UserRole(body.role)
        if body.status is not None:
            with contextlib.suppress(ValueError):
                user.status = UserStatus(body.status)

        await session.commit()

    return {"message": "User updated successfully"}


@router.delete("/{user_id}", response_model=None)
async def delete_user(request: Request, user_id: str) -> Response | dict[str, Any]:
    """Delete a user."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            return JSONResponse(status_code=404, content={"detail": "User not found"})

        await session.delete(user)
        await session.commit()

    return {"message": "User deleted successfully"}


@router.patch("/{user_id}/role", response_model=None)
async def assign_role(request: Request, user_id: str, body: dict[str, Any]) -> Response | dict[str, Any]:
    """Assign a role to a user."""
    session_factory = request.app.state.session_factory
    role_value = body.get("role", "analyst")

    async with session_factory() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            return JSONResponse(status_code=404, content={"detail": "User not found"})

        try:
            user.role = UserRole(role_value)
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid role"})

        await session.commit()

    return {"message": f"Role updated to {role_value}"}


@router.patch("/{user_id}/status", response_model=None)
async def toggle_status(request: Request, user_id: str, body: dict[str, Any]) -> Response | dict[str, Any]:
    """Activate or deactivate a user."""
    session_factory = request.app.state.session_factory
    status_value = body.get("status", "active")

    async with session_factory() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            return JSONResponse(status_code=404, content={"detail": "User not found"})

        try:
            user.status = UserStatus(status_value)
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "Invalid status"})

        await session.commit()

    return {"message": f"User status updated to {status_value}"}
