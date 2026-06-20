"""Alerts router — list, filter, update, resolve alerts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from starlette.responses import Response

from anomaly_detection.db.models import Alert, AlertSeverity, AlertStatus
from anomaly_detection.schemas.common import AlertResponse, AlertUpdateRequest

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


@router.get("")
async def list_alerts(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
    severity: str | None = None,
    status: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    """List alerts with pagination and filtering."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        query = select(Alert)
        count_query = select(func.count(Alert.id))

        if severity:
            query = query.where(Alert.severity == AlertSeverity(severity))
            count_query = count_query.where(Alert.severity == AlertSeverity(severity))
        if status:
            query = query.where(Alert.status == AlertStatus(status))
            count_query = count_query.where(Alert.status == AlertStatus(status))
        if search:
            search_filter = (
                Alert.title.ilike(f"%{search}%")
                | Alert.source_ip.ilike(f"%{search}%")
                | Alert.description.ilike(f"%{search}%")
            )
            query = query.where(search_filter)
            count_query = count_query.where(search_filter)

        total = (await session.execute(count_query)).scalar() or 0
        offset = (page - 1) * per_page
        result = await session.execute(
            query.order_by(Alert.created_at.desc()).offset(offset).limit(per_page)
        )
        alerts = result.scalars().all()

        return {
            "items": [
                AlertResponse(
                    id=a.id,
                    title=a.title,
                    description=a.description,
                    severity=a.severity.value,
                    status=a.status.value,
                    source_ip=a.source_ip,
                    attack_type=a.attack_type,
                    is_read=a.is_read,
                    created_at=a.created_at,
                    resolved_at=a.resolved_at,
                ).model_dump()
                for a in alerts
            ],
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": max(1, (total + per_page - 1) // per_page),
        }


@router.get("/count")
async def alert_counts(request: Request) -> dict[str, Any]:
    """Get counts by severity and status."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        total = (await session.execute(select(func.count(Alert.id)))).scalar() or 0
        critical = (
            await session.execute(
                select(func.count(Alert.id)).where(Alert.severity == AlertSeverity.CRITICAL)
            )
        ).scalar() or 0
        high = (
            await session.execute(
                select(func.count(Alert.id)).where(Alert.severity == AlertSeverity.HIGH)
            )
        ).scalar() or 0
        unread = (
            await session.execute(select(func.count(Alert.id)).where(Alert.is_read.is_(False)))
        ).scalar() or 0
        open_count = (
            await session.execute(
                select(func.count(Alert.id)).where(Alert.status == AlertStatus.OPEN)
            )
        ).scalar() or 0

        return {
            "total": total,
            "critical": critical,
            "high": high,
            "unread": unread,
            "open": open_count,
        }


@router.put("/{alert_id}", response_model=None)
async def update_alert(request: Request, alert_id: str, body: AlertUpdateRequest) -> Response | dict[str, Any]:
    """Update alert status or read state."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(Alert).where(Alert.id == alert_id))
        alert = result.scalar_one_or_none()
        if not alert:
            return JSONResponse(status_code=404, content={"detail": "Alert not found"})

        if body.status:
            try:
                alert.status = AlertStatus(body.status)
                if body.status == "resolved":
                    alert.resolved_at = datetime.now(UTC)
            except ValueError:
                pass
        if body.is_read is not None:
            alert.is_read = body.is_read

        await session.commit()

    return {"message": "Alert updated"}


@router.post("/mark-all-read")
async def mark_all_read(request: Request) -> dict[str, Any]:
    """Mark all alerts as read."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(Alert).where(Alert.is_read.is_(False)))
        for alert in result.scalars().all():
            alert.is_read = True
        await session.commit()

    return {"message": "All alerts marked as read"}
