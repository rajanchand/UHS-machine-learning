"""Alert API routers — list, detail, status updates, feedback, and CSV export."""

from __future__ import annotations

import csv
import io
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.orm import selectinload

from anomaly_detection.constants import FEATURE_COLUMNS
from anomaly_detection.db.models import Alert, AlertStatus, Feedback, Flow
from anomaly_detection.schemas.common import AlertDetailResponse, AlertResponse, AlertStatusUpdate

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


class VerdictEnum(StrEnum):
    TRUE_POSITIVE = "true_positive"
    FALSE_POSITIVE = "false_positive"


class FeedbackSubmit(BaseModel):
    verdict: VerdictEnum


def _get_session(request: Request) -> AsyncSession:
    return cast("AsyncSession", request.app.state.session_factory())


def _require_auth(request: Request) -> str:
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return cast(str, user)


@router.get("", response_model=list[AlertResponse])
async def list_alerts(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None, pattern="^(open|acknowledged|resolved)$"),
    severity: str | None = Query(None, pattern="^(low|medium|high|critical)$"),
    attack_type: str | None = Query(None),
) -> list[AlertResponse]:
    async with _get_session(request) as session:
        query = select(Alert).order_by(desc(Alert.created_at))

        if status:
            query = query.where(Alert.status == AlertStatus(status))
        if severity:
            from anomaly_detection.db.models import AlertSeverity
            query = query.where(Alert.severity == AlertSeverity(severity))
        if attack_type:
            query = query.where(Alert.suspected_attack_type == attack_type)

        query = query.limit(limit).offset(offset)
        alerts = (await session.execute(query)).scalars().all()

        feedback_map: dict[uuid.UUID, str] = {}
        if alerts:
            alert_ids = [a.id for a in alerts]
            feedbacks = (
                await session.execute(select(Feedback).where(Feedback.alert_id.in_(alert_ids)))
            ).scalars().all()
            feedback_map = {f.alert_id: f.verdict for f in feedbacks}

        return [
            AlertResponse(
                id=a.id,
                flow_id=a.flow_id,
                severity=a.severity.value,
                suspected_attack_type=a.suspected_attack_type,
                status=a.status.value,
                created_at=a.created_at,
                feedback_verdict=feedback_map.get(a.id),
            )
            for a in alerts
        ]


@router.get("/feedback/export")
async def export_feedback(request: Request) -> StreamingResponse:
    """Download analyst-labelled feedback as CSV for model retraining."""
    _require_auth(request)

    async with _get_session(request) as session:
        feedbacks = (
            await session.execute(
                select(Feedback).options(
                    selectinload(Feedback.alert).selectinload(Alert.flow)
                )
            )
        ).scalars().all()

    output = io.StringIO()
    writer = csv.writer(output)

    headers = [
        "feedback_id", "alert_id", "flow_id", "verdict", "user", "created_at",
        "original_label", "suspected_attack_type", "corrected_label",
        *FEATURE_COLUMNS,
    ]
    writer.writerow(headers)

    for fb in feedbacks:
        alert = fb.alert
        flow: Flow | None = alert.flow if alert else None
        if flow is None:
            continue

        corrected_label = (
            alert.suspected_attack_type or "ANOMALY"
            if fb.verdict == VerdictEnum.TRUE_POSITIVE
            else "BENIGN"
        )

        row = [
            str(fb.id), str(fb.alert_id), str(flow.id), fb.verdict, fb.user,
            fb.created_at.isoformat(),
            flow.label or "UNKNOWN", alert.suspected_attack_type or "", corrected_label,
            *[str(getattr(flow, col, 0.0)) for col in FEATURE_COLUMNS],
        ]
        writer.writerow(row)

    output.seek(0)
    response = StreamingResponse(iter([output.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=analyst_feedback_dataset.csv"
    return response


@router.get("/{alert_id}", response_model=AlertDetailResponse)
async def get_alert_detail(request: Request, alert_id: str) -> AlertDetailResponse:
    try:
        alert_uuid = uuid.UUID(alert_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid alert ID")

    async with _get_session(request) as session:
        alert = (
            await session.execute(
                select(Alert)
                .options(selectinload(Alert.flow).selectinload(Flow.predictions))
                .where(Alert.id == alert_uuid)
            )
        ).scalar_one_or_none()

        if alert is None:
            raise HTTPException(status_code=404, detail="Alert not found")

        feedback = (
            await session.execute(select(Feedback).where(Feedback.alert_id == alert_uuid))
        ).scalar_one_or_none()

        return AlertDetailResponse(
            id=alert.id,
            flow_id=alert.flow_id,
            severity=alert.severity.value,
            suspected_attack_type=alert.suspected_attack_type,
            status=alert.status.value,
            created_at=alert.created_at,
            feedback_verdict=feedback.verdict if feedback else None,
        )


@router.patch("/{alert_id}/status", response_model=AlertResponse)
async def update_alert_status(
    request: Request,
    alert_id: str,
    update: AlertStatusUpdate,
) -> AlertResponse:
    try:
        alert_uuid = uuid.UUID(alert_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid alert ID")

    async with _get_session(request) as session:
        alert = (
            await session.execute(select(Alert).where(Alert.id == alert_uuid))
        ).scalar_one_or_none()

        if alert is None:
            raise HTTPException(status_code=404, detail="Alert not found")

        alert.status = AlertStatus(update.status)
        await session.commit()

        return AlertResponse(
            id=alert.id,
            flow_id=alert.flow_id,
            severity=alert.severity.value,
            suspected_attack_type=alert.suspected_attack_type,
            status=alert.status.value,
            created_at=alert.created_at,
        )


@router.post("/{alert_id}/feedback")
async def submit_alert_feedback(
    request: Request,
    alert_id: str,
    payload: FeedbackSubmit,
) -> dict[str, str]:
    try:
        alert_uuid = uuid.UUID(alert_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid alert ID")

    username = request.session.get("user", "anonymous")

    async with _get_session(request) as session:
        alert = (
            await session.execute(select(Alert).where(Alert.id == alert_uuid))
        ).scalar_one_or_none()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        feedback = (
            await session.execute(select(Feedback).where(Feedback.alert_id == alert_uuid))
        ).scalar_one_or_none()

        if feedback:
            feedback.verdict = payload.verdict.value
            feedback.user = username
            feedback.created_at = datetime.now(UTC)
        else:
            session.add(Feedback(
                alert_id=alert_uuid,
                verdict=payload.verdict.value,
                user=username,
            ))

        await session.commit()

    return {"status": "feedback_submitted", "verdict": payload.verdict.value}
