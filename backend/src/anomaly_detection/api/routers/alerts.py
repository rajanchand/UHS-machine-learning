"""Alert API routers — list, detail, status updates."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, HTTPException, Query, Request
from sqlalchemy import desc, select
from sqlalchemy.orm import selectinload

from anomaly_detection.db.models import Alert, AlertStatus, Flow
from anomaly_detection.schemas.common import AlertDetailResponse, AlertResponse, AlertStatusUpdate

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


def _get_session(request: Request) -> AsyncSession:
    """Get session from app state."""
    factory = request.app.state.session_factory
    return cast("AsyncSession", factory())


@router.get("", response_model=list[AlertResponse])
async def list_alerts(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    status: str | None = Query(None, pattern="^(open|acknowledged|resolved)$"),
    severity: str | None = Query(None, pattern="^(low|medium|high|critical)$"),
    attack_type: str | None = Query(None),
) -> list[AlertResponse]:
    """List alerts with optional filters."""
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
        result = await session.execute(query)
        alerts = result.scalars().all()

        from anomaly_detection.db.models import Feedback
        if alerts:
            alert_ids = [a.id for a in alerts]
            feedback_result = await session.execute(
                select(Feedback).where(Feedback.alert_id.in_(alert_ids))
            )
            feedback_map = {f.alert_id: f.verdict for f in feedback_result.scalars().all()}
        else:
            feedback_map = {}

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


@router.get("/{alert_id}", response_model=AlertDetailResponse)
async def get_alert_detail(
    request: Request,
    alert_id: str,
) -> AlertDetailResponse:
    """Get alert detail with associated flow and predictions."""
    import uuid

    try:
        alert_uuid = uuid.UUID(alert_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid alert ID format")

    async with _get_session(request) as session:
        result = await session.execute(
            select(Alert)
            .options(selectinload(Alert.flow).selectinload(Flow.predictions))
            .where(Alert.id == alert_uuid)
        )
        alert = result.scalar_one_or_none()

        if alert is None:
            raise HTTPException(status_code=404, detail="Alert not found")

        from anomaly_detection.db.models import Feedback
        feedback_result = await session.execute(
            select(Feedback).where(Feedback.alert_id == alert_uuid)
        )
        feedback = feedback_result.scalar_one_or_none()
        feedback_verdict = feedback.verdict if feedback else None

        return AlertDetailResponse(
            id=alert.id,
            flow_id=alert.flow_id,
            severity=alert.severity.value,
            suspected_attack_type=alert.suspected_attack_type,
            status=alert.status.value,
            created_at=alert.created_at,
            feedback_verdict=feedback_verdict,
        )


@router.patch("/{alert_id}/status", response_model=AlertResponse)
async def update_alert_status(
    request: Request,
    alert_id: str,
    update: AlertStatusUpdate,
) -> AlertResponse:
    """Update alert status (acknowledge/resolve)."""
    import uuid

    try:
        alert_uuid = uuid.UUID(alert_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid alert ID format")

    async with _get_session(request) as session:
        result = await session.execute(
            select(Alert).where(Alert.id == alert_uuid)
        )
        alert = result.scalar_one_or_none()

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


from pydantic import BaseModel


class FeedbackSubmit(BaseModel):
    """Schema for submitting alert feedback."""

    verdict: str  # "true_positive" or "false_positive"


@router.post("/{alert_id}/feedback")
async def submit_alert_feedback(
    request: Request,
    alert_id: str,
    payload: FeedbackSubmit,
) -> dict[str, str]:
    """Submit analyst feedback (true positive / false positive) for an alert."""
    import uuid
    from pydantic import ValidationError
    from anomaly_detection.db.models import Feedback

    if payload.verdict not in ("true_positive", "false_positive"):
        raise HTTPException(status_code=400, detail="Invalid verdict")

    try:
        alert_uuid = uuid.UUID(alert_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid alert ID format")

    username = request.session.get("user", "anonymous")

    async with _get_session(request) as session:
        # Check if alert exists
        alert_result = await session.execute(
            select(Alert).where(Alert.id == alert_uuid)
        )
        alert = alert_result.scalar_one_or_none()
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")

        # Upsert feedback
        result = await session.execute(
            select(Feedback).where(Feedback.alert_id == alert_uuid)
        )
        feedback = result.scalar_one_or_none()

        if feedback:
            feedback.verdict = payload.verdict
            feedback.user = username
            feedback.created_at = datetime.now()
        else:
            feedback = Feedback(
                alert_id=alert_uuid,
                verdict=payload.verdict,
                user=username,
            )
            session.add(feedback)

        await session.commit()
        return {"status": "feedback_submitted", "verdict": payload.verdict}


@router.get("/feedback/export")
async def export_feedback(request: Request):
    """Export the accumulated analyst feedback along with features as a downloadable CSV."""
    import csv
    import io
    from fastapi.responses import StreamingResponse
    from sqlalchemy.orm import selectinload
    from anomaly_detection.db.models import Feedback

    feature_cols = [
        'flow_duration', 'total_fwd_packets', 'total_bwd_packets', 'total_len_fwd_packets', 'total_len_bwd_packets',
        'fwd_packet_len_max', 'fwd_packet_len_min', 'fwd_packet_len_mean', 'fwd_packet_len_std', 'bwd_packet_len_max',
        'bwd_packet_len_min', 'bwd_packet_len_mean', 'bwd_packet_len_std', 'flow_bytes_per_s', 'flow_packets_per_s',
        'fin_flag_count', 'syn_flag_count', 'rst_flag_count', 'psh_flag_count', 'ack_flag_count', 'urg_flag_count',
        'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max',
        'fwd_iat_min', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min', 'down_up_ratio', 'avg_packet_size',
        'avg_fwd_segment_size', 'avg_bwd_segment_size'
    ]

    async with _get_session(request) as session:
        # Load feedback with alerts and flows
        result = await session.execute(
            select(Feedback)
            .options(selectinload(Feedback.alert).selectinload(Alert.flow))
        )
        feedbacks = result.scalars().all()

    # Generate CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)

    # Headers
    headers = [
        "feedback_id", "alert_id", "flow_id", "verdict", "user", "created_at",
        "original_label", "suspected_attack_type", "corrected_label"
    ] + feature_cols
    writer.writerow(headers)

    for f in feedbacks:
        alert = f.alert
        flow = alert.flow if alert else None
        
        if not flow:
            continue

        # Corrected label logic: if true positive, the anomaly is real (we use suspected attack type or generic ANOMALY)
        # If false positive, we correct it to BENIGN.
        if f.verdict == "true_positive":
            corrected_label = alert.suspected_attack_type or "ANOMALY"
        else:
            corrected_label = "BENIGN"

        row = [
            str(f.id), str(f.alert_id), str(flow.id), f.verdict, f.user, str(f.created_at),
            flow.label or "UNKNOWN", alert.suspected_attack_type or "", corrected_label
        ]
        # Append all 37 numerical features
        for col in feature_cols:
            val = getattr(flow, col, 0.0)
            row.append(str(val))
            
        writer.writerow(row)

    output.seek(0)
    response = StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv"
    )
    response.headers["Content-Disposition"] = "attachment; filename=analyst_feedback_dataset.csv"
    return response

# Standard schema imports
from pydantic import BaseModel
from datetime import datetime

