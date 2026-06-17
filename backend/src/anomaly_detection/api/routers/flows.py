"""Flow API routers — batch inference, streaming, and live feed."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import desc, select

from anomaly_detection.db.models import Alert, AlertSeverity, AlertStatus, Flow, Prediction
from anomaly_detection.schemas.common import PredictionResponse, StreamEvent
from anomaly_detection.schemas.flows import (
    BatchFlowRequest,
    FlowCreate,
    FlowResponse,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/v1/flows", tags=["flows"])

# In-memory broadcast channel for SSE
_sse_subscribers: list[asyncio.Queue[str]] = []


def _get_session(request: Request) -> AsyncSession:
    """Get session from app state."""
    factory = request.app.state.session_factory
    return cast("AsyncSession", factory())


def _get_inference_service(request: Request) -> object:
    """Get inference service from app state."""
    return request.app.state.inference_service


def _determine_severity(score: float, threshold: float) -> AlertSeverity:
    """Determine alert severity based on how far above threshold the score is."""
    excess = score - threshold
    if excess > 0.3:
        return AlertSeverity.CRITICAL
    if excess > 0.15:
        return AlertSeverity.HIGH
    if excess > 0.05:
        return AlertSeverity.MEDIUM
    return AlertSeverity.LOW


async def _broadcast_event(event: StreamEvent) -> None:
    """Send event to all SSE subscribers."""
    import json
    data = f"data: {json.dumps(event.model_dump(), default=str)}\n\n"
    dead: list[asyncio.Queue[str]] = []
    for queue in _sse_subscribers:
        try:
            queue.put_nowait(data)
        except asyncio.QueueFull:
            dead.append(queue)
    for q in dead:
        _sse_subscribers.remove(q)


@router.get("", response_model=list[FlowResponse])
async def list_flows(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[FlowResponse]:
    """List recent flows with pagination."""
    async with _get_session(request) as session:
        result = await session.execute(
            select(Flow).order_by(desc(Flow.ts)).limit(limit).offset(offset)
        )
        flows = result.scalars().all()
        return [FlowResponse.model_validate(f) for f in flows]


@router.post("/batch", response_model=list[PredictionResponse])
async def batch_inference(
    request: Request,
    batch: BatchFlowRequest,
) -> list[PredictionResponse]:
    """Score a batch of flows and persist results."""
    inference_svc = _get_inference_service(request)
    from anomaly_detection.services.inference import InferenceService
    assert isinstance(inference_svc, InferenceService)

    predictions: list[PredictionResponse] = []

    async with _get_session(request) as session:
        for flow_create in batch.flows:
            # Create flow record
            flow = Flow(
                ts=flow_create.ts,
                src_ip=flow_create.src_ip,
                src_port=flow_create.src_port,
                dst_ip=flow_create.dst_ip,
                dst_port=flow_create.dst_port,
                protocol=flow_create.protocol,
                label=flow_create.label,
                **flow_create.features.model_dump(),
            )
            session.add(flow)
            await session.flush()

            # Score
            import time
            start_time = time.perf_counter()
            feature_vec = flow_create.features.to_feature_vector()
            score, is_anomaly, model_name, threshold = inference_svc.score_flow(feature_vec)
            latency = time.perf_counter() - start_time

            # Update metrics
            app = request.app
            app.state.metrics_inference_count = getattr(app.state, "metrics_inference_count", 0) + 1
            app.state.metrics_inference_sum = getattr(app.state, "metrics_inference_sum", 0.0) + latency

            # Create prediction
            prediction = Prediction(
                flow_id=flow.id,
                model_name=model_name,
                model_version="v1",
                score=score,
                is_anomaly=is_anomaly,
                threshold=threshold,
            )
            session.add(prediction)

            # Create alert if anomaly
            if is_anomaly:
                severity = _determine_severity(score, threshold)
                alert = Alert(
                    flow_id=flow.id,
                    severity=severity,
                    suspected_attack_type=flow_create.label,
                    status=AlertStatus.OPEN,
                )
                session.add(alert)

            await session.flush()
            predictions.append(PredictionResponse.model_validate(prediction))

        await session.commit()

    return predictions


@router.post("/stream")
async def stream_inference(
    request: Request,
    flow_create: FlowCreate,
) -> PredictionResponse:
    """Score a single flow (for simulator) and broadcast to SSE subscribers."""
    inference_svc = _get_inference_service(request)
    from anomaly_detection.services.inference import InferenceService
    assert isinstance(inference_svc, InferenceService)

    async with _get_session(request) as session:
        # Create flow record
        flow = Flow(
            ts=flow_create.ts,
            src_ip=flow_create.src_ip,
            src_port=flow_create.src_port,
            dst_ip=flow_create.dst_ip,
            dst_port=flow_create.dst_port,
            protocol=flow_create.protocol,
            label=flow_create.label,
            **flow_create.features.model_dump(),
        )
        session.add(flow)
        await session.flush()

        # Score
        import time
        start_time = time.perf_counter()
        feature_vec = flow_create.features.to_feature_vector()
        score, is_anomaly, model_name, threshold = inference_svc.score_flow(feature_vec)
        latency = time.perf_counter() - start_time

        # Update metrics
        app = request.app
        app.state.metrics_inference_count = getattr(app.state, "metrics_inference_count", 0) + 1
        app.state.metrics_inference_sum = getattr(app.state, "metrics_inference_sum", 0.0) + latency

        # Create prediction
        prediction = Prediction(
            flow_id=flow.id,
            model_name=model_name,
            model_version="v1",
            score=score,
            is_anomaly=is_anomaly,
            threshold=threshold,
        )
        session.add(prediction)

        alert_id: uuid.UUID | None = None
        severity_str: str | None = None

        if is_anomaly:
            severity = _determine_severity(score, threshold)
            alert = Alert(
                flow_id=flow.id,
                severity=severity,
                suspected_attack_type=flow_create.label,
                status=AlertStatus.OPEN,
            )
            session.add(alert)
            await session.flush()
            alert_id = alert.id
            severity_str = severity.value

        await session.commit()

        # Broadcast to SSE subscribers
        event = StreamEvent(
            event_type="flow",
            flow_id=flow.id,
            ts=flow.ts,
            src_ip=flow.src_ip,
            dst_ip=flow.dst_ip,
            protocol=flow.protocol,
            score=score,
            is_anomaly=is_anomaly,
            model_name=model_name,
            alert_id=alert_id,
            severity=severity_str,
            suspected_attack_type=flow_create.label,
        )
        await _broadcast_event(event)

        return PredictionResponse.model_validate(prediction)


@router.get("/feed")
async def live_feed(request: Request) -> StreamingResponse:
    """SSE endpoint for real-time flow + prediction updates."""
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
    _sse_subscribers.append(queue)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield data
                except TimeoutError:
                    # Send keep-alive
                    yield ": keepalive\n\n"
        finally:
            if queue in _sse_subscribers:
                _sse_subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
