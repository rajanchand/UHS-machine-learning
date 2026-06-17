"""Pydantic v2 schemas for predictions, alerts, models, and stats."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from anomaly_detection.schemas.flows import FlowDetailResponse

# --- Prediction schemas ---


class PredictionResponse(BaseModel):
    """Schema for prediction API responses."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    flow_id: uuid.UUID
    model_name: str
    model_version: str
    score: float
    is_anomaly: bool
    threshold: float
    created_at: datetime


# --- Alert schemas ---


class AlertResponse(BaseModel):
    """Schema for alert API responses."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    flow_id: uuid.UUID
    severity: str
    suspected_attack_type: str | None = None
    status: str
    created_at: datetime
    feedback_verdict: str | None = None


class AlertDetailResponse(AlertResponse):
    """Alert detail including flow features and predictions."""

    model_config = ConfigDict(from_attributes=True)

    flow: FlowDetailResponse | None = None
    predictions: list[PredictionResponse] = Field(default_factory=list)


class AlertStatusUpdate(BaseModel):
    """Request schema for updating alert status."""

    status: str = Field(
        pattern="^(open|acknowledged|resolved)$",
        description="New alert status",
    )


# --- Model schemas ---


class ModelResponse(BaseModel):
    """Schema for ML model API responses."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    version: str
    trained_at: datetime
    metrics_json: dict[str, Any] | None = None
    artifact_path: str
    threshold: float
    is_active: bool
    description: str | None = None


class ThresholdUpdate(BaseModel):
    """Request schema for updating model threshold."""

    threshold: float = Field(ge=0.0, le=1.0, description="New anomaly threshold")


# --- Stats schemas ---


class TopTalker(BaseModel):
    """IP address with flow count for top-talkers display."""

    ip: str
    flow_count: int


class KPIResponse(BaseModel):
    """Key performance indicators for the dashboard."""

    total_flows: int = Field(ge=0)
    total_alerts: int = Field(ge=0)
    open_alerts: int = Field(ge=0)
    estimated_fpr: float = Field(ge=0.0, le=1.0)
    top_talkers: list[TopTalker] = Field(default_factory=list)


class TimelinePoint(BaseModel):
    """Single point in anomaly score timeline."""

    timestamp: datetime
    avg_score: float
    max_score: float
    flow_count: int
    anomaly_count: int


class TimelineResponse(BaseModel):
    """Anomaly score timeline data for charts."""

    points: list[TimelinePoint] = Field(default_factory=list)
    threshold: float


# --- SSE event schemas ---


class StreamEvent(BaseModel):
    """Schema for SSE stream events sent to the dashboard."""

    event_type: str = Field(description="Event type: flow, prediction, alert")
    flow_id: uuid.UUID
    ts: datetime
    src_ip: str
    dst_ip: str
    protocol: int
    score: float | None = None
    is_anomaly: bool | None = None
    model_name: str | None = None
    alert_id: uuid.UUID | None = None
    severity: str | None = None
    suspected_attack_type: str | None = None


# Forward reference resolution
from anomaly_detection.schemas.flows import FlowDetailResponse  # noqa: E402

AlertDetailResponse.model_rebuild()


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"
