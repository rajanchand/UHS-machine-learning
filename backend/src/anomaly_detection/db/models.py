"""SQLAlchemy ORM models for the anomaly detection database.

Tables:
- flows: Network flow records (TimescaleDB hypertable on ts)
- predictions: Model inference results (TimescaleDB hypertable on created_at)
- alerts: Anomaly alerts with status tracking
- ml_models: Registered model metadata and metrics
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class AlertStatus(enum.Enum):
    """Status lifecycle for alerts."""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class AlertSeverity(enum.Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Flow(Base):
    """Network flow record — TimescaleDB hypertable partitioned on ts."""

    __tablename__ = "flows"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # Network identifiers
    src_ip: Mapped[str] = mapped_column(String(45), nullable=False)
    src_port: Mapped[int] = mapped_column(Integer, nullable=False)
    dst_ip: Mapped[str] = mapped_column(String(45), nullable=False)
    dst_port: Mapped[int] = mapped_column(Integer, nullable=False)
    protocol: Mapped[int] = mapped_column(Integer, nullable=False)

    # Flow features - NSL-KDD 41 standard + 7 derived features
    duration: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    protocol_type: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    service: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flag: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    src_bytes: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    dst_bytes: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    land: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    wrong_fragment: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    urgent: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    hot: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    num_failed_logins: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    logged_in: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    num_compromised: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    root_shell: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    su_attempted: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    num_root: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    num_file_creations: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    num_shells: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    num_access_files: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    num_outbound_cmds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_host_login: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    is_guest_login: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    count: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    srv_count: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    serror_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    srv_serror_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    rerror_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    srv_rerror_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    same_srv_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    diff_srv_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    srv_diff_host_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    dst_host_count: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    dst_host_srv_count: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    dst_host_same_srv_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    dst_host_diff_srv_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    dst_host_same_src_port_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    dst_host_srv_diff_host_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    dst_host_serror_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    dst_host_srv_serror_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    dst_host_rerror_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    dst_host_srv_rerror_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    packet_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    byte_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    avg_packet_size: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    flow_duration: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    inter_arrival_time: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fwd_bwd_ratio: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    port_entropy: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Label (if known from benchmark)
    label: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Relationships
    predictions: Mapped[list[Prediction]] = relationship(back_populates="flow", lazy="selectin")
    alerts: Mapped[list[Alert]] = relationship(back_populates="flow", lazy="selectin")


class Prediction(Base):
    """Model prediction for a flow — TimescaleDB hypertable on created_at."""

    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    flow_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("flows.id", ondelete="CASCADE"), nullable=False
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    # Relationships
    flow: Mapped[Flow] = relationship(back_populates="predictions")


class Alert(Base):
    """Anomaly alert with lifecycle status tracking."""

    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    flow_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("flows.id", ondelete="CASCADE"), nullable=False
    )
    severity: Mapped[AlertSeverity] = mapped_column(
        Enum(AlertSeverity), nullable=False, default=AlertSeverity.MEDIUM
    )
    suspected_attack_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[AlertStatus] = mapped_column(
        Enum(AlertStatus), nullable=False, default=AlertStatus.OPEN
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        Index("ix_alert_status", "status"),
        Index("ix_alert_severity", "severity"),
        Index("ix_alert_created_at", "created_at"),
    )

    # Relationships
    flow: Mapped[Flow] = relationship(back_populates="alerts")


class MLModel(Base):
    """Registered ML model metadata and evaluation metrics."""

    __tablename__ = "ml_models"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)


class User(Base):
    """Analyst user model for authentication."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    username: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class Feedback(Base):
    """Analyst feedback loop model (True Positive / False Positive labels)."""

    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    alert_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("alerts.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)  # "true_positive" or "false_positive"
    user: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationship to Alert
    alert: Mapped[Alert] = relationship(lazy="joined")


class DriftReading(Base):
    """Historical record of population stability index (drift) readings."""

    __tablename__ = "drift_readings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    overall_psi: Mapped[float] = mapped_column(Float, nullable=False)
    feature_psis: Mapped[dict[str, float]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=False
    )

