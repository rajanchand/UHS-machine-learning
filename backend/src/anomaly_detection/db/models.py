"""SQLAlchemy ORM models for the Network Anomaly Detection System.

Production-grade database schema with 15+ tables covering:
- Authentication & RBAC (users, roles)
- Network monitoring (packets, packet_captures)
- ML pipeline (predictions, ml_models, datasets)
- Security (attacks, alerts, blocked_ips)
- Reporting (reports)
- Logging (login_logs, system_logs, audit_logs)
- Configuration (settings)
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


# =============================================================================
# Base
# =============================================================================
class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


# =============================================================================
# Enumerations
# =============================================================================
class UserRole(enum.Enum):
    """User roles for RBAC."""

    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class UserStatus(enum.Enum):
    """User account status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class AlertSeverity(enum.Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(enum.Enum):
    """Alert lifecycle status."""

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class ModelStatus(enum.Enum):
    """ML model status."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    FAILED = "failed"


class CaptureStatus(enum.Enum):
    """Packet capture session status."""

    RUNNING = "running"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


# =============================================================================
# Users & RBAC
# =============================================================================
class User(Base):
    """User accounts with role-based access control."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(256), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    full_name: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), nullable=False, default=UserRole.ANALYST)
    status: Mapped[UserStatus] = mapped_column(
        Enum(UserStatus), nullable=False, default=UserStatus.ACTIVE
    )
    avatar_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    last_login: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    login_logs: Mapped[list[LoginLog]] = relationship(back_populates="user", lazy="selectin")
    audit_logs: Mapped[list[AuditLog]] = relationship(back_populates="user", lazy="selectin")

    __table_args__ = (
        Index("ix_user_role", "role"),
        Index("ix_user_status", "status"),
    )


# =============================================================================
# Packets & Capture
# =============================================================================
class PacketCapture(Base):
    """Packet capture session metadata."""

    __tablename__ = "packet_captures"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    interface: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[CaptureStatus] = mapped_column(
        Enum(CaptureStatus), nullable=False, default=CaptureStatus.RUNNING
    )
    packet_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Relationships
    packets: Mapped[list[Packet]] = relationship(back_populates="capture", lazy="selectin")


class Packet(Base):
    """Individual captured network packet records."""

    __tablename__ = "packets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    capture_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("packet_captures.id", ondelete="CASCADE"), nullable=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    src_ip: Mapped[str] = mapped_column(String(45), nullable=False, index=True)
    dst_ip: Mapped[str] = mapped_column(String(45), nullable=False, index=True)
    protocol: Mapped[str] = mapped_column(String(16), nullable=False, default="TCP")
    src_port: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dst_port: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    packet_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ttl: Mapped[int] = mapped_column(Integer, nullable=False, default=64)
    flags: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="Normal")

    # Relationships
    capture: Mapped[PacketCapture | None] = relationship(back_populates="packets")

    __table_args__ = (
        Index("ix_packet_protocol", "protocol"),
        Index("ix_packet_status", "status"),
    )


# =============================================================================
# ML Models & Predictions
# =============================================================================
class MLModel(Base):
    """Registered ML model metadata, metrics, and feature importance."""

    __tablename__ = "ml_models"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    version: Mapped[str] = mapped_column(String(64), nullable=False, default="v1")
    status: Mapped[ModelStatus] = mapped_column(
        Enum(ModelStatus), nullable=False, default=ModelStatus.INACTIVE
    )
    accuracy: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    precision_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    recall: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    f1_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )
    feature_importance: Mapped[dict[str, float] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )
    confusion_matrix: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )
    training_params: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    predictions: Mapped[list[Prediction]] = relationship(back_populates="model", lazy="selectin")


class Prediction(Base):
    """ML model prediction results for network flows."""

    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("ml_models.id", ondelete="SET NULL"), nullable=True
    )
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    prediction_label: Mapped[str] = mapped_column(String(64), nullable=False, default="Normal")
    features_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )
    src_ip: Mapped[str] = mapped_column(String(45), nullable=False, default="")
    dst_ip: Mapped[str] = mapped_column(String(45), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    # Relationships
    model: Mapped[MLModel | None] = relationship(back_populates="predictions")


class Dataset(Base):
    """Uploaded dataset metadata for ML training."""

    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    columns_json: Mapped[list[str] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


# =============================================================================
# Attacks & Alerts
# =============================================================================
class Attack(Base):
    """Detected network attack records."""

    __tablename__ = "attacks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    attack_type: Mapped[str] = mapped_column(String(64), nullable=False, default="Unknown")
    severity: Mapped[AlertSeverity] = mapped_column(
        Enum(AlertSeverity), nullable=False, default=AlertSeverity.MEDIUM
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    src_ip: Mapped[str] = mapped_column(String(45), nullable=False, index=True)
    dst_ip: Mapped[str] = mapped_column(String(45), nullable=False)
    src_port: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    dst_port: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    protocol: Mapped[str] = mapped_column(String(16), nullable=False, default="TCP")
    recommendation: Mapped[str] = mapped_column(
        Text, nullable=False, default="Monitor traffic patterns"
    )
    is_blocked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    __table_args__ = (
        Index("ix_attack_type", "attack_type"),
        Index("ix_attack_severity", "severity"),
    )


class Alert(Base):
    """Anomaly alerts with lifecycle status tracking."""

    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(256), nullable=False, default="Anomaly Detected")
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    severity: Mapped[AlertSeverity] = mapped_column(
        Enum(AlertSeverity), nullable=False, default=AlertSeverity.MEDIUM
    )
    status: Mapped[AlertStatus] = mapped_column(
        Enum(AlertStatus), nullable=False, default=AlertStatus.OPEN
    )
    source_ip: Mapped[str] = mapped_column(String(45), nullable=False, default="")
    attack_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_emailed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_read: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_alert_status", "status"),
        Index("ix_alert_severity", "severity"),
        Index("ix_alert_created_at", "created_at"),
    )


class BlockedIP(Base):
    """Blocked IP addresses from attack detection."""

    __tablename__ = "blocked_ips"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ip_address: Mapped[str] = mapped_column(String(45), nullable=False, unique=True, index=True)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    attack_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    blocked_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    blocked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


# =============================================================================
# Reports
# =============================================================================
class Report(Base):
    """Generated report metadata."""

    __tablename__ = "reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    report_type: Mapped[str] = mapped_column(String(32), nullable=False, default="daily")
    report_format: Mapped[str] = mapped_column(String(16), nullable=False, default="pdf")
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    date_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    date_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    generated_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


# =============================================================================
# Logging & Audit
# =============================================================================
class LoginLog(Base):
    """Login attempt audit trail."""

    __tablename__ = "login_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    username: Mapped[str] = mapped_column(String(128), nullable=False)
    ip_address: Mapped[str] = mapped_column(String(45), nullable=False, default="")
    user_agent: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    failure_reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    # Relationships
    user: Mapped[User | None] = relationship(back_populates="login_logs")


class SystemLog(Base):
    """Application system event log."""

    __tablename__ = "system_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    level: Mapped[str] = mapped_column(String(16), nullable=False, default="info")
    source: Mapped[str] = mapped_column(String(128), nullable=False, default="system")
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )


class AuditLog(Base):
    """User action audit trail for compliance."""

    __tablename__ = "audit_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    resource: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    resource_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip_address: Mapped[str] = mapped_column(String(45), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )

    # Relationships
    user: Mapped[User | None] = relationship(back_populates="audit_logs")


# =============================================================================
# Settings
# =============================================================================
class Setting(Base):
    """Key-value system configuration store."""

    __tablename__ = "settings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    value: Mapped[str] = mapped_column(Text, nullable=False, default="")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
