"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


# =============================================================================
# Authentication
# =============================================================================
class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=128)
    password: str = Field(..., min_length=6, max_length=256)


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=128)
    email: str = Field(..., min_length=5, max_length=256)
    password: str = Field(..., min_length=6, max_length=256)
    full_name: str = Field(default="", max_length=256)


class ForgotPasswordRequest(BaseModel):
    email: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse | None = None


class RefreshRequest(BaseModel):
    refresh_token: str


# =============================================================================
# Users
# =============================================================================
class UserResponse(BaseModel):
    id: UUID
    username: str
    email: str
    full_name: str
    role: str
    status: str
    avatar_url: str | None = None
    last_login: datetime | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=128)
    email: str = Field(..., min_length=5, max_length=256)
    password: str = Field(..., min_length=6, max_length=256)
    full_name: str = Field(default="", max_length=256)
    role: str = Field(default="analyst")


class UserUpdateRequest(BaseModel):
    full_name: str | None = None
    email: str | None = None
    phone: str | None = None
    role: str | None = None
    status: str | None = None


# =============================================================================
# Dashboard
# =============================================================================
class DashboardStats(BaseModel):
    total_packets: int = 0
    normal_traffic: int = 0
    detected_anomalies: int = 0
    todays_attacks: int = 0
    threat_level: str = "Low"
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    bandwidth_usage: float = 0.0
    packets_per_second: float = 0.0
    active_devices: int = 0
    model_accuracy: float = 0.0
    prediction_time: float = 0.0
    detection_rate: float = 0.0


class RecentAlert(BaseModel):
    id: UUID
    title: str
    severity: str
    status: str
    source_ip: str
    created_at: datetime

    model_config = {"from_attributes": True}


class RecentPrediction(BaseModel):
    id: UUID
    model_name: str
    is_anomaly: bool
    confidence: float
    prediction_label: str
    created_at: datetime

    model_config = {"from_attributes": True}


# =============================================================================
# Packets
# =============================================================================
class PacketResponse(BaseModel):
    id: UUID
    timestamp: datetime
    src_ip: str
    dst_ip: str
    protocol: str
    src_port: int
    dst_port: int
    packet_size: int
    ttl: int
    flags: str
    status: str

    model_config = {"from_attributes": True}


class PacketCaptureResponse(BaseModel):
    id: UUID
    interface: str
    status: str
    packet_count: int
    started_at: datetime
    stopped_at: datetime | None = None

    model_config = {"from_attributes": True}


# =============================================================================
# ML Models & Predictions
# =============================================================================
class ModelResponse(BaseModel):
    id: UUID
    name: str
    model_type: str
    version: str
    status: str
    accuracy: float
    precision_score: float
    recall: float
    f1_score: float
    threshold: float
    description: str | None = None
    feature_importance: dict[str, float] | None = None
    confusion_matrix: dict[str, Any] | None = None
    trained_at: datetime

    model_config = {"from_attributes": True}


class TrainRequest(BaseModel):
    model_type: str = Field(
        ..., description="One of: random_forest, isolation_forest, decision_tree, xgboost"
    )
    dataset_id: str | None = None
    params: dict[str, Any] | None = None


class PredictRequest(BaseModel):
    model_name: str = "random_forest"
    features: dict[str, float] = {}


class PredictionResponse(BaseModel):
    id: UUID
    model_name: str
    is_anomaly: bool
    confidence: float
    prediction_label: str
    src_ip: str
    dst_ip: str
    created_at: datetime

    model_config = {"from_attributes": True}


# =============================================================================
# Attacks
# =============================================================================
class AttackResponse(BaseModel):
    id: UUID
    attack_type: str
    severity: str
    confidence: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    recommendation: str
    is_blocked: bool
    detected_at: datetime

    model_config = {"from_attributes": True}


class BlockIPRequest(BaseModel):
    ip_address: str
    reason: str = ""
    attack_type: str | None = None


# =============================================================================
# Alerts
# =============================================================================
class AlertResponse(BaseModel):
    id: UUID
    title: str
    description: str
    severity: str
    status: str
    source_ip: str
    attack_type: str | None = None
    is_read: bool
    created_at: datetime
    resolved_at: datetime | None = None

    model_config = {"from_attributes": True}


class AlertUpdateRequest(BaseModel):
    status: str | None = None
    is_read: bool | None = None


# =============================================================================
# Reports
# =============================================================================
class ReportGenerateRequest(BaseModel):
    report_type: str = Field(default="daily", description="daily, weekly, monthly, custom")
    report_format: str = Field(default="pdf", description="pdf, excel, csv")
    date_from: str | None = None
    date_to: str | None = None


class ReportResponse(BaseModel):
    id: UUID
    name: str
    report_type: str
    report_format: str
    file_size: int
    created_at: datetime

    model_config = {"from_attributes": True}


# =============================================================================
# Datasets
# =============================================================================
class DatasetResponse(BaseModel):
    id: UUID
    name: str
    filename: str
    file_size: int
    row_count: int
    column_count: int
    columns_json: list[str] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


# =============================================================================
# Logs
# =============================================================================
class LoginLogResponse(BaseModel):
    id: UUID
    username: str
    ip_address: str
    success: bool
    failure_reason: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class SystemLogResponse(BaseModel):
    id: UUID
    level: str
    source: str
    message: str
    created_at: datetime

    model_config = {"from_attributes": True}


class AuditLogResponse(BaseModel):
    id: UUID
    action: str
    resource: str
    resource_id: str | None = None
    details: str | None = None
    ip_address: str
    created_at: datetime

    model_config = {"from_attributes": True}


# =============================================================================
# Settings
# =============================================================================
class SettingResponse(BaseModel):
    key: str
    value: str
    description: str | None = None

    model_config = {"from_attributes": True}


class SettingUpdateRequest(BaseModel):
    settings: dict[str, str]



