"""Pydantic v2 schemas for NSL-KDD network flow data."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from anomaly_detection.constants import FEATURE_COLUMNS


class FlowFeatures(BaseModel):
    """NSL-KDD flow features (41 standard + 7 derived = 48 total)."""

    model_config = ConfigDict(populate_by_name=True)

    # Basic connection features
    duration: float = Field(ge=0, default=0.0)
    protocol_type: float = Field(ge=0, default=1.0)
    service: float = Field(ge=0, default=21.0)
    flag: float = Field(ge=0, default=9.0)
    src_bytes: float = Field(ge=0, default=0.0)
    dst_bytes: float = Field(ge=0, default=0.0)
    land: float = Field(ge=0, default=0.0)
    wrong_fragment: float = Field(ge=0, default=0.0)
    urgent: float = Field(ge=0, default=0.0)
    # Content / login features
    hot: float = Field(ge=0, default=0.0)
    num_failed_logins: float = Field(ge=0, default=0.0)
    logged_in: float = Field(ge=0, default=0.0)
    num_compromised: float = Field(ge=0, default=0.0)
    root_shell: float = Field(ge=0, default=0.0)
    su_attempted: float = Field(ge=0, default=0.0)
    num_root: float = Field(ge=0, default=0.0)
    num_file_creations: float = Field(ge=0, default=0.0)
    num_shells: float = Field(ge=0, default=0.0)
    num_access_files: float = Field(ge=0, default=0.0)
    num_outbound_cmds: float = Field(ge=0, default=0.0)
    is_host_login: float = Field(ge=0, default=0.0)
    is_guest_login: float = Field(ge=0, default=0.0)
    # Time-based features
    count: float = Field(ge=0, default=1.0)
    srv_count: float = Field(ge=0, default=1.0)
    serror_rate: float = Field(ge=0, default=0.0)
    srv_serror_rate: float = Field(ge=0, default=0.0)
    rerror_rate: float = Field(ge=0, default=0.0)
    srv_rerror_rate: float = Field(ge=0, default=0.0)
    same_srv_rate: float = Field(ge=0, default=1.0)
    diff_srv_rate: float = Field(ge=0, default=0.0)
    srv_diff_host_rate: float = Field(ge=0, default=0.0)
    # Host-based features
    dst_host_count: float = Field(ge=0, default=1.0)
    dst_host_srv_count: float = Field(ge=0, default=1.0)
    dst_host_same_srv_rate: float = Field(ge=0, default=1.0)
    dst_host_diff_srv_rate: float = Field(ge=0, default=0.0)
    dst_host_same_src_port_rate: float = Field(ge=0, default=0.0)
    dst_host_srv_diff_host_rate: float = Field(ge=0, default=0.0)
    dst_host_serror_rate: float = Field(ge=0, default=0.0)
    dst_host_srv_serror_rate: float = Field(ge=0, default=0.0)
    dst_host_rerror_rate: float = Field(ge=0, default=0.0)
    dst_host_srv_rerror_rate: float = Field(ge=0, default=0.0)
    # Derived features (7 extra)
    packet_rate: float = Field(ge=0, default=0.0)
    byte_rate: float = Field(ge=0, default=0.0)
    avg_packet_size: float = Field(ge=0, default=0.0)
    flow_duration: float = Field(ge=0, default=0.0)
    inter_arrival_time: float = Field(ge=0, default=0.0)
    fwd_bwd_ratio: float = Field(ge=0, default=1.0)
    port_entropy: float = Field(ge=0, default=0.0)

    def to_feature_vector(self) -> list[float]:
        return [getattr(self, col) for col in FEATURE_COLUMNS]


class FlowCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    ts: datetime
    src_ip: str = Field(min_length=1)
    src_port: int = Field(ge=0, le=65535)
    dst_ip: str = Field(min_length=1)
    dst_port: int = Field(ge=0, le=65535)
    protocol: int = Field(ge=0)
    features: FlowFeatures
    label: str | None = Field(default=None)


class FlowResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    ts: datetime
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    protocol: int
    label: str | None = None
    duration: float
    src_bytes: float
    dst_bytes: float
    count: float
    byte_rate: float


class BatchFlowRequest(BaseModel):
    flows: list[FlowCreate] = Field(min_length=1, max_length=1000)
