"""Pydantic v2 schemas for network flow data."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class FlowFeatures(BaseModel):
    """Core flow features used for ML scoring."""

    model_config = ConfigDict(populate_by_name=True)

    flow_duration: float = Field(ge=0, description="Duration of the flow in microseconds")
    total_fwd_packets: int = Field(ge=0, description="Total forward packets")
    total_bwd_packets: int = Field(ge=0, description="Total backward packets")
    total_len_fwd_packets: float = Field(ge=0, description="Total length of forward packets")
    total_len_bwd_packets: float = Field(ge=0, description="Total length of backward packets")

    # Packet size statistics
    fwd_packet_len_max: float = Field(ge=0, default=0.0)
    fwd_packet_len_min: float = Field(ge=0, default=0.0)
    fwd_packet_len_mean: float = Field(ge=0, default=0.0)
    fwd_packet_len_std: float = Field(ge=0, default=0.0)
    bwd_packet_len_max: float = Field(ge=0, default=0.0)
    bwd_packet_len_min: float = Field(ge=0, default=0.0)
    bwd_packet_len_mean: float = Field(ge=0, default=0.0)
    bwd_packet_len_std: float = Field(ge=0, default=0.0)

    # Flow rates
    flow_bytes_per_s: float = Field(ge=0, default=0.0)
    flow_packets_per_s: float = Field(ge=0, default=0.0)

    # TCP flags
    fin_flag_count: int = Field(ge=0, default=0)
    syn_flag_count: int = Field(ge=0, default=0)
    rst_flag_count: int = Field(ge=0, default=0)
    psh_flag_count: int = Field(ge=0, default=0)
    ack_flag_count: int = Field(ge=0, default=0)
    urg_flag_count: int = Field(ge=0, default=0)

    # Inter-arrival time statistics
    flow_iat_mean: float = Field(ge=0, default=0.0)
    flow_iat_std: float = Field(ge=0, default=0.0)
    flow_iat_max: float = Field(ge=0, default=0.0)
    flow_iat_min: float = Field(ge=0, default=0.0)
    fwd_iat_mean: float = Field(ge=0, default=0.0)
    fwd_iat_std: float = Field(ge=0, default=0.0)
    fwd_iat_max: float = Field(ge=0, default=0.0)
    fwd_iat_min: float = Field(ge=0, default=0.0)
    bwd_iat_mean: float = Field(ge=0, default=0.0)
    bwd_iat_std: float = Field(ge=0, default=0.0)
    bwd_iat_max: float = Field(ge=0, default=0.0)
    bwd_iat_min: float = Field(ge=0, default=0.0)

    # Miscellaneous
    down_up_ratio: float = Field(ge=0, default=0.0)
    avg_packet_size: float = Field(ge=0, default=0.0)
    avg_fwd_segment_size: float = Field(ge=0, default=0.0)
    avg_bwd_segment_size: float = Field(ge=0, default=0.0)

    def to_feature_vector(self) -> list[float]:
        """Convert to ordered feature vector for model input."""
        return [
            self.flow_duration,
            self.total_fwd_packets,
            self.total_bwd_packets,
            self.total_len_fwd_packets,
            self.total_len_bwd_packets,
            self.fwd_packet_len_max,
            self.fwd_packet_len_min,
            self.fwd_packet_len_mean,
            self.fwd_packet_len_std,
            self.bwd_packet_len_max,
            self.bwd_packet_len_min,
            self.bwd_packet_len_mean,
            self.bwd_packet_len_std,
            self.flow_bytes_per_s,
            self.flow_packets_per_s,
            self.fin_flag_count,
            self.syn_flag_count,
            self.rst_flag_count,
            self.psh_flag_count,
            self.ack_flag_count,
            self.urg_flag_count,
            self.flow_iat_mean,
            self.flow_iat_std,
            self.flow_iat_max,
            self.flow_iat_min,
            self.fwd_iat_mean,
            self.fwd_iat_std,
            self.fwd_iat_max,
            self.fwd_iat_min,
            self.bwd_iat_mean,
            self.bwd_iat_std,
            self.bwd_iat_max,
            self.bwd_iat_min,
            self.down_up_ratio,
            self.avg_packet_size,
            self.avg_fwd_segment_size,
            self.avg_bwd_segment_size,
        ]


# Column names matching the feature vector order — used for DataFrame operations
FEATURE_COLUMNS: list[str] = [
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_len_fwd_packets",
    "total_len_bwd_packets",
    "fwd_packet_len_max",
    "fwd_packet_len_min",
    "fwd_packet_len_mean",
    "fwd_packet_len_std",
    "bwd_packet_len_max",
    "bwd_packet_len_min",
    "bwd_packet_len_mean",
    "bwd_packet_len_std",
    "flow_bytes_per_s",
    "flow_packets_per_s",
    "fin_flag_count",
    "syn_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "ack_flag_count",
    "urg_flag_count",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    "fwd_iat_mean",
    "fwd_iat_std",
    "fwd_iat_max",
    "fwd_iat_min",
    "bwd_iat_mean",
    "bwd_iat_std",
    "bwd_iat_max",
    "bwd_iat_min",
    "down_up_ratio",
    "avg_packet_size",
    "avg_fwd_segment_size",
    "avg_bwd_segment_size",
]


class FlowCreate(BaseModel):
    """Schema for creating a new flow record."""

    model_config = ConfigDict(populate_by_name=True)

    ts: datetime = Field(description="Flow timestamp")
    src_ip: str = Field(min_length=1, description="Source IP address")
    src_port: int = Field(ge=0, le=65535, description="Source port")
    dst_ip: str = Field(min_length=1, description="Destination IP address")
    dst_port: int = Field(ge=0, le=65535, description="Destination port")
    protocol: int = Field(ge=0, description="IP protocol number")
    features: FlowFeatures = Field(description="Flow feature values")
    label: str | None = Field(default=None, description="Ground truth label if known")


class FlowResponse(BaseModel):
    """Schema for flow API responses."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    ts: datetime
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    protocol: int
    label: str | None = None

    # Include key features inline for display
    flow_duration: float
    total_fwd_packets: int
    total_bwd_packets: int
    flow_bytes_per_s: float
    avg_packet_size: float


class FlowDetailResponse(FlowResponse):
    """Full flow response including all features and predictions."""

    model_config = ConfigDict(from_attributes=True)

    total_len_fwd_packets: float
    total_len_bwd_packets: float
    fwd_packet_len_max: float
    fwd_packet_len_min: float
    fwd_packet_len_mean: float
    fwd_packet_len_std: float
    bwd_packet_len_max: float
    bwd_packet_len_min: float
    bwd_packet_len_mean: float
    bwd_packet_len_std: float
    flow_packets_per_s: float
    fin_flag_count: int
    syn_flag_count: int
    rst_flag_count: int
    psh_flag_count: int
    ack_flag_count: int
    urg_flag_count: int
    flow_iat_mean: float
    flow_iat_std: float
    flow_iat_max: float
    flow_iat_min: float
    fwd_iat_mean: float
    fwd_iat_std: float
    fwd_iat_max: float
    fwd_iat_min: float
    bwd_iat_mean: float
    bwd_iat_std: float
    bwd_iat_max: float
    bwd_iat_min: float
    down_up_ratio: float
    avg_fwd_segment_size: float
    avg_bwd_segment_size: float


class BatchFlowRequest(BaseModel):
    """Request schema for batch flow inference."""

    flows: list[FlowCreate] = Field(min_length=1, max_length=1000)
