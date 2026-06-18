"""Drift monitoring router — calculates Population Stability Index (PSI)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from anomaly_detection.db.models import DriftReading, Flow

router = APIRouter(prefix="/api/v1/drift", tags=["drift"])


class DriftResponse(BaseModel):
    """Schema for drift calculation response."""

    overall_psi: float
    status: str  # "stable" or "drifting"
    threshold: float
    system_notice: str | None
    ts: datetime
    feature_psis: dict[str, float]


def _get_session(request: Request) -> AsyncSession:
    """Get database session from request state."""
    factory = request.app.state.session_factory
    return cast("AsyncSession", factory())


def calculate_feature_psi(expected_values: pd.Series, actual_values: list[float], num_bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) for a single feature."""
    if len(expected_values) == 0 or len(actual_values) == 0:
        return 0.0

    actual_arr = np.array(actual_values)

    # Calculate quantiles of training data
    percentiles = np.linspace(0, 100, num_bins + 1)[1:-1]
    bins = np.percentile(expected_values, percentiles)
    bins = np.unique(bins)  # Make bins unique to avoid digitize error

    # Digitize both expected and actual to get bin indexes
    expected_idx = np.digitize(expected_values, bins)
    actual_idx = np.digitize(actual_arr, bins)

    num_bins_actual = len(bins) + 1
    expected_counts = np.bincount(expected_idx, minlength=num_bins_actual)
    actual_counts = np.bincount(actual_idx, minlength=num_bins_actual)

    # Convert to percentages
    expected_pct = expected_counts / len(expected_values)
    actual_pct = actual_counts / len(actual_arr)

    # Handle zero counts (add epsilon)
    eps = 1e-4
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)

    # Re-normalize to sum to 1.0
    expected_pct = expected_pct / expected_pct.sum()
    actual_pct = actual_pct / actual_pct.sum()

    # Calculate PSI
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_value)


@router.get("", response_model=DriftResponse)
async def get_drift(request: Request) -> DriftResponse:
    """Calculate population stability index (drift) comparing recent flows to training data."""
    settings = request.app.state.settings

    # 1. Fetch training baseline
    train_parquet_path = Path(settings.data_dir) / "processed" / "train.parquet"
    if not train_parquet_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Training baseline dataset (train.parquet) not found.",
        )

    try:
        train_df = pd.read_parquet(train_parquet_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read training baseline: {e}",
        )

    # 2. Fetch recent actual flows from DB
    async with _get_session(request) as session:
        result = await session.execute(
            select(Flow).order_by(desc(Flow.ts)).limit(500)
        )
        recent_flows = result.scalars().all()

    # Define numeric features list
    feature_cols = [
        'flow_duration', 'total_fwd_packets', 'total_bwd_packets', 'total_len_fwd_packets', 'total_len_bwd_packets',
        'fwd_packet_len_max', 'fwd_packet_len_min', 'fwd_packet_len_mean', 'fwd_packet_len_std', 'bwd_packet_len_max',
        'bwd_packet_len_min', 'bwd_packet_len_mean', 'bwd_packet_len_std', 'flow_bytes_per_s', 'flow_packets_per_s',
        'fin_flag_count', 'syn_flag_count', 'rst_flag_count', 'psh_flag_count', 'ack_flag_count', 'urg_flag_count',
        'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max',
        'fwd_iat_min', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min', 'down_up_ratio', 'avg_packet_size',
        'avg_fwd_segment_size', 'avg_bwd_segment_size'
    ]

    # If not enough actual data (< 10 flows), return default stable response
    if len(recent_flows) < 10:
        feature_psis = {col: 0.0 for col in feature_cols}
        return DriftResponse(
            overall_psi=0.0,
            status="stable",
            threshold=0.25,
            system_notice=None,
            ts=datetime.now(),
            feature_psis=feature_psis
        )

    # 3. Calculate PSI per feature
    feature_psis = {}
    for col in feature_cols:
        if col in train_df.columns:
            # Extract actual values for this feature
            actual_vals = [float(getattr(f, col, 0.0)) for f in recent_flows]
            psi = calculate_feature_psi(train_df[col].dropna(), actual_vals)
            feature_psis[col] = psi
        else:
            feature_psis[col] = 0.0

    # Calculate overall PSI as average
    overall_psi = float(np.mean(list(feature_psis.values())))
    threshold = 0.25
    is_drifting = overall_psi >= threshold
    status = "drifting" if is_drifting else "stable"
    system_notice = "model may be stale — consider retraining" if is_drifting else None
    now = datetime.now()

    # 4. Persist drift reading to DB
    async with _get_session(request) as session:
        reading = DriftReading(
            ts=now,
            overall_psi=overall_psi,
            feature_psis=feature_psis
        )
        session.add(reading)
        await session.commit()

    return DriftResponse(
        overall_psi=overall_psi,
        status=status,
        threshold=threshold,
        system_notice=system_notice,
        ts=now,
        feature_psis=feature_psis
    )
