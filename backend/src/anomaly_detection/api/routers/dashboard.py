"""Dashboard router — KPIs, system health, recent alerts/predictions, chart data."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import psutil
from fastapi import APIRouter, Request
from sqlalchemy import func, select

from anomaly_detection.db.models import (
    Alert,
    Attack,
    MLModel,
    ModelStatus,
    Packet,
    Prediction,
)
from anomaly_detection.schemas.common import DashboardStats, RecentAlert, RecentPrediction

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats(request: Request) -> DashboardStats:
    """Get all dashboard KPI metrics."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        # Total packets
        total_packets = (await session.execute(select(func.count(Packet.id)))).scalar() or 0

        # Total predictions
        total_predictions = (await session.execute(select(func.count(Prediction.id)))).scalar() or 0

        # Anomalies
        anomalies = (
            await session.execute(
                select(func.count(Prediction.id)).where(Prediction.is_anomaly.is_(True))
            )
        ).scalar() or 0

        normal_traffic = total_predictions - anomalies

        # Today's attacks
        today_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        todays_attacks = (
            await session.execute(
                select(func.count(Attack.id)).where(Attack.detected_at >= today_start)
            )
        ).scalar() or 0

        # Detection rate
        detection_rate = (anomalies / total_predictions * 100) if total_predictions > 0 else 0

        # Threat level calculation
        if todays_attacks > 50:
            threat_level = "Critical"
        elif todays_attacks > 20:
            threat_level = "High"
        elif todays_attacks > 5:
            threat_level = "Medium"
        else:
            threat_level = "Low"

        # System metrics
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        # Active model accuracy
        active_model = (
            await session.execute(select(MLModel).where(MLModel.status == ModelStatus.ACTIVE))
        ).scalar_one_or_none()
        model_accuracy = active_model.accuracy * 100 if active_model else 0.0

        # Unique active devices (unique IPs in last hour)
        one_hour_ago = datetime.now(UTC) - timedelta(hours=1)
        active_src = (
            await session.execute(
                select(func.count(func.distinct(Packet.src_ip))).where(
                    Packet.timestamp >= one_hour_ago
                )
            )
        ).scalar() or 0

        return DashboardStats(
            total_packets=total_packets,
            normal_traffic=normal_traffic,
            detected_anomalies=anomalies,
            todays_attacks=todays_attacks,
            threat_level=threat_level,
            cpu_usage=round(cpu_usage, 1),
            memory_usage=round(memory_usage, 1),
            bandwidth_usage=0.0,  # Placeholder
            packets_per_second=0.0,  # Updated by monitoring service
            active_devices=active_src,
            model_accuracy=round(model_accuracy, 1),
            prediction_time=0.0,
            detection_rate=round(detection_rate, 1),
        )


@router.get("/recent-alerts")
async def get_recent_alerts(request: Request) -> list[RecentAlert]:
    """Get the 10 most recent alerts."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(Alert).order_by(Alert.created_at.desc()).limit(10))
        alerts = result.scalars().all()
        return [
            RecentAlert(
                id=a.id,
                title=a.title,
                severity=a.severity.value,
                status=a.status.value,
                source_ip=a.source_ip,
                created_at=a.created_at,
            )
            for a in alerts
        ]


@router.get("/recent-predictions")
async def get_recent_predictions(request: Request) -> list[RecentPrediction]:
    """Get the 10 most recent predictions."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(
            select(Prediction).order_by(Prediction.created_at.desc()).limit(10)
        )
        predictions = result.scalars().all()
        return [
            RecentPrediction(
                id=p.id,
                model_name=p.model_name,
                is_anomaly=p.is_anomaly,
                confidence=p.confidence,
                prediction_label=p.prediction_label,
                created_at=p.created_at,
            )
            for p in predictions
        ]


@router.get("/system-health")
async def get_system_health(request: Request) -> dict[str, Any]:
    """Get system health metrics."""
    cpu = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    return {
        "cpu_usage": round(cpu, 1),
        "memory_usage": round(memory.percent, 1),
        "memory_total_gb": round(memory.total / (1024**3), 1),
        "memory_used_gb": round(memory.used / (1024**3), 1),
        "disk_usage": round(disk.percent, 1),
        "disk_total_gb": round(disk.total / (1024**3), 1),
        "disk_used_gb": round(disk.used / (1024**3), 1),
    }


@router.get("/charts/traffic")
async def get_traffic_chart(request: Request) -> list[dict[str, Any]]:
    """Get traffic data for line chart (last 24 hours, hourly)."""
    session_factory = request.app.state.session_factory
    now = datetime.now(UTC)
    data = []

    async with session_factory() as session:
        for i in range(24, 0, -1):
            hour_start = now - timedelta(hours=i)
            hour_end = now - timedelta(hours=i - 1)
            count = (
                await session.execute(
                    select(func.count(Packet.id)).where(
                        Packet.timestamp >= hour_start,
                        Packet.timestamp < hour_end,
                    )
                )
            ).scalar() or 0
            data.append(
                {
                    "time": hour_start.strftime("%H:%M"),
                    "packets": count,
                }
            )

    return data


@router.get("/charts/protocols")
async def get_protocol_chart(request: Request) -> list[dict[str, Any]]:
    """Get protocol distribution for pie chart."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(
            select(
                Packet.protocol,
                func.count(Packet.id).label("count"),
            )
            .group_by(Packet.protocol)
            .order_by(func.count(Packet.id).desc())
            .limit(10)
        )
        rows = result.all()
        return [{"protocol": row[0], "count": row[1]} for row in rows]


@router.get("/charts/attacks")
async def get_attack_chart(request: Request) -> list[dict[str, Any]]:
    """Get attack type distribution for bar chart."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(
            select(
                Attack.attack_type,
                func.count(Attack.id).label("count"),
            )
            .group_by(Attack.attack_type)
            .order_by(func.count(Attack.id).desc())
        )
        rows = result.all()
        return [{"type": row[0], "count": row[1]} for row in rows]
