"""Model and stats API routers."""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING, cast

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import desc, func, select

from anomaly_detection.db.models import Alert, AlertStatus, Flow, Prediction
from anomaly_detection.schemas.common import (
    KPIResponse,
    ModelResponse,
    ThresholdUpdate,
    TimelinePoint,
    TimelineResponse,
    TopTalker,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

models_router = APIRouter(prefix="/api/v1/models", tags=["models"])
stats_router = APIRouter(prefix="/api/v1/stats", tags=["stats"])


def _get_session(request: Request) -> AsyncSession:
    factory = request.app.state.session_factory
    return cast("AsyncSession", factory())


# --- Model endpoints ---


@models_router.get("", response_model=list[ModelResponse])
async def list_models(request: Request) -> list[ModelResponse]:
    """List all registered models with their metrics."""
    from anomaly_detection.db.models import MLModel
    async with _get_session(request) as session:
        result = await session.execute(
            select(MLModel).order_by(desc(MLModel.trained_at))
        )
        models = result.scalars().all()
        return [ModelResponse.model_validate(m) for m in models]


@models_router.put("/{model_name}/threshold")
async def update_threshold(
    request: Request,
    model_name: str,
    update: ThresholdUpdate,
) -> dict[str, object]:
    """Update the scoring threshold for a model."""
    inference_svc = request.app.state.inference_service
    from anomaly_detection.services.inference import InferenceService
    assert isinstance(inference_svc, InferenceService)

    if model_name not in inference_svc.available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {inference_svc.available_models}",
        )

    inference_svc.set_threshold(model_name, update.threshold)

    # Also persist to DB so the threshold survives restarts
    from sqlalchemy import update as sql_update

    from anomaly_detection.db.models import MLModel
    async with _get_session(request) as session:
        await session.execute(
            sql_update(MLModel)
            .where(MLModel.name == model_name)
            .values(threshold=update.threshold)
        )
        await session.commit()

    return {"model_name": model_name, "threshold": update.threshold}




class ActiveModelUpdate(BaseModel):
    """Request schema for setting active model."""
    name: str


@models_router.put("/active")
async def set_active_model(
    request: Request,
    update: ActiveModelUpdate,
) -> dict[str, object]:
    """Set the active ML model."""
    inference_svc = request.app.state.inference_service
    from anomaly_detection.services.inference import InferenceService
    assert isinstance(inference_svc, InferenceService)

    if update.name not in inference_svc.available_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{update.name}' not found. Available: {inference_svc.available_models}",
        )

    inference_svc.set_active_model(update.name)

    # Also update in database
    from sqlalchemy import update as sql_update

    from anomaly_detection.db.models import MLModel
    async with _get_session(request) as session:
        # Deactivate all models
        await session.execute(
            sql_update(MLModel).values(is_active=False)
        )
        # Activate the chosen model
        await session.execute(
            sql_update(MLModel).where(MLModel.name == update.name).values(is_active=True)
        )
        await session.commit()

    return {"active_model": update.name}


# --- Stats endpoints ---


@stats_router.get("/kpi", response_model=KPIResponse)
async def get_kpis(request: Request) -> KPIResponse:
    """Get current KPIs for the dashboard."""
    async with _get_session(request) as session:
        # Total flows
        total_flows_result = await session.execute(select(func.count(Flow.id)))
        total_flows = total_flows_result.scalar() or 0

        # Total alerts and open alerts
        total_alerts_result = await session.execute(select(func.count(Alert.id)))
        total_alerts = total_alerts_result.scalar() or 0

        open_alerts_result = await session.execute(
            select(func.count(Alert.id)).where(Alert.status == AlertStatus.OPEN)
        )
        open_alerts = open_alerts_result.scalar() or 0

        # Estimated FPR (ratio of anomalies flagged in flows labelled BENIGN)
        benign_total_result = await session.execute(
            select(func.count(Flow.id)).where(Flow.label == "BENIGN")
        )
        benign_total = benign_total_result.scalar() or 0

        false_positives_result = await session.execute(
            select(func.count(Prediction.id))
            .join(Flow, Prediction.flow_id == Flow.id)
            .where(Flow.label == "BENIGN")
            .where(Prediction.is_anomaly.is_(True))
        )
        false_positives = false_positives_result.scalar() or 0

        estimated_fpr = (false_positives / benign_total) if benign_total > 0 else 0.0

        # Top talkers (by flow count)
        top_talkers_result = await session.execute(
            select(Flow.src_ip, func.count(Flow.id).label("cnt"))
            .group_by(Flow.src_ip)
            .order_by(desc("cnt"))
            .limit(5)
        )
        top_talkers = [
            TopTalker(ip=row[0], flow_count=row[1])
            for row in top_talkers_result.all()
        ]

        return KPIResponse(
            total_flows=total_flows,
            total_alerts=total_alerts,
            open_alerts=open_alerts,
            estimated_fpr=min(estimated_fpr, 1.0),
            top_talkers=top_talkers,
        )


@stats_router.get("/timeline", response_model=TimelineResponse)
async def get_timeline(request: Request) -> TimelineResponse:
    """Get anomaly score timeline data for charts."""
    inference_svc = request.app.state.inference_service
    from anomaly_detection.services.inference import InferenceService
    assert isinstance(inference_svc, InferenceService)

    async with _get_session(request) as session:
        # Aggregate scores in 1-minute buckets
        result = await session.execute(
            select(
                func.date_trunc("minute", Prediction.created_at).label("bucket"),
                func.avg(Prediction.score).label("avg_score"),
                func.max(Prediction.score).label("max_score"),
                func.count(Prediction.id).label("flow_count"),
                func.sum(
                    func.cast(Prediction.is_anomaly, type_=Prediction.score.type)
                ).label("anomaly_count"),
            )
            .group_by("bucket")
            .order_by(desc("bucket"))
            .limit(120)  # Last 2 hours of minutes
        )
        rows = result.all()

        points = [
            TimelinePoint(
                timestamp=row[0],
                avg_score=float(row[1] or 0),
                max_score=float(row[2] or 0),
                flow_count=int(row[3] or 0),
                anomaly_count=int(row[4] or 0),
            )
            for row in reversed(rows)
        ]

        threshold = inference_svc.get_threshold()

        return TimelineResponse(points=points, threshold=threshold)


@models_router.get("/comparison")
async def get_model_comparison(request: Request) -> list[dict[str, Any]]:
    """Return evaluation metrics for all models side by side."""
    settings = request.app.state.settings
    metrics_path = settings.data_dir.parent / "evaluation" / "metrics.json"

    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation metrics not found. Run the evaluation script first.",
        )

    try:
        all_metrics: dict[str, Any] = json.loads(metrics_path.read_text())
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to read metrics file") from exc

    result = []
    for name, m in all_metrics.items():
        cm = m.get("confusion_matrix", {})
        result.append({
            "name": name,
            "model_type": m.get("model_type", "unsupervised"),
            "accuracy": m.get("accuracy", 0.0),
            "precision": m.get("precision", 0.0),
            "recall": m.get("recall", 0.0),
            "f1": m.get("f1", 0.0),
            "roc_auc": m.get("roc_auc", 0.0),
            "pr_auc": m.get("pr_auc", 0.0),
            "fpr": m.get("fpr", 0.0),
            "confusion_matrix": cm,
            "per_attack_recall": m.get("per_attack_recall", {}),
        })
    return result
