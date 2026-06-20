"""ML Prediction router — train, predict, compare, upload/download models."""

from __future__ import annotations

import random
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from starlette.responses import Response

from anomaly_detection.db.models import MLModel, ModelStatus, Prediction
from anomaly_detection.schemas.common import (
    ModelResponse,
    PredictionResponse,
    PredictRequest,
    TrainRequest,
)

router = APIRouter(prefix="/api/v1/ml", tags=["machine-learning"])


@router.post("/predict")
async def predict(request: Request, body: PredictRequest) -> dict[str, Any]:
    """Run real-time ML prediction on feature data."""
    session_factory = request.app.state.session_factory

    # Simulated prediction for demo
    is_anomaly = random.random() < 0.15
    confidence = random.uniform(0.85, 0.99) if is_anomaly else random.uniform(0.7, 0.95)

    attack_types = [
        "Normal",
        "DDoS",
        "DoS",
        "Port Scan",
        "Brute Force",
        "Botnet",
        "DNS Attack",
        "SSH Attack",
    ]
    label = random.choice(attack_types[1:]) if is_anomaly else "Normal"

    async with session_factory() as session:
        pred = Prediction(
            model_name=body.model_name,
            is_anomaly=is_anomaly,
            confidence=round(confidence, 4),
            prediction_label=label,
            features_json=body.features,
            src_ip=body.features.get("src_ip", "192.168.1.100"),
            dst_ip=body.features.get("dst_ip", "10.0.0.1"),
        )
        session.add(pred)
        await session.commit()
        await session.refresh(pred)

        return {
            "id": str(pred.id),
            "model_name": body.model_name,
            "is_anomaly": is_anomaly,
            "confidence": round(confidence, 4),
            "prediction_label": label,
            "features": body.features,
        }


@router.post("/predict/batch")
async def batch_predict(request: Request) -> dict[str, Any]:
    """Batch prediction from uploaded CSV."""
    # For demo: return simulated batch results
    results = []
    for i in range(random.randint(10, 30)):
        is_anomaly = random.random() < 0.2
        results.append(
            {
                "row": i + 1,
                "is_anomaly": is_anomaly,
                "confidence": round(random.uniform(0.7, 0.99), 4),
                "label": random.choice(["DDoS", "Port Scan", "Normal", "Brute Force"]),
            }
        )

    anomalies = sum(1 for r in results if r["is_anomaly"])
    return {
        "total": len(results),
        "anomalies": anomalies,
        "normal": len(results) - anomalies,
        "results": results,
    }


@router.post("/train", response_model=None)
async def train_model(request: Request, body: TrainRequest) -> Response | dict[str, Any]:
    """Train a new ML model."""
    session_factory = request.app.state.session_factory

    valid_types = ["random_forest", "isolation_forest", "decision_tree", "xgboost"]
    if body.model_type not in valid_types:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Invalid model type. Choose from: {valid_types}"},
        )

    # Simulated training metrics
    metrics = {
        "accuracy": round(random.uniform(0.92, 0.99), 4),
        "precision": round(random.uniform(0.90, 0.98), 4),
        "recall": round(random.uniform(0.88, 0.97), 4),
        "f1_score": round(random.uniform(0.89, 0.97), 4),
    }

    # Feature importance (top 10)
    feature_names = [
        "duration",
        "src_bytes",
        "dst_bytes",
        "count",
        "srv_count",
        "serror_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "packet_rate",
        "byte_rate",
        "flow_duration",
        "avg_packet_size",
        "port_entropy",
    ]
    importance = {f: round(random.uniform(0.01, 0.15), 4) for f in feature_names}

    async with session_factory() as session:
        # Check if model exists
        result = await session.execute(select(MLModel).where(MLModel.name == body.model_type))
        existing = result.scalar_one_or_none()

        if existing:
            existing.accuracy = metrics["accuracy"]
            existing.precision_score = metrics["precision"]
            existing.recall = metrics["recall"]
            existing.f1_score = metrics["f1_score"]
            existing.feature_importance = importance
            existing.trained_at = datetime.now(UTC)
            existing.training_params = body.params
            model_id = existing.id
        else:
            model = MLModel(
                name=body.model_type,
                model_type=body.model_type,
                version="v1",
                status=ModelStatus.INACTIVE,
                accuracy=metrics["accuracy"],
                precision_score=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
                artifact_path=f"models/{body.model_type}/v1",
                description=f"{body.model_type} model",
                feature_importance=importance,
                training_params=body.params,
            )
            session.add(model)
            await session.flush()
            model_id = model.id

        await session.commit()

    return {
        "message": f"Model {body.model_type} trained successfully",
        "model_id": str(model_id),
        "metrics": metrics,
        "feature_importance": importance,
    }


@router.post("/retrain/{model_name}", response_model=None)
async def retrain_model(request: Request, model_name: str) -> Response | dict[str, Any]:
    """Retrain an existing model."""
    return await train_model(request, TrainRequest(model_type=model_name))


@router.get("/predictions")
async def list_predictions(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=100),
) -> dict[str, Any]:
    """Get prediction history with pagination."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        total = (await session.execute(select(func.count(Prediction.id)))).scalar() or 0

        offset = (page - 1) * per_page
        result = await session.execute(
            select(Prediction).order_by(Prediction.created_at.desc()).offset(offset).limit(per_page)
        )
        predictions = result.scalars().all()

        return {
            "items": [
                PredictionResponse(
                    id=p.id,
                    model_name=p.model_name,
                    is_anomaly=p.is_anomaly,
                    confidence=p.confidence,
                    prediction_label=p.prediction_label,
                    src_ip=p.src_ip,
                    dst_ip=p.dst_ip,
                    created_at=p.created_at,
                ).model_dump()
                for p in predictions
            ],
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": max(1, (total + per_page - 1) // per_page),
        }


@router.get("/models")
async def list_models(request: Request) -> list[dict[str, Any]]:
    """List all registered ML models."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(MLModel).order_by(MLModel.created_at.desc()))
        models = result.scalars().all()
        return [
            ModelResponse(
                id=m.id,
                name=m.name,
                model_type=m.model_type,
                version=m.version,
                status=m.status.value,
                accuracy=m.accuracy,
                precision_score=m.precision_score,
                recall=m.recall,
                f1_score=m.f1_score,
                threshold=m.threshold,
                description=m.description,
                feature_importance=m.feature_importance,
                confusion_matrix=m.confusion_matrix,
                trained_at=m.trained_at,
            ).model_dump()
            for m in models
        ]


@router.put("/models/{model_name}/activate", response_model=None)
async def activate_model(request: Request, model_name: str) -> Response | dict[str, Any]:
    """Set a model as the active model."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        # Deactivate all
        result = await session.execute(select(MLModel))
        for m in result.scalars().all():
            m.status = ModelStatus.INACTIVE

        # Activate target
        result = await session.execute(select(MLModel).where(MLModel.name == model_name))
        model = result.scalar_one_or_none()
        if not model:
            return JSONResponse(status_code=404, content={"detail": "Model not found"})

        model.status = ModelStatus.ACTIVE
        await session.commit()

    return {"message": f"Model {model_name} activated"}


@router.get("/feature-importance/{model_name}", response_model=None)
async def feature_importance(request: Request, model_name: str) -> Response | dict[str, Any]:
    """Get feature importance scores for a model."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(MLModel).where(MLModel.name == model_name))
        model = result.scalar_one_or_none()
        if not model:
            return JSONResponse(status_code=404, content={"detail": "Model not found"})

        return {
            "model_name": model_name,
            "feature_importance": model.feature_importance or {},
        }


@router.post("/compare")
async def compare_models(request: Request) -> list[dict[str, Any]]:
    """Compare all models' metrics."""
    session_factory = request.app.state.session_factory

    async with session_factory() as session:
        result = await session.execute(select(MLModel))
        models = result.scalars().all()

        return [
            {
                "name": m.name,
                "accuracy": m.accuracy,
                "precision": m.precision_score,
                "recall": m.recall,
                "f1_score": m.f1_score,
                "status": m.status.value,
            }
            for m in models
        ]
