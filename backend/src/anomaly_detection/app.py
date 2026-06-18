"""FastAPI application factory with lifespan management."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from sqlalchemy import func, select, text
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.middleware.sessions import SessionMiddleware

from anomaly_detection.config import get_settings
from anomaly_detection.constants import FEATURE_COLUMNS
from anomaly_detection.db.engine import create_engine
from anomaly_detection.db.models import Alert, Flow, MLModel
from anomaly_detection.db.session import create_session_factory
from anomaly_detection.logging import get_logger, setup_logging
from anomaly_detection.schemas.common import HealthResponse
from anomaly_detection.services.inference import InferenceService

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = get_logger(__name__)

_OPEN_PATHS = {"/health", "/ready", "/metrics", "/api/v1/auth/login"}
_SIMULATOR_ROUTES = {"/api/v1/flows/stream", "/api/v1/flows/batch", "/simulate"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info("starting_application", log_level=settings.log_level, env=settings.environment)

    engine = create_engine(settings)
    session_factory = create_session_factory(engine)

    inference_service = InferenceService(
        model_registry_path=settings.model_registry_path,
        data_dir=settings.data_dir,
    )
    inference_service.load_models()

    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.inference_service = inference_service

    active_model_from_db: str | None = None

    async with session_factory() as session:
        for model_name in inference_service.available_models:
            result = await session.execute(select(MLModel).where(MLModel.name == model_name))
            existing = result.scalar_one_or_none()

            if not existing:
                metrics: dict[str, Any] = {}
                metrics_path = settings.data_dir.parent / "evaluation" / "metrics.json"
                if metrics_path.exists():
                    try:
                        all_metrics = json.loads(metrics_path.read_text())
                        metrics = all_metrics.get(model_name, {})
                    except Exception:
                        logger.warning("metrics_json_unreadable", path=str(metrics_path))

                seed_threshold = (
                    metrics.get("threshold_at_1pct_fpr")
                    or settings.default_thresholds.get(model_name)
                    or 0.5
                )
                inference_service.set_threshold(model_name, seed_threshold)

                model_path = settings.model_registry_path / model_name / "v1"
                session.add(MLModel(
                    name=model_name,
                    version="v1",
                    metrics_json=metrics,
                    artifact_path=str(model_path),
                    threshold=seed_threshold,
                    is_active=(model_name == inference_service.active_model_name),
                    description=f"Auto-registered {model_name} v1",
                ))
            else:
                inference_service.set_threshold(model_name, existing.threshold)
                if existing.is_active:
                    active_model_from_db = model_name

        await session.commit()

    if active_model_from_db and active_model_from_db in inference_service.available_models:
        inference_service.set_active_model(active_model_from_db)
        logger.info("active_model_restored_from_db", model=active_model_from_db)

    # Precompute baseline quantile bins for drift detection
    train_parquet = settings.data_dir / "processed" / "train.parquet"
    if train_parquet.exists():
        try:
            logger.info("loading_training_baseline", path=str(train_parquet))
            train_df = pd.read_parquet(train_parquet)
            reference_quantiles: dict[str, list[float]] = {}
            for col in FEATURE_COLUMNS:
                if col in train_df.columns:
                    series = train_df[col].dropna()
                    deciles = np.percentile(series, [10, 20, 30, 40, 50, 60, 70, 80, 90])
                    reference_quantiles[col] = np.sort(np.unique(deciles)).tolist()
            app.state.reference_quantiles = reference_quantiles
            logger.info("training_baseline_loaded", feature_count=len(reference_quantiles))
        except Exception as exc:
            logger.error("training_baseline_load_failed", error=str(exc))
            app.state.reference_quantiles = {}
    else:
        logger.warning("train_parquet_not_found", path=str(train_parquet))
        app.state.reference_quantiles = {}

    app.state.metrics_inference_count = 0
    app.state.metrics_inference_sum = 0.0

    logger.info(
        "application_ready",
        models=inference_service.available_models,
        active_model=inference_service.active_model_name,
    )
    yield

    logger.info("shutting_down")
    await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Network Anomaly Detection API",
        description="ML-powered network traffic anomaly detection system",
        version="0.1.0",
        lifespan=lifespan,
    )

    class AuthGatingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
            path = request.url.path

            if (
                path in _OPEN_PATHS
                or path.startswith("/docs")
                or path.startswith("/openapi.json")
            ):
                return await call_next(request)

            is_simulator_route = path in _SIMULATOR_ROUTES
            is_local = request.client is not None and request.client.host in (
                "127.0.0.1", "::1", "localhost"
            )
            has_api_key = (
                request.headers.get("X-API-Key") == settings.simulator_api_key
            )

            if is_simulator_route and (is_local or has_api_key):
                return await call_next(request)

            if not request.session.get("user"):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Not authenticated"},
                )

            return await call_next(request)

    app.add_middleware(AuthGatingMiddleware)

    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret_key,
        same_site="lax",
        https_only=settings.environment == "production",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.active_scenario = None

    @app.post("/simulate")
    async def set_simulate_scenario(payload: dict[str, Any]) -> dict[str, Any]:
        scenario = payload.get("scenario")
        if scenario not in ("port_scan", "ddos", "brute_force", None, ""):
            raise HTTPException(status_code=400, detail="Invalid scenario")
        app.state.active_scenario = scenario or None
        return {"active_scenario": app.state.active_scenario}

    @app.get("/simulate")
    async def get_simulate_scenario() -> dict[str, Any]:
        return {"active_scenario": getattr(app.state, "active_scenario", None)}

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health_check() -> HealthResponse:
        return HealthResponse()

    @app.get("/ready", tags=["system"])
    async def readiness_check() -> dict[str, str]:
        try:
            async with app.state.session_factory() as session:
                await session.execute(text("SELECT 1"))
        except Exception:
            raise HTTPException(status_code=503, detail="Database not reachable")

        if (
            not getattr(app.state, "inference_service", None)
            or not app.state.inference_service.active_model_name
        ):
            raise HTTPException(status_code=503, detail="No active model loaded")

        return {"status": "ready"}

    @app.get("/metrics", tags=["system"])
    async def metrics_endpoint() -> Response:
        flows_count = 0
        alerts_count = 0
        try:
            async with app.state.session_factory() as session:
                flows_count = (await session.execute(select(func.count(Flow.id)))).scalar() or 0
                alerts_count = (await session.execute(select(func.count(Alert.id)))).scalar() or 0
        except Exception:
            pass

        active_model = "none"
        if getattr(app.state, "inference_service", None):
            active_model = app.state.inference_service.active_model_name or "none"

        latency_count = getattr(app.state, "metrics_inference_count", 0)
        latency_sum = getattr(app.state, "metrics_inference_sum", 0.0)

        body = "\n".join([
            "# HELP flows_processed_total Total network flows processed.",
            "# TYPE flows_processed_total counter",
            f"flows_processed_total {flows_count}",
            "# HELP alerts_raised_total Total anomaly alerts raised.",
            "# TYPE alerts_raised_total counter",
            f"alerts_raised_total {alerts_count}",
            "# HELP inference_latency_seconds_count Inference measurement count.",
            "# TYPE inference_latency_seconds_count counter",
            f"inference_latency_seconds_count {latency_count}",
            "# HELP inference_latency_seconds_sum Sum of inference latencies.",
            "# TYPE inference_latency_seconds_sum counter",
            f"inference_latency_seconds_sum {latency_sum:.6f}",
            "# HELP active_model_version Active model gauge.",
            "# TYPE active_model_version gauge",
            f'active_model_version{{model="{active_model}",version="v1"}} 1',
        ])
        return Response(content=body + "\n", media_type="text/plain")

    from anomaly_detection.api.routers.alerts import router as alerts_router
    from anomaly_detection.api.routers.auth import router as auth_router
    from anomaly_detection.api.routers.drift import router as drift_router
    from anomaly_detection.api.routers.flows import router as flows_router
    from anomaly_detection.api.routers.models_stats import models_router, stats_router

    app.include_router(flows_router)
    app.include_router(alerts_router)
    app.include_router(models_router)
    app.include_router(stats_router)
    app.include_router(auth_router)
    app.include_router(drift_router)

    return app
