"""FastAPI application factory with lifespan management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from anomaly_detection.config import get_settings
from anomaly_detection.db.engine import create_engine
from anomaly_detection.db.session import create_session_factory
from anomaly_detection.logging import get_logger, setup_logging
from anomaly_detection.schemas.common import HealthResponse
from anomaly_detection.services.inference import InferenceService

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan — initialize and cleanup resources."""
    settings = get_settings()
    setup_logging(settings.log_level)

    logger.info("starting_application", log_level=settings.log_level)

    # Create database engine and session factory
    engine = create_engine(settings)
    session_factory = create_session_factory(engine)

    # Create and load inference service
    inference_service = InferenceService(
        model_registry_path=settings.model_registry_path,
        data_dir=settings.data_dir,
    )
    inference_service.load_models()

    # Store in app state for dependency injection
    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory
    app.state.inference_service = inference_service

    # Auto-register loaded models in DB and sync stored thresholds/active model
    import json

    from sqlalchemy import select

    from anomaly_detection.db.models import MLModel

    # Evaluated thresholds at ~1% FPR from the training harness (fallback defaults)
    EVAL_THRESHOLDS: dict[str, float] = {
        "autoencoder": 0.0035,
        "isolation_forest": 0.39,
        "halfspace_trees": 0.976,
        "lightgbm_benchmark": 0.56,
    }

    active_model_from_db: str | None = None

    async with session_factory() as session:
        for model_name in inference_service.available_models:
            result = await session.execute(
                select(MLModel).where(MLModel.name == model_name)
            )
            existing = result.scalar_one_or_none()
            if not existing:
                # New model — load metrics from evaluation output and seed threshold
                metrics: dict = {}
                metrics_path = settings.data_dir.parent / "evaluation" / "metrics.json"
                if metrics_path.exists():
                    try:
                        all_metrics = json.loads(metrics_path.read_text())
                        metrics = all_metrics.get(model_name, {})
                    except Exception:
                        pass

                # Use evaluated threshold from metrics JSON, then hard-coded map, then 0.5
                seed_thr = (
                    metrics.get("threshold_at_1pct_fpr")
                    or EVAL_THRESHOLDS.get(model_name)
                    or 0.5
                )
                inference_service.set_threshold(model_name, seed_thr)

                model_path = settings.model_registry_path / model_name / "v1"
                new_model = MLModel(
                    name=model_name,
                    version="v1",
                    metrics_json=metrics,
                    artifact_path=str(model_path),
                    threshold=seed_thr,
                    is_active=(model_name == inference_service.active_model_name),
                    description=f"Auto-registered {model_name} v1",
                )
                session.add(new_model)
            else:
                # Existing model — restore stored threshold and active flag into service
                inference_service.set_threshold(model_name, existing.threshold)
                if existing.is_active:
                    active_model_from_db = model_name

        await session.commit()

    # Restore active model from DB (overrides in-memory default)
    if active_model_from_db and active_model_from_db in inference_service.available_models:
        inference_service.set_active_model(active_model_from_db)
        logger.info("active_model_restored_from_db", model=active_model_from_db)

    # Precalculate baseline quantile bins (deciles) for each feature
    import pandas as pd
    import numpy as np
    train_parquet_path = settings.data_dir / "processed" / "train.parquet"
    if train_parquet_path.exists():
        try:
            logger.info("loading_training_baseline", path=str(train_parquet_path))
            train_df = pd.read_parquet(train_parquet_path)
            feature_cols = [
                'flow_duration', 'total_fwd_packets', 'total_bwd_packets', 'total_len_fwd_packets', 'total_len_bwd_packets',
                'fwd_packet_len_max', 'fwd_packet_len_min', 'fwd_packet_len_mean', 'fwd_packet_len_std', 'bwd_packet_len_max',
                'bwd_packet_len_min', 'bwd_packet_len_mean', 'bwd_packet_len_std', 'flow_bytes_per_s', 'flow_packets_per_s',
                'fin_flag_count', 'syn_flag_count', 'rst_flag_count', 'psh_flag_count', 'ack_flag_count', 'urg_flag_count',
                'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max',
                'fwd_iat_min', 'bwd_iat_mean', 'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min', 'down_up_ratio', 'avg_packet_size',
                'avg_fwd_segment_size', 'avg_bwd_segment_size'
            ]
            reference_quantiles = {}
            for col in feature_cols:
                if col in train_df.columns:
                    series = train_df[col].dropna()
                    deciles = np.percentile(series, [10, 20, 30, 40, 50, 60, 70, 80, 90])
                    unique_deciles = np.sort(np.unique(deciles))
                    reference_quantiles[col] = unique_deciles.tolist()
            app.state.reference_quantiles = reference_quantiles
            logger.info("training_baseline_loaded", features_count=len(reference_quantiles))
        except Exception as e:
            logger.error("failed_to_load_training_baseline", error=str(e))
            app.state.reference_quantiles = {}
    else:
        logger.warning("train_parquet_not_found", path=str(train_parquet_path))
        app.state.reference_quantiles = {}

    app.state.metrics_inference_count = 0
    app.state.metrics_inference_sum = 0.0

    logger.info(
        "application_ready",
        models_loaded=inference_service.available_models,
        active_model=inference_service.active_model_name,
    )
    yield

    # Cleanup
    logger.info("shutting_down")
    await engine.dispose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from fastapi import HTTPException, Request, Response
    from fastapi.responses import JSONResponse
    from starlette.middleware.sessions import SessionMiddleware

    settings = get_settings()

    app = FastAPI(
        title="Network Anomaly Detection API",
        description="ML-powered network traffic anomaly detection system",
        version="0.1.0",
        lifespan=lifespan,
    )

    from starlette.middleware.base import BaseHTTPMiddleware
    from fastapi.responses import JSONResponse

    class AuthGatingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            path = request.url.path
            
            # Health, ready, metrics, and login are open
            open_paths = {
                "/health",
                "/ready",
                "/metrics",
                "/api/v1/auth/login",
            }
            if path in open_paths or path.startswith("/docs") or path.startswith("/openapi.json"):
                return await call_next(request)
                
            # Bypass for simulator stream/simulate endpoints
            is_simulator_route = path in ("/api/v1/flows/stream", "/api/v1/flows/batch", "/simulate")
            is_local = request.client is not None and request.client.host in ("127.0.0.1", "::1", "localhost")
            has_api_key = request.headers.get("X-API-Key") == "simulator-secret"
            
            if is_simulator_route and (is_local or has_api_key):
                return await call_next(request)
                
            # Check session
            if not request.session.get("user"):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Not authenticated"},
                )
                
            return await call_next(request)

    app.add_middleware(AuthGatingMiddleware)

    # Session middleware
    app.add_middleware(
        SessionMiddleware,
        secret_key="super-secret-key-change-in-production",
        same_site="lax",
        https_only=False,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize simulation active scenario state
    app.state.active_scenario = None

    @app.post("/simulate")
    async def set_simulate_scenario(payload: dict) -> dict:
        scenario = payload.get("scenario")
        if scenario not in ["port_scan", "ddos", "brute_force", None, ""]:
            raise HTTPException(status_code=400, detail="Invalid scenario")
        app.state.active_scenario = scenario or None
        return {"active_scenario": app.state.active_scenario}

    @app.get("/simulate")
    async def get_simulate_scenario() -> dict:
        return {"active_scenario": getattr(app.state, "active_scenario", None)}

    # Health check
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health_check() -> HealthResponse:
        """Health check endpoint for Docker and load balancers."""
        return HealthResponse()

    @app.get("/ready", tags=["system"])
    async def readiness_check() -> dict[str, str]:
        """Check if DB is reachable and models are loaded."""
        try:
            from sqlalchemy import text
            async with app.state.session_factory() as session:
                await session.execute(text("SELECT 1"))
        except Exception:
            raise HTTPException(status_code=503, detail="Database not reachable")
            
        if not getattr(app.state, "inference_service", None) or not app.state.inference_service.active_model_name:
            raise HTTPException(status_code=503, detail="No active model loaded")
            
        return {"status": "ready"}

    @app.get("/metrics", tags=["system"])
    async def metrics_endpoint() -> Response:
        """Prometheus metrics endpoint."""
        flows_count = 0
        alerts_count = 0
        try:
            from sqlalchemy import func, select
            from anomaly_detection.db.models import Alert, Flow
            async with app.state.session_factory() as session:
                flows_result = await session.execute(select(func.count(Flow.id)))
                flows_count = flows_result.scalar() or 0
                
                alerts_result = await session.execute(select(func.count(Alert.id)))
                alerts_count = alerts_result.scalar() or 0
        except Exception:
            pass
            
        active_model = "none"
        active_version = "v1"
        if getattr(app.state, "inference_service", None):
            active_model = app.state.inference_service.active_model_name or "none"
            
        latency_count = getattr(app.state, "metrics_inference_count", 0)
        latency_sum = getattr(app.state, "metrics_inference_sum", 0.0)
        
        metrics_text = (
            f"# HELP flows_processed_total Total number of network flows processed.\n"
            f"# TYPE flows_processed_total counter\n"
            f"flows_processed_total {flows_count}\n"
            f"# HELP alerts_raised_total Total number of anomaly alerts raised.\n"
            f"# TYPE alerts_raised_total counter\n"
            f"alerts_raised_total {alerts_count}\n"
            f"# HELP inference_latency_seconds_count Number of inference latency measurements.\n"
            f"# TYPE inference_latency_seconds_count counter\n"
            f"inference_latency_seconds_count {latency_count}\n"
            f"# HELP inference_latency_seconds_sum Sum of inference latency in seconds.\n"
            f"# TYPE inference_latency_seconds_sum counter\n"
            f"inference_latency_seconds_sum {latency_sum:.6f}\n"
            f"# HELP active_model_version Active model version metric.\n"
            f"# TYPE active_model_version gauge\n"
            f'active_model_version{{model="{active_model}",version="{active_version}"}} 1\n'
        )
        return Response(content=metrics_text, media_type="text/plain")

    # Include routers
    from anomaly_detection.api.routers.alerts import router as alerts_router
    from anomaly_detection.api.routers.auth import router as auth_router
    from anomaly_detection.api.routers.drift import router as drift_router
    from anomaly_detection.api.routers.flows import router as flows_router
    from anomaly_detection.api.routers.models_stats import (
        models_router,
        stats_router,
    )

    app.include_router(flows_router)
    app.include_router(alerts_router)
    app.include_router(models_router)
    app.include_router(stats_router)
    app.include_router(auth_router)
    app.include_router(drift_router)

    return app
