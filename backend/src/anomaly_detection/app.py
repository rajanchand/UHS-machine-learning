"""FastAPI application factory with JWT auth, static frontend, and all routers."""

from __future__ import annotations

import random
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select, text
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from anomaly_detection.authentication import get_user_from_token, hash_password
from anomaly_detection.config import get_settings
from anomaly_detection.db.engine import create_engine
from anomaly_detection.db.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
    Attack,
    Base,
    BlockedIP,
    MLModel,
    ModelStatus,
    Packet,
    Prediction,
    Setting,
    SystemLog,
    User,
    UserRole,
    UserStatus,
)
from anomaly_detection.db.session import create_session_factory
from anomaly_detection.logging import get_logger, setup_logging

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = get_logger(__name__)

# Paths that don't require authentication
_OPEN_PATHS = {
    "/health", "/ready", "/api/health", "/api/ready", "/api/v1/auth/login", "/api/v1/auth/register",
    "/api/v1/auth/forgot-password", "/api/v1/auth/refresh",
    "/api/v1/seed",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application startup and shutdown lifecycle."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("starting_application", env=settings.environment)

    engine = create_engine(settings)
    session_factory = create_session_factory(engine)

    app.state.settings = settings
    app.state.engine = engine
    app.state.session_factory = session_factory

    # Create tables
    from sqlalchemy import inspect as sa_inspect
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed default admin user if none exists
    async with session_factory() as session:
        result = await session.execute(select(func.count(User.id)))
        if (result.scalar() or 0) == 0:
            admin = User(
                username="admin",
                email="admin@anomalyguard.local",
                password_hash=hash_password("admin123"),
                full_name="System Administrator",
                role=UserRole.ADMIN,
                status=UserStatus.ACTIVE,
            )
            analyst = User(
                username="analyst",
                email="analyst@anomalyguard.local",
                password_hash=hash_password("analyst123"),
                full_name="Security Analyst",
                role=UserRole.ANALYST,
                status=UserStatus.ACTIVE,
            )
            session.add_all([admin, analyst])
            await session.commit()
            logger.info("default_users_seeded")

    # Initialize Inference Service
    from anomaly_detection.services.inference import InferenceService
    from pathlib import Path
    
    inference_service = InferenceService(
        model_registry_path=Path(settings.model_registry_path),
        data_dir=Path(settings.data_dir),
    )
    inference_service.load_models()
    app.state.inference_service = inference_service
    app.state.active_scenario = None
    app.state.metrics_inference_count = 0
    app.state.metrics_inference_sum = 0.0

    logger.info("application_ready")
    yield
    logger.info("shutting_down")
    await engine.dispose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="AnomalyGuard — Network Anomaly Detection API",
        description="ML-powered network traffic anomaly detection system for MSc Dissertation",
        version="1.0.0",
        lifespan=lifespan,
    )

    # JWT Authentication Middleware
    class JWTAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(
            self, request: Request, call_next: RequestResponseEndpoint
        ) -> Response:
            path = request.url.path

            # Skip auth for open paths, docs, and static files
            if (
                path in _OPEN_PATHS
                or path.startswith("/docs")
                or path.startswith("/openapi")
                or path.startswith("/static")
                or path.startswith("/css")
                or path.startswith("/js")
                or path.startswith("/img")
                or path.endswith(".html")
                or path.endswith(".ico")
                or path == "/"
            ):
                return await call_next(request)

            # Bypass for simulator routes with valid API key
            is_simulator_route = path in ("/api/v1/flows/stream", "/api/v1/flows/batch", "/simulate", "/api/simulate")
            has_api_key = request.headers.get("X-API-Key") == settings.simulator_api_key
            if is_simulator_route and has_api_key:
                return await call_next(request)

            # Extract JWT token from Authorization header
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                user_info = get_user_from_token(
                    token, settings.jwt_secret_key, settings.jwt_algorithm
                )
                if user_info and user_info.get("token_type") == "access":
                    request.state.user = user_info
                    return await call_next(request)

            return JSONResponse(
                status_code=401,
                content={"detail": "Not authenticated"},
            )

    app.add_middleware(JWTAuthMiddleware)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check
    @app.get("/health", tags=["system"])
    @app.get("/api/health", tags=["system"])
    async def health_check() -> dict:
        return {"status": "ok"}

    @app.get("/ready", tags=["system"])
    @app.get("/api/ready", tags=["system"])
    async def readiness_check() -> dict:
        try:
            async with app.state.session_factory() as session:
                await session.execute(text("SELECT 1"))
        except Exception:
            raise HTTPException(status_code=503, detail="Database not reachable")
        return {"status": "ready"}

    # Simulator scenario endpoints
    @app.post("/simulate", include_in_schema=False)
    @app.post("/api/simulate", include_in_schema=False)
    async def set_simulate_scenario(payload: dict) -> dict:
        scenario = payload.get("scenario")
        if scenario not in ["port_scan", "ddos", "brute_force", None, ""]:
            raise HTTPException(status_code=400, detail="Invalid scenario")
        app.state.active_scenario = scenario or None
        return {"active_scenario": app.state.active_scenario}

    @app.get("/simulate", include_in_schema=False)
    @app.get("/api/simulate", include_in_schema=False)
    async def get_simulate_scenario() -> dict:
        return {"active_scenario": getattr(app.state, "active_scenario", None)}

    # Seed demo data endpoint
    @app.post("/api/v1/seed", tags=["system"])
    async def seed_demo_data() -> dict:
        """Seed the database with demo data for demonstration."""
        session_factory = app.state.session_factory

        async with session_factory() as session:
            now = datetime.now(timezone.utc)

            # Seed ML Models
            result = await session.execute(select(func.count(MLModel.id)))
            if (result.scalar() or 0) == 0:
                models_data = [
                    ("random_forest", "Random Forest", 0.9934, 0.9892, 0.9125, 0.9493, ModelStatus.ACTIVE),
                    ("isolation_forest", "Isolation Forest", 0.9412, 0.6000, 0.7140, 0.6520, ModelStatus.INACTIVE),
                    ("decision_tree", "Decision Tree", 0.9801, 0.9710, 0.8890, 0.9282, ModelStatus.INACTIVE),
                    ("xgboost", "XGBoost", 0.9956, 0.9921, 0.9340, 0.9622, ModelStatus.INACTIVE),
                ]
                for name, mtype, acc, prec, rec, f1, status in models_data:
                    features = {
                        "duration": round(random.uniform(0.05, 0.15), 4),
                        "src_bytes": round(random.uniform(0.08, 0.18), 4),
                        "dst_bytes": round(random.uniform(0.06, 0.14), 4),
                        "count": round(random.uniform(0.04, 0.12), 4),
                        "srv_count": round(random.uniform(0.03, 0.10), 4),
                        "serror_rate": round(random.uniform(0.05, 0.13), 4),
                        "dst_host_count": round(random.uniform(0.04, 0.11), 4),
                        "packet_rate": round(random.uniform(0.06, 0.14), 4),
                        "byte_rate": round(random.uniform(0.05, 0.12), 4),
                        "port_entropy": round(random.uniform(0.03, 0.09), 4),
                    }
                    session.add(MLModel(
                        name=name, model_type=mtype, version="v1",
                        status=status, accuracy=acc, precision_score=prec,
                        recall=rec, f1_score=f1, threshold=0.5,
                        artifact_path=f"models/{name}/v1",
                        description=f"{mtype} classifier for network anomaly detection",
                        feature_importance=features,
                        confusion_matrix={
                            "true_positive": random.randint(800, 1200),
                            "false_positive": random.randint(10, 50),
                            "true_negative": random.randint(3000, 5000),
                            "false_negative": random.randint(20, 80),
                        },
                    ))

            # Seed Packets
            result = await session.execute(select(func.count(Packet.id)))
            if (result.scalar() or 0) < 100:
                protocols = ["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "DNS", "SSH", "FTP"]
                for i in range(200):
                    ts = now - timedelta(hours=random.randint(0, 168))
                    session.add(Packet(
                        timestamp=ts,
                        src_ip=f"192.168.{random.randint(1, 10)}.{random.randint(1, 254)}",
                        dst_ip=f"10.0.{random.randint(0, 5)}.{random.randint(1, 254)}",
                        protocol=random.choice(protocols),
                        src_port=random.randint(1024, 65535),
                        dst_port=random.choice([22, 53, 80, 443, 3306, 5432, 8080]),
                        packet_size=random.randint(40, 1500),
                        ttl=random.choice([64, 128, 255]),
                        flags=random.choice(["SYN", "ACK", "SYN-ACK", "FIN", "RST", "PSH-ACK", ""]),
                        status=random.choices(["Normal", "Suspicious", "Malicious"], weights=[80, 15, 5])[0],
                    ))

            # Seed Attacks
            result = await session.execute(select(func.count(Attack.id)))
            if (result.scalar() or 0) < 20:
                attack_types = [
                    ("DDoS", AlertSeverity.CRITICAL, "Block source IP and enable rate limiting"),
                    ("DoS", AlertSeverity.HIGH, "Enable SYN cookies and rate limit connections"),
                    ("Port Scan", AlertSeverity.MEDIUM, "Monitor source IP and update firewall rules"),
                    ("Brute Force", AlertSeverity.HIGH, "Lock account and block IP after 5 failed attempts"),
                    ("Botnet", AlertSeverity.CRITICAL, "Isolate affected hosts and scan for malware"),
                    ("ARP Spoofing", AlertSeverity.HIGH, "Enable dynamic ARP inspection"),
                    ("DNS Attack", AlertSeverity.MEDIUM, "Verify DNS responses and enable DNSSEC"),
                    ("ICMP Flood", AlertSeverity.MEDIUM, "Rate limit ICMP traffic at firewall"),
                    ("SSH Attack", AlertSeverity.HIGH, "Disable password auth, use key-based only"),
                    ("FTP Attack", AlertSeverity.MEDIUM, "Disable FTP, switch to SFTP"),
                ]
                for i in range(40):
                    atype, severity, rec = random.choice(attack_types)
                    ts = now - timedelta(hours=random.randint(0, 168))
                    session.add(Attack(
                        attack_type=atype, severity=severity,
                        confidence=round(random.uniform(0.65, 0.99), 4),
                        src_ip=f"192.168.{random.randint(1, 20)}.{random.randint(1, 254)}",
                        dst_ip=f"10.0.{random.randint(0, 5)}.{random.randint(1, 254)}",
                        src_port=random.randint(1024, 65535),
                        dst_port=random.choice([22, 53, 80, 443, 3306, 8080]),
                        protocol=random.choice(["TCP", "UDP", "ICMP"]),
                        recommendation=rec,
                        detected_at=ts,
                    ))

            # Seed Alerts
            result = await session.execute(select(func.count(Alert.id)))
            if (result.scalar() or 0) < 20:
                alert_titles = [
                    "Suspicious traffic pattern detected",
                    "Potential DDoS attack in progress",
                    "Port scan activity from external IP",
                    "Multiple failed login attempts",
                    "Anomalous packet size detected",
                    "Unusual protocol usage detected",
                    "High bandwidth consumption alert",
                    "Brute force attack on SSH service",
                    "DNS query anomaly detected",
                    "Potential data exfiltration detected",
                ]
                for i in range(30):
                    ts = now - timedelta(hours=random.randint(0, 168))
                    sev = random.choice(list(AlertSeverity))
                    session.add(Alert(
                        title=random.choice(alert_titles),
                        description=f"Automated alert #{i+1} triggered by anomaly detection engine",
                        severity=sev,
                        status=random.choice(list(AlertStatus)),
                        source_ip=f"192.168.{random.randint(1, 20)}.{random.randint(1, 254)}",
                        attack_type=random.choice(["DDoS", "Port Scan", "Brute Force", "DoS", None]),
                        is_read=random.choice([True, False]),
                        created_at=ts,
                    ))

            # Seed Predictions
            result = await session.execute(select(func.count(Prediction.id)))
            if (result.scalar() or 0) < 50:
                model_names = ["random_forest", "isolation_forest", "decision_tree", "xgboost"]
                for i in range(100):
                    is_anomaly = random.random() < 0.2
                    ts = now - timedelta(hours=random.randint(0, 168))
                    session.add(Prediction(
                        model_name=random.choice(model_names),
                        is_anomaly=is_anomaly,
                        confidence=round(random.uniform(0.7, 0.99), 4),
                        prediction_label="Anomaly" if is_anomaly else "Normal",
                        src_ip=f"192.168.{random.randint(1, 10)}.{random.randint(1, 254)}",
                        dst_ip=f"10.0.{random.randint(0, 5)}.{random.randint(1, 254)}",
                        created_at=ts,
                    ))

            # Seed System Logs
            result = await session.execute(select(func.count(SystemLog.id)))
            if (result.scalar() or 0) < 20:
                log_messages = [
                    ("info", "system", "Application started successfully"),
                    ("info", "ml_engine", "Model loaded: random_forest v1"),
                    ("info", "monitoring", "Packet capture started on eth0"),
                    ("warning", "security", "Rate limit threshold reached for 192.168.1.105"),
                    ("error", "database", "Connection pool exhausted, expanding"),
                    ("info", "ml_engine", "Batch prediction completed: 500 flows processed"),
                    ("info", "auth", "New user registered: analyst@anomalyguard.local"),
                    ("warning", "monitoring", "High bandwidth usage detected: 85%"),
                    ("info", "reports", "Daily report generated successfully"),
                    ("critical", "security", "Multiple failed login attempts from 10.0.2.33"),
                ]
                for level, source, msg in log_messages:
                    ts = now - timedelta(hours=random.randint(0, 48))
                    session.add(SystemLog(level=level, source=source, message=msg, created_at=ts))

            await session.commit()

        return {"message": "Demo data seeded successfully"}

    # Register all API routers
    from anomaly_detection.api.routers.alerts import router as alerts_router
    from anomaly_detection.api.routers.analytics import router as analytics_router
    from anomaly_detection.api.routers.attacks import router as attacks_router
    from anomaly_detection.api.routers.auth import router as auth_router
    from anomaly_detection.api.routers.dashboard import router as dashboard_router
    from anomaly_detection.api.routers.datasets import router as datasets_router
    from anomaly_detection.api.routers.flows import router as flows_router
    from anomaly_detection.api.routers.logs import router as logs_router
    from anomaly_detection.api.routers.ml_prediction import router as ml_router
    from anomaly_detection.api.routers.monitoring import router as monitoring_router
    from anomaly_detection.api.routers.packets import router as packets_router
    from anomaly_detection.api.routers.profile import router as profile_router
    from anomaly_detection.api.routers.reports import router as reports_router
    from anomaly_detection.api.routers.settings import router as settings_router
    from anomaly_detection.api.routers.users import router as users_router

    app.include_router(auth_router)
    app.include_router(dashboard_router)
    app.include_router(monitoring_router)
    app.include_router(packets_router)
    app.include_router(ml_router)
    app.include_router(attacks_router)
    app.include_router(alerts_router)
    app.include_router(analytics_router)
    app.include_router(reports_router)
    app.include_router(datasets_router)
    app.include_router(users_router)
    app.include_router(logs_router)
    app.include_router(settings_router)
    app.include_router(profile_router)
    app.include_router(flows_router)

    # Serve frontend static files
    frontend_path = Path(__file__).resolve().parent.parent.parent.parent / "frontend"
    if frontend_path.exists():
        # Mount static asset directories
        for subdir in ["css", "js", "img"]:
            asset_path = frontend_path / subdir
            if asset_path.exists():
                app.mount(f"/{subdir}", StaticFiles(directory=str(asset_path)), name=subdir)

        # Serve HTML pages
        @app.get("/", include_in_schema=False)
        async def serve_index():
            index = frontend_path / "index.html"
            if index.exists():
                return FileResponse(str(index))
            raise HTTPException(status_code=404, detail="Frontend not found")

        @app.get("/{page_name}.html", include_in_schema=False)
        async def serve_page(page_name: str):
            page = frontend_path / f"{page_name}.html"
            if page.exists():
                return FileResponse(str(page))
            raise HTTPException(status_code=404, detail="Page not found")

    return app
