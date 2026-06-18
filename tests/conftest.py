"""Shared test fixtures and configuration."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator
import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

# ---------------------------------------------------------------------------
# Point settings at the real local model/data directories BEFORE importing
# anything from anomaly_detection (which triggers get_settings() at import time
# in some code paths).
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["MODEL_REGISTRY_PATH"] = str(_PROJECT_ROOT / "models")
os.environ["DATA_DIR"] = str(_PROJECT_ROOT / "data")
# Use SQLite for tests instead of PostgreSQL
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
# Bypass rate limiter in tests
os.environ["LOGIN_RATE_LIMIT"] = "9999"

from anomaly_detection.app import create_app  # noqa: E402
from anomaly_detection.db.models import Base  # noqa: E402
from anomaly_detection.services.inference import InferenceService  # noqa: E402
from anomaly_detection.config import get_settings  # noqa: E402

# Fixture data path
FIXTURE_DIR = _PROJECT_ROOT / "data" / "fixtures"
FIXTURE_CSV = FIXTURE_DIR / "cicids2017_sample.csv"



@pytest.fixture
def fixture_csv_path() -> Path:
    """Path to the fixture CSV for testing."""
    assert FIXTURE_CSV.exists(), (
        f"Fixture CSV not found at {FIXTURE_CSV}. "
        "Run: python -m anomaly_detection.pipeline.generate_fixture"
    )
    return FIXTURE_CSV


@pytest.fixture
def fixture_dir() -> Path:
    """Path to the fixtures directory."""
    return FIXTURE_DIR


@pytest.fixture(scope="session")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
async def db_engine():
    """Create a SQLite in-memory database engine for testing."""
    # Use SQLite + aiosqlite for testing
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    
    @event.listens_for(engine.sync_engine, "connect")
    def register_sqlite_functions(dbapi_connection, connection_record):
        # Register custom date_trunc for SQLite minute bucket aggregation
        def date_trunc(field, dt_str):
            if not dt_str:
                return dt_str
            # Truncate to minute 'YYYY-MM-DD HH:MM:00'
            return str(dt_str)[:16] + ":00"
        dbapi_connection.create_function("date_trunc", 2, date_trunc)
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    yield engine
    
    await engine.dispose()


@pytest.fixture
async def session_factory(db_engine) -> async_sessionmaker[AsyncSession]:
    """Create an async session factory bound to the testing database engine."""
    return async_sessionmaker(
        bind=db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@pytest.fixture
async def db_session(session_factory) -> AsyncGenerator[AsyncSession, None]:
    """Provide a transactional database session for a single test."""
    async with session_factory() as session:
        yield session
        # Roll back changes to keep tests independent and fast
        await session.rollback()


@pytest.fixture
async def app_client(session_factory) -> AsyncGenerator[AsyncClient, None]:
    """Provide an HTTPX client pointing to the FastAPI application with mocked DB."""
    app = create_app()
    
    # Disable the real lifespan to prevent it connecting to PostgreSQL
    app.router.lifespan_context = None
    
    # Manually configure state
    app.state.session_factory = session_factory
    
    # Setup test inference service using local registry
    settings = get_settings()
    inference_service = InferenceService(
        model_registry_path=settings.model_registry_path,
        data_dir=settings.data_dir,
    )
    inference_service.load_models()
    app.state.inference_service = inference_service
    app.state.settings = settings
    app.state.metrics_inference_count = 0
    app.state.metrics_inference_sum = 0.0

    # Register models in database for list_models test if not already present
    from anomaly_detection.db.models import MLModel, User
    from anomaly_detection.utils.auth import hash_password
    from sqlalchemy import select
    async with session_factory() as session:
        for model_name in inference_service.available_models:
            result = await session.execute(
                select(MLModel).where(MLModel.name == model_name)
            )
            if not result.scalar_one_or_none():
                model_path = settings.model_registry_path / model_name / "v1"
                new_model = MLModel(
                    name=model_name,
                    version="v1",
                    metrics_json={},
                    artifact_path=str(model_path),
                    threshold=inference_service.get_threshold(model_name),
                    is_active=(model_name == inference_service.active_model_name),
                    description=f"Test registered {model_name} v1",
                )
                session.add(new_model)
        
        # Seed test user
        result_user = await session.execute(
            select(User).where(User.username == "test_analyst")
        )
        if not result_user.scalar_one_or_none():
            user = User(
                username="test_analyst",
                password_hash=hash_password("test_pass")
            )
            session.add(user)
            
        await session.commit()
    
    # We use ASGITransport to call FastAPI in-process without spinning up a real server
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        # Perform session login
        await client.post(
            "/api/v1/auth/login",
            json={"username": "test_analyst", "password": "test_pass"}
        )
        yield client
