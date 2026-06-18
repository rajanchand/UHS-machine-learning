"""Integration tests for session-based authentication and route gating."""

import pytest
from httpx import AsyncClient, ASGITransport
from anomaly_detection.app import create_app
from anomaly_detection.db.models import User
from anomaly_detection.utils.auth import hash_password
from anomaly_detection.config import get_settings
from anomaly_detection.services.inference import InferenceService


@pytest.fixture
def unauthenticated_app(session_factory, db_engine):
    """Provide a fresh app instance for auth tests with loaded models."""
    app = create_app()
    app.router.lifespan_context = None
    app.state.session_factory = session_factory

    settings = get_settings()
    inference_service = InferenceService(
        model_registry_path=settings.model_registry_path,
        data_dir=settings.data_dir,
    )
    inference_service.load_models()

    # If no models were loaded (e.g. in CI due to gitignored folders), seed with dummy models for testing
    if not inference_service.available_models:
        import numpy as np

        class DummyDetector:
            def __init__(self, name: str) -> None:
                self.name = name

            def score(self, X: np.ndarray) -> np.ndarray:
                return np.zeros(X.shape[0])

        for model_name in [
            "isolation_forest",
            "autoencoder",
            "halfspace_trees",
            "lightgbm_benchmark",
            "random_forest",
            "xgboost",
        ]:
            inference_service._models[model_name] = DummyDetector(model_name)  # type: ignore[assignment]
            inference_service._thresholds[model_name] = 0.5
        inference_service._active_model = "isolation_forest"

    app.state.inference_service = inference_service
    app.state.settings = settings
    app.state.metrics_inference_count = 0
    app.state.metrics_inference_sum = 0.0
    return app


@pytest.mark.anyio
async def test_auth_gating_unauthenticated(unauthenticated_app):
    """Verify that gated routes return 401 when request is unauthenticated."""
    async with AsyncClient(
        transport=ASGITransport(app=unauthenticated_app), base_url="http://testserver"
    ) as client:
        # A gated endpoint
        response = await client.get("/api/v1/alerts")
        assert response.status_code == 401
        assert response.json()["detail"] == "Not authenticated"


@pytest.mark.anyio
async def test_auth_gating_open_endpoints(unauthenticated_app):
    """Verify that open endpoints do not require authentication."""
    async with AsyncClient(
        transport=ASGITransport(app=unauthenticated_app), base_url="http://testserver"
    ) as client:
        # Open endpoints
        res_health = await client.get("/health")
        assert res_health.status_code == 200

        res_ready = await client.get("/ready")
        assert res_ready.status_code == 200

        res_metrics = await client.get("/metrics")
        assert res_metrics.status_code == 200


@pytest.mark.anyio
async def test_auth_gating_simulator_bypass(unauthenticated_app):
    """Verify that simulator routes bypass auth with correct headers or host."""
    # Use remote IP address (8.8.8.8) to simulate non-localhost client
    transport = ASGITransport(app=unauthenticated_app, client=("8.8.8.8", 1234))

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Gated simulation endpoints without header or localhost -> 401
        res_no_key = await client.post("/api/v1/flows/stream", json={})
        assert res_no_key.status_code == 401

        # Bypass using X-API-Key simulator-secret
        res_with_key = await client.post(
            "/api/v1/flows/stream", json={}, headers={"X-API-Key": "simulator-secret"}
        )
        # Bypasses auth: returns 422 because of empty request body validation rather than 401
        assert res_with_key.status_code == 422


@pytest.mark.anyio
async def test_auth_lifecycle(unauthenticated_app, session_factory):
    """Verify login, authentication persistence, and logout flow."""
    # Seed a test user in DB
    async with session_factory() as session:
        user = User(
            username="analyst_bob", password_hash=hash_password("bob_secure_pwd")
        )
        session.add(user)
        await session.commit()

    async with AsyncClient(
        transport=ASGITransport(app=unauthenticated_app), base_url="http://testserver"
    ) as client:
        # Me check initially -> 401
        res_me_init = await client.get("/api/v1/auth/me")
        assert res_me_init.status_code == 401

        # Invalid login -> 401
        res_login_bad = await client.post(
            "/api/v1/auth/login",
            json={"username": "analyst_bob", "password": "wrong_password"},
        )
        assert res_login_bad.status_code == 401

        # Valid login -> 200
        res_login_good = await client.post(
            "/api/v1/auth/login",
            json={"username": "analyst_bob", "password": "bob_secure_pwd"},
        )
        assert res_login_good.status_code == 200
        data = res_login_good.json()
        assert data["username"] == "analyst_bob"
        assert data["status"] == "authenticated"

        # Me check after login -> 200
        res_me_after = await client.get("/api/v1/auth/me")
        assert res_me_after.status_code == 200
        assert res_me_after.json()["username"] == "analyst_bob"

        # Logout -> 200
        res_logout = await client.post("/api/v1/auth/logout")
        assert res_logout.status_code == 200

        # Me check after logout -> 401
        res_me_final = await client.get("/api/v1/auth/me")
        assert res_me_final.status_code == 401
