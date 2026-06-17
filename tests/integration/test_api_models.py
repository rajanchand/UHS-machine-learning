import pytest
from httpx import AsyncClient

@pytest.mark.anyio
async def test_list_models(app_client: AsyncClient):
    response = await app_client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # The models that are successfully loaded should be in this list
    # Since we trained them, at least some should be loaded
    assert len(data) >= 1
    assert any(m["name"] == "isolation_forest" for m in data)

@pytest.mark.anyio
async def test_update_threshold(app_client: AsyncClient):
    # Update threshold for isolation_forest
    payload = {"threshold": 0.75}
    response = await app_client.put("/api/v1/models/isolation_forest/threshold", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "isolation_forest"
    assert data["threshold"] == 0.75

@pytest.mark.anyio
async def test_update_threshold_not_found(app_client: AsyncClient):
    payload = {"threshold": 0.75}
    response = await app_client.put("/api/v1/models/non_existent_model/threshold", json=payload)
    assert response.status_code == 404

@pytest.mark.anyio
async def test_stats_kpi(app_client: AsyncClient):
    response = await app_client.get("/api/v1/stats/kpi")
    assert response.status_code == 200
    data = response.json()
    assert "total_flows" in data
    assert "total_alerts" in data
    assert "open_alerts" in data
    assert "estimated_fpr" in data
    assert "top_talkers" in data

@pytest.mark.anyio
async def test_stats_timeline(app_client: AsyncClient):
    response = await app_client.get("/api/v1/stats/timeline")
    assert response.status_code == 200
    data = response.json()
    assert "points" in data
    assert "threshold" in data
