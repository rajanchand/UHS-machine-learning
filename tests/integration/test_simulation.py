"""Integration tests for attack simulation endpoints."""

import pytest


@pytest.mark.anyio
async def test_simulation_scenario_lifecycle(app_client):
    """Test getting, setting, and clearing active simulation scenarios."""
    # 1. Initially, no scenario is active
    res_get_init = await app_client.get("/simulate")
    assert res_get_init.status_code == 200
    assert res_get_init.json()["active_scenario"] is None

    # 2. Set active scenario to DDoS
    res_set_ddos = await app_client.post("/simulate", json={"scenario": "ddos"})
    assert res_set_ddos.status_code == 200
    assert res_set_ddos.json()["active_scenario"] == "ddos"

    # Get should now return ddos
    res_get_ddos = await app_client.get("/simulate")
    assert res_get_ddos.json()["active_scenario"] == "ddos"

    # 3. Set active scenario to an invalid value -> 400
    res_set_invalid = await app_client.post("/simulate", json={"scenario": "invalid_hack"})
    assert res_set_invalid.status_code == 400

    # 4. Clear/Stop simulation scenario (set to None or empty string)
    res_clear = await app_client.post("/simulate", json={"scenario": None})
    assert res_clear.status_code == 200
    assert res_clear.json()["active_scenario"] is None

    # Get should return None
    res_get_clear = await app_client.get("/simulate")
    assert res_get_clear.json()["active_scenario"] is None
