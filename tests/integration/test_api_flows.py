from datetime import datetime
import pytest
from httpx import AsyncClient

@pytest.mark.anyio
async def test_stream_flow_inference(app_client: AsyncClient):
    flow_payload = {
        "ts": datetime.now().isoformat(),
        "src_ip": "192.168.1.50",
        "src_port": 1234,
        "dst_ip": "10.0.0.5",
        "dst_port": 80,
        "protocol": 6,
        "features": {
            "duration": 100.0,
            "protocol_type": 1.0,
            "service": 21.0,
            "flag": 9.0,
            "src_bytes": 100.0,
            "dst_bytes": 200.0,
        },
        "label": "BENIGN",
    }
    
    response = await app_client.post("/api/v1/flows/stream", json=flow_payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "score" in data
    assert "is_anomaly" in data
    assert "model_name" in data
    assert "threshold" in data

@pytest.mark.anyio
async def test_batch_flow_inference(app_client: AsyncClient):
    batch_payload = {
        "flows": [
            {
                "ts": datetime.now().isoformat(),
                "src_ip": "192.168.1.100",
                "src_port": 80,
                "dst_ip": "10.0.0.1",
                "dst_port": 443,
                "protocol": 6,
                "features": {
                    "duration": 100.0,
                    "protocol_type": 1.0,
                    "service": 21.0,
                    "flag": 9.0,
                    "src_bytes": 100.0,
                    "dst_bytes": 200.0,
                },
                "label": "BENIGN",
            }
        ]
    }
    
    response = await app_client.post("/api/v1/flows/batch", json=batch_payload)
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert "score" in data[0]

@pytest.mark.anyio
async def test_list_flows(app_client: AsyncClient):
    response = await app_client.get("/api/v1/flows")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
