import uuid
from datetime import datetime, timezone
import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession

from anomaly_detection.db.models import Flow, Alert, AlertSeverity, AlertStatus

@pytest.mark.anyio
async def test_alerts_lifecycle(app_client: AsyncClient, session_factory: async_sessionmaker[AsyncSession]):
    # Setup: Insert a dummy flow and alert directly in the database
    flow_id = uuid.uuid4()
    alert_id = uuid.uuid4()
    
    async with session_factory() as session:
        flow = Flow(
            id=flow_id,
            ts=datetime.now(timezone.utc),
            src_ip="192.168.1.25",
            src_port=80,
            dst_ip="10.0.0.2",
            dst_port=443,
            protocol=6,
            label="BENIGN",
        )
        alert = Alert(
            id=alert_id,
            flow_id=flow_id,
            severity=AlertSeverity.MEDIUM,
            status=AlertStatus.OPEN,
            suspected_attack_type="DDoS",
        )
        session.add(flow)
        session.add(alert)
        await session.commit()

    # 1. Get alert list
    response = await app_client.get("/api/v1/alerts")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    
    # Check filters
    response_filt = await app_client.get("/api/v1/alerts?status=open")
    assert response_filt.status_code == 200
    
    # 2. Get alert details
    response_det = await app_client.get(f"/api/v1/alerts/{alert_id}")
    assert response_det.status_code == 200
    detail = response_det.json()
    assert detail["id"] == str(alert_id)
    assert detail["flow_id"] == str(flow_id)
    assert detail["severity"] == "medium"
    assert detail["status"] == "open"
    
    # 3. Update alert status
    response_patch = await app_client.patch(
        f"/api/v1/alerts/{alert_id}/status",
        json={"status": "acknowledged"}
    )
    assert response_patch.status_code == 200
    patched_data = response_patch.json()
    assert patched_data["status"] == "acknowledged"
    
    # Verify in DB
    async with session_factory() as session:
        db_alert = await session.get(Alert, alert_id)
        assert db_alert is not None
        assert db_alert.status == AlertStatus.ACKNOWLEDGED
