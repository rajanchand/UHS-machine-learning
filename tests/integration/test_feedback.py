"""Integration tests for analyst feedback loop and CSV export."""

import pytest
from datetime import datetime
import uuid
from sqlalchemy import select
from anomaly_detection.db.models import Alert, AlertSeverity, AlertStatus, Flow, Feedback


@pytest.mark.anyio
async def test_feedback_loop_and_csv_export(app_client, session_factory):
    """Test submitting analyst feedback for an alert and exporting feedback CSV."""
    flow_uuid = uuid.uuid4()
    alert_uuid = uuid.uuid4()

    # 1. Seed Flow and Alert in database
    async with session_factory() as session:
        flow = Flow(
            id=flow_uuid,
            ts=datetime.now(),
            src_ip="192.168.1.100",
            src_port=443,
            dst_ip="10.0.0.1",
            dst_port=80,
            protocol=6,
            flow_duration=123.45,
            total_fwd_packets=10,
            total_bwd_packets=5,
            label="BENIGN",
        )
        alert = Alert(
            id=alert_uuid,
            flow_id=flow_uuid,
            severity=AlertSeverity.HIGH,
            suspected_attack_type="DDoS",
            status=AlertStatus.OPEN,
        )
        session.add(flow)
        session.add(alert)
        await session.commit()

    # 2. Submit feedback verdict: true_positive
    res_feedback1 = await app_client.post(
        f"/api/v1/alerts/{alert_uuid}/feedback",
        json={"verdict": "true_positive"}
    )
    assert res_feedback1.status_code == 200
    assert res_feedback1.json()["status"] == "feedback_submitted"
    assert res_feedback1.json()["verdict"] == "true_positive"

    # Verify database record
    async with session_factory() as session:
        result = await session.execute(
            select(Feedback).where(Feedback.alert_id == alert_uuid)
        )
        feedback = result.scalar_one_or_none()
        assert feedback is not None
        assert feedback.verdict == "true_positive"
        assert feedback.user == "test_analyst"

    # 3. Update feedback verdict: false_positive
    res_feedback2 = await app_client.post(
        f"/api/v1/alerts/{alert_uuid}/feedback",
        json={"verdict": "false_positive"}
    )
    assert res_feedback2.status_code == 200
    assert res_feedback2.json()["verdict"] == "false_positive"

    # Verify updated database record
    async with session_factory() as session:
        result = await session.execute(
            select(Feedback).where(Feedback.alert_id == alert_uuid)
        )
        feedback = result.scalar_one_or_none()
        assert feedback.verdict == "false_positive"

    # 4. Export CSV and verify structure
    res_export = await app_client.get("/api/v1/alerts/feedback/export")
    assert res_export.status_code == 200
    assert res_export.headers["Content-Type"].startswith("text/csv")
    assert "attachment; filename=analyst_feedback_dataset.csv" in res_export.headers["Content-Disposition"]

    csv_content = res_export.text
    # Verify CSV has headers and correct content rows
    lines = csv_content.splitlines()
    assert len(lines) >= 2
    
    headers = lines[0].split(",")
    assert "feedback_id" in headers
    assert "alert_id" in headers
    assert "verdict" in headers
    assert "corrected_label" in headers
    assert "flow_duration" in headers

    row_values = lines[1].split(",")
    assert str(alert_uuid) in row_values
    assert "false_positive" in row_values
    assert "BENIGN" in row_values  # corrected_label for false positive is BENIGN
