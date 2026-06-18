from datetime import datetime
import pytest
from pydantic import ValidationError

from anomaly_detection.schemas.flows import FlowCreate, FlowFeatures, BatchFlowRequest
from anomaly_detection.schemas.common import ThresholdUpdate, AlertStatusUpdate

def test_flow_features_validation():
    # Valid features
    features = FlowFeatures(
        duration=100.0,
        protocol_type=1.0,
        service=21.0,
        flag=9.0,
        src_bytes=100.0,
        dst_bytes=200.0,
    )
    assert features.duration == 100.0
    assert features.protocol_type == 1.0
    assert features.service == 21.0
    assert features.flag == 9.0
    assert features.src_bytes == 100.0
    assert features.dst_bytes == 200.0
    
    # Negative values should fail
    with pytest.raises(ValidationError):
        FlowFeatures(
            duration=-1.0,
            protocol_type=1.0,
            service=21.0,
            flag=9.0,
            src_bytes=100.0,
            dst_bytes=200.0,
        )

def test_flow_create_validation():
    features = FlowFeatures(
        duration=100.0,
        protocol_type=1.0,
        service=21.0,
        flag=9.0,
        src_bytes=100.0,
        dst_bytes=200.0,
    )
    
    flow = FlowCreate(
        ts=datetime.now(),
        src_ip="192.168.1.10",
        src_port=443,
        dst_ip="10.0.0.1",
        dst_port=80,
        protocol=6,
        features=features,
        label="BENIGN",
    )
    assert flow.src_ip == "192.168.1.10"
    assert flow.src_port == 443
    assert flow.features.duration == 100.0
    
    invalid_port = 999999
    with pytest.raises(ValidationError):
        FlowCreate(
            ts=flow.ts,
            src_ip="192.168.1.10",
            src_port=invalid_port, # out of port range
            dst_ip="10.0.0.1",
            dst_port=80,
            protocol=6,
            features=features,
            label="BENIGN",
        )

def test_batch_flow_request():
    features = FlowFeatures(
        duration=100.0,
        protocol_type=1.0,
        service=21.0,
        flag=9.0,
        src_bytes=100.0,
        dst_bytes=200.0,
    )
    flow = FlowCreate(
        ts=datetime.now(),
        src_ip="192.168.1.1",
        src_port=80,
        dst_ip="192.168.1.2",
        dst_port=80,
        protocol=6,
        features=features,
    )
    
    # Empty batch should fail
    with pytest.raises(ValidationError):
        BatchFlowRequest(flows=[])
        
    # Valid batch
    batch = BatchFlowRequest(flows=[flow])
    assert len(batch.flows) == 1

def test_threshold_update():
    # ge=0.0, le=1.0
    assert ThresholdUpdate(threshold=0.5).threshold == 0.5
    assert ThresholdUpdate(threshold=0.0).threshold == 0.0
    assert ThresholdUpdate(threshold=1.0).threshold == 1.0
    
    with pytest.raises(ValidationError):
        ThresholdUpdate(threshold=-0.1)
    with pytest.raises(ValidationError):
        ThresholdUpdate(threshold=1.1)

def test_alert_status_update():
    assert AlertStatusUpdate(status="open").status == "open"
    assert AlertStatusUpdate(status="acknowledged").status == "acknowledged"
    assert AlertStatusUpdate(status="resolved").status == "resolved"
    
    with pytest.raises(ValidationError):
        AlertStatusUpdate(status="invalid_status")
