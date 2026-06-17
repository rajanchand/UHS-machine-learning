from datetime import datetime
import pytest
from pydantic import ValidationError

from anomaly_detection.schemas.flows import FlowCreate, FlowFeatures, BatchFlowRequest
from anomaly_detection.schemas.common import ThresholdUpdate, AlertStatusUpdate

def test_flow_features_validation():
    # Valid features
    valid_data = {
        "flow_duration": 1000.0,
        "total_fwd_packets": 10,
        "total_bwd_packets": 5,
        "total_len_fwd_packets": 500.0,
        "total_len_bwd_packets": 250.0,
    }
    features = FlowFeatures(**valid_data)
    assert features.flow_duration == 1000.0
    assert features.total_fwd_packets == 10
    assert features.total_bwd_packets == 5
    assert features.total_len_fwd_packets == 500.0
    assert features.total_len_bwd_packets == 250.0
    
    # Negative values should fail
    invalid_data = valid_data.copy()
    invalid_data["total_fwd_packets"] = -1
    with pytest.raises(ValidationError):
        FlowFeatures(**invalid_data)

def test_flow_create_validation():
    features_data = {
        "flow_duration": 1000.0,
        "total_fwd_packets": 10,
        "total_bwd_packets": 5,
        "total_len_fwd_packets": 500.0,
        "total_len_bwd_packets": 250.0,
    }
    
    valid_flow = {
        "ts": datetime.now(),
        "src_ip": "192.168.1.10",
        "src_port": 443,
        "dst_ip": "10.0.0.1",
        "dst_port": 80,
        "protocol": 6,
        "features": features_data,
        "label": "BENIGN",
    }
    
    flow = FlowCreate(**valid_flow)
    assert flow.src_ip == "192.168.1.10"
    assert flow.src_port == 443
    assert flow.features.flow_duration == 1000.0
    
    # Invalid IP or empty should pass but check bounds
    invalid_flow = valid_flow.copy()
    invalid_flow["src_port"] = 999999 # out of port range
    with pytest.raises(ValidationError):
        FlowCreate(**invalid_flow)

def test_batch_flow_request():
    features_data = {
        "flow_duration": 1000.0,
        "total_fwd_packets": 10,
        "total_bwd_packets": 5,
        "total_len_fwd_packets": 500.0,
        "total_len_bwd_packets": 250.0,
    }
    flow_data = {
        "ts": datetime.now(),
        "src_ip": "192.168.1.1",
        "src_port": 80,
        "dst_ip": "192.168.1.2",
        "dst_port": 80,
        "protocol": 6,
        "features": features_data,
    }
    
    # Empty batch should fail
    with pytest.raises(ValidationError):
        BatchFlowRequest(flows=[])
        
    # Valid batch
    batch = BatchFlowRequest(flows=[flow_data])
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
