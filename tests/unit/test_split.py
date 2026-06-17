from datetime import datetime, timedelta
import pandas as pd
import pytest

from anomaly_detection.pipeline.split import time_aware_split, verify_no_session_overlap, filter_benign_only

def test_time_aware_split():
    # Construct a dataframe sorted chronologically
    base_time = datetime(2017, 7, 3, 10, 0, 0)
    data = {
        "timestamp": [base_time + timedelta(seconds=i) for i in range(100)],
        "val": range(100),
    }
    df = pd.DataFrame(data)
    
    train, test = time_aware_split(df, train_ratio=0.8)
    
    assert len(train) == 80
    assert len(test) == 20
    assert train["timestamp"].max() < test["timestamp"].min()

def test_verify_no_session_overlap():
    base_time = datetime(2017, 7, 3, 10, 0, 0)
    
    # Non-overlapping timestamps (valid)
    train_df = pd.DataFrame({
        "timestamp": [base_time, base_time + timedelta(seconds=1)],
        "Flow ID": ["A", "B"],
    })
    test_df = pd.DataFrame({
        "timestamp": [base_time + timedelta(seconds=2), base_time + timedelta(seconds=3)],
        "Flow ID": ["A", "C"], # Some flow IDs might repeat, but timestamps don't overlap
    })
    
    # Should pass
    verify_no_session_overlap(train_df, test_df)
    
    # Overlapping timestamps (invalid)
    train_df2 = pd.DataFrame({
        "timestamp": [base_time, base_time + timedelta(seconds=5)],
        "Flow ID": ["A", "B"],
    })
    test_df2 = pd.DataFrame({
        "timestamp": [base_time + timedelta(seconds=2), base_time + timedelta(seconds=8)],
        "Flow ID": ["C", "D"],
    })
    
    with pytest.raises(AssertionError):
        verify_no_session_overlap(train_df2, test_df2)

def test_filter_benign_only():
    df = pd.DataFrame({
        "label": ["BENIGN", "DDoS", "BENIGN", "PortScan"],
        "val": [1, 2, 3, 4],
    })
    benign_df = filter_benign_only(df)
    assert len(benign_df) == 2
    assert list(benign_df["val"]) == [1, 3]
