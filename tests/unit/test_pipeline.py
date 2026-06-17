import pandas as pd
import numpy as np
import pytest

from anomaly_detection.pipeline.ingest import clean_dataframe, infer_day_from_filename
from anomaly_detection.pipeline.features import extract_features

def test_infer_day_from_filename():
    assert infer_day_from_filename("Monday-WorkingHours.pcap_ISCX.csv") == "2017-07-03"
    assert infer_day_from_filename("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv") == "2017-07-07"
    assert infer_day_from_filename("Unknown.csv") == "2017-07-03"

def test_clean_dataframe():
    # Construct a raw dataframe containing typical CICIDS anomalies
    raw_data = {
        " Flow Duration": [100, 200, 300, 400],
        "Flow Bytes/s": ["12.5", "Infinity", "NaN", "33.3"],
        "Flow Packets/s": ["1.0", "2.0", "3.0", "4.0"],
        "Source IP": ["192.168.1.1", "192.168.1.2", "192.168.1.3", "192.168.1.1"],
        "Source Port": [80, 80, 80, 80],
        "Destination IP": ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.1"],
        "Destination Port": [443, 443, 443, 443],
        "Protocol": [6, 6, 6, 6],
        "Label": ["BENIGN", "DDoS", "BENIGN", "BENIGN"],
        "timestamp": pd.date_range("2017-07-03 10:00:00", periods=4, freq="s"),
    }
    df = pd.DataFrame(raw_data)
    df.columns = df.columns.str.strip()
    
    # Run cleaning
    cleaned = clean_dataframe(df)
    
    # Row 1: valid numeric, valid float -> kept
    # Row 2: contains 'Infinity' -> replaced with NaN -> dropped
    # Row 3: contains 'NaN' -> dropped
    # Row 4: duplicate of Row 1 (all features + network IDs same, except timestamp, wait, index duplicate or row duplicate?)
    # Wait, df.duplicated() checks all columns. Row 1 and Row 4 have different timestamps so they are NOT duplicates!
    # Let's check: only row 1 and row 4 should remain after dropping NaN/Inf
    assert len(cleaned) == 2
    assert "label" in cleaned.columns
    assert "Flow Duration" in cleaned.columns # Column names stripped
    
    # Ensure types are correct
    assert cleaned["Flow Bytes/s"].dtype == np.float64
    assert cleaned["Flow Packets/s"].dtype == np.float64

def test_extract_features():
    # Test feature selection and documentation
    raw_data = {
        "Flow Duration": [100],
        "Total Fwd Packets": [10],
        "Total Backward Packets": [5],
        "Total Length of Fwd Packets": [500.0],
        "Total Length of Bwd Packets": [250.0],
        "Fwd Packet Length Max": [100.0],
        "Fwd Packet Length Min": [10.0],
        "Fwd Packet Length Mean": [50.0],
        "Fwd Packet Length Std": [15.0],
        "Bwd Packet Length Max": [100.0],
        "Bwd Packet Length Min": [10.0],
        "Bwd Packet Length Mean": [50.0],
        "Bwd Packet Length Std": [15.0],
        "Flow Bytes/s": [10.0],
        "Flow Packets/s": [1.0],
        "FIN Flag Count": [0],
        "SYN Flag Count": [1],
        "RST Flag Count": [0],
        "PSH Flag Count": [1],
        "ACK Flag Count": [0],
        "URG Flag Count": [0],
        "Flow IAT Mean": [10.0],
        "Flow IAT Std": [2.0],
        "Flow IAT Max": [15.0],
        "Flow IAT Min": [5.0],
        "Fwd IAT Mean": [10.0],
        "Fwd IAT Std": [2.0],
        "Fwd IAT Max": [15.0],
        "Fwd IAT Min": [5.0],
        "Bwd IAT Mean": [10.0],
        "Bwd IAT Std": [2.0],
        "Bwd IAT Max": [15.0],
        "Bwd IAT Min": [5.0],
        "Down/Up Ratio": [0.5],
        "Average Packet Size": [50.0],
        "Avg Fwd Segment Size": [50.0],
        "Avg Bwd Segment Size": [50.0],
        "Source IP": ["192.168.1.1"],
        "Source Port": [80],
        "Destination IP": ["10.0.0.1"],
        "Destination Port": [443],
        "Protocol": [6],
        "label": ["BENIGN"],
        "timestamp": [pd.Timestamp("2017-07-03 10:00:00")],
    }
    df = pd.DataFrame(raw_data)
    features = extract_features(df)
    
    # Feature columns should match target format
    assert "flow_duration" in features.columns
    assert "total_fwd_packets" in features.columns
    assert "avg_fwd_segment_size" in features.columns
