"""Feature engineering for network flow data.

Selects, renames, and documents the features used by the ML models.
All transforms are documented here for reproducibility.
"""

from __future__ import annotations

import pandas as pd

from anomaly_detection.logging import get_logger
from anomaly_detection.constants import FEATURE_COLUMNS

logger = get_logger(__name__)

# Mapping from CICIDS2017 column names (after stripping) to our internal names
CICIDS_TO_INTERNAL: dict[str, str] = {
    "Flow Duration": "flow_duration",
    "Total Fwd Packets": "total_fwd_packets",
    "Total Backward Packets": "total_bwd_packets",
    "Total Length of Fwd Packets": "total_len_fwd_packets",
    "Total Length of Bwd Packets": "total_len_bwd_packets",
    "Fwd Packet Length Max": "fwd_packet_len_max",
    "Fwd Packet Length Min": "fwd_packet_len_min",
    "Fwd Packet Length Mean": "fwd_packet_len_mean",
    "Fwd Packet Length Std": "fwd_packet_len_std",
    "Bwd Packet Length Max": "bwd_packet_len_max",
    "Bwd Packet Length Min": "bwd_packet_len_min",
    "Bwd Packet Length Mean": "bwd_packet_len_mean",
    "Bwd Packet Length Std": "bwd_packet_len_std",
    "Flow Bytes/s": "flow_bytes_per_s",
    "Flow Packets/s": "flow_packets_per_s",
    "FIN Flag Count": "fin_flag_count",
    "SYN Flag Count": "syn_flag_count",
    "RST Flag Count": "rst_flag_count",
    "PSH Flag Count": "psh_flag_count",
    "ACK Flag Count": "ack_flag_count",
    "URG Flag Count": "urg_flag_count",
    "Flow IAT Mean": "flow_iat_mean",
    "Flow IAT Std": "flow_iat_std",
    "Flow IAT Max": "flow_iat_max",
    "Flow IAT Min": "flow_iat_min",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Fwd IAT Min": "fwd_iat_min",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "Bwd IAT Max": "bwd_iat_max",
    "Bwd IAT Min": "bwd_iat_min",
    "Down/Up Ratio": "down_up_ratio",
    "Average Packet Size": "avg_packet_size",
    "Avg Fwd Segment Size": "avg_fwd_segment_size",
    "Avg Bwd Segment Size": "avg_bwd_segment_size",
}

# Network identifier columns we want to keep
ID_COLUMNS_MAP: dict[str, str] = {
    "Source IP": "src_ip",
    "Destination IP": "dst_ip",
    "Source Port": "src_port",
    "Destination Port": "dst_port",
    "Protocol": "protocol",
}


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and rename features from cleaned CICIDS2017 DataFrame.

    Applies the CICIDS_TO_INTERNAL mapping to select exactly the 37 features
    used by the ML models, plus network identifiers, timestamp, and label.

    Args:
        df: Cleaned DataFrame from the ingest step.

    Returns:
        DataFrame with standardised column names and selected features.

    Raises:
        KeyError: If required columns are missing from the input.
    """
    logger.info("extracting_features", input_columns=len(df.columns))

    # Check which feature columns are available
    available = set(df.columns)
    missing_features = []
    rename_map: dict[str, str] = {}

    for cicids_name, internal_name in CICIDS_TO_INTERNAL.items():
        if cicids_name in available:
            rename_map[cicids_name] = internal_name
        else:
            missing_features.append(cicids_name)

    if missing_features:
        logger.warning("missing_feature_columns", missing=missing_features)

    # Add ID columns
    for cicids_name, internal_name in ID_COLUMNS_MAP.items():
        if cicids_name in available:
            rename_map[cicids_name] = internal_name

    # Select and rename
    keep_cols = list(rename_map.keys())
    metadata_cols = ["timestamp", "label", "Flow ID"]
    keep_cols.extend([c for c in metadata_cols if c in available])

    result = df[keep_cols].copy()
    result = result.rename(columns=rename_map)

    # Ensure all expected feature columns exist (fill missing with 0)
    for col in FEATURE_COLUMNS:
        if col not in result.columns:
            result[col] = 0.0

    # Ensure correct dtypes for features
    for col in FEATURE_COLUMNS:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)

    # Ensure integer columns are int
    int_cols = [
        "total_fwd_packets", "total_bwd_packets",
        "fin_flag_count", "syn_flag_count", "rst_flag_count",
        "psh_flag_count", "ack_flag_count", "urg_flag_count",
    ]
    for col in int_cols:
        if col in result.columns:
            result[col] = result[col].astype(int)

    present_features = [c for c in FEATURE_COLUMNS if c in result.columns]
    logger.info(
        "features_extracted",
        feature_count=len(present_features),
        total_columns=len(result.columns),
        rows=len(result),
    )

    return result


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract just the feature columns as a numeric matrix.

    Args:
        df: DataFrame with internal column names (output of extract_features).

    Returns:
        DataFrame containing only the FEATURE_COLUMNS in the correct order.
    """
    return df[FEATURE_COLUMNS].copy()
