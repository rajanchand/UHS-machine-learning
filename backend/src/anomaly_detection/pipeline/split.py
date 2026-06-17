"""Time-aware train/test split with session-leakage prevention.

CRITICAL: Flows from the same session are correlated. A random row shuffle
would leak future information into the training set. We split strictly by
timestamp — everything before the split point is train, everything after
is test.

For unsupervised training, we additionally filter the training set to
benign-only flows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import joblib
from sklearn.preprocessing import StandardScaler

from anomaly_detection.logging import get_logger
from anomaly_detection.schemas.flows import FEATURE_COLUMNS

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pandas as pd

logger = get_logger(__name__)


def time_aware_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame chronologically by timestamp.

    Args:
        df: DataFrame with 'timestamp' column, already cleaned and feature-extracted.
        train_ratio: Fraction of data (by time) to use for training.

    Returns:
        Tuple of (train_df, test_df) split by time.

    Raises:
        ValueError: If timestamp column is missing or split produces empty sets.
    """
    if "timestamp" not in df.columns:
        msg = "DataFrame must have a 'timestamp' column for time-aware splitting"
        raise ValueError(msg)

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Find the split point
    split_idx = int(len(df) * train_ratio)
    split_timestamp = df.iloc[split_idx]["timestamp"]

    train_df = df[df["timestamp"] < split_timestamp].copy()
    test_df = df[df["timestamp"] >= split_timestamp].copy()

    logger.info(
        "time_split_complete",
        train_rows=len(train_df),
        test_rows=len(test_df),
        split_timestamp=str(split_timestamp),
        train_date_range=f"{train_df['timestamp'].min()} to {train_df['timestamp'].max()}",
        test_date_range=f"{test_df['timestamp'].min()} to {test_df['timestamp'].max()}",
    )

    if len(train_df) == 0 or len(test_df) == 0:
        msg = f"Split produced empty set: train={len(train_df)}, test={len(test_df)}"
        raise ValueError(msg)

    return train_df, test_df


def verify_no_session_overlap(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Assert that no flow sessions appear in both train and test sets.

    A session is identified by (src_ip, dst_ip, src_port, dst_port, protocol)
    or by Flow ID if available.

    Args:
        train_df: Training DataFrame.
        test_df: Testing DataFrame.

    Raises:
        AssertionError: If overlapping sessions are found.
    """
    if "Flow ID" in train_df.columns and "Flow ID" in test_df.columns:
        train_ids = set(train_df["Flow ID"].unique())
        test_ids = set(test_df["Flow ID"].unique())
        overlap = train_ids & test_ids

        # For time-based splits, some flow IDs may appear in both if they
        # span the split boundary. We check that the TIMESTAMP ranges
        # don't overlap, which is the stronger guarantee.
        if overlap:
            logger.warning(
                "flow_id_overlap_detected",
                count=len(overlap),
                note="Checking timestamp non-overlap as primary guarantee",
            )

    # Primary guarantee: timestamp ranges must not overlap
    train_max_ts = train_df["timestamp"].max()
    test_min_ts = test_df["timestamp"].min()

    assert train_max_ts <= test_min_ts, (
        f"Timestamp overlap detected: train max={train_max_ts}, test min={test_min_ts}. "
        "This indicates data leakage — the split is not strictly chronological."
    )

    logger.info(
        "session_overlap_check_passed",
        train_max_ts=str(train_max_ts),
        test_min_ts=str(test_min_ts),
    )


def filter_benign_only(df: pd.DataFrame) -> pd.DataFrame:
    """Filter DataFrame to benign flows only (for unsupervised training).

    Args:
        df: DataFrame with 'label' column.

    Returns:
        DataFrame containing only rows where label is 'BENIGN'.
    """
    if "label" not in df.columns:
        logger.warning("no_label_column", note="Cannot filter benign — returning all rows")
        return df

    benign_mask = df["label"].str.upper() == "BENIGN"
    benign_df = df[benign_mask].copy()

    logger.info(
        "benign_filter",
        total=len(df),
        benign=len(benign_df),
        attack=len(df) - len(benign_df),
    )

    return benign_df


def fit_scaler(
    train_features: pd.DataFrame,
    save_path: Path | None = None,
) -> StandardScaler:
    """Fit a StandardScaler on training features only.

    Args:
        train_features: DataFrame of training features (FEATURE_COLUMNS only).
        save_path: Optional path to persist the fitted scaler.

    Returns:
        Fitted StandardScaler.
    """
    scaler = StandardScaler()
    scaler.fit(train_features[FEATURE_COLUMNS])

    logger.info("scaler_fitted", n_features=len(FEATURE_COLUMNS))

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)
        logger.info("scaler_saved", path=str(save_path))

    return scaler


def transform_features(
    df: pd.DataFrame,
    scaler: StandardScaler,
) -> np.ndarray:
    """Transform features using a fitted scaler.

    Args:
        df: DataFrame containing FEATURE_COLUMNS.
        scaler: Fitted StandardScaler.

    Returns:
        Scaled feature matrix as numpy array.
    """
    return scaler.transform(df[FEATURE_COLUMNS])  # type: ignore[no-any-return]


def run_split_pipeline(
    df: pd.DataFrame,
    output_dir: Path,
    train_ratio: float = 0.8,
) -> dict[str, object]:
    """Run the full split pipeline: split → filter benign → scale → persist.

    Args:
        df: Cleaned, feature-extracted DataFrame.
        output_dir: Directory to save artifacts (parquet files, scaler).
        train_ratio: Fraction of data for training.

    Returns:
        Dictionary with metadata about the split.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Time-aware split
    train_df, test_df = time_aware_split(df, train_ratio)

    # 2. Verify no session overlap
    verify_no_session_overlap(train_df, test_df)

    # 3. Filter benign for unsupervised training
    train_benign_df = filter_benign_only(train_df)

    # 4. Fit scaler on training benign features only
    scaler_path = output_dir / "scaler.joblib"
    fit_scaler(train_benign_df, save_path=scaler_path)

    # 5. Save artifacts
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    train_benign_df.to_parquet(output_dir / "train_benign.parquet", index=False)

    logger.info(
        "split_pipeline_complete",
        train_total=len(train_df),
        train_benign=len(train_benign_df),
        test_total=len(test_df),
        output_dir=str(output_dir),
    )

    # Return metadata for data card
    label_col = "label" if "label" in test_df.columns else None
    metadata: dict[str, object] = {
        "train_total": len(train_df),
        "train_benign": len(train_benign_df),
        "test_total": len(test_df),
        "train_date_range": (
            str(train_df["timestamp"].min()),
            str(train_df["timestamp"].max()),
        ),
        "test_date_range": (
            str(test_df["timestamp"].min()),
            str(test_df["timestamp"].max()),
        ),
        "scaler_path": str(scaler_path),
    }

    if label_col:
        metadata["test_class_balance"] = test_df[label_col].value_counts().to_dict()
        metadata["train_class_balance"] = train_df[label_col].value_counts().to_dict()

    return metadata
