"""Ingest raw CICIDS2017 CSVs — clean, fix dtypes, concatenate.

Known issues in the raw data:
- Column names have leading spaces (e.g., ' Flow Duration')
- 'Flow Bytes/s' and 'Flow Packets/s' contain 'Infinity' strings
- Some rows have NaN values in numeric columns
- Duplicate rows exist

This module handles all of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from anomaly_detection.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

# CICIDS2017 files are named by day of capture
CICIDS2017_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
]

# Day mapping for timestamp synthesis when loading from multiple files
DAY_MAP = {
    "Monday": "2017-07-03",
    "Tuesday": "2017-07-04",
    "Wednesday": "2017-07-05",
    "Thursday": "2017-07-06",
    "Friday": "2017-07-07",
}


def load_single_csv(filepath: Path) -> pd.DataFrame:
    """Load a single CICIDS2017 CSV with column name cleaning.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with stripped column names.
    """
    logger.info("loading_csv", path=str(filepath))
    df = pd.read_csv(filepath, encoding="utf-8", low_memory=False)

    # Strip leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()

    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    logger.info("loaded_csv", path=str(filepath), rows=len(df), columns=len(df.columns))
    return df


def infer_day_from_filename(filename: str) -> str:
    """Extract the day name from a CICIDS2017 filename.

    Args:
        filename: The CSV filename.

    Returns:
        ISO date string (e.g., '2017-07-03').
    """
    for day, date in DAY_MAP.items():
        if filename.startswith(day):
            return date
    return "2017-07-03"  # Default to Monday


def load_and_merge(raw_dir: Path) -> pd.DataFrame:
    """Load all CICIDS2017 CSVs from a directory and merge them.

    Also works with a single fixture CSV.

    Args:
        raw_dir: Directory containing CSV files, or path to a single CSV.

    Returns:
        Merged and cleaned DataFrame.
    """
    if raw_dir.is_file():
        # Single file mode (fixture)
        df = load_single_csv(raw_dir)
        return _parse_timestamps(df)

    frames: list[pd.DataFrame] = []
    csv_files = sorted(raw_dir.glob("*.csv"))

    if not csv_files:
        msg = f"No CSV files found in {raw_dir}"
        raise FileNotFoundError(msg)

    for csv_path in csv_files:
        frame = load_single_csv(csv_path)
        # Add date context from filename if no Timestamp column
        if "Timestamp" not in frame.columns:
            date_str = infer_day_from_filename(csv_path.name)
            frame["_date"] = date_str
        frames.append(frame)

    logger.info("merging_dataframes", file_count=len(frames))
    df = pd.concat(frames, ignore_index=True)
    return _parse_timestamps(df)


def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Parse or synthesize timestamp column.

    CICIDS2017 has a 'Timestamp' column with format like '3/7/2017 8:47:17'.

    Args:
        df: DataFrame with potential Timestamp column.

    Returns:
        DataFrame with parsed 'timestamp' column (datetime64).
    """
    if "Timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["Timestamp"],
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )
        # Drop rows where timestamp parsing failed
        bad_ts = df["timestamp"].isna().sum()
        if bad_ts > 0:
            logger.warning("dropped_bad_timestamps", count=int(bad_ts))
            df = df.dropna(subset=["timestamp"])
    elif "_date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["_date"])
        df = df.drop(columns=["_date"])
    else:
        # Fallback: use index-based synthetic timestamp
        df["timestamp"] = pd.date_range(
            start="2017-07-03 08:00:00",
            periods=len(df),
            freq="s",
        )

    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the DataFrame: fix dtypes, handle NaN/Inf, drop duplicates.

    Args:
        df: Raw DataFrame from CSV loading.

    Returns:
        Cleaned DataFrame ready for feature engineering.

    Raises:
        ValueError: If cleaning results in zero rows.
    """
    initial_rows = len(df)
    logger.info("cleaning_start", rows=initial_rows)

    # Standardise the label column name
    if "Label" in df.columns:
        df = df.rename(columns={"Label": "label"})

    # Fix numeric columns that may contain 'Infinity' or 'NaN' strings
    problematic_cols = ["Flow Bytes/s", "Flow Packets/s"]
    for col in problematic_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Convert all feature columns to numeric where possible
    exclude_cols = {"Flow ID", "Source IP", "Destination IP", "label", "timestamp", "Timestamp",
                    "Source Port", "Destination Port", "Protocol"}
    for col in df.columns:
        if col not in exclude_cols and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace infinities with NaN, then drop NaN rows
    df = df.replace([np.inf, -np.inf], np.nan)

    nan_rows = df.isna().any(axis=1).sum()
    if nan_rows > 0:
        logger.info("dropping_nan_rows", count=int(nan_rows))
        df = df.dropna()

    # Drop duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        logger.info("dropping_duplicates", count=int(dup_count))
        df = df.drop_duplicates()

    final_rows = len(df)
    dropped = initial_rows - final_rows
    logger.info(
        "cleaning_complete",
        initial=initial_rows,
        final=final_rows,
        dropped=dropped,
        drop_pct=round(dropped / initial_rows * 100, 2) if initial_rows > 0 else 0,
    )

    if final_rows == 0:
        msg = "Cleaning resulted in zero rows — check the input data"
        raise ValueError(msg)

    return df.reset_index(drop=True)
