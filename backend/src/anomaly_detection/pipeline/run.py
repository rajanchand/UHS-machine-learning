"""Run the complete data pipeline: ingest → clean → features → split → data card.

Usage:
    python -m anomaly_detection.pipeline.run --input data/fixtures/cicids2017_sample.csv
    python -m anomaly_detection.pipeline.run --input data/raw/  # full dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anomaly_detection.pipeline.data_card import generate_data_card
from anomaly_detection.pipeline.features import extract_features
from anomaly_detection.pipeline.ingest import clean_dataframe, load_and_merge
from anomaly_detection.pipeline.split import run_split_pipeline


def run_pipeline(input_path: Path, output_dir: Path) -> None:
    """Execute the full data pipeline.

    Args:
        input_path: Path to raw CSV file(s) — single file or directory.
        output_dir: Directory for processed artifacts.
    """
    # 1. Ingest
    raw_df = load_and_merge(input_path)
    total_raw = len(raw_df)

    # 2. Clean
    clean_df = clean_dataframe(raw_df)
    total_clean = len(clean_df)

    # 3. Feature engineering
    feature_df = extract_features(clean_df)

    # 4. Split + scale + persist
    metadata = run_split_pipeline(feature_df, output_dir)

    # 5. Data card
    generate_data_card(
        metadata=metadata,
        total_raw_rows=total_raw,
        total_clean_rows=total_clean,
        output_path=output_dir / "data_card.md",
    )


def main() -> None:
    """CLI entrypoint for the data pipeline."""
    parser = argparse.ArgumentParser(description="Run the data pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/fixtures/cicids2017_sample.csv"),
        help="Path to raw CSV(s) — file or directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for processed artifacts",
    )
    args = parser.parse_args()
    run_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
