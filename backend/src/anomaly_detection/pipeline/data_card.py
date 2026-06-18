"""Generate a data card documenting the dataset, features, and split."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from anomaly_detection.logging import get_logger
from anomaly_detection.constants import FEATURE_COLUMNS

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


def generate_data_card(
    metadata: dict[str, Any],
    total_raw_rows: int,
    total_clean_rows: int,
    output_path: Path,
) -> None:
    """Generate a markdown data card describing the dataset.

    Args:
        metadata: Split pipeline metadata dict.
        total_raw_rows: Number of rows before cleaning.
        total_clean_rows: Number of rows after cleaning.
        output_path: Path to save the data card markdown.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Data Card — CICIDS2017 Network Flow Dataset",
        "",
        "## Overview",
        "",
        f"- **Total raw rows:** {total_raw_rows:,}",
        f"- **Total clean rows:** {total_clean_rows:,}",
        f"- **Rows dropped:** {total_raw_rows - total_clean_rows:,} "
        f"({(total_raw_rows - total_clean_rows) / total_raw_rows * 100:.1f}%)"
        if total_raw_rows > 0 else "",
        f"- **Feature count:** {len(FEATURE_COLUMNS)}",
        "",
        "## Feature List",
        "",
        "| # | Feature | Category |",
        "|---|---------|----------|",
    ]

    # Categorise features
    categories = {
        "Duration": ["flow_duration"],
        "Volume": ["total_fwd_packets", "total_bwd_packets",
                    "total_len_fwd_packets", "total_len_bwd_packets"],
        "Packet Size": [c for c in FEATURE_COLUMNS if "packet_len" in c],
        "Flow Rate": ["flow_bytes_per_s", "flow_packets_per_s"],
        "TCP Flags": [c for c in FEATURE_COLUMNS if "flag" in c],
        "Inter-Arrival Time": [c for c in FEATURE_COLUMNS if "iat" in c],
        "Miscellaneous": ["down_up_ratio", "avg_packet_size",
                          "avg_fwd_segment_size", "avg_bwd_segment_size"],
    }

    feature_to_cat = {}
    for cat, feats in categories.items():
        for f in feats:
            feature_to_cat[f] = cat

    for i, col in enumerate(FEATURE_COLUMNS, 1):
        cat = feature_to_cat.get(col, "Other")
        lines.append(f"| {i} | `{col}` | {cat} |")

    lines.extend([
        "",
        "## Train/Test Split",
        "",
        "**Method:** Chronological (time-aware) — no random shuffle.",
        "All flows before the split timestamp go to train; all after go to test.",
        "This prevents session-level data leakage.",
        "",
        f"- **Train rows (total):** {metadata.get('train_total', 'N/A'):,}",
        f"- **Train rows (benign only, for unsupervised):** {metadata.get('train_benign', 'N/A'):,}",
        f"- **Test rows:** {metadata.get('test_total', 'N/A'):,}",
    ])

    train_range = metadata.get("train_date_range", ("N/A", "N/A"))
    test_range = metadata.get("test_date_range", ("N/A", "N/A"))
    lines.extend([
        f"- **Train date range:** {train_range[0]} → {train_range[1]}",
        f"- **Test date range:** {test_range[0]} → {test_range[1]}",
    ])

    # Class balance tables
    for split_name in ["train_class_balance", "test_class_balance"]:
        balance = metadata.get(split_name)
        if balance and isinstance(balance, dict):
            total = sum(balance.values())
            pretty_name = split_name.replace("_class_balance", "").title()
            lines.extend([
                "",
                f"## {pretty_name} Class Distribution",
                "",
                "| Attack Type | Count | Percentage |",
                "|-------------|------:|----------:|",
            ])
            for attack_type, count in sorted(balance.items(), key=lambda x: -x[1]):
                pct = count / total * 100 if total > 0 else 0
                lines.append(f"| {attack_type} | {count:,} | {pct:.2f}% |")

    lines.extend([
        "",
        "## Scaler",
        "",
        "- **Type:** StandardScaler (sklearn)",
        "- **Fitted on:** Training benign data only",
        f"- **Path:** `{metadata.get('scaler_path', 'N/A')}`",
        "",
        "## Notes",
        "",
        "- Unsupervised models are trained on benign traffic only.",
        "- The supervised LightGBM benchmark uses all labelled training data.",
        "- Test set contains both benign and attack flows for evaluation.",
    ])

    card_text = "\n".join(lines) + "\n"
    output_path.write_text(card_text)
    logger.info("data_card_generated", path=str(output_path))
