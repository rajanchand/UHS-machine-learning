"""Evaluation harness — the highest-value deliverable.

Produces:
- metrics.json with all metrics for all models
- PR/ROC curves overlaid per model
- Confusion matrices
- Markdown comparison table
- Per-attack-type recall breakdown

Usage:
    python -m anomaly_detection.ml.evaluate --data-dir data/processed --model-dir models --output-dir evaluation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from anomaly_detection.logging import get_logger, setup_logging
from anomaly_detection.constants import FEATURE_COLUMNS
from anomaly_detection.ml.autoencoder import AutoEncoderDetector
from anomaly_detection.ml.base import AnomalyDetector
from anomaly_detection.ml.halfspace_trees import HalfSpaceTreesDetector
from anomaly_detection.ml.isolation_forest import IsolationForestDetector
from anomaly_detection.ml.lightgbm_model import LightGBMBenchmark
from anomaly_detection.ml.random_forest import RandomForestDetector
from anomaly_detection.ml.xgboost_model import XGBoostDetector

matplotlib.use("Agg")  # Non-interactive backend

logger = get_logger(__name__)

SEED = 42


def find_threshold_at_target_fpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_fpr: float = 0.01,
) -> float:
    """Find the threshold that achieves a target FPR.

    Args:
        y_true: Binary labels.
        scores: Anomaly scores.
        target_fpr: Target false positive rate.

    Returns:
        Threshold value.
    """
    fpr_values, _, thresholds = roc_curve(y_true, scores)
    # Find the threshold closest to target FPR without exceeding it
    valid_mask = fpr_values <= target_fpr
    if not valid_mask.any():
        return float(thresholds[0])
    idx = np.where(valid_mask)[0][-1]
    return float(thresholds[min(idx, len(thresholds) - 1)])


def find_threshold_at_target_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_recall: float = 0.90,
) -> float:
    """Find the threshold that achieves a target recall.

    Args:
        y_true: Binary labels.
        scores: Anomaly scores.
        target_recall: Target recall.

    Returns:
        Threshold value.
    """
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, scores)
    valid_mask = recall_vals[:-1] >= target_recall
    if not valid_mask.any():
        return float(thresholds[0])
    # Among thresholds meeting target recall, pick the one with highest precision
    valid_indices = np.where(valid_mask)[0]
    best_idx = valid_indices[np.argmax(precision_vals[valid_indices])]
    return float(thresholds[best_idx])


def compute_per_attack_recall(
    labels: pd.Series,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Compute recall for each attack type.

    Args:
        labels: Original string labels.
        predictions: Binary predictions (0=benign, 1=anomaly).

    Returns:
        Dict mapping attack type to recall.
    """
    results: dict[str, float] = {}
    for attack_type in labels.unique():
        if attack_type.upper() == "BENIGN":
            continue
        mask = labels == attack_type
        if mask.sum() == 0:
            continue
        attack_preds = predictions[mask]
        attack_recall = float(attack_preds.sum() / len(attack_preds))
        results[attack_type] = round(attack_recall, 4)
    return results


def evaluate_model(
    name: str,
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: pd.Series,
    target_fpr: float = 0.01,
    target_recall: float = 0.90,
) -> dict[str, object]:
    """Evaluate a single model comprehensively.

    Args:
        name: Model name.
        model: Fitted model with score() method.
        X_test: Scaled test features.
        y_test: Binary test labels.
        labels: Original string labels for per-attack analysis.
        target_fpr: Target FPR for threshold selection.
        target_recall: Target recall for threshold selection.

    Returns:
        Dictionary of all metrics.
    """
    from anomaly_detection.ml.base import AnomalyDetector

    assert isinstance(model, AnomalyDetector)

    logger.info("evaluating_model", model=name)

    # Score
    scores = model.score(X_test)

    # Threshold selection
    threshold_fpr = find_threshold_at_target_fpr(y_test, scores, target_fpr)
    threshold_recall = find_threshold_at_target_recall(y_test, scores, target_recall)

    # Use the FPR-based threshold as primary
    primary_threshold = threshold_fpr
    predictions = (scores >= primary_threshold).astype(int)

    # Core metrics
    roc_auc = float(roc_auc_score(y_test, scores))
    pr_auc = float(average_precision_score(y_test, scores))
    precision = float(precision_score(y_test, predictions, zero_division=0))
    recall = float(recall_score(y_test, predictions, zero_division=0))
    f1 = float(f1_score(y_test, predictions, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    fpr_actual = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    accuracy = float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0

    # Per-attack-type recall
    per_attack = compute_per_attack_recall(labels, predictions)

    # FPR at target recall
    predictions_at_recall = (scores >= threshold_recall).astype(int)
    cm_recall = confusion_matrix(y_test, predictions_at_recall)
    tn_r, fp_r, _, _ = cm_recall.ravel()
    fpr_at_recall = float(fp_r / (fp_r + tn_r)) if (fp_r + tn_r) > 0 else 0.0

    # Detection latency for streaming model
    detection_latency_ms: float | None = None
    if hasattr(model, "score_one_with_latency"):
        latencies: list[float] = []
        sample_count = min(100, X_test.shape[0])
        for i in range(sample_count):
            sample = {
                FEATURE_COLUMNS[j]: float(X_test[i, j])
                for j in range(X_test.shape[1])
            }
            _, latency = model.score_one_with_latency(sample)
            latencies.append(latency)
        detection_latency_ms = float(np.mean(latencies) * 1000)

    metrics: dict[str, object] = {
        "model_name": name,
        "model_type": model.model_type,
        "accuracy": round(accuracy, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr_actual, 4),
        "threshold_at_1pct_fpr": round(threshold_fpr, 4),
        "threshold_at_90pct_recall": round(threshold_recall, 4),
        "fpr_at_90pct_recall": round(fpr_at_recall, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "per_attack_recall": per_attack,
        "test_samples": len(y_test),
        "test_positive_rate": round(float(y_test.sum() / len(y_test)), 4),
    }

    if detection_latency_ms is not None:
        metrics["detection_latency_ms"] = round(detection_latency_ms, 3)

    logger.info(
        "evaluation_complete",
        model=name,
        pr_auc=metrics["pr_auc"],
        roc_auc=metrics["roc_auc"],
        fpr=metrics["fpr"],
    )

    return metrics


def plot_roc_curves(
    all_scores: dict[str, np.ndarray],
    y_test: np.ndarray,
    output_path: Path,
) -> None:
    """Plot overlaid ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["#3b82f6", "#ef4444", "#f59e0b", "#10b981", "#8b5cf6", "#f97316"]
    for (name, scores), color in zip(all_scores.items(), colors, strict=False):
        fpr_values, tpr_values, _ = roc_curve(y_test, scores)
        roc_auc_val = auc(fpr_values, tpr_values)
        ax.plot(fpr_values, tpr_values, color=color, lw=2,
                label=f"{name} (AUC = {roc_auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("roc_curves_saved", path=str(output_path))


def plot_pr_curves(
    all_scores: dict[str, np.ndarray],
    y_test: np.ndarray,
    output_path: Path,
) -> None:
    """Plot overlaid Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["#3b82f6", "#ef4444", "#f59e0b", "#10b981", "#8b5cf6", "#f97316"]
    for (name, scores), color in zip(all_scores.items(), colors, strict=False):
        prec, rec, _ = precision_recall_curve(y_test, scores)
        pr_auc_val = average_precision_score(y_test, scores)
        ax.plot(rec, prec, color=color, lw=2,
                label=f"{name} (PR-AUC = {pr_auc_val:.3f})")

    baseline = y_test.sum() / len(y_test)
    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Models", fontsize=14)
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("pr_curves_saved", path=str(output_path))


def plot_confusion_matrices(
    all_metrics: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    """Plot grid of confusion matrices (2 columns, auto rows)."""
    n = len(all_metrics)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    axes_flat = axes.flatten() if n > 1 else [axes]

    for idx, (name, metrics) in enumerate(all_metrics.items()):
        ax = axes_flat[idx]
        cm = metrics["confusion_matrix"]
        assert isinstance(cm, dict)
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])

        im = ax.imshow(matrix, cmap="Blues", aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,}",
                        ha="center", va="center", fontsize=14,
                        color="white" if matrix[i, j] > matrix.max() / 2 else "black")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Predicted\nNormal", "Predicted\nAnomaly"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Actual\nNormal", "Actual\nAnomaly"])

        model_type_label = " (supervised)" if "benchmark" in name else ""
        ax.set_title(f"{name}{model_type_label}", fontsize=13)
        fig.colorbar(im, ax=ax, shrink=0.6)

    # Hide unused subplots when count is odd
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Confusion Matrices — All Models", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("confusion_matrices_saved", path=str(output_path))


def generate_comparison_table(
    all_metrics: dict[str, dict[str, object]],
    output_path: Path,
) -> None:
    """Generate a markdown comparison table."""
    lines: list[str] = [
        "# Model Comparison Table",
        "",
        "| Model | Type | PR-AUC | ROC-AUC | Precision | Recall | F1 | FPR | FPR@90%R |",
        "|-------|------|--------|---------|-----------|--------|-----|-----|---------|",
    ]

    for name, m in all_metrics.items():
        model_type = str(m.get("model_type", ""))
        suffix = " ⚠️" if model_type == "supervised" else ""
        lines.append(
            f"| {name}{suffix} | {model_type} | "
            f"{m['pr_auc']} | {m['roc_auc']} | {m['precision']} | "
            f"{m['recall']} | {m['f1']} | {m['fpr']} | {m['fpr_at_90pct_recall']} |"
        )

    lines.extend([
        "",
        "> ⚠️ = Supervised upper-bound benchmark (uses labels unavailable in production).",
        "> All unsupervised models trained on benign traffic only.",
        "> Threshold selected at target FPR ≤ 1%.",
        "> PR-AUC is the lead metric for this imbalanced dataset.",
    ])

    # Per-attack recall table
    lines.extend(["", "## Per-Attack-Type Recall", ""])

    attack_types: set[str] = set()
    for m in all_metrics.values():
        par = m.get("per_attack_recall", {})
        assert isinstance(par, dict)
        attack_types.update(par.keys())

    if attack_types:
        header = "| Attack Type |"
        separator = "|-------------|"
        for name in all_metrics:
            header += f" {name} |"
            separator += "--------|"
        lines.extend([header, separator])

        for attack in sorted(attack_types):
            row = f"| {attack} |"
            for m in all_metrics.values():
                par = m.get("per_attack_recall", {})
                assert isinstance(par, dict)
                val = par.get(attack, "N/A")
                row += f" {val} |"
            lines.append(row)

    table_text = "\n".join(lines) + "\n"
    output_path.write_text(table_text)
    logger.info("comparison_table_saved", path=str(output_path))


def run_evaluation(
    data_dir: Path,
    model_dir: Path,
    output_dir: Path,
) -> None:
    """Run the full evaluation pipeline.

    Args:
        data_dir: Directory with processed data (test.parquet, scaler.joblib).
        model_dir: Model registry directory.
        output_dir: Directory for evaluation outputs.
    """
    np.random.seed(SEED)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test data and scaler
    test_df = pd.read_parquet(data_dir / "test.parquet")
    scaler = joblib.load(data_dir / "scaler.joblib")

    X_test = scaler.transform(test_df[FEATURE_COLUMNS].values)
    y_test = (test_df["label"].str.upper() != "BENIGN").astype(int).values
    labels = test_df["label"]

    logger.info(
        "evaluation_data_loaded",
        test_samples=len(y_test),
        positive_rate=float(y_test.sum() / len(y_test)),
    )

    # Load models — skip any whose artifacts are missing
    model_registry: dict[str, tuple[type[AnomalyDetector], Path]] = {
        "isolation_forest": (IsolationForestDetector, model_dir / "isolation_forest" / "v1"),
        "autoencoder": (AutoEncoderDetector, model_dir / "autoencoder" / "v1"),
        "halfspace_trees": (HalfSpaceTreesDetector, model_dir / "halfspace_trees" / "v1"),
        "lightgbm_benchmark": (LightGBMBenchmark, model_dir / "lightgbm_benchmark" / "v1"),
        "random_forest": (RandomForestDetector, model_dir / "random_forest" / "v1"),
        "xgboost": (XGBoostDetector, model_dir / "xgboost" / "v1"),
    }
    models: dict[str, object] = {}
    for model_name, (cls, model_path) in model_registry.items():
        if (model_path / "model.joblib").exists():
            models[model_name] = cls.load(model_path)
        else:
            logger.warning("model_artifact_missing_skipping", model=model_name, path=str(model_path))

    # Evaluate each model
    all_metrics: dict[str, dict[str, object]] = {}
    all_scores: dict[str, np.ndarray] = {}

    for name, model in models.items():
        from anomaly_detection.ml.base import AnomalyDetector

        assert isinstance(model, AnomalyDetector)
        metrics = evaluate_model(name, model, X_test, y_test, labels)
        all_metrics[name] = metrics
        all_scores[name] = model.score(X_test)

    # Save metrics.json
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2, default=str))
    logger.info("metrics_saved", path=str(metrics_path))

    # Generate plots
    plot_roc_curves(all_scores, y_test, output_dir / "roc_curves.png")
    plot_pr_curves(all_scores, y_test, output_dir / "pr_curves.png")
    plot_confusion_matrices(all_metrics, output_dir / "confusion_matrices.png")

    # Generate comparison table
    generate_comparison_table(all_metrics, output_dir / "comparison_table.md")

    logger.info("evaluation_complete", output_dir=str(output_dir))


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection models")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation"))
    args = parser.parse_args()

    setup_logging("INFO")
    run_evaluation(args.data_dir, args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
