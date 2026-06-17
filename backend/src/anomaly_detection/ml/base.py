"""Abstract base class for anomaly detection models.

All models implement a common interface so they're swappable in the
inference pipeline. This enables the model selector in the dashboard.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


class AnomalyDetector(ABC):
    """Common interface for all anomaly detection models."""

    name: str = "base"
    version: str = "v1"
    model_type: str = "unsupervised"  # or "supervised"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        """Train the model.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Optional labels (only used by supervised models).
        """
        ...

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for each sample.

        Higher scores = more anomalous.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Array of anomaly scores, shape (n_samples,).
        """
        ...

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """Predict binary anomaly labels.

        Args:
            X: Feature matrix (n_samples, n_features).
            threshold: Score threshold — samples with score >= threshold
                      are classified as anomalies (1).

        Returns:
            Array of binary predictions, shape (n_samples,).
        """
        scores = self.score(X)
        return (scores >= threshold).astype(int)

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the model to disk.

        Args:
            path: Directory to save model artifacts to.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> AnomalyDetector:
        """Load a saved model from disk.

        Args:
            path: Directory containing model artifacts.

        Returns:
            Loaded model instance.
        """
        ...

    def save_metadata(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        """Save model metadata alongside the model artifact.

        Args:
            path: Directory to save metadata to.
            extra: Additional metadata to include.
        """
        metadata: dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "trained_at": datetime.now(UTC).isoformat(),
        }
        if extra:
            metadata.update(extra)

        meta_path = path / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))
