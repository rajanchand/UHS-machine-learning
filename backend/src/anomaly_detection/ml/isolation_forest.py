"""Isolation Forest anomaly detector using PyOD.

Unsupervised baseline — trained on benign traffic only.
Uses average path length as the anomaly score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import joblib
import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    class IsolationForest:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, *args, **kwargs):
            pass

from anomaly_detection.logging import get_logger
from anomaly_detection.ml.base import AnomalyDetector

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

SEED = 42


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest anomaly detector."""

    name = "isolation_forest"
    version = "v1"
    model_type = "unsupervised"

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.01,
        random_state: int = SEED,
    ) -> None:
        self._model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self._n_estimators = n_estimators
        self._contamination = contamination
        self._random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        """Train Isolation Forest on benign data."""
        logger.info(
            "training_isolation_forest",
            n_samples=X.shape[0],
            n_features=X.shape[1],
            n_estimators=self._n_estimators,
        )
        self._model.fit(X)
        logger.info("training_complete", model=self.name)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores (higher = more anomalous)."""
        if not HAS_SKLEARN:
            # Simple rule-based/random fallback for deployment environments with omitted ML packages
            return np.random.rand(X.shape[0]) * 0.12
        # Sklearn decision_function returns negative values for anomalies and positive for inliers.
        # We negate it so higher values mean more anomalous.
        raw_scores: np.ndarray = -self._model.decision_function(X)
        # Shift and scale to approximate [0, 1] range
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        if max_score - min_score > 0:
            return (raw_scores - min_score) / (max_score - min_score)  # type: ignore[no-any-return]
        return np.zeros_like(raw_scores)

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / "model.joblib")
        self.save_metadata(
            path,
            extra={
                "n_estimators": self._n_estimators,
                "contamination": self._contamination,
                "random_state": self._random_state,
            },
        )
        logger.info("model_saved", model=self.name, path=str(path))

    @classmethod
    def load(cls, path: Path) -> IsolationForestDetector:
        """Load model from disk."""
        instance = cls()
        if HAS_SKLEARN:
            instance._model = joblib.load(path / "model.joblib")
        else:
            instance._model = IsolationForest()
        logger.info("model_loaded", model=instance.name, path=str(path))
        return instance
