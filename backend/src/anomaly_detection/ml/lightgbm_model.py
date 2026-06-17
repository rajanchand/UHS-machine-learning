"""LightGBM supervised benchmark model.

This is a SUPERVISED model trained on the full labelled dataset.
It serves as an UPPER-BOUND BENCHMARK ONLY — it is NOT representative
of real-world anomaly detection where labels are unavailable.

Clearly labelled as such in all outputs.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import joblib
import lightgbm as lgb
import numpy as np

from anomaly_detection.logging import get_logger
from anomaly_detection.ml.base import AnomalyDetector

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

SEED = 42


class LightGBMBenchmark(AnomalyDetector):
    """LightGBM supervised benchmark — upper-bound, not production model."""

    name = "lightgbm_benchmark"
    version = "v1"
    model_type = "supervised"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        random_state: int = SEED,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._num_leaves = num_leaves
        self._random_state = random_state
        self._model: lgb.LGBMClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        """Train LightGBM classifier on labelled data.

        Args:
            X: Feature matrix.
            y: Binary labels (0=benign, 1=attack). REQUIRED for this model.
        """
        if y is None:
            msg = "LightGBM benchmark requires labels (y) — this is a supervised model"
            raise ValueError(msg)

        logger.info(
            "training_lightgbm_benchmark",
            n_samples=X.shape[0],
            n_features=X.shape[1],
            positive_rate=float(y.sum() / len(y)),
            note="SUPERVISED UPPER-BOUND BENCHMARK",
        )

        self._model = lgb.LGBMClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            num_leaves=self._num_leaves,
            random_state=self._random_state,
            is_unbalance=True,  # Handle class imbalance
            verbose=-1,
        )
        self._model.fit(X, y)
        logger.info("training_complete", model=self.name)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly probability scores.

        Returns predicted probability of the anomaly (attack) class.
        """
        if self._model is None:
            msg = "Model not fitted — call fit() first"
            raise RuntimeError(msg)

        # predict_proba returns [P(benign), P(attack)]
        proba: np.ndarray = self._model.predict_proba(X)
        return proba[:, 1]  # P(attack)

    def save(self, path: Path) -> None:
        """Save model to disk."""
        if self._model is None:
            msg = "Cannot save unfitted model"
            raise RuntimeError(msg)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / "model.joblib")
        self.save_metadata(path, extra={
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "learning_rate": self._learning_rate,
            "num_leaves": self._num_leaves,
            "note": "SUPERVISED UPPER-BOUND BENCHMARK — NOT for production anomaly detection",
        })
        logger.info("model_saved", model=self.name, path=str(path))

    @classmethod
    def load(cls, path: Path) -> LightGBMBenchmark:
        """Load model from disk."""
        instance = cls()
        instance._model = joblib.load(path / "model.joblib")

        meta_path = path / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            instance._n_estimators = meta.get("n_estimators", 200)
            instance._max_depth = meta.get("max_depth", 8)

        logger.info("model_loaded", model=instance.name, path=str(path))
        return instance
