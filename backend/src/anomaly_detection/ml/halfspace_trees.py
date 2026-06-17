"""HalfSpaceTrees streaming anomaly detector using River.

Online/streaming detector — can learn incrementally from one sample at a time.
Wraps River's learn_one/score_one into the batch interface.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import joblib
import numpy as np
from river import anomaly, compose, preprocessing

from anomaly_detection.logging import get_logger
from anomaly_detection.ml.base import AnomalyDetector
from anomaly_detection.schemas.flows import FEATURE_COLUMNS

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

SEED = 42


class HalfSpaceTreesDetector(AnomalyDetector):
    """HalfSpaceTrees streaming anomaly detector."""

    name = "halfspace_trees"
    version = "v1"
    model_type = "unsupervised"

    def __init__(
        self,
        n_trees: int = 25,
        height: int = 8,
        window_size: int = 250,
        seed: int = SEED,
    ) -> None:
        self._n_trees = n_trees
        self._height = height
        self._window_size = window_size
        self._seed = seed

        # Pipeline: MinMaxScaler → HalfSpaceTrees
        self._model = compose.Pipeline(
            preprocessing.MinMaxScaler(),
            anomaly.HalfSpaceTrees(
                n_trees=n_trees,
                height=height,
                window_size=window_size,
                seed=seed,
            ),
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        """Train by streaming through all samples.

        HalfSpaceTrees is online — we call learn_one for each sample.

        Args:
            X: Feature matrix.
            y: Ignored for unsupervised model.
        """
        logger.info(
            "training_halfspace_trees",
            n_samples=X.shape[0],
            n_trees=self._n_trees,
            window_size=self._window_size,
        )

        for i in range(X.shape[0]):
            sample = {FEATURE_COLUMNS[j]: float(X[i, j]) for j in range(X.shape[1])}
            self._model.learn_one(sample)

        logger.info("training_complete", model=self.name)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for each sample.

        River's score_one returns values in [0, 1] where 1 = anomalous.
        """
        scores = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            sample = {FEATURE_COLUMNS[j]: float(X[i, j]) for j in range(X.shape[1])}
            scores[i] = self._model.score_one(sample)
        return scores

    def score_one(self, x: dict[str, float]) -> float:
        """Score a single sample (for streaming inference).

        Args:
            x: Dict mapping feature names to values.

        Returns:
            Anomaly score in [0, 1].
        """
        return float(self._model.score_one(x))

    def learn_one(self, x: dict[str, float]) -> None:
        """Update model with a single sample (online learning).

        Args:
            x: Dict mapping feature names to values.
        """
        self._model.learn_one(x)

    def score_one_with_latency(self, x: dict[str, float]) -> tuple[float, float]:
        """Score a single sample and measure latency.

        Args:
            x: Dict mapping feature names to values.

        Returns:
            Tuple of (anomaly_score, latency_seconds).
        """
        start = time.perf_counter()
        score = self.score_one(x)
        latency = time.perf_counter() - start
        return score, latency

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / "model.joblib")
        self.save_metadata(path, extra={
            "n_trees": self._n_trees,
            "height": self._height,
            "window_size": self._window_size,
            "seed": self._seed,
        })
        logger.info("model_saved", model=self.name, path=str(path))

    @classmethod
    def load(cls, path: Path) -> HalfSpaceTreesDetector:
        """Load model from disk."""
        instance = cls()
        instance._model = joblib.load(path / "model.joblib")

        # Load hyperparams from metadata
        meta_path = path / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            instance._n_trees = meta.get("n_trees", 25)
            instance._height = meta.get("height", 8)
            instance._window_size = meta.get("window_size", 250)

        logger.info("model_loaded", model=instance.name, path=str(path))
        return instance
