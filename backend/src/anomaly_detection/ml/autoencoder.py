"""AutoEncoder anomaly detector using PyOD.

Unsupervised detector — trained on benign traffic only.
Uses reconstruction error as the anomaly score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import joblib
import numpy as np
from pyod.models.auto_encoder import AutoEncoder

from anomaly_detection.logging import get_logger
from anomaly_detection.ml.base import AnomalyDetector

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

SEED = 42


class AutoEncoderDetector(AnomalyDetector):
    """AutoEncoder anomaly detector using reconstruction error."""

    name = "autoencoder"
    version = "v1"
    model_type = "unsupervised"

    def __init__(
        self,
        hidden_neurons: list[int] | None = None,
        epochs: int = 50,
        batch_size: int = 64,
        contamination: float = 0.01,
        random_state: int = SEED,
    ) -> None:
        self._hidden_neurons = hidden_neurons or [64, 32, 16, 32, 64]
        self._epochs = epochs
        self._batch_size = batch_size
        self._contamination = contamination
        self._random_state = random_state
        self._model: AutoEncoder | None = None
        # Score range from training set — used to normalise raw scores at inference
        self._train_score_min: float = 0.0
        self._train_score_max: float = 1.0

    def _create_model(self, n_features: int) -> AutoEncoder:
        """Create the AutoEncoder model with appropriate architecture."""
        # Ensure hidden layers are smaller than input
        hidden = [min(h, n_features) for h in self._hidden_neurons]
        return AutoEncoder(
            hidden_neuron_list=hidden,
            epoch_num=self._epochs,
            batch_size=self._batch_size,
            contamination=self._contamination,
            random_state=self._random_state,
            verbose=0,
        )

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        """Train AutoEncoder on benign data."""
        logger.info(
            "training_autoencoder",
            n_samples=X.shape[0],
            n_features=X.shape[1],
            hidden_neurons=self._hidden_neurons,
            epochs=self._epochs,
        )
        self._model = self._create_model(X.shape[1])
        self._model.fit(X)

        # Record training-set score range so inference is consistent.
        # decision_function returns raw reconstruction error; we normalise
        # using the *training* min/max so single-sample scoring is never 0.
        train_raw: np.ndarray = self._model.decision_function(X)
        self._train_score_min = float(train_raw.min())
        self._train_score_max = float(train_raw.max())
        logger.info(
            "training_complete",
            model=self.name,
            score_min=self._train_score_min,
            score_max=self._train_score_max,
        )

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores normalised by the training-set score range.

        Using the training-set min/max (instead of the per-batch min/max)
        ensures single-sample inference returns a meaningful score instead
        of always producing 0.0.
        """
        if self._model is None:
            msg = "Model not fitted — call fit() first"
            raise RuntimeError(msg)

        raw_scores: np.ndarray = self._model.decision_function(X)
        score_range = self._train_score_max - self._train_score_min
        if score_range > 0:
            normalised = (raw_scores - self._train_score_min) / score_range
            return np.clip(normalised, 0.0, 1.0)
        return np.zeros_like(raw_scores)

    def save(self, path: Path) -> None:
        """Save model to disk."""
        if self._model is None:
            msg = "Cannot save unfitted model"
            raise RuntimeError(msg)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / "model.joblib")
        self.save_metadata(path, extra={
            "hidden_neurons": self._hidden_neurons,
            "epochs": self._epochs,
            "batch_size": self._batch_size,
            "contamination": self._contamination,
            "train_score_min": self._train_score_min,
            "train_score_max": self._train_score_max,
        })
        logger.info("model_saved", model=self.name, path=str(path))

    @classmethod
    def load(cls, path: Path) -> AutoEncoderDetector:
        """Load model from disk."""
        import json
        instance = cls()
        instance._model = joblib.load(path / "model.joblib")
        # Restore training-set score range if available
        meta_path = path / "metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                instance._train_score_min = float(meta.get("train_score_min", 0.0))
                instance._train_score_max = float(meta.get("train_score_max", 1.0))
            except Exception:
                pass
        logger.info(
            "model_loaded",
            model=instance.name,
            path=str(path),
            score_min=instance._train_score_min,
            score_max=instance._train_score_max,
        )
        return instance
