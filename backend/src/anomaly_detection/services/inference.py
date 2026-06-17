"""Inference service — loads models and scores flows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import joblib
import numpy as np

from anomaly_detection.logging import get_logger
from anomaly_detection.ml.autoencoder import AutoEncoderDetector
from anomaly_detection.ml.halfspace_trees import HalfSpaceTreesDetector
from anomaly_detection.ml.isolation_forest import IsolationForestDetector
from anomaly_detection.ml.lightgbm_model import LightGBMBenchmark

if TYPE_CHECKING:
    from pathlib import Path

    from anomaly_detection.ml.base import AnomalyDetector

logger = get_logger(__name__)

MODEL_LOADERS: dict[str, type[AnomalyDetector]] = {
    "isolation_forest": IsolationForestDetector,
    "autoencoder": AutoEncoderDetector,
    "halfspace_trees": HalfSpaceTreesDetector,
    "lightgbm_benchmark": LightGBMBenchmark,
}


class InferenceService:
    """Loads models and scalers, provides scoring interface."""

    def __init__(self, model_registry_path: Path, data_dir: Path) -> None:
        self._registry_path = model_registry_path
        self._data_dir = data_dir
        self._models: dict[str, AnomalyDetector] = {}
        self._scaler: Any = None
        self._active_model: str | None = None
        self._thresholds: dict[str, float] = {}

    def load_models(self) -> None:
        """Load all available models from the registry."""
        scaler_path = self._data_dir / "processed" / "scaler.joblib"
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)
            logger.info("scaler_loaded", path=str(scaler_path))
        else:
            logger.warning("scaler_not_found", path=str(scaler_path))

        for model_name, loader_cls in MODEL_LOADERS.items():
            model_path = self._registry_path / model_name / "v1"
            if model_path.exists() and (model_path / "model.joblib").exists():
                try:
                    model = loader_cls.load(model_path)
                    self._models[model_name] = model
                    self._thresholds[model_name] = 0.5  # Default threshold
                    logger.info("model_loaded", model=model_name)
                except Exception:
                    logger.exception("model_load_failed", model=model_name)
            else:
                logger.info("model_not_found", model=model_name, path=str(model_path))

        # Set first available model as active
        if self._models:
            self._active_model = next(iter(self._models))
            logger.info("active_model_set", model=self._active_model)

    @property
    def available_models(self) -> list[str]:
        """List of loaded model names."""
        return list(self._models.keys())

    @property
    def active_model_name(self) -> str | None:
        """Currently active model name."""
        return self._active_model

    def set_active_model(self, name: str) -> None:
        """Set the active model for scoring."""
        if name not in self._models:
            msg = f"Model '{name}' not loaded. Available: {self.available_models}"
            raise ValueError(msg)
        self._active_model = name
        logger.info("active_model_changed", model=name)

    def get_threshold(self, model_name: str | None = None) -> float:
        """Get the threshold for a model."""
        name = model_name or self._active_model
        if name is None:
            return 0.5
        return self._thresholds.get(name, 0.5)

    def set_threshold(self, model_name: str, threshold: float) -> None:
        """Set the threshold for a model."""
        self._thresholds[model_name] = threshold
        logger.info("threshold_updated", model=model_name, threshold=threshold)

    def scale_features(self, feature_vector: list[float]) -> np.ndarray:
        """Scale a single feature vector using the fitted scaler."""
        arr = np.array(feature_vector).reshape(1, -1)
        if self._scaler is not None:
            return self._scaler.transform(arr)  # type: ignore[no-any-return]
        return arr

    def score_flow(
        self,
        feature_vector: list[float],
        model_name: str | None = None,
    ) -> tuple[float, bool, str, float]:
        """Score a single flow.

        Args:
            feature_vector: Ordered feature values.
            model_name: Model to use (None = active model).

        Returns:
            Tuple of (score, is_anomaly, model_name, threshold).
        """
        name = model_name or self._active_model
        if name is None or name not in self._models:
            msg = f"No model available for scoring. Loaded: {self.available_models}"
            raise RuntimeError(msg)

        model = self._models[name]
        threshold = self._thresholds.get(name, 0.5)

        scaled = self.scale_features(feature_vector)
        score = float(model.score(scaled)[0])
        is_anomaly = score >= threshold

        return score, is_anomaly, name, threshold

    def score_batch(
        self,
        feature_vectors: list[list[float]],
        model_name: str | None = None,
    ) -> list[tuple[float, bool]]:
        """Score a batch of flows.

        Args:
            feature_vectors: List of ordered feature value lists.
            model_name: Model to use (None = active model).

        Returns:
            List of (score, is_anomaly) tuples.
        """
        name = model_name or self._active_model
        if name is None or name not in self._models:
            msg = f"No model available for scoring. Loaded: {self.available_models}"
            raise RuntimeError(msg)

        model = self._models[name]
        threshold = self._thresholds.get(name, 0.5)

        arr = np.array(feature_vectors)
        if self._scaler is not None:
            arr = self._scaler.transform(arr)

        scores = model.score(arr)
        return [(float(s), bool(s >= threshold)) for s in scores]
