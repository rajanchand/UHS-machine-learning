"""Inference service — loads models and scores network flows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import joblib
import numpy as np

from anomaly_detection.constants import EXPECTED_FEATURE_COUNT
from anomaly_detection.logging import get_logger
from anomaly_detection.ml.autoencoder import AutoEncoderDetector
from anomaly_detection.ml.halfspace_trees import HalfSpaceTreesDetector
from anomaly_detection.ml.isolation_forest import IsolationForestDetector
from anomaly_detection.ml.lightgbm_model import LightGBMBenchmark
from anomaly_detection.ml.random_forest import RandomForestDetector
from anomaly_detection.ml.xgboost_model import XGBoostDetector

if TYPE_CHECKING:
    from pathlib import Path

    from anomaly_detection.ml.base import AnomalyDetector

logger = get_logger(__name__)

MODEL_LOADERS: dict[str, type[AnomalyDetector]] = {
    "isolation_forest": IsolationForestDetector,
    "autoencoder": AutoEncoderDetector,
    "halfspace_trees": HalfSpaceTreesDetector,
    "lightgbm_benchmark": LightGBMBenchmark,
    "random_forest": RandomForestDetector,
    "xgboost": XGBoostDetector,
}


class InferenceService:
    """Loads models and scalers, provides a scoring interface."""

    def __init__(self, model_registry_path: Path, data_dir: Path) -> None:
        self._registry_path = model_registry_path
        self._data_dir = data_dir
        self._models: dict[str, AnomalyDetector] = {}
        self._scaler: Any = None
        self._active_model: str | None = None
        self._thresholds: dict[str, float] = {}

    def load_models(self) -> None:
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
                    self._thresholds[model_name] = 0.5
                    logger.info("model_loaded", model=model_name)
                except Exception:
                    logger.exception("model_load_failed", model=model_name)
            else:
                logger.info("model_not_found", model=model_name, path=str(model_path))

        if self._models:
            self._active_model = next(iter(self._models))
            logger.info("active_model_set", model=self._active_model)

    @property
    def available_models(self) -> list[str]:
        return list(self._models.keys())

    @property
    def active_model_name(self) -> str | None:
        return self._active_model

    def set_active_model(self, name: str) -> None:
        if name not in self._models:
            raise ValueError(f"Model '{name}' not loaded. Available: {self.available_models}")
        self._active_model = name
        logger.info("active_model_changed", model=name)

    def get_threshold(self, model_name: str | None = None) -> float:
        name = model_name or self._active_model
        return self._thresholds.get(name, 0.5) if name else 0.5

    def set_threshold(self, model_name: str, threshold: float) -> None:
        self._thresholds[model_name] = threshold
        logger.info("threshold_updated", model=model_name, threshold=threshold)

    def scale_features(self, feature_vector: list[float]) -> np.ndarray:
        arr = np.array(feature_vector).reshape(1, -1)
        if self._scaler is not None:
            return self._scaler.transform(arr)  # type: ignore[no-any-return]
        return arr

    def score_flow(
        self,
        feature_vector: list[float],
        model_name: str | None = None,
    ) -> tuple[float, bool, str, float]:
        """Score a single flow vector.

        Returns:
            (score, is_anomaly, model_name, threshold)

        Raises:
            RuntimeError: No model is loaded or active.
            ValueError: Feature vector has wrong length.
        """
        if len(feature_vector) != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Expected {EXPECTED_FEATURE_COUNT} features, got {len(feature_vector)}"
            )

        name = model_name or self._active_model
        if name is None or name not in self._models:
            raise RuntimeError(f"No model available. Loaded: {self.available_models}")

        model = self._models[name]
        threshold = self._thresholds.get(name, 0.5)

        scaled = self.scale_features(feature_vector)
        score = float(model.score(scaled)[0])
        return score, score >= threshold, name, threshold

    def score_batch(
        self,
        feature_vectors: list[list[float]],
        model_name: str | None = None,
    ) -> list[tuple[float, bool]]:
        name = model_name or self._active_model
        if name is None or name not in self._models:
            raise RuntimeError(f"No model available. Loaded: {self.available_models}")

        model = self._models[name]
        threshold = self._thresholds.get(name, 0.5)

        arr = np.array(feature_vectors)
        if self._scaler is not None:
            arr = self._scaler.transform(arr)

        scores = model.score(arr)
        return [(float(s), bool(s >= threshold)) for s in scores]
