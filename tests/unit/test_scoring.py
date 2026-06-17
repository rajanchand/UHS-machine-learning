import numpy as np
import pytest

from anomaly_detection.ml.isolation_forest import IsolationForestDetector

class DummyDetector:
    def score(self, X: np.ndarray) -> np.ndarray:
        # returns simple scores
        return X.flatten()
        
    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        scores = self.score(X)
        return (scores >= threshold).astype(int)

def test_binary_prediction_thresholding():
    detector = DummyDetector()
    X = np.array([0.1, 0.4, 0.5, 0.8])
    
    # Threshold at 0.5
    preds = detector.predict(X, threshold=0.5)
    assert np.array_equal(preds, [0, 0, 1, 1])
    
    # Threshold at 0.3
    preds2 = detector.predict(X, threshold=0.3)
    assert np.array_equal(preds2, [0, 1, 1, 1])
    
    # Threshold at 0.9
    preds3 = detector.predict(X, threshold=0.9)
    assert np.array_equal(preds3, [0, 0, 0, 0])

def test_isolation_forest_prediction():
    # Verify that the detector class correctly inherits predict() from base
    iforest = IsolationForestDetector(n_estimators=10)
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    
    # base AnomalyDetector.predict uses >= threshold
    # Let's mock score on iforest
    iforest.score = lambda X: scores # type: ignore
    
    preds = iforest.predict(np.zeros((4, 2)), threshold=0.5)
    assert np.array_equal(preds, [0, 0, 1, 1])
