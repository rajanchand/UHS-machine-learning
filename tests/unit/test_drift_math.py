"""Unit tests for Population Stability Index (PSI) drift calculation math."""

import numpy as np
import pandas as pd
from anomaly_detection.api.routers.drift import calculate_feature_psi


def test_calculate_feature_psi_no_drift():
    """Verify that identical distributions result in a very low PSI score."""
    np.random.seed(42)
    # Generate identical normal distributions
    expected = pd.Series(np.random.normal(0, 1, 1000))
    actual = np.random.normal(0, 1, 1000).tolist()

    psi = calculate_feature_psi(expected, actual)
    assert psi < 0.1  # Identical distributions should have low PSI (typically < 0.1)


def test_calculate_feature_psi_significant_drift():
    """Verify that shifted distributions result in a high PSI score."""
    np.random.seed(42)
    # Shift the actual distribution mean significantly
    expected = pd.Series(np.random.normal(0, 1, 1000))
    actual = np.random.normal(2, 1, 1000).tolist()

    psi = calculate_feature_psi(expected, actual)
    assert psi >= 0.25  # Shifted distributions should flag drift (threshold >= 0.25)


def test_calculate_feature_psi_empty_inputs():
    """Verify that empty inputs handle gracefully and return 0.0."""
    expected = pd.Series([])
    actual = []
    psi = calculate_feature_psi(expected, actual)
    assert psi == 0.0
