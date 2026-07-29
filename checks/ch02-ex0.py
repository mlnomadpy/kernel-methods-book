#!/usr/bin/env python3
"""Gaussian feature-distance calculation from ch02, Example 3-1."""
import numpy as np


def feature_distance(radius: float, sigma: float) -> float:
    return float(np.sqrt(2 * (1 - np.exp(-(radius**2) / (2 * sigma**2)))))


assert feature_distance(0.0, 1.0) == 0.0
np.testing.assert_allclose(feature_distance(10.0, 1.0), np.sqrt(2), rtol=1e-12)
print("PASS ch02 Gaussian feature distance")
