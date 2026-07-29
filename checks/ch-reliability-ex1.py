"""Executable check for the shift-and-conformal deployment calculation."""

import numpy as np


outcomes = np.array([0.0, 0.0, 1.0, 1.0])
weights = np.array([0.2, 0.6, 1.2, 2.0])
weighted_mean = np.mean(weights * outcomes)
effective_n = weights.sum() ** 2 / np.sum(weights**2)

residuals = np.array([0.1, 0.2, 0.4, 0.7, 1.1, 1.6, 2.3, 4.0, 7.0])
coverage = 0.75
rank = int(np.ceil((len(residuals) + 1) * coverage))
radius = np.sort(residuals)[rank - 1]

np.testing.assert_allclose(weights.mean(), 1.0)
np.testing.assert_allclose(weighted_mean, 0.8)
np.testing.assert_allclose(effective_n, 16.0 / 5.84)
assert rank == 8
np.testing.assert_allclose(radius, 4.0)
assert effective_n < len(weights)

print("ch-reliability example 1 passed")
