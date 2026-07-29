#!/usr/bin/env python3
"""Two-point Gaussian KRR calculation from ch03, Example 4-4."""
import numpy as np

off_diagonal = np.exp(-1.0)
gram = np.array([[1.0, off_diagonal], [off_diagonal, 1.0]])
alpha = np.linalg.solve(gram + 0.5 * np.eye(2), np.array([1.0, 0.0]))
fitted = gram @ alpha

np.testing.assert_allclose(alpha, [0.709332, -0.173966], rtol=1e-5)
np.testing.assert_allclose(fitted, [0.645334, 0.086983], rtol=1e-5)
print("PASS ch03 two-point KRR")
