"""Numerical witness for ch00's three-object Gram compatibility example."""

import numpy as np


phi = np.array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
K = phi @ phi.T
K_bad = K.copy()
K_bad[0, 2] = K_bad[2, 0] = -0.4

eig_good = np.linalg.eigvalsh(K)
eig_bad = np.linalg.eigvalsh(K_bad)
distance_sq = K[0, 0] + K[1, 1] - 2.0 * K[0, 1]

assert np.allclose(K, K.T)
assert eig_good.min() >= -1e-12
assert eig_bad.min() < -1e-3
assert np.isclose(distance_sq, 0.4)
print("PASS intro-gram-compatibility")
