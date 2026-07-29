"""Deterministic verification for ordinary versus simple kriging."""

import numpy as np


K = np.array([[1.0, np.exp(-2.0)], [np.exp(-2.0), 1.0]])
k0 = np.full(2, np.exp(-1.0))
y = np.array([0.0, 2.0])

augmented = np.block(
    [[K, np.ones((2, 1))], [np.ones((1, 2)), np.zeros((1, 1))]]
)
solution = np.linalg.solve(augmented, np.r_[k0, 1.0])
w_ok, eta = solution[:2], solution[2]
w_sk = np.linalg.solve(K, k0)
variance_ok = 1.0 - w_ok @ k0 - eta
variance_sk = 1.0 - k0 @ w_sk

assert np.allclose(w_ok, [0.5, 0.5])
assert np.isclose(w_ok.sum(), 1.0)
assert np.isclose(w_ok @ y, 1.0)
assert np.allclose(w_sk, [0.32402714, 0.32402714])
assert np.isclose(eta, -0.1997882)
assert np.isclose(variance_ok, 0.8319074)
assert np.isclose(variance_sk, np.tanh(1.0))
print("PASS spatial-kriging")
