"""Executable check for the two-state coverage failure in ch-dynamics."""

import numpy as np


gamma = 0.8
transition = np.array([[1.0, 0.0], [0.5, 0.5]])
reward = np.array([0.0, 1.0])

true_value = np.linalg.solve(np.eye(2) - gamma * transition, reward)
bad_value = np.array([0.0, 100.0])


def bellman_residual(value):
    return value - (reward + gamma * transition @ value)


true_residual = bellman_residual(true_value)
bad_residual = bellman_residual(bad_value)

behavior = np.array([1.0, 0.0])
evaluation = np.array([0.0, 1.0])
observed_bad_residual = np.sqrt(np.sum(behavior * bad_residual**2))
target_absolute_error = np.sum(evaluation * np.abs(bad_value - true_value))

np.testing.assert_allclose(true_value, [0.0, 5.0 / 3.0], atol=1e-14)
np.testing.assert_allclose(true_residual, [0.0, 0.0], atol=1e-14)
np.testing.assert_allclose(observed_bad_residual, 0.0)
np.testing.assert_allclose(target_absolute_error, 100.0 - 5.0 / 3.0)
assert behavior[1] == 0.0 and evaluation[1] > 0.0

print("ch-dynamics example 1 passed")
