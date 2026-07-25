"""Deterministic finite-width failure witness for Chapter 14."""

import numpy as np


def normalized_relu_kernel(x: np.ndarray) -> np.ndarray:
    correlations = np.clip(x @ x.T, -1.0, 1.0)
    angles = np.arccos(correlations)
    return (
        np.sqrt(np.maximum(0.0, 1.0 - correlations**2))
        + (np.pi - angles) * correlations
    ) / np.pi


rng = np.random.default_rng(1401)
inputs = rng.normal(size=(16, 12))
inputs /= np.linalg.norm(inputs, axis=1, keepdims=True)
exact = normalized_relu_kernel(inputs)

expected = {
    8: (0.512743, 8),
    32: (0.419790, 16),
    128: (0.188895, 16),
    512: (0.116120, 16),
    2048: (0.051238, 16),
}

for width, (expected_error, expected_rank) in expected.items():
    weights = np.random.default_rng(1401 + width).normal(size=(width, 12))
    features = np.sqrt(2.0 / width) * np.maximum(inputs @ weights.T, 0.0)
    empirical = features @ features.T
    error = np.linalg.norm(empirical - exact, ord="fro") / np.linalg.norm(exact, ord="fro")
    rank = np.linalg.matrix_rank(empirical, tol=1e-10)
    assert np.isclose(error, expected_error, atol=5e-7), (width, error)
    assert rank == expected_rank, (width, rank)

exact_rank = np.linalg.matrix_rank(exact, tol=1e-10)
exact_minimum_eigenvalue = np.linalg.eigvalsh(exact)[0]
assert exact_rank == 16
assert np.isclose(exact_minimum_eigenvalue, 0.093207, atol=5e-7)

print(
    "finite-width witness verified: "
    f"exact_rank={exact_rank}, min_eigenvalue={exact_minimum_eigenvalue:.6f}"
)
