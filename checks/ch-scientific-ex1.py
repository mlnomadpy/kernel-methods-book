"""Executable check for the polynomial-kernel Poisson collocation example."""

import numpy as np


gram = np.array(
    [
        [1.0, 1.0, 0.0],
        [1.0, 4.0, -2.0],
        [0.0, -2.0, 4.0],
    ]
)
information = np.array([0.0, 0.0, 2.0])
coefficients = np.linalg.solve(gram, information)


def recovered(x):
    representers = np.array([1.0, (1.0 + x) ** 2, -2.0 * x**2])
    return coefficients @ representers


grid = np.linspace(0.0, 1.0, 101)
exact = grid * (1.0 - grid)

np.testing.assert_allclose(coefficients, [-0.5, 0.5, 0.75])
np.testing.assert_allclose([recovered(x) for x in grid], exact, atol=2e-15)
assert np.linalg.eigvalsh(gram)[0] > 0.0

print("ch-scientific example 1 passed")
