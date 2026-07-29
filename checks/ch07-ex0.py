"""Recover the Brownian/min kernel from derivative features."""

import numpy as np

x = np.array([0.1, 0.4, 0.9])
gram = np.minimum(x[:, None], x[None, :])
coefficients = np.array([1.0, -2.0, 0.5])

grid = np.linspace(0.0, 1.0, 10_001)
derivative = (grid[None, :] <= x[:, None]).T @ coefficients
energy = np.trapezoid(derivative**2, grid)
quadratic_form = coefficients @ gram @ coefficients

assert np.linalg.eigvalsh(gram).min() > 0.0
assert np.isclose(energy, quadratic_form, atol=2e-4)
