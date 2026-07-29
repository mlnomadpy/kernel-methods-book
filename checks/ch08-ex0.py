"""Fourier coefficients for the two extremal translation-invariant kernels."""

import numpy as np

n = np.arange(-4, 5)

# Flat spectral measure: the integral of exp(i n t) over one period.
grid = np.linspace(-np.pi, np.pi, 200_001)
diagonal_coefficients = np.trapezoid(
    np.exp(1j * n[:, None] * grid), grid, axis=1
) / (2 * np.pi)
assert np.allclose(diagonal_coefficients, (n == 0).astype(float), atol=1e-10)

# A point mass at frequency zero evaluates every character at one.
C = 2.5
constant_coefficients = C * np.exp(1j * n * 0.0)
assert np.allclose(constant_coefficients, C)
