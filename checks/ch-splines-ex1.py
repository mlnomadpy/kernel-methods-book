"""Deterministic verification for the spline null-space worked example."""

import numpy as np


grid = np.linspace(0.0, 1.0, 200_001)
second_derivative = np.where(
    grid <= 0.5,
    -24.0 * grid,
    -24.0 * (1.0 - grid),
)
roughness = np.trapezoid(second_derivative**2, grid)

lam = 0.01
curvature_contrast = np.array([-1.0, 2.0, -1.0])
shrinkage = 1.0 / (1.0 + 216.0 * lam)
fitted = np.ones(3) / 3.0 + shrinkage * curvature_contrast / 3.0
effective_df = 2.0 + shrinkage

assert np.isclose(roughness, 48.0, atol=1e-8)
assert np.isclose(shrinkage, 0.3164556962025316)
assert np.allclose(fitted, [0.2278481, 0.5443038, 0.2278481])
assert np.isclose(effective_df, 2.3164556962025316)
print("PASS spline-nullspace")
