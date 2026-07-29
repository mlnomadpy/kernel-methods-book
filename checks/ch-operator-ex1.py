"""Executable check for the two-task operator-valued transfer example."""

import numpy as np


r = 0.25
rho = 0.75
ridge = 0.5
observed_gram = np.array([[1.0, r * rho], [r * rho, 1.0]])
coefficients = np.linalg.solve(observed_gram + ridge * np.eye(2), np.ones(2))

output_matrix = np.array([[1.0, rho], [rho, 1.0]])
assert np.linalg.eigvalsh(output_matrix)[0] > 0.0
np.testing.assert_allclose(coefficients, [16.0 / 27.0] * 2)

prediction_x1 = (
    output_matrix @ np.array([coefficients[0], 0.0])
    + r * output_matrix @ np.array([0.0, coefficients[1]])
)
np.testing.assert_allclose(prediction_x1, [19.0 / 27.0, 16.0 / 27.0])

print("ch-operator example 1 passed")
