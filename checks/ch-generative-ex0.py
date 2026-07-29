"""The off-diagonal probability distribution is not a positive-semidefinite kernel."""

import numpy as np

gram = np.array([[0.0, 0.5], [0.5, 0.0]])
eigenvalues = np.linalg.eigvalsh(gram)

assert np.allclose(eigenvalues, [-0.5, 0.5])
assert eigenvalues[0] < 0
