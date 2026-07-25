"""Deterministic failure witness for independently estimated quantum kernels."""

import numpy as np

K_exact = np.ones((3, 3))
K_hat = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, 0.1], [0.9, 0.1, 1.0]])

exact_eigs = np.linalg.eigvalsh(K_exact)
estimated_eigs, estimated_vecs = np.linalg.eigh(K_hat)
K_psd = (estimated_vecs * np.maximum(estimated_eigs, 0.0)) @ estimated_vecs.T
repair_distance = np.linalg.norm(K_hat - K_psd, ord="fro")

assert exact_eigs.min() >= -1e-12
assert estimated_eigs.min() < -0.22
assert np.allclose(repair_distance, -estimated_eigs.min(), atol=1e-12)
assert np.linalg.eigvalsh(K_psd).min() >= -1e-12

print("exact eigenvalues:", np.round(exact_eigs, 6))
print("estimated eigenvalues:", np.round(estimated_eigs, 6))
print("PSD-projection Frobenius distance:", round(float(repair_distance), 6))
