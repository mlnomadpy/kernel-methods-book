"""Deterministic kernel-PCA denoising and eigenspace-stability experiment.

The linear kernel is used deliberately: kernel PCA then has an exact input-space
pre-image, so the experiment isolates the statistical denoising claim from the
separate nonlinear pre-image optimization problem.
"""

import numpy as np


def make_clean_images(count, side, rng):
    grid = np.linspace(-1.0, 1.0, side)
    xx, yy = np.meshgrid(grid, grid)
    images = []
    for _ in range(count):
        shift_x, shift_y = rng.normal(0.0, 0.10, size=2)
        width = rng.uniform(0.28, 0.38)
        blob = np.exp(-((xx - shift_x) ** 2 + (yy - shift_y) ** 2) / (2.0 * width**2))
        shoulder = 0.45 * np.exp(
            -((xx + 0.35 - shift_x) ** 2 + (yy - 0.25 - shift_y) ** 2)
            / (2.0 * (0.55 * width) ** 2)
        )
        images.append((blob + shoulder).ravel())
    return np.asarray(images)


def top_subspace(centered, rank):
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    basis = right_vectors[:rank].T
    covariance_eigenvalues = singular_values**2 / centered.shape[0]
    return basis, covariance_eigenvalues


def projector_distance(basis_a, basis_b):
    projector_a = basis_a @ basis_a.T
    projector_b = basis_b @ basis_b.T
    return np.linalg.norm(projector_a - projector_b, ord=2)


rng = np.random.default_rng(20260725)
side = 24
train_count = 180
test_count = 60
rank = 2

clean_train = make_clean_images(train_count, side, rng)
clean_test = make_clean_images(test_count, side, rng)
noise_sd = 0.22
noisy_train = clean_train + rng.normal(0.0, noise_sd, clean_train.shape)
noisy_test = clean_test + rng.normal(0.0, noise_sd, clean_test.shape)

mean_image = noisy_train.mean(axis=0)
basis, eigenvalues = top_subspace(noisy_train - mean_image, rank)
denoised = mean_image + (noisy_test - mean_image) @ basis @ basis.T

raw_mse = np.mean((noisy_test - clean_test) ** 2)
denoised_mse = np.mean((denoised - clean_test) ** 2)
improvement = 100.0 * (raw_mse - denoised_mse) / raw_mse
eigengap = eigenvalues[rank - 1] - eigenvalues[rank]

# A failure regime: corruption lies in the learned signal subspace. Projection
# retains it, so the low-rank denoiser cannot remove it.
aligned_coefficients = rng.normal(0.0, 0.22, size=(test_count, rank))
aligned_noise = aligned_coefficients @ basis.T
aligned_observed = clean_test + aligned_noise
aligned_denoised = mean_image + (aligned_observed - mean_image) @ basis @ basis.T
aligned_raw_mse = np.mean((aligned_observed - clean_test) ** 2)
aligned_denoised_mse = np.mean((aligned_denoised - clean_test) ** 2)

# Finite-sample stability: compare subspaces fitted to nested samples and verify
# the Davis-Kahan diagnostic ||P_hat-P_ref|| <= 2||C_hat-C_ref||/gap whenever
# the perturbation is below half the reference eigengap.
centered_full = noisy_train - noisy_train.mean(axis=0)
reference_basis, reference_values = top_subspace(centered_full, rank)
reference_covariance = centered_full.T @ centered_full / train_count
small_count = 170
centered_small = noisy_train[:small_count] - noisy_train[:small_count].mean(axis=0)
small_basis, _ = top_subspace(centered_small, rank)
small_covariance = centered_small.T @ centered_small / small_count
operator_error = np.linalg.norm(small_covariance - reference_covariance, ord=2)
reference_gap = reference_values[rank - 1] - reference_values[rank]
subspace_error = projector_distance(small_basis, reference_basis)
dk_bound = 2.0 * operator_error / reference_gap

print(f"image_shape={side}x{side}")
print(f"train_count={train_count} test_count={test_count} retained_rank={rank}")
print(f"raw_test_mse={raw_mse:.6f}")
print(f"denoised_test_mse={denoised_mse:.6f}")
print(f"relative_improvement_percent={improvement:.2f}")
print(f"empirical_eigengap_at_rank={eigengap:.6f}")
print(f"aligned_noise_raw_mse={aligned_raw_mse:.6f}")
print(f"aligned_noise_after_projection_mse={aligned_denoised_mse:.6f}")
print(f"reference_eigengap={reference_gap:.6f}")
print(f"covariance_operator_error={operator_error:.6f}")
print(f"projector_operator_error={subspace_error:.6f}")
print(f"davis_kahan_diagnostic_bound={dk_bound:.6f}")

assert denoised_mse < 0.20 * raw_mse
assert aligned_denoised_mse > 0.90 * aligned_raw_mse
assert reference_gap > 0.0
assert subspace_error <= dk_bound + 1e-12
assert np.all(np.isfinite(denoised))
