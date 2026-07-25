"""Deterministic release-gate fixture for ch-accountable.

This is a regression test for the chapter's numerical invariants and decision
logic. It is not a power study or evidence that a deployed model is valid.
"""

import numpy as np


SEED = 5506
TOL_SYMMETRY = 1e-12
TOL_SOLVE = 1e-10
TOL_LOO = 1e-10
LEVEL = 0.05
PERMUTATIONS = 499
rng = np.random.default_rng(SEED)


def rbf(x, z, length_scale=0.35):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    z = np.asarray(z, dtype=float).reshape(-1, 1)
    return np.exp(-0.5 * ((x - z.T) / length_scale) ** 2)


def mmd2_u(x, y):
    kxx = rbf(x, x)
    kyy = rbf(y, y)
    kxy = rbf(x, y)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return (
        kxx.sum() / (len(x) * (len(x) - 1))
        + kyy.sum() / (len(y) * (len(y) - 1))
        - 2.0 * kxy.mean()
    )


def permutation_pvalue(statistic, left, right, permutations):
    observed = statistic(left, right)
    pooled = np.concatenate([left, right])
    split = len(left)
    exceed = 0
    for permutation in permutations:
        permuted = pooled[permutation]
        exceed += statistic(permuted[:split], permuted[split:]) >= observed
    return observed, (exceed + 1.0) / (len(permutations) + 1.0)


def hsic_biased(score, attribute):
    n = len(score)
    h = np.eye(n) - np.ones((n, n)) / n
    k = rbf(score, score, 0.5)
    l = rbf(attribute, attribute, 0.5)
    return float(np.trace(k @ h @ l @ h) / (n - 1) ** 2)


def hsic_pvalue(score, attribute, permutations):
    observed = hsic_biased(score, attribute)
    exceed = 0
    for permutation in permutations:
        exceed += hsic_biased(score, attribute[permutation]) >= observed
    return observed, (exceed + 1.0) / (len(permutations) + 1.0)


# Fixed kernel-ridge fit and numerical gates.
x_train = np.linspace(-1.0, 1.0, 31)
y_train = np.sin(2.2 * x_train) + 0.03 * np.cos(7.0 * x_train)
ridge = 0.08
k_train = rbf(x_train, x_train)
a = k_train + ridge * np.eye(len(x_train))
coef = np.linalg.solve(a, y_train)
relative_residual = np.linalg.norm(a @ coef - y_train) / np.linalg.norm(y_train)
symmetry_error = np.max(np.abs(k_train - k_train.T))

# Verify the exact deletion identity at a fixed record and query.
query = np.array([0.37])
k_query = rbf(x_train, query).ravel()
a_inv = np.linalg.solve(a, np.eye(len(a)))
hat = k_train @ a_inv
y_hat = hat @ y_train
delete_index = 17
loo_residual = (y_train[delete_index] - y_hat[delete_index]) / (
    1.0 - hat[delete_index, delete_index]
)
closed_form_change = (a_inv @ k_query)[delete_index] * loo_residual
keep = np.arange(len(x_train)) != delete_index
deleted_coef = np.linalg.solve(
    rbf(x_train[keep], x_train[keep]) + ridge * np.eye(keep.sum()),
    y_train[keep],
)
full_prediction = float(k_query @ coef)
deleted_prediction = float(rbf(query, x_train[keep]).ravel() @ deleted_coef)
loo_error = abs(closed_form_change - (full_prediction - deleted_prediction))

# Exercise the finite-sample conformal rank, including its declared finite case.
x_cal = np.linspace(-0.95, 0.95, 39)
y_cal = np.sin(2.2 * x_cal) + 0.11 * np.sin(13.0 * x_cal)
scores = np.abs(y_cal - rbf(x_cal, x_train) @ coef)
alpha = 0.10
rank = int(np.ceil((len(scores) + 1) * (1.0 - alpha)))
q_hat = float(np.sort(scores)[rank - 1])

# Reuse identical permutation schedules within each diagnostic family.
n_audit = 80
mmd_permutations = np.array(
    [rng.permutation(2 * n_audit) for _ in range(PERMUTATIONS)]
)
reference = rng.normal(0.0, 1.0, n_audit)
same = rng.normal(0.0, 1.0, n_audit)
shifted = rng.normal(1.0, 1.0, n_audit)
mmd_same, p_mmd_same = permutation_pvalue(
    mmd2_u, reference, same, mmd_permutations
)
mmd_shifted, p_mmd_shifted = permutation_pvalue(
    mmd2_u, reference, shifted, mmd_permutations
)

attribute = rng.normal(size=n_audit)
independent_score = rng.normal(size=n_audit)
dependent_score = attribute + 0.15 * rng.normal(size=n_audit)
hsic_permutations = np.array(
    [rng.permutation(n_audit) for _ in range(PERMUTATIONS)]
)
hsic_independent, p_hsic_independent = hsic_pvalue(
    independent_score, attribute, hsic_permutations
)
hsic_dependent, p_hsic_dependent = hsic_pvalue(
    dependent_score, attribute, hsic_permutations
)

assert symmetry_error < TOL_SYMMETRY
assert relative_residual < TOL_SOLVE
assert loo_error < TOL_LOO
assert rank == 36 and np.isfinite(q_hat)
assert p_mmd_same > LEVEL
assert p_mmd_shifted <= LEVEL
assert p_hsic_independent > LEVEL
assert p_hsic_dependent <= LEVEL

print(f"seed={SEED} permutations={PERMUTATIONS}")
print(f"symmetry_error={symmetry_error:.3e}")
print(f"relative_solve_residual={relative_residual:.3e}")
print(f"loo_refit_discrepancy={loo_error:.3e}")
print(f"conformal_rank={rank} q_hat={q_hat:.6f}")
print(f"mmd_same={mmd_same:.6f} p={p_mmd_same:.3f}")
print(f"mmd_shifted={mmd_shifted:.6f} p={p_mmd_shifted:.3f}")
print(f"hsic_independent={hsic_independent:.6f} p={p_hsic_independent:.3f}")
print(f"hsic_dependent={hsic_dependent:.6f} p={p_hsic_dependent:.3f}")
