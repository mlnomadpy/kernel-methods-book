"""Regression checks for numerical values introduced in the public solution companion."""

from math import erf, exp, isclose, log, sqrt

import numpy as np


def close(actual, expected, tolerance=1e-6):
    assert isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance), (
        actual,
        expected,
    )


# Applications: corrected positive-definite Gram, normalization, and centering.
K_app = np.array([[4.0, 2.0, 0.0], [2.0, 9.0, 3.0], [0.0, 3.0, 2.0]])
assert np.linalg.eigvalsh(K_app).min() > 0
scale = np.sqrt(np.diag(K_app))
K_normalized = K_app / scale[:, None] / scale[None, :]
assert np.allclose(np.diag(K_normalized), 1.0)
H = np.eye(3) - np.ones((3, 3)) / 3
K_centered = H @ K_app @ H
assert np.allclose(K_centered.sum(axis=0), 0.0)
assert np.allclose(K_centered.sum(axis=1), 0.0)


# Applications: the added sigma=1 bandwidth calculation.
x_raw = np.arange(6, dtype=float)
x = (x_raw - x_raw.mean()) / x_raw.std()
y = np.sin(x_raw)
fold_a, fold_b = np.array([0, 2, 4]), np.array([1, 3, 5])


def rbf(left, right, sigma):
    return np.exp(-((left[:, None] - right[None, :]) ** 2) / (2 * sigma**2))


errors = []
for train, test in ((fold_a, fold_b), (fold_b, fold_a)):
    alpha = np.linalg.solve(rbf(x[train], x[train], 1.0) + 0.1 * np.eye(3), y[train])
    prediction = rbf(x[test], x[train], 1.0) @ alpha
    errors.append(np.mean((prediction - y[test]) ** 2))
close(np.mean(errors), 0.10464254494467633)
eigenvalues = np.linalg.eigvalsh(rbf(x, x, 1.0))
close(np.sum(eigenvalues / (eigenvalues + 0.1)), 3.5386615804184784)


# Bayesian optimization: PI and two-point information gain.
normal_cdf = lambda z: 0.5 * (1.0 + erf(z / sqrt(2.0)))
close(normal_cdf(-0.0030 / 0.7973), 0.49849890377885747)
close(normal_cdf(-0.0374 / 0.9817), 0.484805100508983)
K_bo = np.array([[1.0, exp(-2.0)], [exp(-2.0), 1.0]])
information = 0.5 * log(np.linalg.det(np.eye(2) + 100.0 * K_bo))
close(information, 4.6060615715032)


# Entropic OT uses KL(pi || a b^T), so the self-cost is positive and debiases to zero.
r = exp(-1.0)
u = sqrt(0.5 / (1.0 + r))
plan = u * u * np.array([[1.0, r], [r, 1.0]])
reference = np.full((2, 2), 0.25)
cost = np.array([[0.0, 1.0], [1.0, 0.0]])
ot_self = np.sum(plan * cost) + np.sum(plan * np.log(plan / reference))
close(ot_self, 0.37988549304172253)
close(ot_self - 0.5 * ot_self - 0.5 * ot_self, 0.0)


# Generative HMM and Poisson-Fisher computations.
close(0.172032 + 0.010368 + 0.009216 + 0.001944, 0.19356)
theta = 2.0
fisher_kernel = lambda a, b: (a - theta) * (b - theta) / theta
assert fisher_kernel(3, 4) > 0 and fisher_kernel(1, 4) < 0 and fisher_kernel(2, 4) == 0


# Signature/global-alignment values and observed convergence direction.
local = exp(-0.5)
close(1 + 3 * local + local**2, 3.1874714203093424)
pde_values = np.array(
    [3.6666666667, 3.5789629630, 3.5637291466, 3.5602864923, 3.5594481010,
     3.5592398866, 3.5591879194, 3.5591749330]
)
limit = 3.5591706047
assert np.all(pde_values > limit)
assert np.all(np.diff(pde_values) < 0)


# Text WMD indefinite similarity spectrum.
wmd = np.array(
    [[0.0, 1.0690, 7.9134], [1.0690, 0.0, 6.9597], [7.9134, 6.9597, 0.0]]
)
assert np.linalg.eigvalsh(-wmd).min() < -11.0

print("Solution companion numerical regression checks passed.")
