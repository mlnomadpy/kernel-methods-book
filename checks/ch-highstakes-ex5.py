"""Grouped splitting and coverage for a spatial retrieval.

This deterministic simulation is a leakage witness, not a satellite benchmark.
Each region has an offset. Random pixel splitting lets a region-indicator kernel
recover test-region offsets; leave-one-region-out deployment cannot.
"""
import numpy as np

rng = np.random.default_rng(29)
n_regions, per_region = 8, 40
region = np.repeat(np.arange(n_regions), per_region)
x = rng.uniform(-1.0, 1.0, n_regions * per_region)
offset = np.array([-1.4, -0.9, -0.5, -0.1, 0.2, 0.6, 1.0, 1.5])
y = 2.0 * x + offset[region] + 0.12 * rng.standard_normal(x.size)

def design(indices, include_region=True):
    cols = [np.ones(len(indices)), x[indices]]
    if include_region:
        cols.extend((region[indices] == g).astype(float) for g in range(n_regions))
    return np.column_stack(cols)

def fit_predict(train, test, include_region=True):
    xt = design(train, include_region)
    coef = np.linalg.solve(xt.T @ xt + 1e-8 * np.eye(xt.shape[1]), xt.T @ y[train])
    pred_train = xt @ coef
    pred_test = design(test, include_region) @ coef
    sigma = np.sqrt(np.mean((y[train] - pred_train) ** 2))
    rmse = np.sqrt(np.mean((y[test] - pred_test) ** 2))
    coverage = np.mean(np.abs(y[test] - pred_test) <= 1.96 * sigma)
    return float(rmse), float(coverage), float(sigma)

perm = rng.permutation(y.size)
random_test = perm[: y.size // 4]
random_train = perm[y.size // 4 :]
group_test = np.flatnonzero(region == n_regions - 1)
group_train = np.flatnonzero(region != n_regions - 1)

random_rmse, random_cov, sigma = fit_predict(random_train, random_test)
group_rmse, group_cov, _ = fit_predict(group_train, group_test)

print("spatial split leakage fixture")
print(f"  random-pixel RMSE / coverage : {random_rmse:.3f} / {random_cov:.3f}")
print(f"  held-region RMSE / coverage  : {group_rmse:.3f} / {group_cov:.3f}")
print(f"  leaked residual sigma        : {sigma:.3f}")

assert random_rmse < 0.15
assert random_cov > 0.90
assert group_rmse > 1.0
assert group_cov < 0.10
assert np.isfinite([random_rmse, random_cov, group_rmse, group_cov, sigma]).all()
