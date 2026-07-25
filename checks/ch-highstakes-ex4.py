"""ch-highstakes-ex4: GP uncertainty sampling reaches accuracy in fewer points.

The calibrated posterior variance is what turns a fixed compute budget into an
adaptive one: query the next expensive evaluation where the GP is least sure. On a
1-D toy potential-energy surface we compare active learning (place the next sample
at argmax posterior variance) with random sampling, and count how many points each
needs to reach a target RMSE.
"""
import numpy as np

rng = np.random.default_rng(13)

def f(x):                                       # a smooth double-well energy
    return (x ** 2 - 1) ** 2 + 0.3 * np.sin(4 * x)

def rbf(A, B, ell):
    A = np.asarray(A).reshape(-1, 1); B = np.asarray(B).reshape(-1, 1)
    d = A - B.T
    return np.exp(-(d ** 2) / (2 * ell ** 2))

grid = np.linspace(-2, 2, 400)
ytrue = f(grid)
ell, noise = 0.35, 1e-4

def gp_fit_predict(xs, ys):
    K = rbf(xs, xs, ell) + noise * np.eye(len(xs))
    L = np.linalg.cholesky(K)
    a = np.linalg.solve(L.T, np.linalg.solve(L, ys))
    ks = rbf(grid, xs, ell)
    mu = ks @ a
    v = np.linalg.solve(L, ks.T)
    var = 1.0 - np.sum(v ** 2, axis=0)
    return mu, np.sqrt(np.maximum(var, 0))

def rmse(mu): return float(np.sqrt(np.mean((mu - ytrue) ** 2)))

target = 0.05
budget = 25

# active learning: start with 3 points, add at argmax variance
def run_active():
    xs = np.array([-2.0, 0.0, 2.0]); ys = f(xs)
    for step in range(budget):
        mu, sd = gp_fit_predict(xs, ys)
        if rmse(mu) < target:
            return len(xs)
        xnext = grid[np.argmax(sd)]
        xs = np.append(xs, xnext); ys = np.append(ys, f(xnext))
    return len(xs)

def run_random(seed):
    r = np.random.default_rng(seed)
    xs = np.array([-2.0, 0.0, 2.0]); ys = f(xs)
    for step in range(budget):
        mu, _ = gp_fit_predict(xs, ys)
        if rmse(mu) < target:
            return len(xs)
        xnext = r.uniform(-2, 2)
        xs = np.append(xs, xnext); ys = np.append(ys, f(xnext))
    return len(xs)

n_active = run_active()
n_random = np.mean([run_random(s) for s in range(30)])

# Misspecification witness: train only on the left of a narrow, unmodelled bump.
# The stationary GP variance is small near an observed point although the error is large.
xs_miss = np.array([-2.0, -1.0, 0.0, 0.8, 1.0, 1.2, 2.0])
ys_miss = f(xs_miss)
mu_miss, sd_miss = gp_fit_predict(xs_miss, ys_miss)
truth_miss = ytrue + 1.5 * np.exp(-((grid - 1.05) / 0.035) ** 2)
i_bump = int(np.argmin(np.abs(grid - 1.05)))
miss_error = abs(mu_miss[i_bump] - truth_miss[i_bump])
miss_sd = sd_miss[i_bump]
print(f"GP active learning vs random on a 1-D PES (RBF ell={ell}, target RMSE={target})")
print(f"  points to target, uncertainty sampling : {n_active}")
print(f"  points to target, random (mean of 30)  : {n_random:.1f}")
print(f"  fraction of the random budget          : {n_active / n_random:.2f}")
print(f"  misspecified bump error / model sd      : {miss_error:.3f} / {miss_sd:.3f}")

assert n_active == 19
assert abs(n_random - 26.6) < 0.05
assert miss_error > 1.0 and miss_sd < 0.2
assert np.isfinite([n_random, miss_error, miss_sd]).all()
