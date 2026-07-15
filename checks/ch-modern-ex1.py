"""Worked example: double descent for ridgeless (minimum-norm) least squares in
the isotropic linear model of Hastie, Montanari, Rosasco, and Tibshirani (2019).

Model.  x ~ N(0, I_p),  y = x^T beta + eps,  eps ~ N(0, sigma^2),
with ||beta||^2 = r^2.  From n samples (X, y) we form the minimum-norm least
squares estimator  beta_hat = pinv(X) y  (numpy lstsq returns the min-norm
solution).  The excess prediction risk of a linear predictor is

    Risk(beta_hat) = E_x[(x^T beta_hat - x^T beta)^2] = ||beta_hat - beta||^2 .

We sweep the aspect ratio gamma = p / n across the interpolation threshold
gamma = 1 and average over deterministic seeds.  We compare the simulation to
the EXACT finite-sample risk, which follows from the inverse-Wishart mean
E[W^{-1}] = I/(m - d - 1) for W ~ Wishart_d(I, m):

    gamma < 1 :  Risk = sigma^2 * p / (n - p - 1)                        (pure variance)
    gamma > 1 :  Risk = r^2 * (p - n)/p + sigma^2 * n / (p - n - 1)      (bias + variance)

Both branches diverge as p -> n: the double-descent peak.  As n, p -> infinity
with p/n -> gamma these reduce to the Marchenko-Pastur limits of Hastie,
Montanari, Rosset, and Tibshirani (2019):
    gamma < 1 : sigma^2 gamma/(1-gamma);   gamma > 1 : r^2(1-1/gamma) + sigma^2/(gamma-1).
"""
import numpy as np

n = 40                    # sample size (fixed)
sigma2 = 0.25             # noise variance  (sigma = 0.5)
r2 = 1.0                  # signal energy ||beta||^2
seeds = list(range(200))  # deterministic averaging

sigma = np.sqrt(sigma2)
ps = [4, 12, 20, 28, 36, 38, 44, 52, 60, 80, 120, 200, 400]

def exact_risk(p):
    if p < n - 1:                                   # underparameterized
        return sigma2 * p / (n - p - 1)
    if p > n + 1:                                   # overparameterized (min-norm)
        return r2 * (p - n) / p + sigma2 * n / (p - n - 1)
    return float("inf")                             # blows up at p = n

print(f"{'p':>5} {'gamma':>7} {'sim risk':>12} {'exact theory':>14}")
rows = []
for p in ps:
    beta = np.ones(p) / np.sqrt(p) * np.sqrt(r2)     # ||beta||^2 = r2
    risks = []
    for s in seeds:
        rng = np.random.default_rng(7000 + s)
        X = rng.standard_normal((n, p))
        y = X @ beta + sigma * rng.standard_normal(n)
        beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)   # min-norm LS
        risks.append(float(np.sum((beta_hat - beta) ** 2)))
    gamma = p / n
    sim = float(np.mean(risks))
    exact = exact_risk(p)
    rows.append((p, gamma, sim, exact))
    ex_s = f"{exact:14.4f}" if np.isfinite(exact) else f"{'inf':>14}"
    print(f"{p:>5} {gamma:>7.2f} {sim:>12.4f} {ex_s}")

peak = max(rows, key=lambda t: t[2])
print(f"\npeak of simulated risk at p = {peak[0]} (gamma = {peak[1]:.2f}), "
      f"risk = {peak[2]:.4f}")
print(f"underparameterized p = 20 (gamma = 0.5): sim = {rows[2][2]:.4f}, "
      f"exact sigma^2 p/(n-p-1) = {rows[2][3]:.4f}")
print(f"overparam minimum near p = 80 (gamma = 2.0): sim = {rows[9][2]:.4f}, "
      f"exact = {rows[9][3]:.4f}")
print(f"far overparameterized p = 400 (gamma = 10): sim = {rows[-1][2]:.4f}, "
      f"exact = {rows[-1][3]:.4f}")
