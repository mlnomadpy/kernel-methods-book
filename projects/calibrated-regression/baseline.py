"""Calibrated Regression Challenge -- reference baseline.

Kernel ridge regression (RBF, Nystrom-approximated so it runs in seconds on
6000 points) plus split conformal for a certified 90% interval. Prints the
leaderboard metric (mean Winkler interval score at 90%), the empirical coverage,
and the RMSE. These are the numbers a submission must beat, and they are what the
book's Success bar quotes.

The metric (lower is better): for a 90% interval [l, u] and truth y,
  score = (u - l) + (2/alpha) (l - y) 1[y<l] + (2/alpha) (y - u) 1[y>u], alpha=0.1.
"""
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.10

tr = np.loadtxt(os.path.join(HERE, "train.csv"), delimiter=",", skiprows=1)
X, y = tr[:, :-1], tr[:, -1]
rng = np.random.default_rng(0)

# standardize
mu, sd = X.mean(0), X.std(0)
Xs = (X - mu) / sd

# split: fit / calibrate / test
n = len(Xs)
perm = rng.permutation(n)
i_fit, i_cal, i_te = perm[:n // 2], perm[n // 2:3 * n // 4], perm[3 * n // 4:]

# Nystrom RBF features for a fast KRR fit
M = 300
land = Xs[rng.choice(i_fit, M, replace=False)]
gamma = 1.0 / (2 * np.median(np.sum((Xs[rng.choice(n, 500)][:, None] - land[None, :]) ** 2, -1)))
def feat(A):
    d2 = np.sum(A * A, 1)[:, None] + np.sum(land * land, 1)[None, :] - 2 * A @ land.T
    return np.exp(-gamma * d2)

Ff = feat(Xs[i_fit]); Fc = feat(Xs[i_cal]); Ft = feat(Xs[i_te])
lam = 1e-2
W = np.linalg.solve(Ff.T @ Ff + lam * np.eye(M), Ff.T @ y[i_fit])
def predict(F): return F @ W

pred = predict(Ft); yt = y[i_te]
rmse = np.sqrt(np.mean((pred - yt) ** 2))

def winkler(lo, hi):
    return np.mean((hi - lo) + (2 / ALPHA) * (lo - yt) * (yt < lo) + (2 / ALPHA) * (yt - hi) * (yt > hi))

# (1) provided baseline: GLOBAL split conformal, one constant width q
res = np.abs(y[i_cal] - predict(Fc))
q = np.sort(res)[int(np.ceil((len(res) + 1) * (1 - ALPHA))) - 1]
g_lo, g_hi = pred - q, pred + q
g_cover, g_score = np.mean((yt >= g_lo) & (yt <= g_hi)), winkler(g_lo, g_hi)

# (2) target: NORMALIZED conformal -- a second KRR predicts the residual scale,
# so the band width adapts to the local noise. This is the move that wins.
sfit = np.abs(y[i_fit] - predict(Ff))
Ws = np.linalg.solve(Ff.T @ Ff + lam * np.eye(M), Ff.T @ sfit)
def scale(F): return np.maximum(F @ Ws, 0.05)
ncal = np.abs(y[i_cal] - predict(Fc)) / scale(Fc)
qn = np.sort(ncal)[int(np.ceil((len(ncal) + 1) * (1 - ALPHA))) - 1]
w = qn * scale(Ft)
n_lo, n_hi = pred - w, pred + w
n_cover, n_score = np.mean((yt >= n_lo) & (yt <= n_hi)), winkler(n_lo, n_hi)

print("Calibrated Regression -- reference baselines (Nystrom KRR)")
print(f"  RMSE                                 : {rmse:.3f}")
print(f"  GLOBAL conformal  score {g_score:.3f}  coverage {g_cover:.3f}   <- provided baseline (leaderboard metric, lower better)")
print(f"  NORMALIZED conformal score {n_score:.3f}  coverage {n_cover:.3f}   <- locally-adaptive target to beat it")
