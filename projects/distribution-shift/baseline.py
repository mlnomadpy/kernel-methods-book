"""Distribution Shift Detection -- reference baselines.

Two detectors scored by AUC over the test pairs: a mean-difference statistic (the
weak baseline, blind to same-mean shape changes) and the unbiased MMD^2 with an
RBF kernel at the median-heuristic bandwidth (catches all three shift types). The
book's Success bar quotes these AUCs. numpy only.
"""
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))
N_PER, D = 60, 4

def load(fn, labeled):
    arr = np.loadtxt(os.path.join(HERE, fn), delimiter=",", skiprows=1)
    ids = arr[:, 0].astype(int)
    if labeled:
        X, y = arr[:, 1:-1], arr[:, -1].astype(int)
    else:
        X, y = arr[:, 1:], None
    return ids, X, y

tids, Xte, _ = load("test.csv", False)
sol = {}
with open(os.path.join(HERE, "solution.csv")) as f:
    next(f)
    for line in f:
        i, y, u = line.strip().split(","); sol[int(i)] = int(y)
yte = np.array([sol[i] for i in tids])

def split_pair(row):
    half = N_PER * D
    return row[:half].reshape(N_PER, D), row[half:].reshape(N_PER, D)

def mmd2(A, B):
    def k(X, Y, g):
        d2 = np.sum(X * X, 1)[:, None] + np.sum(Y * Y, 1)[None, :] - 2 * X @ Y.T
        return np.exp(-g * d2)
    Z = np.vstack([A, B])
    d2 = np.sum(Z * Z, 1)[:, None] + np.sum(Z * Z, 1)[None, :] - 2 * Z @ Z.T
    med = np.median(d2[d2 > 0]); g = 1.0 / med
    n = len(A)
    Kaa, Kbb, Kab = k(A, A, g), k(B, B, g), k(A, B, g)
    np.fill_diagonal(Kaa, 0); np.fill_diagonal(Kbb, 0)
    return Kaa.sum() / (n * (n - 1)) + Kbb.sum() / (n * (n - 1)) - 2 * Kab.mean()

def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1; npos, nneg = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)

mean_score = np.zeros(len(Xte)); mmd_score = np.zeros(len(Xte))
for i, row in enumerate(Xte):
    A, B = split_pair(row)
    mean_score[i] = np.sum((A.mean(0) - B.mean(0)) ** 2)
    mmd_score[i] = mmd2(A, B)

print("Distribution Shift Detection -- reference baselines (AUC over test pairs)")
print(f"  mean-difference statistic (weak baseline) : AUC = {auc(mean_score, yte):.3f}")
print(f"  MMD^2, RBF median bandwidth               : AUC = {auc(mmd_score, yte):.3f}   <- leaderboard metric")
