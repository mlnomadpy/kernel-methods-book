"""Worked example 2: the bandwidth controls test power, and a wider vs
narrower bandwidth changes the test outcome.

Same exact permutation MMD test as Example 1, on two small overlapping samples
with a moderate mean shift:
  X = [-3, -2, -1, 0, 1] ~ P   (n = 5)
  Y = [ 0,  1,  2, 3, 4] ~ Q   (m = 5)      (they overlap on [0, 1])

Gaussian kernel k(x,x') = exp(-(x-x')^2 / (2 sigma^2)). We sweep the bandwidth
and, for each, compute the unbiased U-statistic, the exact permutation null over
all C(10,5)=252 relabelings, its standard deviation, the power proxy
t(sigma) = MMD^2_U / std_null (the signal-to-noise ratio that governs power),
and the exact p-value. A too-narrow bandwidth washes the signal into noise and
fails to reject; the median-heuristic bandwidth resolves the shift and rejects.
"""
import numpy as np
from itertools import combinations

X = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])
Y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
pool = np.concatenate([X, Y])
n = len(X)
N = len(pool)

dists = [abs(pool[i] - pool[j]) for i, j in combinations(range(N), 2)]
sigma_med = float(np.median(dists))
print(f"mean(X)={X.mean():.2f} mean(Y)={Y.mean():.2f} (shift 3, overlap on [0,1])")
print(f"median-heuristic sigma = median of {len(dists)} pairwise distances = {sigma_med:.4f}")

def mmd2_unbiased(a, b, s):
    na, nb = len(a), len(b)
    Kxx = np.exp(-((a[:, None] - a[None, :]) ** 2) / (2 * s ** 2))
    Kyy = np.exp(-((b[:, None] - b[None, :]) ** 2) / (2 * s ** 2))
    Kxy = np.exp(-((a[:, None] - b[None, :]) ** 2) / (2 * s ** 2))
    sxx = (Kxx.sum() - np.trace(Kxx)) / (na * (na - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (nb * (nb - 1))
    return sxx + syy - 2 * Kxy.mean()

def exact_test(s):
    T0 = mmd2_unbiased(X, Y, s)
    stats = []
    for idx in combinations(range(N), n):
        idx = list(idx)
        comp = [i for i in range(N) if i not in idx]
        stats.append(mmd2_unbiased(pool[idx], pool[comp], s))
    stats = np.array(stats)
    count = int(np.sum(stats >= T0 - 1e-12))
    return T0, stats.std(), count, count / len(stats), len(stats)

print("\n sigma   MMD^2_U    std_null   t=MMD^2/std   p-value    decision(0.05)")
for s in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0]:
    T0, sd, count, p, B = exact_test(s)
    t = T0 / sd if sd > 0 else float("nan")
    dec = "REJECT" if p <= 0.05 else "fail"
    marker = "  <- median heuristic" if abs(s - sigma_med) < 1e-9 else ""
    print(f" {s:5.2f} {T0:9.5f}  {sd:9.5f}   {t:9.4f}   {p:.4f} ({count}/{B})  {dec}{marker}")

print("\n--- the two bandwidths quoted in the worked example ---")
for label, s in [("narrow", 0.5), ("median-heuristic", 2.0)]:
    T0, sd, count, p, B = exact_test(s)
    print(f"{label:16s} sigma={s}: MMD^2_U={T0:.5f}, t={T0/sd:.4f}, "
          f"p={count}/{B}={p:.4f}, {'REJECT' if p <= 0.05 else 'fail to reject'} at 0.05")
