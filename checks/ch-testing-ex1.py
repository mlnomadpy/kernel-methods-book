"""Worked example 1: an exact permutation MMD two-sample test on two tiny
samples, with the median-heuristic Gaussian kernel.

Samples (1-D):
  X = [0, 1, 2, 3]   ~ P     (n = 4)
  Y = [7, 8, 9, 10]  ~ Q     (m = 4)

Kernel: Gaussian k(x,x') = exp(-(x-x')^2 / (2 sigma^2)) with sigma set by the
median heuristic, sigma = median{ |z_i - z_j| : i < j } over the pooled sample.

We compute:
  - the biased V-statistic   MMD^2_V (diagonal included),
  - the unbiased U-statistic MMD^2_U (diagonal dropped),
  - the exact permutation null: enumerate all C(8,4) = 70 ways to relabel the
    pooled points into two groups of 4, recompute MMD^2_U for each, and report
    the p-value = (# permuted statistics >= observed) / 70.
Every number printed here is embedded in the worked example.
"""
import numpy as np
from itertools import combinations

X = np.array([0.0, 1.0, 2.0, 3.0])
Y = np.array([7.0, 8.0, 9.0, 10.0])
pool = np.concatenate([X, Y])
n = len(X)
N = len(pool)

# --- median-heuristic bandwidth over the pooled sample -----------------------
dists = [abs(pool[i] - pool[j]) for i, j in combinations(range(N), 2)]
sigma = float(np.median(dists))
print(f"pooled pairwise |z_i - z_j| (28 values), median sigma = {sigma:.4f}")

def kmat(a, b):
    d = a[:, None] - b[None, :]
    return np.exp(-(d ** 2) / (2 * sigma ** 2))

def mmd2_biased(a, b):
    na, nb = len(a), len(b)
    Kxx, Kyy, Kxy = kmat(a, a), kmat(b, b), kmat(a, b)
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

def mmd2_unbiased(a, b):
    na, nb = len(a), len(b)
    Kxx, Kyy, Kxy = kmat(a, a), kmat(b, b), kmat(a, b)
    sxx = (Kxx.sum() - np.trace(Kxx)) / (na * (na - 1))
    syy = (Kyy.sum() - np.trace(Kyy)) / (nb * (nb - 1))
    sxy = Kxy.mean()
    return sxx + syy - 2 * sxy

# --- block sums that make up the statistic -----------------------------------
Kxx, Kyy, Kxy = kmat(X, X), kmat(Y, Y), kmat(X, Y)
print("\nwithin-X kernel matrix K(x_i,x_j) =")
print(np.round(Kxx, 4))
print("cross kernel matrix K(x_i,y_j) =")
print(np.round(Kxy, 6))
sxx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
syy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
sxy = Kxy.mean()
print(f"\noff-diagonal within-X average  = {sxx:.6f}")
print(f"off-diagonal within-Y average  = {syy:.6f}")
print(f"cross average                  = {sxy:.6e}")

V = mmd2_biased(X, Y)
U = mmd2_unbiased(X, Y)
print(f"\nbiased   V-statistic MMD^2_V = {V:.6f}")
print(f"unbiased U-statistic MMD^2_U = {U:.6f}")

# --- exact permutation null ---------------------------------------------------
T0 = U
perm_stats = []
for idx in combinations(range(N), n):
    idx = list(idx)
    comp = [i for i in range(N) if i not in idx]
    A, B = pool[idx], pool[comp]
    perm_stats.append(mmd2_unbiased(A, B))
perm_stats = np.array(perm_stats)
count = int(np.sum(perm_stats >= T0 - 1e-12))
pval = count / len(perm_stats)
print(f"\nexact permutation null over C(8,4) = {len(perm_stats)} relabelings")
print(f"null mean = {perm_stats.mean():.6f}, null std = {perm_stats.std():.6f}")
print(f"null max  = {perm_stats.max():.6f}  (observed T0 = {T0:.6f})")
print(f"# relabelings with T >= T0 : {count}")
print(f"p-value = {count}/{len(perm_stats)} = {pval:.4f}")
print(f"decision at alpha=0.05 : {'REJECT H0' if pval <= 0.05 else 'fail to reject'}")
