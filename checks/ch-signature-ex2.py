"""Worked example: the Global Alignment (GA) kernel of Cuturi (2011) between two
short real sequences, by the forward dynamic program, cross-checked against a
brute-force sum over all alignments.

Sequences (1-D):   x = (1, 2, 3),   y = (1, 3).
Local similarity:  kappa(a,b) = exp(-(a-b)^2 / (2 sigma^2)),  sigma = 1.

The GA kernel sums, over every alignment pi (a monotone path of matched index
pairs from (1,1) to (n,m) using moves down/right/diagonal), the product of the
local similarities along pi:

    k_GA(x,y) = sum_pi  prod_{(i,j) in pi} kappa(x_i, y_j).

This soft sum (contrast with DTW's single-best min) is computed by the recursion
    M[i,j] = kappa(x_i,y_j) * ( M[i-1,j] + M[i-1,j-1] + M[i,j-1] ),
    M[0,0] = 1,   M[i,0] = M[0,j] = 0  (i,j > 0),   k_GA = M[n,m].
"""

import numpy as np
from math import exp

x = [1.0, 2.0, 3.0]
y = [1.0, 3.0]
sigma = 1.0


def kap(a, b):
    return exp(-(a - b) ** 2 / (2 * sigma ** 2))


n, m = len(x), len(y)

print("=== local similarity matrix kappa(x_i, y_j) ===")
K = np.zeros((n, m))
for i in range(n):
    for j in range(m):
        K[i, j] = kap(x[i], y[j])
        print(f"  kappa(x{i+1}={x[i]:.0f}, y{j+1}={y[j]:.0f}) = {K[i,j]:.6f}")

# ---- forward DP (sum over alignments) --------------------------------------

M = np.zeros((n + 1, m + 1))
M[0, 0] = 1.0
for i in range(1, n + 1):
    for j in range(1, m + 1):
        M[i, j] = kap(x[i - 1], y[j - 1]) * (M[i - 1, j] + M[i - 1, j - 1] + M[i, j - 1])

print("\n=== GA dynamic-programming table M (rows i=0..3, cols j=0..2) ===")
for i in range(n + 1):
    print("  " + "  ".join(f"{M[i,j]:9.6f}" for j in range(m + 1)))
print(f"\nk_GA(x,y) = M[{n},{m}] = {M[n,m]:.6f}")

# ---- brute-force check: enumerate every alignment path ----------------------

def alignments(n, m):
    """All monotone paths of matched pairs from (1,1) to (n,m), moves
    (1,0),(0,1),(1,1). Yields tuples of (i,j) pairs (1-indexed)."""
    def rec(i, j):
        if (i, j) == (n, m):
            yield [(i, j)]
            return
        for di, dj in ((1, 0), (0, 1), (1, 1)):
            ni, nj = i + di, j + dj
            if ni <= n and nj <= m:
                for tail in rec(ni, nj):
                    yield [(i, j)] + tail
    yield from rec(1, 1)


paths = list(alignments(n, m))
brute = 0.0
for pth in paths:
    prod = 1.0
    for (i, j) in pth:
        prod *= K[i - 1, j - 1]
    brute += prod
print(f"\nnumber of alignments enumerated = {len(paths)}")
print(f"brute-force sum over alignments = {brute:.6f}")
print(f"matches DP value?                 {np.isclose(brute, M[n,m])}")

# count of alignments (kappa == 1 everywhere) is the Delannoy-type path count
Mone = np.zeros((n + 1, m + 1)); Mone[0, 0] = 1
for i in range(1, n + 1):
    for j in range(1, m + 1):
        Mone[i, j] = Mone[i-1, j] + Mone[i-1, j-1] + Mone[i, j-1]
print(f"path count (all kappa=1)          = {Mone[n,m]:.0f}")

# ---- Cuturi's PD-guaranteeing local kernel: kappa/(1+kappa) = (1/2) Gaussian --
print("\n=== check Cuturi's identity kappa/(1+kappa) = (1/2) e^{-d^2/2sigma^2} ===")
for a, b in [(1, 1), (1, 3), (2, 3)]:
    g = exp(-(a - b) ** 2 / (2 * sigma ** 2))
    ka = g / (2 - g)                      # kappa built so that kappa/(1+kappa)=g/2
    print(f"  d^2={(a-b)**2}:  kappa/(1+kappa) = {ka/(1+ka):.6f}   (1/2)Gaussian = {g/2:.6f}")
