"""Companion numeric check: plain dynamic time warping (DTW) does not yield a
positive-definite kernel, whereas the Global Alignment kernel does.

We compute the DTW discrepancy between several short 1-D sequences with the
squared-difference local cost and the standard min-plus recursion
    D[i,j] = (x_i - y_j)^2 + min(D[i-1,j], D[i-1,j-1], D[i,j-1]),
form the "DTW kernel" Gram matrix G_ab = exp(-gamma * DTW(s_a, s_b)), and read
off its eigenvalues.  A negative eigenvalue certifies that exp(-gamma DTW) is
not positive definite (Cuturi 2011).  For contrast we build the GA Gram matrix
on the same sequences and confirm it is positive semidefinite.
"""

import numpy as np
from math import exp, inf


def dtw(x, y):
    n, m = len(x), len(y)
    D = np.full((n + 1, m + 1), inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (x[i - 1] - y[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i - 1, j - 1], D[i, j - 1])
    return D[n, m]


def ga(x, y, sigma):
    n, m = len(x), len(y)
    M = np.zeros((n + 1, m + 1))
    M[0, 0] = 1.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            g = exp(-(x[i - 1] - y[j - 1]) ** 2 / (2 * sigma ** 2))
            kap = g / (2 - g)                      # Cuturi's PD local kernel
            M[i, j] = kap * (M[i - 1, j] + M[i - 1, j - 1] + M[i, j - 1])
    return M[n, m]


seqs = [
    [0.0, 3.0, 0.0],
    [0.0, 0.0, 3.0],
    [3.0, 0.0, 0.0],
    [0.0, 3.0, 3.0],
    [3.0, 3.0, 0.0],
]

N = len(seqs)
DT = np.zeros((N, N))
for a in range(N):
    for b in range(N):
        DT[a, b] = dtw(seqs[a], seqs[b])

print("=== pairwise DTW discrepancy (squared-difference cost) ===")
for a in range(N):
    print("  " + "  ".join(f"{DT[a,b]:5.1f}" for b in range(N)))

# the degeneracy at the heart of it: distinct sequences at DTW-distance 0
print("\n=== DTW collapses distinct sequences ===")
print(f"  s1 = {seqs[1]},  s3 = {seqs[3]}  are different, yet DTW(s1,s3) = {DT[1,3]:.1f}")
print(f"  but DTW(s1,s2) = {DT[1,2]:.1f}  while  DTW(s3,s2) = {DT[3,2]:.1f}  (they disagree)")

# scan gamma; report the DTW-kernel Gram and its spectrum
print("\n=== eigenvalues of the DTW kernel Gram  G = exp(-gamma * DTW) ===")
for gamma in [0.1, 0.2, 0.3, 0.5]:
    G = np.exp(-gamma * DT)
    ev = np.linalg.eigvalsh(G)
    print(f"  gamma={gamma:.1f}   min eig = {ev.min():+.5f}   eigs = "
          + ", ".join(f"{e:+.4f}" for e in ev))

gamma = 0.1
G = np.exp(-gamma * DT)
ev = np.linalg.eigvalsh(G)
print(f"\nAt gamma={gamma}: DTW-kernel Gram has minimum eigenvalue {ev.min():+.5f} < 0,")
print("so exp(-gamma DTW) is NOT positive definite.")

# a tiny 3x3 witness: the two DTW-0 sequences s1,s3 and a third s2
sub = [1, 3, 2]
G3 = np.exp(-gamma * DT[np.ix_(sub, sub)])
ev3 = np.linalg.eigvalsh(G3)
print("\n=== 3x3 witness on sequences s1, s3, s2 ===")
for r in range(3):
    print("  " + "  ".join(f"{G3[r,c]:7.5f}" for c in range(3)))
print(f"det = {np.linalg.det(G3):+.6f} = -(a-b)^2 with a=exp(-18g), b=exp(-27g)")
print(f"eigenvalues = " + ", ".join(f"{e:+.5f}" for e in ev3)
      + f"   (min {ev3.min():+.5f})")

# contrast: GA Gram on the same sequences is PSD
print("\n=== Global Alignment Gram on the same sequences (sigma=1) ===")
GA = np.zeros((N, N))
for a in range(N):
    for b in range(N):
        GA[a, b] = ga(seqs[a], seqs[b], sigma=1.0)
evg = np.linalg.eigvalsh(GA)
print("  min eig of GA Gram =", f"{evg.min():+.6f}", " -> PSD:", evg.min() > -1e-9)
