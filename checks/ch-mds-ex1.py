"""Worked example 1: classical MDS from a distance matrix.

Four points sit at the corners of a 4-by-3 rectangle, so the pairwise
distances are the integers 3, 4, 5 (two 3-4-5 right triangles). We are given
ONLY the distance matrix, not the coordinates. Classical MDS double-centers
the squared-distance matrix into a Gram matrix B = -1/2 J D2 J, eigendecomposes
it, and reads the top-2 embedding off the leading eigenpairs. We then verify
the recovered configuration reproduces every original distance.

Prints every number the worked example displays.
"""
import numpy as np

np.set_printoptions(suppress=True)

# Given: distance matrix among 4 points (corners of a 4x3 rectangle).
D = np.array([
    [0.0, 4.0, 5.0, 3.0],
    [4.0, 0.0, 3.0, 5.0],
    [5.0, 3.0, 0.0, 4.0],
    [3.0, 5.0, 4.0, 0.0],
])
n = D.shape[0]
print("D =\n", D)

D2 = D ** 2
print("D^(2) (squared distances) =\n", D2)

J = np.eye(n) - np.ones((n, n)) / n
print("J = I - (1/n) 11^T =\n", np.round(J, 4))

B = -0.5 * J @ D2 @ J
print("B = -1/2 J D2 J =\n", np.round(B, 4))

# Eigendecomposition of the symmetric B.
w, V = np.linalg.eigh(B)          # ascending
order = np.argsort(w)[::-1]       # descending
w = w[order]
V = V[:, order]
print("eigenvalues of B (descending) =", np.round(w, 4))

k = 2
Lk = np.diag(np.sqrt(np.maximum(w[:k], 0)))
Xk = V[:, :k] @ Lk
print("recovered 2D coordinates (rows = points) =\n", np.round(Xk, 4))

# Verify: distances of the recovered configuration match D.
Drec = np.sqrt(((Xk[:, None, :] - Xk[None, :, :]) ** 2).sum(-1))
print("distances of recovered configuration =\n", np.round(Drec, 4))
print("max abs distance error =", round(float(np.abs(Drec - D).max()), 10))

# Trailing eigenvalues are ~0: the data is exactly 2-dimensional.
print("sum of trailing eigenvalues (k+1..n) =", round(float(w[k:].sum()), 10))
