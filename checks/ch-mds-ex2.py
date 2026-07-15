"""Worked example 2: classical MDS on Euclidean distances equals PCA.

Five points in the plane. Route A: center the data and run PCA (eigendecompose
the centered Gram matrix / covariance, project onto the top-2 axes). Route B:
form the Euclidean distance matrix, throw the coordinates away, and run
classical MDS (double-center, eigendecompose). The two 2D embeddings coincide
up to a per-axis sign flip, which is the content of the MDS = PCA equivalence.

Prints every number the worked example displays.
"""
import numpy as np

np.set_printoptions(suppress=True)

# Five points in R^2 (the raw coordinates).
X = np.array([
    [1.0, 0.0],
    [2.0, 1.0],
    [3.0, 0.0],
    [0.0, 2.0],
    [1.0, 3.0],
])
n = X.shape[0]
print("X =\n", X)

# ---- Route A: PCA on centered data ----
Xc = X - X.mean(axis=0)
print("centered X =\n", np.round(Xc, 4))
G = Xc @ Xc.T                      # centered Gram (linear kernel), n x n
print("centered Gram G = Xc Xc^T =\n", np.round(G, 4))
wg, Vg = np.linalg.eigh(G)
order = np.argsort(wg)[::-1]
wg = wg[order]; Vg = Vg[:, order]
print("eigenvalues of G (descending) =", np.round(wg, 4))
k = 2
Ypca = Vg[:, :k] @ np.diag(np.sqrt(np.maximum(wg[:k], 0)))
print("PCA embedding (rows = points) =\n", np.round(Ypca, 4))

# ---- Route B: classical MDS from Euclidean distances ----
D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
print("Euclidean distance matrix D =\n", np.round(D, 4))
D2 = D ** 2
J = np.eye(n) - np.ones((n, n)) / n
B = -0.5 * J @ D2 @ J
print("B = -1/2 J D2 J =\n", np.round(B, 4))
print("max abs |B - G| =", round(float(np.abs(B - G).max()), 10))
wb, Vb = np.linalg.eigh(B)
order = np.argsort(wb)[::-1]
wb = wb[order]; Vb = Vb[:, order]
print("eigenvalues of B (descending) =", np.round(wb, 4))
Ymds = Vb[:, :k] @ np.diag(np.sqrt(np.maximum(wb[:k], 0)))
print("MDS embedding (rows = points) =\n", np.round(Ymds, 4))

# ---- Compare: equal up to a per-axis sign ----
signs = np.sign((Ypca * Ymds).sum(axis=0))
Ymds_aligned = Ymds * signs
print("per-axis sign alignment =", signs.astype(int))
print("max abs difference after sign alignment =",
      round(float(np.abs(Ypca - Ymds_aligned).max()), 10))
