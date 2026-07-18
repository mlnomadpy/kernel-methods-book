"""Worked example 4: Isomap is classical MDS on GEODESIC distances,
i.e. kernel PCA with the double-centered geodesic Gram matrix.

Five points sit on a circular arc (a curved 1-D manifold). A symmetric
2-nearest-neighbor rule wires them into the path 1-2-3-4-5 that follows the
arc. Isomap (Tenenbaum-de Silva-Langford 2000) replaces Euclidean distance by
the GEODESIC (shortest-path) distance along this graph, then runs classical
MDS: double-center the squared geodesic distances into the kernel
K_iso = -1/2 H Dgeo^2 H and read the embedding off its top eigenpair. The
geodesic kernel unrolls the arc into a straight line (one positive eigenvalue),
which the Euclidean chords cannot do (two positive eigenvalues).

Prints every number the worked example displays.
"""
import numpy as np

np.set_printoptions(suppress=True)

# Five points on a unit-circle arc at 0, 40, 80, 120, 160 degrees.
ang = np.deg2rad([0, 40, 80, 120, 160])
P = np.c_[np.cos(ang), np.sin(ang)]
n = P.shape[0]
print("points =\n", np.round(P, 4))

# Euclidean distances (chords).
Deuc = np.sqrt(((P[:, None] - P[None, :]) ** 2).sum(-1))
print("Euclidean chord distances =\n", np.round(Deuc, 4))

# 2-NN path graph edge lengths (adjacent chords); geodesic via Floyd-Warshall.
INF = 1e9
G = np.full((n, n), INF)
np.fill_diagonal(G, 0.0)
for i in range(n - 1):                     # path edges 1-2-...-5
    G[i, i + 1] = G[i + 1, i] = Deuc[i, i + 1]
Dgeo = G.copy()
for m in range(n):                         # Floyd-Warshall shortest paths
    Dgeo = np.minimum(Dgeo, Dgeo[:, [m]] + Dgeo[[m], :])
print("adjacent chord length c =", round(float(Deuc[0, 1]), 4))
print("geodesic (shortest-path) distances =\n", np.round(Dgeo, 4))

# Classical MDS / kernel PCA on geodesic distances.
H = np.eye(n) - np.ones((n, n)) / n
Kiso = -0.5 * H @ (Dgeo ** 2) @ H
print("K_iso = -1/2 H Dgeo^2 H =\n", np.round(Kiso, 4))
wK, UK = np.linalg.eigh(Kiso)
order = np.argsort(wK)[::-1]
wK = wK[order]
UK = UK[:, order]
print("eigenvalues of K_iso (descending) =", np.round(wK, 4))

# 1-D Isomap embedding = top eigenpair.
y = UK[:, 0] * np.sqrt(max(wK[0], 0.0))
if y[0] > y[-1]:                           # orient increasing along the arc
    y = -y
print("1-D Isomap embedding =", np.round(y, 4))
print("strictly increasing along the arc?", bool(np.all(np.diff(y) > 0)))

# Isomap recovers arc length; centered path positions 0,c,2c,3c,4c.
c = Deuc[0, 1]
arclen = np.arange(n) * c
arclen = arclen - arclen.mean()
print("centered arc-length positions =", np.round(arclen, 4))
print("max abs |Isomap - arc length| =",
      round(float(np.abs(y - arclen).max()), 10))

# Contrast: classical MDS on EUCLIDEAN chords needs 2 dimensions.
Beuc = -0.5 * H @ (Deuc ** 2) @ H
weuc = np.sort(np.linalg.eigvalsh(Beuc))[::-1]
print("eigenvalues of Euclidean double-centering (descending) =",
      np.round(weuc, 4))
