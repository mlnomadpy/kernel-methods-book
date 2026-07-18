"""Worked example 3: Laplacian eigenmaps ARE kernel PCA on the
pseudo-inverse graph Laplacian.

Five sample points: a dense clump of three and a short tail of two. A
symmetric k-nearest-neighbor rule wires the clump into a triangle {1,2,3} and
attaches the tail 3-4-5, joined by the bridge edge (3,4); this is a lollipop
(tadpole) graph. With simple 0/1 weights the graph Laplacian is L = D - W.
Belkin-Niyogi Laplacian eigenmaps embed the graph using the eigenvectors of L
with the SMALLEST nonzero eigenvalues. Ham-Lee-Mika-Scholkopf (2004) identify
this with kernel PCA using the kernel K = L^+ (the Moore-Penrose
pseudo-inverse): the bottom nonzero eigenvectors of L are exactly the TOP
eigenvectors of L^+, with reciprocal eigenvalues.

Prints every number the worked example displays.
"""
import numpy as np

np.set_printoptions(suppress=True)

# k-NN adjacency: triangle {1,2,3} bridged (3,4) to a tail 4-5 (lollipop).
W = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0],
], dtype=float)
n = W.shape[0]
deg = W.sum(1)
D = np.diag(deg)
L = D - W
print("degrees =", deg.astype(int))
print("L = D - W =\n", L.astype(int))

# --- Laplacian eigenmaps: bottom nonzero eigenvectors of L ---
wL, UL = np.linalg.eigh(L)                 # ascending
print("eigenvalues of L (ascending) =", np.round(wL, 4))
k = 2
tol = 1e-8
nz = np.where(wL > tol)[0]                 # indices of nonzero eigenvalues
idx_lap = nz[:k]                           # two smallest nonzero
Ylap = UL[:, idx_lap]
print("Laplacian-eigenmap eigenvalues used =", np.round(wL[idx_lap], 4))
print("Laplacian-eigenmap embedding (rows = nodes) =\n", np.round(Ylap, 4))
print("Fiedler vector u2 =", np.round(UL[:, idx_lap[0]], 4))

# --- Kernel K = pseudo-inverse of L ---
K = np.linalg.pinv(L)
print("K = L^+ (pseudo-inverse Laplacian kernel) =\n", np.round(K, 4))

# K is already centered: rows sum to zero, so J K J = K.
J = np.eye(n) - np.ones((n, n)) / n
print("max abs row sum of K =", round(float(np.abs(K.sum(1)).max()), 10))
print("max abs |JKJ - K| =", round(float(np.abs(J @ K @ J - K).max()), 10))

# Rebuild K from L's OWN eigenbasis: same eigenvectors, reciprocal eigenvalues.
Krec = np.zeros((n, n))
for i in nz:
    Krec += (1.0 / wL[i]) * np.outer(UL[:, i], UL[:, i])
print("max abs |pinv(L) - sum 1/lambda u u^T| =",
      round(float(np.abs(K - Krec).max()), 10))

# --- Kernel PCA on K: top eigenvectors ---
wK, UK = np.linalg.eigh(K)
order = np.argsort(wK)[::-1]
wK = wK[order]
UK = UK[:, order]
print("eigenvalues of K (descending) =", np.round(wK, 4))
print("reciprocals 1/lambda of L's nonzero eigenvalues =",
      np.round(1.0 / wL[nz], 4))

# top-k eigenvectors of K equal the bottom-k-nonzero eigenvectors of L
UKk = UK[:, :k]
signs = np.sign((Ylap * UKk).sum(0))
UKk_al = UKk * signs
print("per-axis sign alignment =", signs.astype(int))
print("max abs |L eigvecs - K eigvecs| (aligned) =",
      round(float(np.abs(Ylap - UKk_al).max()), 10))

# kernel-PCA embedding = V_k Lambda_k^{1/2}
Ykpca = UKk_al @ np.diag(np.sqrt(wK[:k]))
print("kernel-PCA embedding V_k Lambda_k^{1/2} (rows = nodes) =\n",
      np.round(Ykpca, 4))
print("per-axis scale sqrt(lambda_K) =", np.round(np.sqrt(wK[:k]), 4))
