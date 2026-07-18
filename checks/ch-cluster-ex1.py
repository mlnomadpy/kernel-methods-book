"""Worked example (ch-cluster, ex1): the normalized-cut spectral relaxation on a
tiny weighted graph, showing that the Fiedler vector's sign splits two clusters.

Graph: two triangles joined by a single weak bridge. Cluster A = {0,1,2} and
cluster B = {3,4,5}. Every intra-cluster pair carries weight 1 (each triangle is
complete); the only inter-cluster edge is the weak bridge {2,3} with weight
eps = 0.1. So the affinity K = W is a 6x6 symmetric matrix, the degree matrix
D = diag(row sums), the unnormalized Laplacian L = D - W, and the symmetric
normalized Laplacian L_sym = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}.

Normalized cut (Shi and Malik 2000): for the split (A, B),
  cut(A,B)   = sum of cross weights = w_{23} = eps,
  vol(A)     = sum of degrees in A,   vol(B) likewise,
  Ncut(A,B)  = cut/vol(A) + cut/vol(B).

Relaxation (von Luxburg 2007): minimizing Ncut over the two-valued indicator f
equals minimizing the generalized Rayleigh quotient f^T L f / f^T D f subject to
Df orthogonal to 1; dropping the two-valued constraint leaves the generalized
eigenproblem L f = lambda D f. Its second-smallest eigenvector (the Fiedler
vector) is the relaxed indicator, and the eigenvalues of L_rw = D^{-1} L equal
those of L_sym. This script prints every number the worked example shows. All of
it is a single 6x6 eigendecomposition, pure numpy.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

n = 6
eps = 0.1
A_nodes = [0, 1, 2]
B_nodes = [3, 4, 5]

# --- build the affinity (weighted adjacency) W -----------------------------
W = np.zeros((n, n))
for i, j in [(0, 1), (0, 2), (1, 2)]:      # triangle A, weight 1
    W[i, j] = W[j, i] = 1.0
for i, j in [(3, 4), (3, 5), (4, 5)]:      # triangle B, weight 1
    W[i, j] = W[j, i] = 1.0
W[2, 3] = W[3, 2] = eps                     # weak bridge
print("affinity W =\n", W)

deg = W.sum(axis=1)
D = np.diag(deg)
print("degrees d_i =", np.round(deg, 4))

# --- volumes, cut, and normalized cut of the true split --------------------
volA = deg[A_nodes].sum()
volB = deg[B_nodes].sum()
volV = deg.sum()
cutAB = W[np.ix_(A_nodes, B_nodes)].sum()
Ncut = cutAB / volA + cutAB / volB
print("vol(A) =", round(float(volA), 4), " vol(B) =", round(float(volB), 4),
      " vol(V) =", round(float(volV), 4))
print("cut(A,B) =", round(float(cutAB), 4))
print("Ncut(A,B) =", round(float(Ncut), 6))

# --- Laplacians ------------------------------------------------------------
L = D - W
Dm12 = np.diag(1.0 / np.sqrt(deg))
Lsym = Dm12 @ L @ Dm12
print("\nunnormalized Laplacian L = D - W =\n", L)
print("symmetric normalized Laplacian L_sym = D^{-1/2} L D^{-1/2} =\n", Lsym)

# eigenpairs of the symmetric normalized Laplacian (ascending)
mu, U = np.linalg.eigh(Lsym)
print("\nL_sym eigenvalues (ascending) =", np.round(mu, 6))

# generalized eigenvalues L f = lambda D f coincide with L_sym eigenvalues;
# generalized eigenvectors are f = D^{-1/2} u.
gen_vecs = Dm12 @ U
print("generalized eigenvalues of (L, D) =", np.round(mu, 6))

# --- the Fiedler vector: second-smallest generalized eigenvector -----------
f = gen_vecs[:, 1].copy()
if f[0] < 0:            # fix a reproducible global sign
    f = -f
f = f / np.abs(f).max()  # scale so the largest entry is 1
print("\nFiedler vector f (2nd generalized eigenvector, scaled) =", np.round(f, 4))
print("sign(f) =", np.sign(f).astype(int))
splitA = [i for i in range(n) if f[i] > 0]
splitB = [i for i in range(n) if f[i] < 0]
print("positive-sign nodes =", splitA, " negative-sign nodes =", splitB)

# also the second-smallest eigenvector of L_sym itself (row-embedding coord)
u2 = U[:, 1].copy()
if u2[0] < 0:
    u2 = -u2
print("L_sym 2nd eigenvector u2 =", np.round(u2, 4))

# --- verify the relaxation identity f^T L f = vol(V) * Ncut ----------------
# two-valued indicator g of von Luxburg: g_i = sqrt(volB/volA) on A, -sqrt(volA/volB) on B
g = np.zeros(n)
g[A_nodes] = np.sqrt(volB / volA)
g[B_nodes] = -np.sqrt(volA / volB)
print("\ntwo-valued indicator g =", np.round(g, 4))
print("g^T D 1 (should be 0) =", round(float(g @ D @ np.ones(n)), 10))
print("g^T D g (should be vol(V)) =", round(float(g @ D @ g), 6))
print("g^T L g =", round(float(g @ L @ g), 6))
print("vol(V) * Ncut =", round(float(volV * Ncut), 6))

# --- Ng-Jordan-Weiss row-normalized embedding (k = 2) ----------------------
# top-2 eigenvectors of the normalized affinity D^{-1/2} W D^{-1/2}
An = Dm12 @ W @ Dm12
val, vec = np.linalg.eigh(An)
X = vec[:, [-1, -2]]                      # two largest eigenvectors
Y = X / np.linalg.norm(X, axis=1, keepdims=True)   # normalize rows to unit length
print("\nNJW normalized-affinity top-2 eigenvalues =", np.round(val[[-1, -2]], 6))
print("NJW row-normalized embedding Y (rows) =\n", np.round(Y, 4))
# rows within a cluster are nearly identical; report the two cluster-mean rows
print("cluster-A mean row =", np.round(Y[A_nodes].mean(axis=0), 4))
print("cluster-B mean row =", np.round(Y[B_nodes].mean(axis=0), 4))
