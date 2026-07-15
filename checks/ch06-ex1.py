"""Worked example 1: kernel PCA on five 2-D points with an RBF kernel.

Five points fall into two visible groups: a tight cluster near the origin
{(0,0),(1,0),(0,1)} and a separated pair {(5,4),(6,4)}. We run kernel PCA
with the Gaussian (RBF) kernel k(x,y)=exp(-||x-y||^2 / (2 sigma^2)), sigma=2:

  1. form the 5x5 Gram matrix K,
  2. double-center it into K_c = J K J with J = I - (1/5) 11^T,
  3. eigendecompose K_c to get eigenvalues Delta_i and eigenvectors u_i,
  4. normalize alpha_i = u_i / sqrt(Delta_i) so that ||f_i||_H = 1,
  5. read the training-point projections K_c alpha_i = sqrt(Delta_i) u_i,
  6. project a fresh test point z=(0.5,0.5) onto the top two components,
     which requires centering its test kernel vector the same way.

Prints every number the worked example displays.
"""
import numpy as np

np.set_printoptions(suppress=True)

# Given: five 2-D points, two groups.
X = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [5.0, 4.0],
    [6.0, 4.0],
])
n = X.shape[0]
sigma = 2.0

def rbf(A, B, sigma):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2.0 * sigma ** 2))

# Step 1: Gram matrix.
K = rbf(X, X, sigma)
print("K =\n", np.round(K, 4))

# Step 2: double-centering.
J = np.eye(n) - np.ones((n, n)) / n
Kc = J @ K @ J
print("K_c = J K J =\n", np.round(Kc, 4))

# Step 3: eigendecomposition (descending).
w, V = np.linalg.eigh(Kc)
order = np.argsort(w)[::-1]
w = w[order]
V = V[:, order]
print("eigenvalues Delta (descending) =", np.round(w, 4))
print("total variance sum Delta =", round(float(w.sum()), 4))
print("variance fraction of PC1, PC2 =",
      np.round(w[:2] / w[:np.sum(w > 1e-9)].sum(), 4))

# Sign convention: make each eigenvector's largest-magnitude entry positive.
for i in range(n):
    j = np.argmax(np.abs(V[:, i]))
    if V[j, i] < 0:
        V[:, i] = -V[:, i]

u1, u2 = V[:, 0], V[:, 1]
print("u1 =", np.round(u1, 4))
print("u2 =", np.round(u2, 4))

# Step 4: normalize expansion coefficients alpha_i = u_i / sqrt(Delta_i).
a1 = u1 / np.sqrt(w[0])
a2 = u2 / np.sqrt(w[1])
print("alpha1 = u1/sqrt(Delta1) =", np.round(a1, 4))
print("alpha2 = u2/sqrt(Delta2) =", np.round(a2, 4))
# Verify unit RKHS norm: alpha_i^T K_c alpha_i = 1.
print("||f1||_H^2 = alpha1^T K_c alpha1 =", round(float(a1 @ Kc @ a1), 6))
print("||f2||_H^2 = alpha2^T K_c alpha2 =", round(float(a2 @ Kc @ a2), 6))

# Step 5: training-point projections onto PC1, PC2 = sqrt(Delta_i) u_i = K_c alpha_i.
proj1 = Kc @ a1
proj2 = Kc @ a2
print("training projections onto PC1 (K_c alpha1) =", np.round(proj1, 4))
print("cross-check sqrt(Delta1) u1 =", np.round(np.sqrt(w[0]) * u1, 4))
print("training projections onto PC2 (K_c alpha2) =", np.round(proj2, 4))

# Step 6: out-of-sample test point z, projected onto PC1 and PC2.
z = np.array([[0.5, 0.5]])
kz = rbf(z, X, sigma).ravel()          # k(z, x_j), length n
print("test point z =", z.ravel())
print("k(z, x_j) =", np.round(kz, 4))
# Center the test kernel vector against the training statistics.
row_mean = K.mean(axis=0)              # (1/n) sum_m k(x_m, x_j)
grand = K.mean()                       # (1/n^2) sum_{m,l} k(x_m, x_l)
kz_c = kz - kz.mean() - row_mean + grand
print("centered k~(z, x_j) =", np.round(kz_c, 4))
pz1 = float(a1 @ kz_c)
pz2 = float(a2 @ kz_c)
print("projection of z onto PC1 =", round(pz1, 4))
print("projection of z onto PC2 =", round(pz2, 4))

# Reading: z sits in the origin cluster, so its PC1 sign matches those points.
print("sign(PC1) of z vs origin-cluster points:",
      np.sign(pz1), np.round(np.sign(proj1[:3]), 0))
