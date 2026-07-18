"""Worked example 2: out-of-sample kernel PCA and the refit check.

We fit kernel PCA on a small TRAINING set of five 2-D points arranged along a
shallow arc, then project a fresh TEST point z out-of-sample onto the top two
kernel principal components using only its (centered) kernel vector to the
training set. Finally we VERIFY the out-of-sample coordinates against the
coordinates z receives when kernel PCA is re-run on the augmented set that
includes z, which should match approximately.

Pipeline:
  1. training Gram matrix K (5x5), Gaussian kernel, sigma = 1.5,
  2. double-center: K_c = J K J with J = I - (1/5) 11^T,
  3. eigendecompose K_c -> eigenvalues Delta_i, eigenvectors u_i,
  4. normalize alpha_i = u_i / sqrt(Delta_i) so ||f_i||_H = 1,
  5. out-of-sample projection of z: center k(z, .) with FROZEN training
     statistics, then coordinate_i = alpha_i^T k~(z),
  6. exact self-consistency: the same formula on a training point x_m
     reproduces its training coordinate sqrt(Delta_i) u_i[m],
  7. Nystrom identity: coordinate_i = (1/sqrt Delta_i) u_i^T k~(z)
     = sqrt(Delta_i) * [ (1/Delta_i) u_i^T k~(z) ], the eigenfunction value,
  8. refit on the augmented 6-point set and compare z's coordinates.

Prints every number the worked example displays. Pure linear algebra.
"""
import numpy as np

np.set_printoptions(suppress=True)

# ---- training data: five points on a shallow arc (distinct from example 1) ----
X = np.array([
    [-2.0, 0.0],
    [-1.0, 1.0],
    [ 0.0, 0.0],
    [ 1.0, 1.0],
    [ 2.0, 0.0],
])
n = X.shape[0]
sigma = 1.5


def rbf(A, B, s):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2.0 * s ** 2))


def fix_signs(V):
    """Deterministic sign: largest-magnitude entry of each column made positive."""
    V = V.copy()
    for i in range(V.shape[1]):
        j = np.argmax(np.abs(V[:, i]))
        if V[j, i] < 0:
            V[:, i] = -V[:, i]
    return V


# Step 1: training Gram matrix.
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
V = fix_signs(V[:, order])
w_pos = w[w > 1e-9]
print("eigenvalues Delta (descending) =", np.round(w, 4))
print("total variance sum Delta =", round(float(w_pos.sum()), 4))
print("variance fraction PC1, PC2 =", np.round(w[:2] / w_pos.sum(), 4))
print("cumulative top-2 fraction =", round(float(w[:2].sum() / w_pos.sum()), 4))

u1, u2 = V[:, 0], V[:, 1]
print("u1 =", np.round(u1, 4))
print("u2 =", np.round(u2, 4))

# Step 4: normalized expansion coefficients.
a1 = u1 / np.sqrt(w[0])
a2 = u2 / np.sqrt(w[1])
print("alpha1 =", np.round(a1, 4))
print("alpha2 =", np.round(a2, 4))
print("||f1||_H^2 = alpha1^T Kc alpha1 =", round(float(a1 @ Kc @ a1), 6))
print("||f2||_H^2 = alpha2^T Kc alpha2 =", round(float(a2 @ Kc @ a2), 6))

# training coordinates on PC1, PC2.
proj1 = Kc @ a1
proj2 = Kc @ a2
print("training coords PC1 (sqrt(D1) u1) =", np.round(proj1, 4))
print("training coords PC2 (sqrt(D2) u2) =", np.round(proj2, 4))

# ---- Step 5: out-of-sample projection of a fresh test point z ----
z = np.array([[-0.5, 0.5]])
kz = rbf(z, X, sigma).ravel()          # k(z, x_j), length n
print("\ntest point z =", z.ravel())
print("k(z, x_j) =", np.round(kz, 4))

# FROZEN training statistics: column mean (per training column j) and grand mean.
col_mean = K.mean(axis=0)              # (1/n) sum_b K(x_b, x_j), from TRAINING K
grand = K.mean()                       # (1/n^2) sum_{a,b} K(x_a, x_b), from TRAINING K
kz_mean = kz.mean()                    # (1/n) sum_a k(z, x_a), the TEST row-mean
print("frozen training column means =", np.round(col_mean, 4))
print("frozen training grand mean =", round(float(grand), 4))
print("test row-mean (1/n) sum_a k(z,x_a) =", round(float(kz_mean), 4))

# centered test kernel vector: two terms use the test row-mean, two use TRAINING stats.
kz_c = kz - kz_mean - col_mean + grand
print("centered k~(z, x_j) =", np.round(kz_c, 4))

p1_oos = float(a1 @ kz_c)
p2_oos = float(a2 @ kz_c)
print("out-of-sample coord of z on PC1 =", round(p1_oos, 4))
print("out-of-sample coord of z on PC2 =", round(p2_oos, 4))

# ---- Step 6: exact self-consistency on a training point (x_1) ----
m = 0
km = K[:, m].copy()                    # k(x_m, x_j) is the m-th column of K
km_c = km - km.mean() - col_mean + grand
p1_train_via_oos = float(a1 @ km_c)
print("\nself-consistency: oos formula on x_1, PC1 coord =", round(p1_train_via_oos, 4))
print("  training coord of x_1 on PC1 (sqrt(D1) u1[0]) =", round(float(proj1[m]), 4))
print("  abs difference =", format(abs(p1_train_via_oos - proj1[m]), ".2e"))

# ---- Step 7: Nystrom identity (coordinate = sqrt(D) * extended eigenfunction) ----
psi1 = (u1 @ kz_c) / w[0]               # Nystrom-extended (centered) eigenfunction value
print("\nNystrom eigenfunction psi1(z) = (1/D1) u1^T k~(z) =", round(float(psi1), 4))
print("sqrt(D1)*psi1(z) =", round(float(np.sqrt(w[0]) * psi1), 4),
      " equals oos coord PC1 =", round(p1_oos, 4))

# ---- Step 8: refit kernel PCA on the augmented 6-point set including z ----
Xa = np.vstack([X, z])
na = Xa.shape[0]
Ka = rbf(Xa, Xa, sigma)
Ja = np.eye(na) - np.ones((na, na)) / na
Kca = Ja @ Ka @ Ja
wa, Va = np.linalg.eigh(Kca)
order_a = np.argsort(wa)[::-1]
wa = wa[order_a]
Va = fix_signs(Va[:, order_a])

# sign-align augmented components to the training components by their overlap on
# the first n (shared) training points.
s1 = np.sign(Va[:n, 0] @ u1) or 1.0
s2 = np.sign(Va[:n, 1] @ u2) or 1.0
ua1 = s1 * Va[:, 0]
ua2 = s2 * Va[:, 1]
proj1_aug = np.sqrt(wa[0]) * ua1       # augmented coords on PC1
proj2_aug = np.sqrt(wa[1]) * ua2
p1_ref = float(proj1_aug[n])           # z is the last (6th) point
p2_ref = float(proj2_aug[n])
print("\naugmented eigenvalues (top 2) =", np.round(wa[:2], 4))
print("refit coord of z on PC1 =", round(p1_ref, 4), " (out-of-sample was", round(p1_oos, 4), ")")
print("refit coord of z on PC2 =", round(p2_ref, 4), " (out-of-sample was", round(p2_oos, 4), ")")
print("abs diff PC1 =", round(abs(p1_ref - p1_oos), 4),
      " rel diff PC1 =", round(abs(p1_ref - p1_oos) / abs(p1_ref), 4))
print("abs diff PC2 =", round(abs(p2_ref - p2_oos), 4))

# the original training points barely move under refit -> justifies "approximately".
shift1 = np.max(np.abs(proj1_aug[:n] - proj1))
shift2 = np.max(np.abs(proj2_aug[:n] - proj2))
print("max shift of the 5 training coords under refit: PC1 =", round(float(shift1), 4),
      " PC2 =", round(float(shift2), 4))

# ---- display precision used in the book (3 decimals) ----
print("\n[display 3dp] training coords PC1 =", np.round(proj1, 3))
print("[display 3dp] training coords PC2 =", np.round(proj2, 3))
print("[display 3dp] oos coord PC1, PC2 =", round(p1_oos, 3), round(p2_oos, 3))
print("[display 3dp] refit coord PC1, PC2 =", round(p1_ref, 3), round(p2_ref, 3))
print("[display 3dp] variance % PC1, PC2, cum =",
      round(100 * w[0] / w_pos.sum(), 1), round(100 * w[1] / w_pos.sum(), 1),
      round(100 * w[:2].sum() / w_pos.sum(), 1))
print("[display 3dp] refit abs diff PC1, PC2 =",
      round(abs(p1_ref - p1_oos), 3), round(abs(p2_ref - p2_oos), 3),
      " rel PC1 % =", round(100 * abs(p1_ref - p1_oos) / abs(p1_ref), 1))
