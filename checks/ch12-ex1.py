"""Worked example: centering is essential for kernel alignment.

Two candidate positive definite (linear/rank-one) kernels on the same n = 4
points, with binary labels y = (+1, +1, -1, -1). Each kernel is the Gram matrix
K = phi phi^T of a scalar feature phi:

  phi_1 = (3, 3, 1, 1)        -> K1   (a PERFECT class separator, but the
                                       feature carries a large constant offset:
                                       phi_1 = (1,1,-1,-1) + 2)
  phi_2 = (1, 0.5, -0.5, -1)  -> K2   (an IMPERFECT separator, feature centered)

The ideal (label) kernel is L = y y^T. We compare two selection criteria:

  uncentered alignment  A(K, L)   = <K, L>_F / (||K||_F ||L||_F)
  centered   alignment  A_c(K, L) = <HKH, HLH>_F / (||HKH||_F ||HLH||_F)

with Frobenius inner product <A,B>_F = sum_ij A_ij B_ij and centering matrix
H = I - (1/n) 1 1^T (Cristianini, Shawe-Taylor, Elisseeff, Kandola 2002;
Cortes, Mohri, Rostamizadeh 2012).

After centering, phi_1 becomes exactly y (H phi_1 = y), so K1 centered EQUALS the
ideal kernel y y^T: K1 is the best possible kernel here. Yet the uncentered
alignment ranks K2 (0.9) far above K1 (0.2), because the constant offset in
phi_1 inflates ||K1||_F. The centered alignment repairs the ranking, giving
K1 = 1.0 above K2 = 0.9. Every number printed below appears in the worked
example.
"""
import numpy as np

np.set_printoptions(suppress=True)

y = np.array([1.0, 1.0, -1.0, -1.0])
n = len(y)
L = np.outer(y, y)                     # ideal / label kernel y y^T

phi1 = np.array([3.0, 3.0, 1.0, 1.0])  # perfect separator + large offset
phi2 = np.array([1.0, 0.5, -0.5, -1.0])  # imperfect separator, centered
K1 = np.outer(phi1, phi1)
K2 = np.outer(phi2, phi2)

H = np.eye(n) - np.ones((n, n)) / n     # centering matrix


def fip(A, B):                          # Frobenius inner product
    return float(np.sum(A * B))


def align(K, M):
    return fip(K, M) / (np.sqrt(fip(K, K)) * np.sqrt(fip(M, M)))


print("y            =", y.astype(int))
print("L = y y^T =\n", L.astype(int))
print("K1 = phi1 phi1^T =\n", K1.astype(int))
print("K2 = phi2 phi2^T =\n", K2)
print("both PSD (rank-one Gram): eig(K1) =", np.round(np.linalg.eigvalsh(K1), 3),
      " eig(K2) =", np.round(np.linalg.eigvalsh(K2), 3))

# --- uncentered alignment ----------------------------------------------------
print("\n-- uncentered --")
print("phi1 . y =", fip(phi1, y), "   <K1,L>_F = (phi1.y)^2 =", fip(K1, L))
print("||K1||_F =", np.sqrt(fip(K1, K1)), "  ||L||_F =", np.sqrt(fip(L, L)))
print("A(K1,L)  =", round(align(K1, L), 4))
print("phi2 . y =", fip(phi2, y), "   <K2,L>_F = (phi2.y)^2 =", fip(K2, L))
print("||K2||_F =", np.sqrt(fip(K2, K2)))
print("A(K2,L)  =", round(align(K2, L), 4))
print("uncentered ranking: K2 (%.1f) > K1 (%.1f)  [WRONG]"
      % (align(K2, L), align(K1, L)))

# --- centered alignment ------------------------------------------------------
K1c, K2c, Lc = H @ K1 @ H, H @ K2 @ H, H @ L @ H
print("\n-- centered --")
print("H phi1 =", np.round(H @ phi1, 3), " (= y), so K1_c = y y^T exactly")
print("K1_c =\n", np.round(K1c, 3))
print("||K1_c||_F =", np.sqrt(fip(K1c, K1c)),
      " (offset gone: 20 -> 4)   ||K2_c||_F =", np.sqrt(fip(K2c, K2c)))
print("A_c(K1,L) =", round(align(K1c, Lc), 4))
print("A_c(K2,L) =", round(align(K2c, Lc), 4))
print("centered ranking: K1 (%.1f) > K2 (%.1f)  [CORRECT: K1 is the ideal kernel]"
      % (align(K1c, Lc), align(K2c, Lc)))
