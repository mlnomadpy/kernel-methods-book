"""Worked example 1: HSIC and a kernel conditional-independence (KCIT) statistic
on a tiny chain X -> Y -> Z (n = 8).

We want a sample that is (a) tiny and hand-checkable, (b) non-degenerate (every
variable genuinely varies within each stratum), and (c) satisfies X _|_ Z | Y
EXACTLY, as a Markov chain must.  A 2x2 product design does all three.  The
conditioner Y has two strata; inside each stratum the (X,Z) pairs form a full
2x2 grid, so X and Z are exactly independent given Y, yet both shift upward with
Y, so all three pairs are marginally dependent:

    row:   1    2    3    4    5    6    7    8
    X  =  [0,   0,   1,   1,   2,   2,   3,   3]
    Y  =  [0,   0,   0,   0,   1,   1,   1,   1]
    Z  =  [0,   1,   0,   1,   2,   3,   2,   3]
    stratum Y=0:  (X,Z) in {0,1} x {0,1}   (balanced grid  ->  X _|_ Z)
    stratum Y=1:  (X,Z) in {2,3} x {2,3}   (balanced grid  ->  X _|_ Z)

Statistics (Gaussian kernels, median-heuristic bandwidth, H = I - 11^T/n):
  * biased HSIC V-statistic  HSIC_b(A,B) = (1/n^2) Tr(A_c B_c),  A_c = H K_A H.
  * KCIT statistic (Zhang, Peters, Janzing, Schoelkopf 2011) for X _|_ Z | Y,
        T_CI = (1/n) Tr( K~_{Xdot|Y} K~_{Z|Y} ),
    with Xdot=(X,Y), residual-maker R_Y = eps (K~_Y + eps I)^{-1}, and
    K~_{Xdot|Y}=R_Y K~_Xdot R_Y,  K~_{Z|Y}=R_Y K~_Z R_Y.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

X = np.array([0, 0, 1, 1, 2, 2, 3, 3], float)
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1], float)
Z = np.array([0, 1, 0, 1, 2, 3, 2, 3], float)
n = len(X)
H = np.eye(n) - np.ones((n, n)) / n


def sqdist(A):
    A = A.reshape(-1, 1) if A.ndim == 1 else A
    s = np.sum(A * A, 1)
    return np.maximum(s[:, None] + s[None, :] - 2 * A @ A.T, 0.0)


def median_sigma(A):
    D = sqdist(A)
    return np.median(np.sqrt(D[np.triu_indices(len(D), 1)]))


def rbf(A):
    return np.exp(-sqdist(A) / (2 * median_sigma(A) ** 2))


def hsic_b(A, B):
    Ac, Bc = H @ rbf(A) @ H, H @ rbf(B) @ H
    return np.trace(Ac @ Bc) / n ** 2


print("median bandwidths: sX=%.4f sY=%.4f sZ=%.4f"
      % (median_sigma(X), median_sigma(Y), median_sigma(Z)))
print("HSIC_b(X,Z) = %.4f   (X, Z marginally dependent)" % hsic_b(X, Z))
print("HSIC_b(X,Y) = %.4f" % hsic_b(X, Y))
print("HSIC_b(Y,Z) = %.4f" % hsic_b(Y, Z))

# --- KCIT statistic for  X _|_ Z | Y ---
eps = 1e-3
Xdot = np.column_stack([X, Y])            # augmented variable (X,Y)
Kxd = H @ rbf(Xdot) @ H
Kz = H @ rbf(Z) @ H
Ky = H @ rbf(Y) @ H
Ry = eps * np.linalg.solve(Ky + eps * np.eye(n), np.eye(n))
T_ci = np.trace((Ry @ Kxd @ Ry) @ (Ry @ Kz @ Ry)) / n
assert np.isclose(hsic_b(X, Z), 0.0844, atol=5e-5)
assert T_ci < 1e-10
print("T_CI(X,Z | Y) = %.6f   (X, Z conditionally independent given Y)" % T_ci)
print("ratio T_CI(X,Z|Y) / HSIC_b(X,Z) = %.4g" % (T_ci / hsic_b(X, Z)))

# Contrast: if Z instead TRACKED X within each stratum, X _|_ Z | Y would fail.
Zdep = np.array([0, 0, 1, 1, 2, 2, 3, 3], float)   # Z = X within each Y-stratum
Kzd = H @ rbf(Zdep) @ H
Xd2 = np.column_stack([X, Y])
Kxd2 = H @ rbf(Xd2) @ H
T_dep = np.trace((Ry @ Kxd2 @ Ry) @ (Ry @ Kzd @ Ry)) / n
assert np.isclose(T_dep, 0.097063, atol=1e-6)
assert T_dep > 1e6 * T_ci
print("T_CI when Z depends on X given Y = %.6f   (statistic is not vacuous)" % T_dep)
