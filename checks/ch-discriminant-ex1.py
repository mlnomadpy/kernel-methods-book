"""Worked example: Kernel Fisher Discriminant on a tiny two-class set.

Four points in R^2, two per class, linear kernel k(x,x') = <x,x'>. We build the
dual Fisher objective on the Gram matrix (Mika et al. 1999): maximise
    (alpha' M alpha) / (alpha' (N + mu I) alpha),
with M = (m+ - m-)(m+ - m-)', (m_c)_j = (1/l_c) sum_{i in c} K_{ji}, and
    N = sum_c K_c (I_{l_c} - (1/l_c) 1 1') K_c',
K_c the l x l_c block of columns for class c. The maximiser is
    alpha ~ (N + mu I)^{-1} (m+ - m-).
Prints the Gram matrix, class-mean vectors, M and N, the direction alpha, and the
projections t_j = (K alpha)_j of the four training points.
"""
import numpy as np

# class +1: x1, x2 ; class -1: x3, x4
X = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [-1.0, 0.0],
              [0.0, -1.0]])
plus = [0, 1]
minus = [2, 3]
mu = 1.0  # regularisation

K = X @ X.T
print("Gram matrix K =\n", K.astype(int))

mplus = K[:, plus].mean(axis=1)
mminus = K[:, minus].mean(axis=1)
print("m+ =", mplus)
print("m- =", mminus)
d = mplus - mminus
print("m+ - m- =", d)

M = np.outer(d, d)
print("M =\n", M.astype(int))


def within(idx):
    Kc = K[:, idx]
    lc = len(idx)
    P = np.eye(lc) - np.ones((lc, lc)) / lc
    return Kc @ P @ Kc.T


N = within(plus) + within(minus)
print("N =\n", N)

A = N + mu * np.eye(4)
alpha = np.linalg.solve(A, d)
print("alpha (unnormalised) =", np.round(alpha, 6))
alpha = alpha / np.linalg.norm(alpha)
print("alpha (unit norm) =", np.round(alpha, 6))

t = K @ alpha
print("projections t = K alpha =", np.round(t, 6))
print("class + projections:", np.round(t[plus], 6))
print("class - projections:", np.round(t[minus], 6))
# Rayleigh quotient value
Jval = (alpha @ M @ alpha) / (alpha @ A @ alpha)
print("objective alpha'M alpha / alpha'(N+muI)alpha =", round(Jval, 6))
