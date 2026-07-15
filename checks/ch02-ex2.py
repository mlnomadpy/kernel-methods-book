"""Worked example 2 (ch02): the representer expansion equals a feature-space dot.

Same degree-2 polynomial kernel K(x,y) = (x . y)^2 with explicit feature map
Phi(x) = (x1^2, sqrt(2) x1 x2, x2^2). Three training points, targets y. The
minimum-RKHS-norm interpolant guaranteed by the representer theorem has the form
f = sum_i alpha_i K(x_i, .) with coefficients solving K alpha = y.

We then evaluate f at a fresh point x* two ways:
  Route 1 (representer expansion):  f(x*) = sum_i alpha_i K(x_i, x*) = k_* . alpha
  Route 2 (feature space):          f(x*) = <Phi(x*), w>,  w = sum_i alpha_i Phi(x_i)
They must agree, and f must reproduce the targets at the training points.
Prints every displayed number.
"""
import numpy as np

np.set_printoptions(suppress=True)

X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])   # x1, x2, x3
yv = np.array([1.0, 2.0, 3.0])
xstar = np.array([2.0, 1.0])


def phi(x):
    return np.array([x[0] ** 2, np.sqrt(2.0) * x[0] * x[1], x[1] ** 2])


def K(x, y):
    return float(x @ y) ** 2


n = X.shape[0]
Gram = np.array([[K(X[i], X[j]) for j in range(n)] for i in range(n)])
print("Gram matrix K =\n", Gram.astype(int))
print("det K =", round(float(np.linalg.det(Gram)), 6))

alpha = np.linalg.solve(Gram, yv)                    # minimum-norm interpolant
print("alpha (solves K alpha = y) =", np.round(alpha, 6))

# interpolation check: f(x_i) = [K alpha]_i = y_i
print("K alpha =", np.round(Gram @ alpha, 6), " target y =", yv)

# kernel row at the test point
kstar = np.array([K(X[i], xstar) for i in range(n)])
print("k_* = [K(x_i, x*)] =", kstar.astype(int))

# --- Route 1: representer expansion ---
f_expansion = float(kstar @ alpha)
print("Route 1  f(x*) = k_* . alpha =", round(f_expansion, 6))

# --- Route 2: feature-space inner product ---
w = sum(alpha[i] * phi(X[i]) for i in range(n))      # w = sum_i alpha_i Phi(x_i)
print("w = sum_i alpha_i Phi(x_i) =", np.round(w, 6))
print("Phi(x*) =", np.round(phi(xstar), 6))
f_feature = float(phi(xstar) @ w)
print("Route 2  f(x*) = <Phi(x*), w> =", round(f_feature, 6))
print("two routes agree:", np.isclose(f_expansion, f_feature))

# RKHS norm two ways: alpha^T K alpha = ||w||^2
print("alpha^T K alpha =", round(float(alpha @ Gram @ alpha), 6))
print("||w||^2         =", round(float(w @ w), 6))
