"""Worked example 3: the SVM dual is non-convex for an indefinite kernel.

The C-SVM dual maximizes  q(alpha) = sum_i alpha_i - 1/2 alpha^T G alpha  over
the box 0 <= alpha_i <= C and the equality sum_i alpha_i y_i = 0, where
G_ij = y_i y_j K(x_i,x_j). For a positive definite K the matrix G is psd, so q
is concave and the dual is a convex QP with a unique solution. We show that for
the indefinite sigmoid Gram matrix of Examples 1-2 this breaks: because
diag(y) is orthogonal (y_i = +/-1), G = diag(y) K diag(y) is orthogonally
similar to K and shares its spectrum EXACTLY, so G inherits K's negative
eigenvalue. We then exhibit a constraint-feasible direction of positive
curvature, certifying that q is not concave.

Prints every number the worked example displays.
"""
import numpy as np

np.set_printoptions(suppress=True, precision=4)

X = np.array([-1.0, 1.0, 3.0])
a, c = 0.5, 0.2
K = np.tanh(a * np.outer(X, X) + c)
y = np.array([1.0, -1.0, 1.0])

G = np.diag(y) @ K @ np.diag(y)
print("G = diag(y) K diag(y) =\n", np.round(G, 4))
print("eig(K) =", np.round(np.linalg.eigvalsh(K), 4))
print("eig(G) =", np.round(np.linalg.eigvalsh(G), 4), " (same spectrum: diag(y) is orthogonal)")

# Hessian of the dual objective q is -G. Concave iff G psd. It is not:
print("min eig(G) =", round(float(np.linalg.eigvalsh(G).min()), 4),
      " < 0  ->  -G has a positive eigenvalue  ->  q is not concave")

# Certify with a feasible ascent direction: y^T v = 0 and v^T G v < 0
# (so the curvature of q along v, namely -v^T G v, is strictly positive).
v = np.array([1.0, 2.0, 1.0])
print("\nfeasible direction v =", v.tolist())
print("y^T v =", round(float(y @ v), 4), " (satisfies the equality constraint)")
print("v^T G v =", round(float(v @ G @ v), 4), " < 0")
print("curvature of q along v = -v^T G v =", round(float(-v @ G @ v), 4), " > 0  (q curves UP)")

# Contrast: a positive definite kernel would give v^T G v >= 0 for every v.
# Here the box 0<=alpha<=C keeps the feasible set compact, so a maximum still
# exists, but it can sit at many stationary points; SMO-type solvers return one
# stationary point, not necessarily the global maximizer.
print("\nconclusion: indefinite K -> indefinite G -> non-concave dual QP")
