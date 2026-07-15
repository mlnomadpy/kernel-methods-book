"""Sample check script (the pattern every checks/<src>-ex<N>.py follows).

Worked example: the linear-kernel Gram matrix on three dependent points, and the
ridge solution (K + lambda I) alpha = y. Prints every quantity the example shows.
"""
import numpy as np

X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # x1, x2, x3
y = np.array([1.0, -1.0, 0.0])
lam = 1.0

K = X @ X.T
print("Gram matrix K =\n", K.astype(int))
print("det K =", round(np.linalg.det(K), 12))          # 0: x3 = x1 + x2
print("null direction K @ (1,1,-1) =", (K @ np.array([1, 1, -1.0])).astype(int))

alpha = np.linalg.solve(K + lam * np.eye(3), y)
print("alpha =", alpha)                                 # (4, -5, -1)/7
print("7 * alpha =", np.round(7 * alpha, 9))
