"""Worked example: one deflation step of primal PLS on a tiny (X, y).

Centered data matrix X (3 x 2), centered target y (3,). Following the PLS
feature-extraction loop (Shawe-Taylor and Cristianini 2004, Algorithm 6.42/6.45):
    u1 = first singular vector of X'y  (for a single-column y, u1 = X'y / ||X'y||)
    tau1 = X u1                         (the score, a direction in sample space)
    p1 = X'X u1 / (u1' X'X u1)          (the loading)
    c1 = y' X u1 / (u1' X'X u1)         (the regression weight)
    X2 = X (I - u1 p1')                 (deflation)
Prints every quantity and checks that tau1 is orthogonal to the deflated X2.
"""
import numpy as np

X = np.array([[1.0, 2.0],
              [1.0, -1.0],
              [-2.0, -1.0]])   # columns already sum to zero (centered)
y = np.array([1.0, -1.0, 0.0])  # centered

print("X =\n", X)
print("column sums of X (centered) =", X.sum(axis=0))
print("y =", y)

Xty = X.T @ y
print("X'y =", Xty)
u1 = Xty / np.linalg.norm(Xty)
print("u1 = X'y / ||X'y|| =", np.round(u1, 6))

tau1 = X @ u1
print("tau1 = X u1 =", np.round(tau1, 6))

XtX = X.T @ X
print("X'X =\n", XtX.astype(int))
denom = u1 @ XtX @ u1
print("u1'X'X u1 =", round(denom, 6))

p1 = (XtX @ u1) / denom
print("p1 = X'X u1 / (u1'X'X u1) =", np.round(p1, 6))

c1 = (y @ X @ u1) / denom
print("c1 = y'X u1 / (u1'X'X u1) =", round(c1, 6))

yhat = tau1 * c1
print("fitted contribution tau1 * c1 =", np.round(yhat, 6))

X2 = X @ (np.eye(2) - np.outer(u1, p1))
print("X2 = X (I - u1 p1') =\n", np.round(X2, 6))
print("X2' tau1 (should be ~0) =", np.round(X2.T @ tau1, 9))
