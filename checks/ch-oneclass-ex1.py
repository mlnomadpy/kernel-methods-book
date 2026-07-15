"""ch-oneclass, Example 1: smallest enclosing hypersphere (SVDD) on 5 points.

Hard SVDD dual (Schoelkopf-Smola 8.18-8.19, Shawe-Taylor-Cristianini 7.1):
    maximize  W(a) = sum_i a_i k(x_i,x_i) - sum_ij a_i a_j k(x_i,x_j)
    subject to sum_i a_i = 1,  a_i >= 0.
With a linear kernel k(x,x') = <x,x'> this is the smallest enclosing ball in
input space. Center c = sum_i a_i x_i (a convex combination); the optimum dual
value equals r^2; a point sits ON the sphere iff a_i > 0 (a support vector).
Every number printed here appears in the worked example.
"""
import numpy as np
from scipy.optimize import minimize

# --- setup: 4 corners of a square plus its center ---
X = np.array([[0.0, 0.0],
              [2.0, 0.0],
              [2.0, 2.0],
              [0.0, 2.0],
              [1.0, 1.0]])
m = len(X)
K = X @ X.T                      # linear kernel Gram matrix
d = np.diag(K).copy()            # k(x_i,x_i) = ||x_i||^2
print("K =\n", K)
print("diag(K) = ||x_i||^2 =", d)

# --- dual QP: maximize W(a) = a.d - a K a,  sum a = 1, a >= 0 ---
def negW(a):
    return -(a @ d - a @ K @ a)

cons = ({"type": "eq", "fun": lambda a: a.sum() - 1.0},)
bnds = [(0.0, None)] * m
a0 = np.full(m, 1.0 / m)
sol = minimize(negW, a0, method="SLSQP", bounds=bnds, constraints=cons,
               options={"ftol": 1e-12, "maxiter": 500})
a = sol.x
a[np.abs(a) < 1e-7] = 0.0
print("alpha =", np.round(a, 4))
print("sum(alpha) =", round(float(a.sum()), 6))

# --- center, radius ---
c = a @ X
Wopt = a @ d - a @ K @ a
r2 = float(Wopt)
r = np.sqrt(r2)
print("center c =", np.round(c, 4))
print("W(alpha*) = r^2 =", round(r2, 4))
print("radius r =", round(float(r), 4))

# --- distances of every point to the center; who is on the sphere ---
dist2 = np.sum((X - c) ** 2, axis=1)
print("||x_i - c||^2 =", np.round(dist2, 4))
for i in range(m):
    on = "ON sphere (SV)" if a[i] > 1e-6 else "strictly inside"
    print(f"  x{i+1}={tuple(X[i])}: dist^2={dist2[i]:.4f}  alpha={a[i]:.4f}  {on}")
nSV = int(np.sum(a > 1e-6))
print("number of SVs =", nSV, " fraction =", nSV / m)
