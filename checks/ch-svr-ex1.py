"""ch-svr, Example 1: a tiny 1-D epsilon-SVR fit with a linear kernel.

We solve the epsilon-SVR dual QP for 5 points on a line, read off w and b,
classify every point as inside / on / outside the epsilon-tube, identify the
support vectors, and check strong duality (primal objective == dual objective).

Convention (Schoelkopf-Smola Ch 9):
  f(x) = w x + b,  w = sum_i (a_i - a*_i) x_i,  beta_i = a_i - a*_i.
  alpha_i attaches to points ABOVE the line (y_i > f), alpha*_i below.
  Box: a_i, a*_i in [0, C/m].
Every number printed here appears in the worked example.
"""
import numpy as np
from scipy.optimize import minimize

# --- setup ---
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
m = len(x)
C = 10.0
eps = 0.5
Cm = C / m
K = np.outer(x, x)          # linear kernel k(x,x') = x x'
print("m, C, C/m, eps =", m, C, Cm, eps)
print("K =\n", K)

# --- dual QP:  maximize W(a,a*) ---
#   W = sum (a_i - a*_i) y_i - eps sum(a_i + a*_i)
#       - 0.5 sum_ij (a_i-a*_i)(a_j-a*_j) K_ij
# variables z = [a (m), a* (m)], bounds [0, C/m], equality sum(a-a*)=0.
def negW(z):
    a, as_ = z[:m], z[m:]
    beta = a - as_
    return -(beta @ y - eps * (a + as_).sum() - 0.5 * beta @ K @ beta)

cons = ({"type": "eq", "fun": lambda z: (z[:m] - z[m:]).sum()},)
bnds = [(0.0, Cm)] * (2 * m)
z0 = np.full(2 * m, Cm / 2)
sol = minimize(negW, z0, method="SLSQP", bounds=bnds, constraints=cons,
               options={"ftol": 1e-12, "maxiter": 500})
a, as_ = sol.x[:m], sol.x[m:]
beta = a - as_
beta[np.abs(beta) < 1e-7] = 0.0
print("alpha   =", np.round(a, 4))
print("alpha*  =", np.round(as_, 4))
print("beta = a - a* =", np.round(beta, 4))

# --- primal quantities ---
w = beta @ x
# b from an in-bound SV: upper edge (beta>0): b = y_i - w x_i - eps
#                        lower edge (beta<0): b = y_i - w x_i + eps
on_edge = np.where((np.abs(beta) > 1e-6) & (np.abs(beta) < Cm - 1e-6))[0]
bs = []
for i in on_edge:
    bs.append(y[i] - w * x[i] - eps * np.sign(beta[i]))
b = float(np.mean(bs))
print("w =", round(w, 4))
print("b =", round(b, 4))

f = w * x + b
resid = y - f
print("f(x)      =", np.round(f, 4))
print("residual y-f =", np.round(resid, 4))

# --- classify points ---
tol = 1e-6
for i in range(m):
    if abs(resid[i]) < eps - tol:
        loc = "inside"
    elif abs(abs(resid[i]) - eps) < tol:
        loc = "on edge (SV)"
    else:
        loc = "outside (SV)"
    print(f"  point x={x[i]:.0f}: resid={resid[i]:+.4f}  {loc}  beta={beta[i]:+.4f}")

nSV = int(np.sum(np.abs(beta) > 1e-6))
print("number of SVs =", nSV, " fraction =", nSV / m)

# --- duality check ---
Wdual = beta @ y - eps * (a + as_).sum() - 0.5 * beta @ K @ beta
xi = np.maximum(0.0, resid - eps) + np.maximum(0.0, -resid - eps)  # total slack
primal = 0.5 * w**2 + Cm * xi.sum()
print("sum(beta) =", round(float(beta.sum()), 6))
print("dual objective   W =", round(float(Wdual), 6))
print("primal objective   =", round(float(primal), 6))
print("duality gap        =", round(float(primal - Wdual), 8))
