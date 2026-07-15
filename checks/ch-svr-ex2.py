"""ch-svr, Example 2: nu-SVR and the nu-property on a tiny data set (RBF kernel).

We solve the nu-SVR dual QP for three values of nu on the same 5 points and read
off the automatically chosen tube width epsilon. We then count the fraction of
errors (points strictly outside the tube) and the fraction of support vectors.
This illustrates Proposition 9.2 (Schoelkopf-Smola-Williamson-Bartlett 2000):
  nu is an UPPER bound on the fraction of errors and a LOWER bound on the
  fraction of support vectors.

nu-SVR dual (SS Ch 9):
  maximize  sum_i beta_i y_i - 0.5 sum_ij beta_i beta_j K_ij,  beta_i = a_i - a*_i
  s.t.      sum_i beta_i = 0,  0 <= a_i,a*_i <= C/m,  sum_i (a_i + a*_i) <= C nu.
The regression is f(x) = sum_i beta_i k(x_i,x) + b. A Gaussian (RBF) kernel is
used so the Gram matrix is full rank and the dual solution is unique.
"""
import numpy as np
from scipy.optimize import minimize

# --- setup ---
x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y = np.array([0.0, 0.9, 0.2, -0.8, 0.3])
m = len(x)
C = float(m)                       # C/m = 1, box [0,1]
Cm = C / m
sigma = 1.5
D2 = (x[:, None] - x[None, :]) ** 2
K = np.exp(-D2 / (2 * sigma**2))   # Gaussian kernel
print("m, C, C/m, sigma =", m, C, Cm, sigma)
print("K =\n", np.round(K, 4))
print("y =", y)


def solve_nu(nu):
    def negW(z):
        beta = z[:m] - z[m:]
        return -(beta @ y - 0.5 * beta @ K @ beta)
    cons = (
        {"type": "eq", "fun": lambda z: (z[:m] - z[m:]).sum()},
        {"type": "ineq", "fun": lambda z: C * nu - (z[:m] + z[m:]).sum()},
    )
    bnds = [(0.0, Cm)] * (2 * m)
    z0 = np.full(2 * m, C * nu / (4 * m))
    sol = minimize(negW, z0, method="SLSQP", bounds=bnds, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 2000})
    beta = sol.x[:m] - sol.x[m:]
    beta[np.abs(beta) < 1e-6] = 0.0
    # in-bound SVs sit exactly on the tube edges:
    #   upper edge (beta>0): y_i - g_i - b = +eps
    #   lower edge (beta<0): y_i - g_i - b = -eps
    g = K @ beta                       # g_i = sum_j beta_j K_ij (= f_i - b)
    up = [i for i in range(m) if 1e-5 < beta[i] < Cm - 1e-5]
    lo = [i for i in range(m) if -Cm + 1e-5 < beta[i] < -1e-5]
    i, j = up[0], lo[0]
    eps = 0.5 * ((y[i] - g[i]) - (y[j] - g[j]))
    b = 0.5 * ((y[i] - g[i]) + (y[j] - g[j]))
    f = g + b
    resid = y - f
    nSV = int(np.sum(np.abs(beta) > 1e-6))
    nErr = int(np.sum(np.abs(beta) > Cm - 1e-5))
    return dict(nu=nu, eps=eps, b=b, beta=beta, resid=resid,
                nSV=nSV, nErr=nErr, budget=float((np.abs(beta)).sum()))


print(f"\n{'nu':>5} {'eps':>7} {'#SV':>4} {'#err':>5} "
      f"{'err/m':>6} {'SV/m':>6}   err/m<=nu<=SV/m ?")
rows = []
for nu in [0.2, 0.4, 0.6]:
    r = solve_nu(nu)
    rows.append(r)
    ok = (r["nErr"] / m) <= r["nu"] + 1e-9 and r["nu"] <= (r["nSV"] / m) + 1e-9
    print(f"{r['nu']:>5.2f} {r['eps']:>7.4f} {r['nSV']:>4d} {r['nErr']:>5d} "
          f"{r['nErr']/m:>6.2f} {r['nSV']/m:>6.2f}   {'yes' if ok else 'NO'}")

# --- detail for nu = 0.6 ---
r = rows[2]
print("\n--- detail, nu = 0.6 ---")
print("beta =", np.round(r["beta"], 4))
print("sum(beta)       =", round(float(r["beta"].sum()), 6))
print("budget sum|beta| =", round(r["budget"], 4), " (C nu =", C * 0.6, ")")
print("eps =", round(r["eps"], 4), "  b =", round(r["b"], 4))
print("residual y-f =", np.round(r["resid"], 4))
for i in range(m):
    rr = r["resid"][i]
    if abs(rr) > r["eps"] + 1e-6:
        loc = "outside (error, |beta|=C/m)"
    elif abs(abs(rr) - r["eps"]) < 1e-4:
        loc = "on edge (in-bound SV)"
    else:
        loc = "inside"
    print(f"  x={x[i]:.0f}: resid={rr:+.4f}  beta={r['beta'][i]:+.4f}  {loc}")
