"""ch-svr, Example 3: quantile (pinball) SV regression at two quantile levels.

We fit the SAME tiny data set with the pinball loss at tau = 0.25 and tau = 0.75
(and tau = 0.5 for the median / LAD cross-check). The dual is the eps-SVR dual
with eps = 0 and, crucially, an ASYMMETRIC box: a_i in [0, (C/m) tau] and
a*_i in [0, (C/m)(1-tau)]. The asymmetry alone tilts the fit toward the
tau-quantile; there is no tube.

Pinball loss of residual u = y - f(x), quantile level tau in (0,1):
    rho_tau(u) = max(tau * u, (tau - 1) * u)
             = tau*u        if u >= 0  (under-prediction, point on/above the fit)
             = (1-tau)*|u|  if u <  0  (over-prediction,  point on/below the fit)

Primal (chapter convention, penalty C/m):
    min 1/2 ||w||^2 + (C/m) sum_i [ tau xi_i + (1-tau) xi*_i ]
    s.t. y_i - f(x_i) <= xi_i,  f(x_i) - y_i <= xi*_i,  xi_i, xi*_i >= 0.

Dual (Takeuchi, Le, Sears, Smola 2006; Koenker and Bassett 1978 for the loss):
    max_{a,a*} sum_i (a_i - a*_i) y_i - 1/2 sum_ij (a_i-a*_i)(a_j-a*_j) K_ij
    s.t. sum_i (a_i - a*_i) = 0,  0 <= a_i <= (C/m)tau,  0 <= a*_i <= (C/m)(1-tau).

With a LINEAR kernel K = x x^T is rank one, so the individual multipliers are not
unique, but the fit (w, b), hence f, the residuals, and the point counts, ARE
uniquely pinned (the primal is strictly convex in w). We print only those.
Every number printed here appears in the worked example.
"""
import numpy as np
from scipy.optimize import minimize

# --- setup: six points, mild upward trend with scatter ---
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
y = np.array([1.0, 2.6, 2.0, 4.2, 3.4, 5.6])
m = len(x)
C = 12.0
Cm = C / m                      # = 2.0
K = np.outer(x, x)              # linear kernel k(x,x') = x x'
print("m, C, C/m =", m, C, Cm)
print("x =", x)
print("y =", y)


def pinball(u, tau):
    return np.maximum(tau * u, (tau - 1.0) * u)


def solve_tau(tau):
    hi_a = Cm * tau             # ceiling on a_i     (above-line side)
    hi_as = Cm * (1.0 - tau)    # ceiling on a*_i    (below-line side)

    def negW(z):
        beta = z[:m] - z[m:]
        return -(beta @ y - 0.5 * beta @ K @ beta)

    cons = ({"type": "eq", "fun": lambda z: (z[:m] - z[m:]).sum()},)
    bnds = [(0.0, hi_a)] * m + [(0.0, hi_as)] * m
    best = None
    for seed in range(8):                       # fixed seeds -> deterministic
        rng = np.random.default_rng(seed)
        z0 = np.concatenate([rng.uniform(0, hi_a, m), rng.uniform(0, hi_as, m)])
        sol = minimize(negW, z0, method="SLSQP", bounds=bnds, constraints=cons,
                       options={"ftol": 1e-12, "maxiter": 1000})
        if best is None or sol.fun < best.fun:
            best = sol
    z = best.x
    a, as_ = z[:m], z[m:]
    beta = a - as_
    w = float(beta @ x)                          # unique
    # b from any in-bound SV (0 < a_i < hi_a, or 0 < a*_i < hi_as): it lies on
    # the fit, so b = y_i - w x_i.  (In-bound SVs are exactly the on-line points.)
    on = [i for i in range(m)
          if 1e-6 < a[i] < hi_a - 1e-6 or 1e-6 < as_[i] < hi_as - 1e-6]
    b = float(np.mean([y[i] - w * x[i] for i in on]))
    f = w * x + b
    resid = y - f
    nbelow = int(np.sum(resid < -1e-6))          # y < f  : point below the fit
    nabove = int(np.sum(resid > 1e-6))           # y > f  : point above the fit
    non = m - nbelow - nabove                     # residual == 0 : on the fit
    dualobj = float(beta @ y - 0.5 * beta @ K @ beta)
    primal = float(0.5 * w**2 + Cm * pinball(resid, tau).sum())
    return dict(tau=tau, hi_a=hi_a, hi_as=hi_as, w=w, b=b, f=f, resid=resid,
                nbelow=nbelow, nabove=nabove, non=non,
                on=[i + 1 for i in on], dualobj=dualobj, primal=primal)


print(f"\n{'tau':>5} {'box a_i':>10} {'box a*_i':>10} {'w':>7} {'b':>7} "
      f"{'#below':>7} {'#on':>4} {'#above':>7} {'below/m':>8}")
res = {}
for tau in [0.25, 0.5, 0.75]:
    r = solve_tau(tau)
    res[tau] = r
    print(f"{tau:>5.2f} [0,{r['hi_a']:>5.2f}] [0,{r['hi_as']:>5.2f}] "
          f"{r['w']:>7.4f} {r['b']:>7.4f} {r['nbelow']:>7d} {r['non']:>4d} "
          f"{r['nabove']:>7d} {r['nbelow']/m:>8.3f}")

# finite-sample quantile bracket (ties on the fit): below/m <= tau <= (below+on)/m
print("\nquantile bracket   below/m <= tau <= (below+on)/m :")
for tau in [0.25, 0.5, 0.75]:
    r = res[tau]
    lo_f = r["nbelow"] / m
    hi_f = (r["nbelow"] + r["non"]) / m
    ok = lo_f - 1e-9 <= tau <= hi_f + 1e-9
    print(f"  tau={tau:.2f}:  {lo_f:.3f} <= {tau:.2f} <= {hi_f:.3f}   "
          f"({'holds' if ok else 'FAILS'})")

for tau in [0.25, 0.75]:
    r = res[tau]
    print(f"\n--- detail, tau = {tau} ---")
    print(f"asymmetric box: a_i in [0,{r['hi_a']:.2f}]  a*_i in [0,{r['hi_as']:.2f}]")
    print(f"fit: f(x) = {r['w']:.4f} x + {r['b']:.4f}")
    print("f(x_i)       =", np.round(r["f"], 4))
    print("residual y-f =", np.round(r["resid"], 4))
    print(f"points on the fit (in-bound SVs) = {r['on']}")
    print(f"#below = {r['nbelow']}  #on = {r['non']}  #above = {r['nabove']}"
          f"   fraction below = {r['nbelow']}/{m}")
    print(f"dual obj = {r['dualobj']:.5f}   primal obj = {r['primal']:.5f}"
          f"   gap = {r['primal'] - r['dualobj']:.2e}")

# --- median (tau=0.5): symmetric box, regularized LAD line ---
r5 = res[0.5]
print("\n--- tau = 0.5 (median / LAD) ---")
print(f"box symmetric: a_i, a*_i in [0,{r5['hi_a']:.2f}]")
print(f"fit: f(x) = {r5['w']:.4f} x + {r5['b']:.4f}   (regularized least-abs-dev)")

# --- quantile ordering: low-tau fit lies below high-tau fit everywhere ---
lo, hi = res[0.25]["f"], res[0.75]["f"]
print("\ntau=0.25 fit below tau=0.75 fit at every x:", bool(np.all(lo <= hi + 1e-9)))
print("f_0.25(x) =", np.round(lo, 4))
print("f_0.75(x) =", np.round(hi, 4))
