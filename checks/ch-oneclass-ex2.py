"""ch-oneclass, Example 2: the nu-one-class SVM and the nu-property.

Separate the data from the origin in feature space (Schoelkopf et al. 2001).
Dual (Schoelkopf-Smola 8.13-8.15):
    minimize  (1/2) sum_ij a_i a_j k(x_i,x_j)
    subject to 0 <= a_i <= 1/(nu m),  sum_i a_i = 1.
Decision f(x) = sgn( sum_i a_i k(x_i,x) - rho ). rho is read off any non-bound
support vector (0 < a_i < 1/(nu m)) via rho = sum_j a_j k(x_j,x_i). An OUTLIER
is a training point with f(x_i) < 0; every outlier sits at the upper bound
a_i = 1/(nu m). The nu-property: nu upper-bounds the outlier fraction and
lower-bounds the SV fraction. We sweep nu on a tiny 1-D set (a cluster plus two
stragglers) with a Gaussian kernel k(x,x') = exp(-|x-x'|^2 / c), c = 1.
Every number printed here appears in the worked example.
"""
import numpy as np
from scipy.optimize import minimize

# --- setup: 8 clustered points plus 2 stragglers ---
x = np.array([0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 3.5, 4.2])
m = len(x)
cwidth = 1.0
K = np.exp(-(x[:, None] - x[None, :]) ** 2 / cwidth)
print("m =", m, " c =", cwidth)

def solve_oneclass(nu):
    ub = 1.0 / (nu * m)
    cons = ({"type": "eq", "fun": lambda a: a.sum() - 1.0},)
    sol = minimize(lambda a: 0.5 * a @ K @ a, np.full(m, 1.0 / m),
                   method="SLSQP", bounds=[(0.0, ub)] * m, constraints=cons,
                   options={"ftol": 1e-12, "maxiter": 3000})
    a = sol.x
    a[np.abs(a) < 1e-7] = 0.0
    nb = np.where((a > 1e-6) & (a < ub - 1e-6))[0]      # non-bound SVs
    scores = K @ a                                       # w . Phi(x_i)
    rho = float(np.mean(scores[nb]))
    f = scores - rho
    outliers = f < -1e-6
    svs = a > 1e-6
    return a, ub, rho, f, outliers, svs

print(f"{'nu':>5} {'1/(nu m)':>9} {'rho':>8} {'#SV':>4} {'#OL':>4} "
      f"{'fracSV':>7} {'fracOL':>7} {'nu-prop OK':>11}")
for nu in [0.2, 0.4, 0.5]:
    a, ub, rho, f, outliers, svs = solve_oneclass(nu)
    fracSV, fracOL = svs.mean(), outliers.mean()
    ok = (fracOL <= nu + 1e-9) and (nu <= fracSV + 1e-9)
    print(f"{nu:>5.1f} {ub:>9.3f} {rho:>8.4f} {int(svs.sum()):>4} "
          f"{int(outliers.sum()):>4} {fracSV:>7.2f} {fracOL:>7.2f} {str(ok):>11}")
    ol_pts = [f"x={x[i]:.1f}" for i in np.where(outliers)[0]]
    print("      outliers:", ol_pts if ol_pts else "none")
