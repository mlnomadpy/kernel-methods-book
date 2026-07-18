"""ch05, Example 4: cheap leave-one-out proxies for model selection.

For a tiny hard-margin SVM (two tight clusters), we compute two leave-one-out
(LOO) error bounds that need only ONE training run each, avoiding the n retrainings
of true LOO:
  (i)  the support-vector count bound      #LOO errors <= #SV,
  (ii) the radius-margin bound             #LOO errors <= R^2 ||w||^2,
where R is the radius of the smallest ball enclosing the data (feature space) and
1/||w|| is the margin. R^2 is obtained from the minimum-enclosing-ball dual QP
    R^2 = max_beta  sum_i beta_i K_ii - beta^T K beta,  beta >= 0, sum beta = 1.
We then verify the true LOO error by actually retraining n times. Every number
printed here appears in the worked example. Pure QP, runs in a second.
"""
import numpy as np
from scipy.optimize import minimize

# --- setup: two tight clusters, well separated along x1 ---
X = np.array([[0.0, 0.0], [0.0, 1.0],      # class -1, left
              [10.0, 0.0], [10.0, 1.0]])   # class +1, right
y = np.array([-1.0, -1.0, +1.0, +1.0])
n = len(y)
K = X @ X.T
print("K =\n", K)


def hard_margin(Xs, ys):
    """Canonical hard-margin SVM via the dual; returns (alpha, w, b)."""
    m = len(ys)
    M = np.outer(ys, ys) * (Xs @ Xs.T)

    def negW(a):
        return -(a.sum() - 0.5 * a @ M @ a)

    cons = ({"type": "eq", "fun": lambda a: a @ ys},)
    sol = minimize(negW, np.ones(m) * 0.1, method="SLSQP",
                   bounds=[(0.0, None)] * m, constraints=cons,
                   options={"ftol": 1e-14, "maxiter": 2000})
    a = sol.x
    a[a < 1e-9] = 0.0
    w = (a * ys) @ Xs
    sv = np.where(a > 1e-6)[0]
    b = float(np.mean([ys[k] - w @ Xs[k] for k in sv]))
    return a, w, b


alpha, w, b = hard_margin(X, y)
nsv = int(np.sum(alpha > 1e-6))
wn2 = float(w @ w)
print("alpha =", np.round(alpha, 6))
print("w =", np.round(w, 6), " b =", round(b, 6))
print("||w||^2 =", round(wn2, 6), "  margin 1/||w|| =", round(1.0 / np.sqrt(wn2), 6))
print("number of support vectors #SV =", nsv)
print("SV-count LOO bound  #SV/n =", round(nsv / n, 6))

# --- minimum enclosing ball radius^2 via its dual QP ---
diagK = np.diag(K)


def negR2(beta):
    return -(beta @ diagK - beta @ K @ beta)


consR = ({"type": "eq", "fun": lambda be: be.sum() - 1.0},)
solR = minimize(negR2, np.ones(n) / n, method="SLSQP",
                bounds=[(0.0, None)] * n, constraints=consR,
                options={"ftol": 1e-14, "maxiter": 2000})
beta = solR.x
R2 = float(beta @ diagK - beta @ K @ beta)
center = beta @ X
print("MEB beta =", np.round(beta, 6), " center =", np.round(center, 6))
print("R^2 =", round(R2, 6))

# --- radius-margin bound ---
T = R2 * wn2
print("radius-margin quantity R^2 ||w||^2 =", round(T, 6))
print("radius-margin LOO rate bound T/n =", round(T / n, 6))

# --- true leave-one-out error, by retraining n times ---
loo_err = 0
for i in range(n):
    idx = [j for j in range(n) if j != i]
    _, wi, bi = hard_margin(X[idx], y[idx])
    pred = np.sign(wi @ X[i] + bi)
    if pred != y[i]:
        loo_err += 1
print("true LOO errors (retrained n times) =", loo_err)
