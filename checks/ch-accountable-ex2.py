"""ch-accountable-ex2: split-conformal prediction on a kernel-ridge regressor.

Native Gaussian error bars fail under a misspecified noise model (ex1). Split
conformal repairs coverage with a distribution-free, finite-sample guarantee on
ANY predictor. Recipe (Vovk-Gammerman-Shafer 2005; Lei et al. 2018): fit f-hat on
a training split, score the calibration residuals s_i = |y_i - f-hat(x_i)|, take
q-hat = the ceil((n+1)(1-alpha))-th smallest score, and predict [f-hat(x)-q-hat,
f-hat(x)+q-hat]. Coverage is guaranteed in [1-alpha, 1-alpha+1/(n+1)].

Same heteroscedastic truth as ex1, so the naive-vs-conformal contrast is direct.
"""
import numpy as np

rng = np.random.default_rng(1)

def rbf(A, B, ell):
    A = np.asarray(A).reshape(-1, 1); B = np.asarray(B).reshape(-1, 1)
    d = A - B.T
    return np.exp(-(d ** 2) / (2 * ell ** 2))

def truth_mean(x): return np.sin(3 * x)
def truth_sd(x):   return 0.05 + 0.45 * x

ell, lam = 0.15, 1e-2
n_tr, n_cal, n_te = 200, 500, 8000
xtr = rng.uniform(0, 1, n_tr); ytr = truth_mean(xtr) + truth_sd(xtr) * rng.standard_normal(n_tr)
xca = rng.uniform(0, 1, n_cal); yca = truth_mean(xca) + truth_sd(xca) * rng.standard_normal(n_cal)
xte = rng.uniform(0, 1, n_te); yte = truth_mean(xte) + truth_sd(xte) * rng.standard_normal(n_te)

# fit KRR on the training split
K = rbf(xtr, xtr, ell) + lam * np.eye(n_tr)
alpha = np.linalg.solve(K, ytr)
def f_hat(xs): return rbf(xs, xtr, ell) @ alpha

alpha_lvl = 0.10
# calibration residuals -> conformal quantile
res = np.abs(yca - f_hat(xca))
k = int(np.ceil((n_cal + 1) * (1 - alpha_lvl)))         # rank of the quantile
qhat = np.sort(res)[k - 1]
print("split conformal on KRR (RBF ell=0.15, ridge 1e-2)")
print(f"  target coverage 1 - alpha        : {1 - alpha_lvl:.2f}")
print(f"  quantile rank k = ceil((n+1)(1-a)): {k} of {n_cal}")
print(f"  q-hat (band half-width)          : {qhat:.3f}")

# conformal coverage on fresh test data
mte = f_hat(xte)
cov_conf = float(np.mean((yte >= mte - qhat) & (yte <= mte + qhat)))
lo_bound, hi_bound = 1 - alpha_lvl, 1 - alpha_lvl + 1.0 / (n_cal + 1)
print(f"  conformal empirical coverage     : {cov_conf:.3f}   (guarantee [{lo_bound:.3f}, {hi_bound:.3f}])")

# naive: assume one homoscedastic sigma estimated from training residuals
sig_hat = float(np.std(ytr - f_hat(xtr)))
z = 1.645
naive_in = (yte >= mte - z * sig_hat) & (yte <= mte + z * sig_hat)
cov_naive = float(np.mean(naive_in))
print(f"  naive +-1.645 sigma-hat coverage : {cov_naive:.3f}   (sigma-hat = {sig_hat:.3f})")
# conditional coverage in the high-noise half exposes the marginal-vs-conditional gap
hi_half = xte > 0.5
conf_in = (yte >= mte - qhat) & (yte <= mte + qhat)
print(f"  high-noise half: conformal {np.mean(conf_in[hi_half]):.3f}  vs naive {np.mean(naive_in[hi_half]):.3f}")
print(f"  conformal band width 2*q-hat     : {2 * qhat:.3f}")
print(f"  naive band width 2*1.645*sigma   : {2 * z * sig_hat:.3f}")
