"""ch-accountable-ex1: a GP credible interval, and where it is honest vs not.

A GP returns a predictive mean and variance in closed form. Under a correctly
specified model those give calibrated (correct-coverage) intervals; under a
misspecified noise model they can under-cover. We show both the honest behaviour
(the band pinches at data, balloons in gaps) and the failure (a homoscedastic
noise assumption under-covers in a high-noise region), which motivates the
conformal fix in ex2.

Pure numpy; every printed number is used verbatim in the worked example.
"""
import numpy as np

rng = np.random.default_rng(0)

def rbf(A, B, ell):
    A = np.asarray(A).reshape(-1, 1)
    B = np.asarray(B).reshape(-1, 1)
    d = A - B.T
    return np.exp(-(d ** 2) / (2 * ell ** 2))

# --- part 1: the honest error bar on a tiny, hand-sized set -------------------
# 8 noiseless points of sin(x) on [0,6] with a deliberate gap in (2.5, 4.5)
xt = np.array([0.3, 0.9, 1.6, 2.2, 4.7, 5.1, 5.6, 6.0])
yt = np.sin(xt)
ell, sig = 1.0, 0.05
K = rbf(xt, xt, ell) + sig ** 2 * np.eye(len(xt))
L = np.linalg.cholesky(K)
alpha = np.linalg.solve(L.T, np.linalg.solve(L, yt))

def gp_predict(xs):
    ks = rbf(xs, xt, ell)
    mu = ks @ alpha
    v = np.linalg.solve(L, ks.T)
    var = 1.0 - np.sum(v ** 2, axis=0)
    return mu, np.sqrt(np.maximum(var, 0))

xs = np.array([0.9, 3.5, 6.5])           # at a datum, in the gap, extrapolating
mu, sd = gp_predict(xs)
print("PART 1: honest error bar (RBF ell=1.0, noise 0.05)")
print(f"  posterior std at a training point x=0.9 : {sd[0]:.3f}")
print(f"  posterior std in the gap        x=3.5 : {sd[1]:.3f}")
print(f"  posterior std extrapolating     x=6.5 : {sd[2]:.3f}")

# --- part 2: coverage of the nominal 90% interval under a misspecified noise --
# truth: wiggly mean + heteroscedastic noise that grows across the domain.
def truth_mean(x):
    return np.sin(3 * x)
def truth_sd(x):
    return 0.05 + 0.45 * x            # small noise at x=0, large at x=1

n_tr = 40
xr = np.sort(rng.uniform(0, 1, n_tr))
yr = truth_mean(xr) + truth_sd(xr) * rng.standard_normal(n_tr)

# fit a GP that ASSUMES one homoscedastic noise = the average sd (misspecified)
ell2 = 0.15
sig2 = float(np.mean(truth_sd(xr)))     # a single number, wrong in both tails
K2 = rbf(xr, xr, ell2) + sig2 ** 2 * np.eye(n_tr)
L2 = np.linalg.cholesky(K2)
a2 = np.linalg.solve(L2.T, np.linalg.solve(L2, yr))

def gp2(xs):
    ks = rbf(xs, xr, ell2)
    mu = ks @ a2
    v = np.linalg.solve(L2, ks.T)
    var = 1.0 - np.sum(v ** 2, axis=0) + sig2 ** 2   # include noise for a predictive interval
    return mu, np.sqrt(np.maximum(var, 0))

# a large fresh test set for a stable coverage estimate
xte = rng.uniform(0, 1, 4000)
yte = truth_mean(xte) + truth_sd(xte) * rng.standard_normal(4000)
mte, ste = gp2(xte)
z = 1.645                               # nominal 90%
lo, hi = mte - z * ste, mte + z * ste
inside = (yte >= lo) & (yte <= hi)
cov_all = float(np.mean(inside))
cov_hi = float(np.mean(inside[xte > 0.5]))    # the high-noise half
print("\nPART 2: nominal 90% GP interval under a homoscedastic (wrong) noise model")
print(f"  assumed noise sigma                 : {sig2:.3f}")
print(f"  empirical coverage, whole domain    : {cov_all:.3f}")
print(f"  empirical coverage, high-noise half : {cov_hi:.3f}   (target 0.90)")
