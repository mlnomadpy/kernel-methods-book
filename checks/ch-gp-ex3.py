"""Worked example: sparse Gaussian-process posteriors on a tiny set.

Full GP on four training points versus two sparse approximations built from two
inducing inputs Z, chosen here as two of the training inputs themselves. The
squared-exponential kernel is k(a,b)=exp(-(a-b)^2 / (2 l^2)) with length scale
l=2, noise variance sigma^2=0.1. Everything follows the unifying Nystrom
notation Q_ab = K_au K_uu^{-1} K_ub of Quinonero-Candela and Rasmussen (2005):

  SoR / DTC : Lambda = sigma^2 I           (deterministic training conditional)
  FITC      : Lambda = diag(K_ff - Q_ff) + sigma^2 I

with shared predictive mean  Q_{*f}(Q_ff+Lambda)^{-1} y  and predictive
variance  K_** - Q_{*f}(Q_ff+Lambda)^{-1} Q_{f*}  (SoR replaces K_** by Q_**).
We also evaluate the Titsias (2009) collapsed variational bound (ELBO) and
compare it to the exact log marginal likelihood. Because two of the inducing
inputs coincide with training inputs, the FITC diagonal correction is exactly
zero at those two rows, which the printout makes visible. Every number the
worked example and surrounding prose display is printed here.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

l = 2.0
sigma2 = 0.1


def k(a, b):
    a = np.atleast_1d(np.asarray(a, float))
    b = np.atleast_1d(np.asarray(b, float))
    return np.exp(-(a[:, None] - b[None, :]) ** 2 / (2 * l * l))


x = np.array([-2.0, -1.0, 0.5, 2.0])   # 4 training inputs
y = np.array([-1.7, -0.8, 0.7, 1.5])   # targets
Z = np.array([-1.0, 0.5])              # 2 inducing inputs = training points 2 and 3
xs = 0.3                               # single test input
n, m = len(x), len(Z)

# ---------- full GP ----------
Kff = k(x, x)
A = Kff + sigma2 * np.eye(n)
ks = k(x, xs)[:, 0]
kss = float(k(xs, xs)[0, 0])           # = 1
alpha = np.linalg.solve(A, y)
mean_full = float(ks @ alpha)
var_full = kss - float(ks @ np.linalg.solve(A, ks))
print("K_ff =\n", np.round(Kff, 4))
print("full: mean(x*) =", round(mean_full, 4), " var(x*) =", round(var_full, 4))

# ---------- sparse pieces (Nystrom) ----------
Kuu = k(Z, Z)
Kuf = k(Z, x)
Kfu = Kuf.T
Kus = k(Z, xs)[:, 0]
Kuu_inv = np.linalg.inv(Kuu)
Qff = Kfu @ Kuu_inv @ Kuf
Qss = float(Kus @ Kuu_inv @ Kus)
diagcorr = np.diag(Kff - Qff)          # >= 0, and 0 at inducing points
print("K_uu =\n", np.round(Kuu, 4))
print("K_uf =\n", np.round(Kuf, 4))
print("Q_ff (Nystrom) =\n", np.round(Qff, 4))
print("diag(K_ff - Q_ff) =", np.round(diagcorr, 4))
print("Q_** =", round(Qss, 4))

# ---------- DTC / SoR (Lambda = sigma^2 I) ----------
Sig_d = np.linalg.inv(Kuu + (1.0 / sigma2) * (Kuf @ Kfu))
mean_dtc = float((1.0 / sigma2) * Kus @ Sig_d @ Kuf @ y)
var_dtc = kss - Qss + float(Kus @ Sig_d @ Kus)   # DTC keeps K_**
var_sor = float(Kus @ Sig_d @ Kus)               # SoR replaces K_** by Q_**
print("DTC/SoR: mean(x*) =", round(mean_dtc, 4))
print("DTC: var(x*) =", round(var_dtc, 4), " (keeps K_**)")
print("SoR: var(x*) =", round(var_sor, 4), " (replaces K_** by Q_**, overconfident)")

# ---------- FITC (Lambda = diag(K_ff - Q_ff) + sigma^2 I) ----------
Lam = np.diag(diagcorr) + sigma2 * np.eye(n)
Lam_inv = np.linalg.inv(Lam)
Sig_f = np.linalg.inv(Kuu + Kuf @ Lam_inv @ Kfu)
mean_fitc = float(Kus @ Sig_f @ Kuf @ Lam_inv @ y)
var_fitc = kss - Qss + float(Kus @ Sig_f @ Kus)
print("Lambda_FITC diag =", np.round(np.diag(Lam), 4))
print("FITC: mean(x*) =", round(mean_fitc, 4), " var(x*) =", round(var_fitc, 4))

# ---------- Titsias collapsed ELBO ----------
Qn = Qff + sigma2 * np.eye(n)
_, logdetQ = np.linalg.slogdet(Qn)
quadQ = float(y @ np.linalg.solve(Qn, y))
logN = -0.5 * quadQ - 0.5 * logdetQ - 0.5 * n * np.log(2 * np.pi)
trace_term = float(np.trace(Kff - Qff))
elbo = logN - trace_term / (2 * sigma2)
print("ELBO: log N(y|0, Q_ff + sig2 I) =", round(logN, 4))
print("ELBO: (1/2 sig2) tr(K_ff - Q_ff) =", round(trace_term / (2 * sigma2), 4),
      " [trace =", round(trace_term, 4), "]")
print("ELBO (variational lower bound) =", round(elbo, 4))

# exact log marginal likelihood for comparison
_, logdetA = np.linalg.slogdet(A)
lml = -0.5 * float(y @ np.linalg.solve(A, y)) - 0.5 * logdetA - 0.5 * n * np.log(2 * np.pi)
print("exact log marginal likelihood =", round(lml, 4))
print("gap (LML - ELBO) =", round(lml - elbo, 4))
