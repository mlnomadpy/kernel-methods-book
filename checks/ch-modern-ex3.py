"""Worked example: benign vs harmful overfitting of the minimum-norm interpolant
(Bartlett, Long, Lugosi, and Tsigler 2020).

Linear model  y = x^T beta* + eps,  x ~ N(0, Sigma),  eps ~ N(0, sigma^2),
Sigma = diag(lambda_1 >= lambda_2 >= ...).  From n < p samples we take the
minimum-norm interpolant  beta_hat = pinv(X) y  and measure the excess risk
    R(beta_hat) = (beta_hat - beta*)^T Sigma (beta_hat - beta*) .

BLLT effective ranks and the key index:
    r_k(Sigma)  = (sum_{i>k} lambda_i) / lambda_{k+1}
    R_k(Sigma)  = (sum_{i>k} lambda_i)^2 / (sum_{i>k} lambda_i^2)
    k*          = min { k >= 0 : r_k(Sigma) >= b n }   (constant b, here b = 1).
Their bound: the variance part of the risk is of order
    sigma^2 ( k*/n + n / R_{k*}(Sigma) ),
so interpolation is benign (risk -> 0) exactly when k*/n -> 0 and R_{k*}/n -> inf:
the tail past k* must have effective rank far exceeding n so it can absorb the
label noise while barely perturbing the fit.

We use a spiked covariance: s signal eigenvalues = 1, then a flat junk tail of
dimension m at level tau.  A wide junk tail (m >> n) is benign; a narrow one is
harmful.  Every displayed number is computed here.
"""
import numpy as np

def eff_ranks(lam, k):
    tail = lam[k:]
    r = tail.sum() / lam[k] if k < len(lam) else 0.0   # lambda_{k+1} is lam[k]
    R = (tail.sum() ** 2) / (tail ** 2).sum()
    return r, R

def kstar(lam, n, b=1.0):
    for k in range(len(lam)):
        r, _ = eff_ranks(lam, k)
        if r >= b * n:
            return k
    return len(lam) - 1

def run(name, s, m, tau, n, p_note=""):
    p = s + m
    lam = np.concatenate([np.ones(s), tau * np.ones(m)])   # spiked covariance
    sigma2 = 0.25
    sigma = np.sqrt(sigma2)
    beta = np.zeros(p)
    beta[:s] = 1.0 / np.sqrt(s)          # signal lives in the spike, ||beta||^2 = 1

    ks = kstar(lam, n, b=1.0)
    r_ks, R_ks = eff_ranks(lam, ks)
    var_pred = sigma2 * (ks / n + n / R_ks)

    risks = []
    for seed in range(120):
        rng = np.random.default_rng(9000 + seed)
        X = rng.standard_normal((n, p)) * np.sqrt(lam)     # rows ~ N(0, Sigma)
        y = X @ beta + sigma * rng.standard_normal(n)
        beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)   # min-norm interpolant
        d = beta_hat - beta
        risks.append(float(np.sum(lam * d * d)))           # d^T Sigma d
    risk = float(np.mean(risks))

    r0, R0 = eff_ranks(lam, 0)
    tail_var = lam[ks:].sum()
    print(f"--- {name} (s={s}, junk dim m={m}, tau={tau}, n={n}, p={p}) {p_note}")
    print(f"    signal energy ||beta*||_Sigma^2 = {float(np.sum(lam*beta*beta)):.3f}  (null-predictor risk)")
    print(f"    tail variance sum_{{i>k*}} lambda_i = {tail_var:.1f}")
    print(f"    r_0 = tr(Sigma)/lambda_1 = {r0:.1f}   (r_0/n = {r0/n:.2f})")
    print(f"    k* = {ks}   (k*/n = {ks/n:.3f})")
    print(f"    R_{{k*}} = {R_ks:.1f}   (n/R_{{k*}} = {n/R_ks:.4f})")
    print(f"    BLLT variance term  sigma^2 (k*/n + n/R_k*) = {var_pred:.4f}")
    print(f"    simulated total excess risk (min-norm interpolant) = {risk:.4f}")
    print()
    return ks, R_ks, var_pred, risk

n = 100
print("BENIGN: a wide, low-energy junk tail absorbs the noise")
run("benign", s=5, m=20000, tau=0.001, n=n, p_note="(R_k* >> n, tail var << n)")
print("HARMFUL: a narrow junk tail, nowhere to hide the noise")
run("harmful", s=5, m=150, tau=0.5, n=n, p_note="(R_k* ~ n)")
