"""Worked example: spectral learning curve of kernel ridge regression, and the
omniscient / replica prediction of Bordelon, Canatar, and Pehlevan (2020) and
Canatar, Bordelon, and Pehlevan (2021).

Setup.  Finite input domain of P points with the uniform measure.  We build an
orthonormal eigenbasis phi_rho (phi_rho(x) = sqrt(P) * U[x, rho] for an
orthogonal U, so (1/P) sum_x phi_rho phi_sigma = delta_{rho sigma}) and a kernel
    K(x, x') = sum_rho eta_rho phi_rho(x) phi_rho(x'),  eta_rho = rho^{-a}.
The target is  f*(x) = sum_rho abar_rho phi_rho(x)  with abar_rho^2 = rho^{-b}.

KRR.  Draw n points S uniformly (with replacement), y_S = f*(x_S) + noise,
predict  fhat = K(.,S) (K(S,S) + n*lam I)^{-1} y_S, and measure
    Eg = (1/P) sum_x (fhat(x) - f*(x))^2,  averaged over deterministic draws.

Omniscient prediction.  Solve for the effective ridge kappa > 0:
    kappa = lam + sum_rho  eta_rho * kappa / (n*eta_rho + kappa).
Set the per-mode learnability  L_rho = n*eta_rho / (n*eta_rho + kappa)  and
    gamma = (1/n) sum_rho L_rho^2 .
Then the predicted error is
    Eg_omni = 1/(1-gamma) * ( sum_rho (1-L_rho)^2 abar_rho^2 + sigma^2 * gamma ).
Each mode contributes  (1/(1-gamma)) (1-L_rho)^2 abar_rho^2 : mode rho is learned
once n*eta_rho crosses kappa (L_rho -> 1), unlearned while n*eta_rho << kappa.
"""
import numpy as np

P = 1200                # size of the (discrete) input domain
a = 1.5                 # eigenvalue decay eta_rho = rho^{-a}
b = 2.0                 # target/source decay abar_rho^2 = rho^{-b}
lam = 1e-4              # ridge
sigma2 = 0.0            # noiseless target (isolate the spectral bias)
draws = 150             # deterministic averaging of KRR draws
ns = [10, 20, 40, 80, 160, 320]

rho = np.arange(1, P + 1)
eta = rho.astype(float) ** (-a)
abar = rho.astype(float) ** (-b / 2.0)     # so abar^2 = rho^{-b}

# fixed orthonormal eigenbasis on the P grid points
rng0 = np.random.default_rng(0)
A = rng0.standard_normal((P, P))
U, _ = np.linalg.qr(A)                     # orthogonal: columns orthonormal
Phi = np.sqrt(P) * U                       # Phi[x, rho] = phi_rho(x)
# (1/P) Phi^T Phi = I  -> eigenfunctions orthonormal wrt uniform measure
# K(x,x') = sum_rho eta_rho phi_rho(x) phi_rho(x'); with (1/P) sum_x phi phi = I
# the integral operator T_K has eigenvalues exactly eta_rho.
K = Phi @ np.diag(eta) @ Phi.T             # = P * U diag(eta) U^T
fstar = Phi @ abar                          # f*(x) = sum_rho abar_rho phi_rho(x)

def solve_kappa(n):
    lo, hi = 1e-12, lam + eta.sum() + 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        h = mid - lam - np.sum(eta * mid / (n * eta + mid))
        if h > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def omniscient(n):
    kappa = solve_kappa(n)
    L = n * eta / (n * eta + kappa)
    gamma = np.sum(L ** 2) / n
    bias = np.sum((1.0 - L) ** 2 * abar ** 2)
    Eg = (bias + sigma2 * gamma) / (1.0 - gamma)
    return kappa, gamma, Eg

print(f"{'n':>5} {'kappa':>12} {'gamma':>8} {'Eg omni':>12} {'Eg sim (KRR)':>14}")
rows = []
for n in ns:
    kappa, gamma, Eg_omni = omniscient(n)
    sims = []
    for d in range(draws):
        rng = np.random.default_rng(5000 + d)
        S = rng.integers(0, P, size=n)              # uniform w/ replacement
        KSS = K[np.ix_(S, S)]
        yS = fstar[S] + (np.sqrt(sigma2) * rng.standard_normal(n) if sigma2 > 0 else 0.0)
        alpha = np.linalg.solve(KSS + n * lam * np.eye(n), yS)
        fhat = K[:, S] @ alpha
        sims.append(np.mean((fhat - fstar) ** 2))
    Eg_sim = float(np.mean(sims))
    rows.append((n, kappa, gamma, Eg_omni, Eg_sim))
    print(f"{n:>5} {kappa:>12.6f} {gamma:>8.4f} {Eg_omni:>12.5f} {Eg_sim:>14.5f}")

# empirical power-law exponent of the omniscient curve (large-n points only,
# where the finite-domain tail cutoff is least contaminating)
ns_arr = np.array([r[0] for r in rows], float)
eg_arr = np.array([r[3] for r in rows], float)
slope = np.polyfit(np.log(ns_arr[-3:]), np.log(eg_arr[-3:]), 1)[0]
print(f"\nfitted power-law exponent (last 3 pts)  Eg_omni ~ n^(beta),  beta = {slope:.3f}")
