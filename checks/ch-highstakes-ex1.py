"""ch-highstakes-ex1: a quasi-periodic GP recovers a stellar rotation period.

Spots rotate into and out of view but evolve, so a stellar light curve is
quasi-periodic, not sinusoidal. The quasi-periodic kernel writes that physics down:
  k(tau) = A exp(-tau^2/(2 l^2) - Gamma sin^2(pi tau / P)),
with period P, spot-coherence length l, and amplitude Gamma. Unlike a periodogram
peak, marginalizing the GP marginal likelihood over P returns a posterior with a
credible interval. We synthesize a curve at a known P and recover it.
"""
import numpy as np

rng = np.random.default_rng(7)

def qp_kernel(t1, t2, A, ell, Gamma, P):
    tau = t1[:, None] - t2[None, :]
    return A * np.exp(-tau ** 2 / (2 * ell ** 2) - Gamma * np.sin(np.pi * np.abs(tau) / P) ** 2)

# truth
P_true, ell_true, Gamma_true, A_true = 10.0, 30.0, 2.0, 1.0
sigma = 0.15
n = 120
t = np.sort(rng.uniform(0, 60, n))
Ktt = qp_kernel(t, t, A_true, ell_true, Gamma_true, P_true) + 1e-8 * np.eye(n)
y = np.linalg.cholesky(Ktt) @ rng.standard_normal(n) + sigma * rng.standard_normal(n)

def log_marg_like(P):
    K = qp_kernel(t, t, A_true, ell_true, Gamma_true, P) + sigma ** 2 * np.eye(n)
    L = np.linalg.cholesky(K)
    a = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return -0.5 * y @ a - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)

grid = np.linspace(5, 20, 601)
dx = grid[1] - grid[0]
ll = np.array([log_marg_like(P) for P in grid])
post = np.exp(ll - ll.max()); post /= post.sum() * dx           # flat prior on P
P_map = grid[np.argmax(ll)]
cdf = np.cumsum(post) * dx
lo = grid[np.searchsorted(cdf, 0.16)]
hi = grid[np.searchsorted(cdf, 0.84)]
mean_P = float(np.sum(grid * post) * dx)
print("quasi-periodic GP period recovery")
print(f"  true period P            : {P_true:.2f} d")
print(f"  MAP period               : {P_map:.2f} d")
print(f"  posterior mean period    : {mean_P:.2f} d")
print(f"  68% credible interval    : [{lo:.2f}, {hi:.2f}] d")

# contrast: a Lomb-Scargle-style single peak of the periodogram (no interval)
freqs = 2 * np.pi / grid
power = np.array([np.abs(np.sum(y * np.exp(-1j * f * t))) ** 2 for f in freqs])
print(f"  periodogram peak period  : {grid[np.argmax(power)]:.2f} d   (a point, no credible interval)")

assert abs(P_map - 10.10) < 1e-12
assert abs(mean_P - 10.10) < 0.01
assert (lo, hi) == (10.0, 10.2)
assert abs(grid[np.argmax(power)] - 10.725) < 1e-12
assert np.isfinite(ll).all()
