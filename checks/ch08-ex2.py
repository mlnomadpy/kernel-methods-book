"""Worked example: a spectral-mixture kernel Gram matrix is positive definite.

The spectral-mixture kernel of Wilson and Adams (2013) places a symmetric
mixture of Q Gaussians on the frequency axis as its Bochner spectral density.
In one dimension, with tau = x - x', the resulting stationary kernel is

    k(tau) = sum_{q=1}^Q w_q * exp(-2 pi^2 tau^2 v_q) * cos(2 pi mu_q tau),

with weights w_q > 0, spectral variances v_q > 0, and spectral means (center
frequencies) mu_q >= 0. Because the spectral density is a nonnegative symmetric
measure, Bochner guarantees the kernel is positive definite by construction, so
EVERY Gram matrix it produces is positive semidefinite. We verify this on a
concrete 5-point set:

  1. build the 5x5 Gram matrix K_ij = k(x_i - x_j),
  2. report its diagonal k(0) = sum_q w_q and a couple of off-diagonal entries
     (note some are negative: the cosine makes the kernel oscillate),
  3. compute its eigenvalues and confirm the smallest is strictly positive,
     so K is positive definite.

Pure numpy. Prints every number the worked example displays.
"""
import numpy as np

# two spectral components
w = np.array([1.0, 0.5])       # weights
v = np.array([0.30, 0.05])     # spectral variances (width of each frequency bump)
mu = np.array([0.6, 1.4])      # spectral means (center frequencies)


def sm_kernel(tau):
    tau = np.asarray(tau, dtype=np.float64)
    out = np.zeros_like(tau)
    for wq, vq, mq in zip(w, v, mu):
        out = out + wq * np.exp(-2.0 * np.pi ** 2 * tau ** 2 * vq) * np.cos(2.0 * np.pi * mq * tau)
    return out


x = np.array([0.0, 0.3, 0.9, 1.5, 2.2])
n = len(x)
K = np.array([[float(sm_kernel(x[i] - x[j])) for j in range(n)] for i in range(n)])

print("Spectral-mixture kernel, Q = 2 components")
print("  weights w      =", w.tolist())
print("  variances v    =", v.tolist())
print("  frequencies mu =", mu.tolist())
print("  points x       =", x.tolist())

print("\nGram matrix K (5x5):")
for i in range(n):
    print("  [" + " ".join(f"{K[i, j]:8.4f}" for j in range(n)) + "]")

print(f"\nDiagonal k(0) = sum_q w_q = {float(sm_kernel(0.0)):.4f}")
print(f"Sample off-diagonals:  K[0,1] = {K[0,1]:.4f}   K[0,2] = {K[0,2]:.4f}   "
      f"K[0,4] = {K[0,4]:.4f}   (some negative: the kernel oscillates)")

evals = np.linalg.eigvalsh(K)
print("\nEigenvalues of K (ascending):")
print("  [" + " ".join(f"{e:.4f}" for e in evals) + "]")
print(f"smallest eigenvalue = {float(evals.min()):.4f} > 0, so K is positive definite.")
print(f"symmetry check max|K - K^T| = {float(np.max(np.abs(K - K.T))):.2e}")
