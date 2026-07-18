"""Worked example: FALKON's preconditioner collapses the CG iteration count.

Nystrom kernel ridge regression solves an m x m system in the landmark
coefficients,
    H beta = v,   H = (1/n) K_nm^T K_nm + lam K_mm,   v = (1/n) K_nm^T y.
With a statistically small ridge lam and a decaying kernel spectrum, H is badly
conditioned and plain conjugate gradient needs many matrix-vector products.
FALKON preconditions with P^{-1} = B B^T, built from two Cholesky
factorizations of m x m matrices:
    K_mm = T^T T,     (1/m) T T^T + lam I = A^T A,     B = T^{-1} A^{-1},
which is exactly the H one would get if the m landmarks were the whole
dataset. CG on the transformed operator M = B^T H B then converges in a
handful of steps. Laplacian kernel on the line, landmarks sampled uniformly
from the data, so every matrix is honestly representable in float64. The
script counts iterations of plain CG on H and preconditioned CG on M to the
same tolerance, reports all three condition numbers, and confirms the two
solvers return the same predictor. Every displayed number is printed here.
"""
import numpy as np

rng = np.random.default_rng(0)
n = 1000         # data points
m = 100          # landmarks (size of the solved system)
sigma = 0.5      # Laplacian kernel width
lam = 1e-5       # ridge parameter
tol = 1e-6       # CG relative-residual tolerance
maxit = 2000

def kern(A, B):
    return np.exp(-np.abs(A[:, None] - B[None, :]) / sigma)

# data spread over [0,4]; landmarks are a uniform subsample of the data
X = rng.uniform(0.0, 4.0, size=n)
Z = np.sort(X[rng.choice(n, size=m, replace=False)])
ytrue = np.sin(3 * X)
y = ytrue + 0.1 * rng.standard_normal(n)

Kmm = kern(Z, Z)
Knm = kern(X, Z)
H = (Knm.T @ Knm) / n + lam * Kmm
v = (Knm.T @ y) / n
print("n =", n, " m =", m, " sigma =", sigma, " lambda =", lam, " tol =", tol)
print("cond(K_mm) =", f"{np.linalg.cond(Kmm):.2e}")
print("cond(H)    =", f"{np.linalg.cond(H):.2e}")

def cg(matvec, b, tol, maxit):
    """Standard conjugate gradient from a zero start; returns (x, iters)."""
    x = np.zeros_like(b)
    r = b - matvec(x)
    pdir = r.copy()
    rs = r @ r
    b_norm = np.linalg.norm(b)
    for it in range(1, maxit + 1):
        Ap = matvec(pdir)
        alpha = rs / (pdir @ Ap)
        x = x + alpha * pdir
        r = r - alpha * Ap
        if np.linalg.norm(r) <= tol * b_norm:
            return x, it
        rs_new = r @ r
        pdir = r + (rs_new / rs) * pdir
        rs = rs_new
    return x, maxit

# plain CG on H beta = v
beta_plain, it_plain = cg(lambda w: H @ w, v, tol, maxit)
print("plain CG iterations =", it_plain)

# FALKON preconditioner: two Cholesky factorizations of m x m matrices
T = np.linalg.cholesky(Kmm).T                               # K_mm = T^T T (T upper)
A = np.linalg.cholesky((T @ T.T) / m + lam * np.eye(m)).T   # (1/m) T T^T + lam I = A^T A
B = np.linalg.inv(T) @ np.linalg.inv(A)                     # P^{-1} = B B^T
M = B.T @ H @ B                                             # transformed operator
print("cond(M)    =", f"{np.linalg.cond(M):.2e}")

# preconditioned CG: solve M u = B^T v, then beta = B u
u, it_pcg = cg(lambda w: M @ w, B.T @ v, tol, maxit)
beta_pcg = B @ u
print("preconditioned CG iterations =", it_pcg)

# both reach the tolerance and return the same predictor
res_plain = np.linalg.norm(H @ beta_plain - v) / np.linalg.norm(v)
res_pcg = np.linalg.norm(H @ beta_pcg - v) / np.linalg.norm(v)
print("relative residual, plain =", f"{res_plain:.1e}", " pcg =", f"{res_pcg:.1e}")
f_plain, f_pcg = Knm @ beta_plain, Knm @ beta_pcg
pred_diff = np.linalg.norm(f_plain - f_pcg) / np.linalg.norm(f_pcg)
print("prediction agreement ||f_plain - f_pcg|| / ||f_pcg|| =", f"{pred_diff:.1e}")
rmse_plain = np.sqrt(np.mean((f_plain - ytrue) ** 2))
rmse_pcg = np.sqrt(np.mean((f_pcg - ytrue) ** 2))
print("fit RMSE to noiseless truth, plain =", f"{rmse_plain:.3f}", " pcg =", f"{rmse_pcg:.3f}")
print("speedup in iterations =", round(it_plain / it_pcg, 1), "x")
