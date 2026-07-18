"""Worked example (ch-geom, ex1): the graph Matern and diffusion (heat) kernels
on the 5-cycle C_5, built from the graph-Laplacian eigenpairs, and the numerical
verification that the diffusion kernel is the nu -> infinity limit of the Matern
family.

Graph: cycle on 5 nodes 0-1-2-3-4-0. Adjacency A, degree D = 2 I, Laplacian
L = D - A (symmetric positive semidefinite). Eigenpairs (lambda_i, u_i) are the
discrete Fourier modes of the cycle.

Matern kernel (Borovitskiy et al. 2021):  K_nu = (2 nu / kappa^2 * I + L)^(-nu),
a spectral filter Phi(lambda) = (2 nu / kappa^2 + lambda)^(-nu) applied to each
Laplacian eigenvalue. We take nu = 1, kappa^2 = 1, so K_1 = (2 I + L)^(-1).

Diffusion (heat) kernel (Kondor and Lafferty 2002):  H = exp(-t L), the filter
Phi(lambda) = exp(-t lambda). We take t = 1.

Every kernel here is positive definite because its eigenvalues Phi(lambda_i) are
strictly positive. The final block confirms the diffusion limit: the normalized
Matern filter (1 + (kappa^2/(2 nu)) lambda)^(-nu) -> exp(-(kappa^2/2) lambda), so
with kappa^2 = 2 (matching t = kappa^2/2 = 1) the Matern Gram approaches exp(-L).
All linear algebra is pure numpy (eigendecomposition of a 5x5 symmetric matrix).
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)
m = 5

# --- build the 5-cycle Laplacian -------------------------------------------
A = np.zeros((m, m))
for i in range(m):
    A[i, (i + 1) % m] = 1.0
    A[i, (i - 1) % m] = 1.0
D = np.diag(A.sum(axis=1))
L = D - A
print("Laplacian L of C_5 =\n", L)

# eigenpairs of L (Fourier modes of the cycle)
lam, U = np.linalg.eigh(L)
print("Laplacian eigenvalues lambda =", np.round(lam, 4))

def spectral_kernel(phi):
    """Assemble sum_i phi(lambda_i) u_i u_i^T from the eigenpairs."""
    return (U * phi(lam)) @ U.T

# --- Matern kernel, nu = 1, kappa^2 = 1  ->  (2 I + L)^(-1) -----------------
nu, kappa2 = 1.0, 1.0
Kmat = spectral_kernel(lambda l: (2 * nu / kappa2 + l) ** (-nu))
print("\nMatern kernel K_1 = (2 I + L)^(-1) =\n", Kmat)
print("eigenvalues of K_1 = 1/(2+lambda) =", np.round(np.sort(1.0 / (2 + lam)), 4))
print("min eigenvalue of K_1 =", round(float(np.min(np.linalg.eigvalsh(Kmat))), 6))

# normalized to unit diagonal (correlation form); C_5 is vertex-transitive so
# every diagonal entry is equal and normalization is a single rescaling.
d = np.sqrt(np.diag(Kmat))
Kmat_n = Kmat / np.outer(d, d)
print("normalized Matern (unit diagonal), row 0 =", np.round(Kmat_n[0], 4))

# --- diffusion (heat) kernel, t = 1  ->  exp(-L) ---------------------------
t = 1.0
H = spectral_kernel(lambda l: np.exp(-t * l))
print("\ndiffusion kernel H = exp(-L) =\n", H)
print("eigenvalues of H = exp(-lambda) =", np.round(np.sort(np.exp(-lam)), 4))
print("min eigenvalue of H =", round(float(np.min(np.linalg.eigvalsh(H))), 6))

# --- diffusion kernel is the nu -> infinity Matern limit --------------------
# normalized Matern filter g_nu(lambda) = (1 + (kappa2/(2 nu)) lambda)^(-nu),
# with kappa2 = 2 so that kappa2/2 = 1 = t. As nu grows, g_nu -> exp(-lambda).
kappa2_lim = 2.0
for nu_big in [2, 10, 50]:
    Gnu = spectral_kernel(
        lambda l, n=nu_big: (1 + (kappa2_lim / (2 * n)) * l) ** (-n)
    )
    print(f"nu={nu_big:>3}: max|G_nu - exp(-L)| =",
          round(float(np.max(np.abs(Gnu - H))), 6))
