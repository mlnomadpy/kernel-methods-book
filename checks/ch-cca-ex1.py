"""Worked example: unregularized kernel CCA overfits to correlation 1, and the
(K + kappa I) shrinkage repairs it.

Three paired objects, two scalar views. Each view is embedded with the Gaussian
(RBF) kernel k(u, v) = exp(-(u - v)^2 / (2 sigma^2)), sigma = 1, which is
universal/characteristic, so both 3x3 Gram matrices K_a, K_b are full rank
(invertible). Kernel CCA solves the generalized eigenproblem

    [ 0      K_a K_b ] [a]        [ B_a   0  ] [a]
    [ K_b K_a   0    ] [b] = rho  [ 0    B_b ] [b],

with B_a = K_a^2, B_b = K_b^2 in the raw (unregularized) problem, and with the
(K + kappa I) shrinkage B_a = (K_a + kappa I)^2, B_b = (K_b + kappa I)^2 in the
regularized problem (Hardoon, Szedmak, Shawe-Taylor 2004; Fukumizu, Bach,
Gretton 2007). The top generalized eigenvalue rho is the leading canonical
correlation.

Prints the two Gram matrices, the unregularized top correlation (which is 1),
the regularized top correlation for kappa = 1, and the recovered dual weights.
"""
import numpy as np

np.set_printoptions(suppress=True)

# --- two scalar views of three objects --------------------------------------
xa = np.array([-1.0, 0.0, 2.0])   # view A
xb = np.array([0.0, 1.0, 1.5])    # view B
n = len(xa)
sigma = 1.0


def rbf_gram(v):
    d2 = (v[:, None] - v[None, :]) ** 2
    return np.exp(-d2 / (2 * sigma ** 2))


Ka = rbf_gram(xa)
Kb = rbf_gram(xb)
print("K_a =\n", np.round(Ka, 3))
print("K_b =\n", np.round(Kb, 3))
print("K_a invertible (det) =", round(float(np.linalg.det(Ka)), 4))
print("K_b invertible (det) =", round(float(np.linalg.det(Kb)), 4))


def top_kcca(kappa):
    """Return (rho, alpha, beta) for the leading pair with (K+kI)^2 shrinkage."""
    A = np.block([[np.zeros((n, n)), Ka @ Kb],
                  [Kb @ Ka, np.zeros((n, n))]])
    if kappa == 0.0:
        Ba, Bb = Ka @ Ka, Kb @ Kb
    else:
        Ra = Ka + kappa * np.eye(n)
        Rb = Kb + kappa * np.eye(n)
        Ba, Bb = Ra @ Ra, Rb @ Rb
    B = np.block([[Ba, np.zeros((n, n))],
                  [np.zeros((n, n)), Bb]])
    # generalized eigenproblem A w = rho B w, B positive definite
    evals, evecs = np.linalg.eig(np.linalg.solve(B, A))
    evals = evals.real
    k = int(np.argmax(evals))
    w = evecs[:, k].real
    alpha, beta = w[:n], w[n:]
    # fix the overall sign of the coupled pair (alpha, beta) for determinism
    s = 1.0 if alpha[0] >= 0 else -1.0
    return evals[k], s * alpha, s * beta


# kappa = 0 is the raw, degenerate problem; kappa > 0 is the shrinkage dial
for kappa in [0.0, 0.1, 0.5, 1.0]:
    rho, _, _ = top_kcca(kappa)
    tag = "unregularized" if kappa == 0.0 else "kappa=%.1f" % kappa
    print("top canonical correlation rho (%s) =" % tag, round(float(rho), 3))

# recover the dual weights at the headline regularization kappa = 1
rho, alpha, beta = top_kcca(1.0)
alpha = alpha / np.linalg.norm(alpha)
beta = beta / np.linalg.norm(beta)
print("regularized (kappa=1) alpha (unit) =", np.round(alpha, 3))
print("regularized (kappa=1) beta  (unit) =", np.round(beta, 3))
