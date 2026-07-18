"""Worked example: expected improvement in closed form.

Same squared-exponential surrogate as ch-bo-ex1 (length scale l = 0.5, noise
variance sigma^2 = 0.01), conditioned on the two initial observations
(0, sin 0) and (2, sin 6). The incumbent is the best value seen so far,
f_plus = max(y) = 0. For a Gaussian posterior f(x) ~ N(mu, sigma^2) the
expected improvement E[max(f(x) - f_plus, 0)] has the closed form

    EI(x) = (mu - f_plus) * Phi(z) + sigma * phi(z),   z = (mu - f_plus) / sigma,

with phi, Phi the standard-normal pdf and cdf. We evaluate it on the grid and
in full detail at the single candidate x = 1.0. Prints every number shown.
"""
import numpy as np
from math import erf

grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
l = 0.5
sigma2 = 0.01

def f(x):
    return np.sin(3.0 * x)

def k(a, b):
    return np.exp(-(a - b) ** 2 / (2.0 * l ** 2))

def Phi(z):
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))

def phi(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)

# posterior after the two initial observations
Xobs = np.array([0.0, 2.0])
Yobs = np.array([float(f(0.0)), float(f(2.0))])
K = k(Xobs[:, None], Xobs[None, :])
A = K + sigma2 * np.eye(2)
Ainv = np.linalg.inv(A)

mu = np.array([k(Xobs, xs) @ Ainv @ Yobs for xs in grid])
var = np.array([k(xs, xs) - k(Xobs, xs) @ Ainv @ k(Xobs, xs) for xs in grid])
sd = np.sqrt(np.maximum(var, 0.0))

fplus = float(Yobs.max())
print("incumbent f_plus = max(y) =", round(fplus, 4))
print("posterior mean   mu(grid)   =", np.round(mu, 4))
print("posterior std    sigma(grid) =", np.round(sd, 4))

def EI(m, s, fp):
    if s <= 0.0:
        return max(m - fp, 0.0)
    z = (m - fp) / s
    return (m - fp) * Phi(z) + s * phi(z)

ei = np.array([EI(mu[i], sd[i], fplus) for i in range(len(grid))])
print("expected improvement EI(grid) =", np.round(ei, 4))
j = int(np.argmax(ei))
print("EI picks x =", grid[j], " (grid index", j, "), EI =", round(float(ei[j]), 4))
print()

# full detail at the candidate x = 1.0 (grid index 2)
i = 2
m, s = float(mu[i]), float(sd[i])
z = (m - fplus) / s
print("--- candidate x = 1.0 ---")
print("mu =", round(m, 4), " sigma =", round(s, 4), " f_plus =", round(fplus, 4))
print("z = (mu - f_plus)/sigma =", round(z, 4))
print("Phi(z) =", round(Phi(z), 4), " phi(z) =", round(phi(z), 4))
term1 = (m - fplus) * Phi(z)
term2 = s * phi(z)
print("(mu - f_plus)*Phi(z) =", round(term1, 4))
print("sigma*phi(z) =", round(term2, 4))
print("EI(1.0) =", round(term1 + term2, 4))
