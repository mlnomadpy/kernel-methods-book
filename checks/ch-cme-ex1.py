"""Worked example 1: the empirical conditional mean embedding.

Joint sample of n = 4 pairs (x_i, y_i) with y roughly tracking x. Gaussian
kernels k on X (bandwidth sigma = 1) and l on Y (bandwidth tau = 1). Form the
regularized CME weight vector beta(x*) = (K + n*lam I)^{-1} k_{x*} at the test
input x* = 1.5, then read off two conditional expectations: the plug-in
estimate of E[Y | X = x*] using g(y) = y, and the rigorous RKHS estimate of
E[l(y0, Y) | X = x*] for the genuine RKHS function g = l(y0, .) with y0 = 2.0.
Prints every number the worked example displays.
"""
import numpy as np

x = np.array([0.0, 1.0, 2.0, 3.0])   # training inputs
y = np.array([0.0, 1.2, 1.8, 3.0])   # training outputs
n = 4
sigma = 1.0
tau = 1.0
lam = 0.125                          # population reg; gamma = n*lam = 0.5
gamma = n * lam
xstar = 1.5
y0 = 2.0

def kx(a, b):
    return np.exp(-(a - b) ** 2 / (2 * sigma ** 2))

def ky(a, b):
    return np.exp(-(a - b) ** 2 / (2 * tau ** 2))

K = kx(x[:, None], x[None, :])
L = ky(y[:, None], y[None, :])
print("K =\n", np.round(K, 4))
print("L =\n", np.round(L, 4))

kxs = kx(x, xstar)                   # k_{x*} = [k(x_i, x*)]
print("k_{x*} =", np.round(kxs, 4))

A = K + gamma * np.eye(n)
print("gamma = n*lam =", round(gamma, 4))
beta = np.linalg.solve(A, kxs)       # CME weights at x*
print("beta(x*) =", np.round(beta, 4))
print("sum beta =", round(float(beta.sum()), 4))

# plug-in conditional mean of Y (g(y) = y, not in the RKHS, but the standard
# predictor for the conditional mean of the response)
EY = float(beta @ y)
print("E[Y | X=1.5] (plug-in) =", round(EY, 4))

# rigorous RKHS estimate for g = l(y0, .): E[g(Y)|X=x*] = sum_i beta_i g(y_i)
g_at_yi = ky(y, y0)
print("g(y_i) = l(y_i, 2.0) =", np.round(g_at_yi, 4))
Eg = float(beta @ g_at_yi)
print("E[l(2.0, Y) | X=1.5] =", round(Eg, 4))

# for contrast: value of that same g at the plug-in conditional mean
print("l(2.0, E[Y|X=1.5]) =", round(float(ky(EY, y0)), 4))
