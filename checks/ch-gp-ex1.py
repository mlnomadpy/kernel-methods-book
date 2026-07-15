"""Worked example: Gaussian-process posterior on three training points.

Gaussian (squared-exponential) kernel k(x,x') = exp(-(x-x')^2 / (2 l^2)) with
l = 1, additive noise variance sigma^2 = 0.1. Train on three points, then
compute the predictive mean and variance at a test location x* = 0.5, plus the
log marginal likelihood. Prints every number the worked example displays.
"""
import numpy as np

x = np.array([0.0, 1.0, 2.0])       # training inputs
y = np.array([1.0, 0.5, -0.5])      # training targets
l = 1.0
sigma2 = 0.1
xstar = 0.5

def k(a, b):
    return np.exp(-(a - b) ** 2 / (2 * l ** 2))

K = k(x[:, None], x[None, :])
print("K =\n", np.round(K, 4))

A = K + sigma2 * np.eye(3)
print("K + sigma^2 I =\n", np.round(A, 4))

kstar = k(x, xstar)
print("k* =", np.round(kstar, 4))

Ainv = np.linalg.inv(A)
alpha = Ainv @ y                    # (K + sigma^2 I)^{-1} y
print("alpha = (K+sigma^2 I)^{-1} y =", np.round(alpha, 4))

mean = kstar @ alpha
print("posterior mean m(x*) =", round(mean, 4))

var = k(xstar, xstar) - kstar @ Ainv @ kstar
print("k(x*,x*) =", round(float(k(xstar, xstar)), 4))
print("posterior variance v(x*) =", round(float(var), 4))
print("posterior std sqrt(v) =", round(float(np.sqrt(var)), 4))

# log marginal likelihood: -1/2 y^T A^{-1} y - 1/2 log det A - (n/2) log 2pi
n = 3
sign, logdet = np.linalg.slogdet(A)
lml = -0.5 * y @ Ainv @ y - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)
print("y^T A^{-1} y =", round(float(y @ Ainv @ y), 4))
print("log det(A) =", round(float(logdet), 4))
print("log marginal likelihood =", round(float(lml), 4))
