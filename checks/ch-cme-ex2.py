"""Worked example 2: Kernel Bayes' Rule posterior update.

Joint sample of n = 5 pairs (x_i, y_i) with y_i = x_i, Gaussian kernels
(sigma = tau = 1). A non-uniform prior on X (weights m, favouring large x) is
combined with the observation Y = 1.0 through the finite-sample Kernel Bayes'
Rule of Fukumizu, Song and Gretton (2013):

  rho   = (K + n*eps I)^{-1} K m                     (kernel sum rule)
  D     = diag(rho)
  w(y)  = D L ((D L)^2 + delta I)^{-1} D l_y          (kernel Bayes' rule)

The posterior embedding is mu_{X|Y=y} = sum_i w_i(y) phi(x_i); the posterior
mean of g(X) = X is sum_i w_i(y) x_i. Prints every displayed number.
"""
import numpy as np

x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
n = 5
sigma = 1.0
tau = 1.0
eps = 0.1        # sum-rule regularization, appears as (K + n*eps I)
delta = 0.01     # Bayes-rule (squared-operator) regularization
yobs = 1.0

def kx(a, b):
    return np.exp(-(a - b) ** 2 / (2 * sigma ** 2))

def ky(a, b):
    return np.exp(-(a - b) ** 2 / (2 * tau ** 2))

K = kx(x[:, None], x[None, :])
L = ky(y[:, None], y[None, :])

m = np.array([0.05, 0.05, 0.10, 0.30, 0.50])   # prior weights on x_i (sum 1)
print("prior weights m =", np.round(m, 4), " sum =", round(float(m.sum()), 4))
print("prior mean E[X] =", round(float(m @ x), 4))

# kernel sum rule: rho weights the predictive embedding of Y under the prior
rho = np.linalg.solve(K + n * eps * np.eye(n), K @ m)
print("rho =", np.round(rho, 4), " sum =", round(float(rho.sum()), 4))

D = np.diag(rho)
l_y = ky(y, yobs)                 # l_{y} = [l(y_i, yobs)]
print("l_y =", np.round(l_y, 4))

DL = D @ L
inner = np.linalg.solve(DL @ DL + delta * np.eye(n), D @ l_y)
w = DL @ inner                    # posterior embedding weights w(y)
print("w(y) =", np.round(w, 4), " sum =", round(float(w.sum()), 4))

post_mean = float(w @ x)
print("posterior mean E[X | Y=1.0] =", round(post_mean, 4))

# normalized posterior weights (project to a probability vector for reading)
wn = w / w.sum()
print("normalized w =", np.round(wn, 4))
print("posterior mean (normalized) =", round(float(wn @ x), 4))
