"""ch-invariance, Example 1: Gaussian-kernel pre-image fixed-point iteration.

We are given a target feature-space vector
    Psi = sum_i beta_i Phi(x_i),   Phi(x) = k(x, .),  k Gaussian with width sigma,
a convex combination (beta_i >= 0, sum beta_i = 1) of three input points, and we
seek an approximate pre-image z minimizing || Psi - Phi(z) ||^2. Since the
Gaussian kernel is normalized (k(z,z)=1), this reduces to maximizing
<Psi, Phi(z)>^2, whose stationarity condition gives the fixed-point map

    z_{n+1} = ( sum_i beta_i exp(-||x_i - z_n||^2 / 2 sigma^2) x_i )
              / ( sum_i beta_i exp(-||x_i - z_n||^2 / 2 sigma^2) ).

Prints every number the worked example displays.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# --- setup ---
X = np.array([[0.0, 0.0],
              [2.0, 0.0],
              [1.0, 1.5]])
beta = np.array([0.5, 0.3, 0.2])
sigma = 1.0
two_s2 = 2.0 * sigma**2

print("X =\n", X)
print("beta =", beta)
print("sigma =", sigma)

def weights(z):
    d2 = np.sum((X - z)**2, axis=1)
    return beta * np.exp(-d2 / two_s2)

def objective(z):
    # <Psi, Phi(z)> = sum_i beta_i k(x_i, z)
    return np.sum(weights(z))

# start at the plain weighted mean of the points
z = beta @ X
print("z0 =", z, " J(z0) = <Psi,Phi(z0)> =", round(objective(z), 6))

for n in range(1, 7):
    w = weights(z)
    z = (w @ X) / w.sum()
    print(f"z{n} =", np.round(z, 6), " J =", round(objective(z), 6))

z_star = z
print("z* =", np.round(z_star, 6))
print("J(z*) =", round(objective(z_star), 6))
