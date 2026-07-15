"""ch-invariance, Example 2: reduced-set approximation of a 3-term expansion by 1 term.

Given an SVM-style feature-space vector
    Psi = sum_{i=1}^3 alpha_i Phi(x_i),  Gaussian kernel, width sigma,
we approximate it by a single-term expansion beta Phi(z). We
  (a) find the pre-image z by the fixed-point iteration (18.22),
  (b) set the optimal coefficient beta = <Psi, Phi(z)> / <Phi(z), Phi(z)>
      = sum_i alpha_i k(x_i, z)   (since k(z,z)=1 for the Gaussian),
  (c) report the feature-space approximation error
      ||Psi - beta Phi(z)||^2 = ||Psi||^2 - beta^2,
      and the relative error  ||Psi - beta Phi(z)|| / ||Psi||.

Prints every number the worked example displays.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# --- setup ---
X = np.array([[0.0, 0.0],
              [1.0, 0.0],
              [0.5, 1.0]])
alpha = np.array([1.0, 0.8, 0.6])
sigma = 1.0
two_s2 = 2.0 * sigma**2

def k(a, b):
    return np.exp(-np.sum((a - b)**2) / two_s2)

# Gram matrix of the x_i
K = np.array([[k(X[i], X[j]) for j in range(3)] for i in range(3)])
print("K =\n", K)

# ||Psi||^2 = sum_ij alpha_i alpha_j k(x_i, x_j)
Psi2 = alpha @ K @ alpha
print("||Psi||^2 =", round(Psi2, 6))
print("||Psi||   =", round(np.sqrt(Psi2), 6))

def weights(z):
    d2 = np.sum((X - z)**2, axis=1)
    return alpha * np.exp(-d2 / two_s2)

# (a) fixed-point iteration for the pre-image
z = alpha @ X / alpha.sum()
print("z0 =", np.round(z, 6))
for n in range(1, 8):
    w = weights(z)
    z = (w @ X) / w.sum()
    print(f"z{n} =", np.round(z, 6))
z_star = z

# (b) optimal single coefficient
kz = np.array([k(X[i], z_star) for i in range(3)])
print("k(x_i, z*) =", np.round(kz, 6))
beta = alpha @ kz          # divided by k(z,z)=1
print("beta =", round(beta, 6))

# (c) approximation error in feature space
err2 = Psi2 - beta**2      # = ||Psi||^2 - 2 beta<Psi,Phi(z)> + beta^2, with <Psi,Phi(z)>=beta
print("||Psi - beta Phi(z*)||^2 =", round(err2, 6))
print("||Psi - beta Phi(z*)||   =", round(np.sqrt(err2), 6))
print("relative error =", round(np.sqrt(err2 / Psi2), 6))
