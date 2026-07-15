"""Worked example: the GP posterior mean equals the kernel-ridge fit.

Same Gaussian kernel k(x,x') = exp(-(x-x')^2 / (2 l^2)), l = 1, on three points.
Kernel ridge regression solves alpha = (K + lambda n I)^{-1} y and predicts
f(x*) = sum_i alpha_i k(x_i, x*) = k*^T alpha. The GP posterior mean is
m(x*) = k*^T (K + sigma^2 I)^{-1} y. Setting sigma^2 = lambda n makes the two
matrices identical, so the predictions coincide. We verify this numerically at
two test points.
"""
import numpy as np

x = np.array([0.0, 1.0, 2.0])
y = np.array([1.0, 0.0, -1.0])
l = 1.0
n = 3
lam = 0.05                          # ridge parameter
lam_n = lam * n                     # = 0.15
sigma2 = lam_n                      # GP noise variance matched to lambda n

def k(a, b):
    return np.exp(-(a - b) ** 2 / (2 * l ** 2))

K = k(x[:, None], x[None, :])
print("K =\n", np.round(K, 4))
print("lambda n = sigma^2 =", round(lam_n, 4))

# Kernel ridge regression
alpha_krr = np.linalg.solve(K + lam_n * np.eye(n), y)
print("alpha_KRR = (K + lambda n I)^{-1} y =", np.round(alpha_krr, 4))

# GP posterior-mean coefficients
alpha_gp = np.linalg.solve(K + sigma2 * np.eye(n), y)
print("alpha_GP  = (K + sigma^2 I)^{-1} y =", np.round(alpha_gp, 4))
print("max |alpha_KRR - alpha_GP| =", float(np.max(np.abs(alpha_krr - alpha_gp))))

for xstar in [0.5, 1.5]:
    kstar = k(x, xstar)
    f_krr = kstar @ alpha_krr
    m_gp = kstar @ alpha_gp
    print(f"x* = {xstar}: KRR f(x*) =", round(float(f_krr), 6),
          "| GP mean m(x*) =", round(float(m_gp), 6),
          "| difference =", float(abs(f_krr - m_gp)))
