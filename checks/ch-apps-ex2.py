"""Worked example: reading the spectrum to diagnose capacity.

The same six standardized 1-D points as ch-apps-ex1. We build the RBF Gram
matrix k(x,x') = exp(-(x-x')^2 / (2 sigma^2)) at a wide and a narrow bandwidth,
take its eigenvalues, and read off the effective dimension at ridge level
lambda = 0.1,

    d_eff(lambda) = sum_i  lambda_i / (lambda_i + lambda),

the number of eigen-directions the kernel actually exposes at that ridge. A wide
bandwidth gives a fast-decaying spectrum and a small effective dimension (few
parameters, underfitting risk); a narrow bandwidth gives a slow-decaying
spectrum and a large effective dimension (many parameters, overfitting risk).
Prints every number the worked example shows.
"""
import numpy as np

x_raw = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
x = (x_raw - x_raw.mean()) / x_raw.std()
lam = 0.1
n = len(x)

def rbf(a, b, sigma):
    return np.exp(-(a[:, None] - b[None, :]) ** 2 / (2.0 * sigma ** 2))

for sigma in [2.0, 0.5]:
    K = rbf(x, x, sigma)
    ev = np.sort(np.linalg.eigvalsh(K))[::-1]      # descending
    deff = float(np.sum(ev / (ev + lam)))
    print(f"sigma = {sigma}")
    print("   trace K (=n) =", round(float(np.trace(K)), 4))
    print("   eigenvalues (desc) =", list(np.round(ev, 4)))
    print("   top-2 share of trace =", round(float(ev[:2].sum() / ev.sum()), 4))
    print("   effective dimension d_eff(0.1) =", round(deff, 4))
    print()
