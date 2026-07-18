"""Worked example 2: a two-stage kernel instrumental-variable (KIV) fit recovering
a structural slope on synthetic confounded data.

Structural causal model (all variables centered, n=6):
    instrument   z = (-1,-1,-1, 1, 1, 1)
    confounder   u = (-1, 1, 0,-1, 1, 0)      chosen sample-uncorrelated with z
    treatment    x = z + u                      (Z is relevant: it moves X)
    outcome      y = beta*x + gamma*u,   beta = 2 (structural), gamma = 3 (confound)

The confounder u enters both x and y, so the least-squares (OLS) regression of y
on x is biased for beta.  A valid instrument z (relevant, and uncorrelated with u
so it affects y only through x) lets two-stage least squares recover beta.  With
LINEAR kernels the KIV estimator of Singh, Sahani, Gretton (2019) reduces exactly
to two-stage least squares, so we can check both the scalar 2SLS slope and the
full KIV matrix formula agree and recover beta = 2, while OLS does not.
"""
import numpy as np

z = np.array([-1, -1, -1, 1, 1, 1], float)
u = np.array([-1, 1, 0, -1, 1, 0], float)   # sample-uncorrelated with z
x = z + u
beta, gamma = 2.0, 3.0
y = beta * x + gamma * u
n = len(x)

xc, yc, zc = x - x.mean(), y - y.mean(), z - z.mean()
print("Cov_hat(z,u) = %.4f   (instrument uncorrelated with confounder)"
      % (zc @ (u - u.mean()) / n))
print("Cov_hat(x,u) = %.4f   (treatment IS confounded)"
      % (xc @ (u - u.mean()) / n))

# --- naive OLS slope (biased) ---
b_ols = (xc @ yc) / (xc @ xc)
varX = (xc @ xc) / n
covXU = xc @ (u - u.mean()) / n
print("Var_hat(X) = %.4f" % varX)
print("beta_OLS  = %.4f  (= beta + gamma*Cov(X,U)/Var(X) = 2 + 3*%.4f/%.4f)"
      % (b_ols, covXU, varX))
print("OLS bias  = %.4f" % (b_ols - beta))

# --- two-stage least squares = KIV with linear kernels ---
a_hat = (zc @ xc) / (zc @ zc)          # stage 1: regress X on Z
xhat = a_hat * zc                      # fitted treatment (the CME, linear case)
b_2sls = (xhat @ yc) / (xhat @ xhat)   # stage 2: regress Y on fitted treatment
print("stage-1 slope a_hat = %.4f" % a_hat)
print("beta_2SLS = %.4f" % b_2sls)

# --- full KIV matrix formula with linear kernels + tiny ridge ---
lam, xi = 1e-6, 1e-6
Kz = np.outer(zc, zc)                   # linear-kernel Gram of the instrument
Kx = np.outer(xc, xc)                   # linear-kernel Gram of the treatment
B = np.linalg.solve(Kz + n * lam * np.eye(n), Kz)   # stage-1 CME coefficients
K2 = B.T @ Kx @ B                       # Gram of the estimated embeddings
c = np.linalg.solve(K2 + n * xi * np.eye(n), yc)    # stage-2 ridge weights
alpha = B @ c                           # structural-function coefficients
slope_kiv = alpha @ xc                  # f(x) = (sum_i alpha_i xc_i) * x
print("beta_KIV(linear kernels, ridge->0) = %.4f" % slope_kiv)
