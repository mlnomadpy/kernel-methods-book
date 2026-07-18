"""Worked example 2: one step of Stein variational gradient descent (SVGD) for
three particles moving toward a standard-Gaussian target.

Target p = N(0,1), score s_p(x) = -x. Base kernel RBF k(x,x') =
exp(-(x-x')^2 / (2 h^2)) with h = 1. SVGD moves each particle x_i by the
empirical optimal perturbation

  phi(x_i) = (1/n) sum_j [ k(x_j, x_i) s_p(x_j)  +  grad_{x_j} k(x_j, x_i) ],

which splits into a driving term D_i (pulls particles toward high-density
regions of p, via the score) and a repulsion term R_i (spreads particles apart,
via the kernel gradient). For the RBF, grad_{x_j} k(x_j,x_i) = (x_i - x_j) k.

Update: x_i <- x_i + eps * phi(x_i). This uses only the score s_p, never the
normalizer of p.
"""
import numpy as np

h = 1.0
eps = 0.1
x = np.array([1.0, 2.0, 3.0])       # initial particles
n = len(x)

def score(v):
    return -v

# kernel matrix K[j,i] = k(x_j, x_i) and gradient of first argument
diff = x[:, None] - x[None, :]                  # diff[j,i] = x_j - x_i
K = np.exp(-(diff ** 2) / (2 * h ** 2))         # K[j,i] = k(x_j, x_i)
gradK = (x[None, :] - x[:, None]) / h ** 2 * K  # (x_i - x_j) k, indexed [j,i]

# driving and repulsion parts of phi(x_i), averaging over j
drive = (K * score(x)[:, None]).sum(axis=0) / n     # D_i
repel = gradK.sum(axis=0) / n                        # R_i
phi = drive + repel                                  # phi(x_i)
x_new = x + eps * phi

print("kernel matrix K[j,i] = k(x_j, x_i) =")
print(np.round(K, 4))
print("driving   D_i =", np.round(drive, 4))
print("repulsion R_i =", np.round(repel, 4))
print("phi(x_i)      =", np.round(phi, 4))
print("x (old)       =", np.round(x, 4))
print("x (new)       =", np.round(x_new, 4))
print(f"mean before   = {x.mean():.4f}")
print(f"mean after    = {x_new.mean():.4f}")
