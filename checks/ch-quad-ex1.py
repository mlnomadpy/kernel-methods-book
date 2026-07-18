"""Worked example 1: optimally weighted kernel quadrature beats uniform weights,
and the worst-case error equals the MMD from P to the weighted node measure.

Target P = N(0,1). Kernel k(x,x') = exp(-(x-x')^2 / 2) (Gaussian, lengthscale 1).
Nodes x = (-1, 0, 1). For this Gaussian-kernel / Gaussian-measure pair the two
integrals we need are closed form:
    kernel mean   z_i = E_{X~P}[k(x_i,X)] = (1/sqrt2) exp(-x_i^2/4)
    initial var   C   = E_{X,X'~P}[k(X,X')] = 1/sqrt3
The worst-case squared error of weights w over the unit RKHS ball is
    e(w)^2 = C - 2 w^T z + w^T K w   (= squared MMD from P to sum_i w_i delta_{x_i}).
Uniform weights w = 1/3 versus optimal weights w* = K^{-1} z (which need not sum
to 1). Prints every number the worked example displays.
"""
import numpy as np

x = np.array([-1.0, 0.0, 1.0])
n = len(x)

def k(a, b):
    return np.exp(-(a - b) ** 2 / 2.0)

K = k(x[:, None], x[None, :])
z = (1.0 / np.sqrt(2.0)) * np.exp(-x ** 2 / 4.0)   # kernel mean at the nodes
C = 1.0 / np.sqrt(3.0)                              # E_{P x P}[k]

# self-check the two closed forms against 1-D numerical integrals under N(0,1)
t = np.linspace(-12, 12, 200001)
dt = t[1] - t[0]
phi = np.exp(-t ** 2 / 2) / np.sqrt(2 * np.pi)              # N(0,1) density
z_num = np.array([np.sum(k(xi, t) * phi) * dt for xi in x])
phi2 = np.exp(-t ** 2 / 4) / np.sqrt(4 * np.pi)            # N(0,2): X - X'
C_num = np.sum(np.exp(-t ** 2 / 2) * phi2) * dt
assert np.allclose(z, z_num, atol=1e-6), (z, z_num)
assert np.allclose(C, C_num, atol=1e-6), (C, C_num)

print("K =\n", np.round(K, 6))
print("z (kernel mean at nodes) =", np.round(z, 6))
print(f"C = E_{{PxP}}[k] = {float(C):.6f}")

# uniform weights
wu = np.full(n, 1.0 / n)
eu2 = C - 2 * wu @ z + wu @ K @ wu
print("uniform w  =", np.round(wu, 6), f" sum = {float(wu.sum()):.6f}")
print(f"uniform e^2 = {float(eu2):.6f}   e = {float(np.sqrt(eu2)):.6f}")

# optimal weights w* = K^{-1} z
ws = np.linalg.solve(K, z)
es2 = C - z @ np.linalg.solve(K, z)                        # = C - z^T K^{-1} z
print("optimal w* =", np.round(ws, 6), f" sum = {float(ws.sum()):.6f}")
print(f"optimal e^2 = {float(es2):.6f}   e = {float(np.sqrt(es2)):.6f}")
print(f"check e*^2 via quad form = {float(C - 2 * ws @ z + ws @ K @ ws):.6f}")
print(f"variance drop e_u^2 - e*^2 = {float(eu2 - es2):.6f}")
print(f"relative reduction = {float((eu2 - es2) / eu2):.4f}")
