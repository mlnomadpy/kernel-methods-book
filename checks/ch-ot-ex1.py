"""Worked example 1: exact 1-Wasserstein between two tiny 1-D empirical
measures by sorting, cross-checked against the CDF-difference integral, and
contrasted with an MMD (the energy distance) on the same pair.

P = (1/3)(d_0 + d_2 + d_5),  Q = (1/3)(d_1 + d_3 + d_4)   (3 atoms each, uniform)

Ground cost c(x,y) = |x - y| for W_1 (and squared for W_2). In one dimension
the optimal coupling matches the sorted samples, so

  W_1(P,Q) = (1/n) sum_i |x_(i) - y_(i)|        (monotone / sorted pairing)

A non-monotone pairing must cost at least as much; we show the reversed one.
The CDF form  W_1 = integral |F_P(t) - F_Q(t)| dt  gives the same number.

The energy distance E(P,Q) = 2 E|X-Y| - E|X-X'| - E|Y-Y'| is the squared MMD
with kernel k(x,y) = -|x-y| (Sejdinovic et al. 2013); it is also the epsilon ->
infinity limit (up to the 1/2 factor) of the Sinkhorn divergence with this cost
(Feydy et al. 2019). Every printed number appears in the worked example.
"""
import numpy as np

x = np.array([0.0, 2.0, 5.0])   # atoms of P
y = np.array([1.0, 3.0, 4.0])   # atoms of Q
n = 3

xs = np.sort(x)
ys = np.sort(y)
print("sorted x =", xs)
print("sorted y =", ys)

# --- W_1 by the monotone (sorted) coupling ---
abs_sorted = np.abs(xs - ys)
print("|x_(i) - y_(i)| =", abs_sorted)
W1 = abs_sorted.sum() / n
print("W_1 (sorted coupling)      =", round(float(W1), 4))

# --- a reversed (non-monotone) coupling costs more ---
abs_rev = np.abs(xs - ys[::-1])
print("|x_(i) - y_(n+1-i)| =", abs_rev)
W1_rev = abs_rev.sum() / n
assert np.isclose(W1, 1.0)
assert np.isclose(W1_rev, 3.0)
assert W1 < W1_rev
print("cost of reversed coupling  =", round(float(W1_rev), 4))

# --- W_2 by the same sorted coupling ---
W2 = np.sqrt((abs_sorted ** 2).sum() / n)
print("W_2 (sorted coupling)      =", round(float(W2), 4))

# --- CDF-difference integral (sanity check for W_1) ---
# breakpoints of the two step CDFs, on a fine grid the integral is exact here
grid = np.linspace(-1, 6, 700001)
def cdf(atoms, t):
    return np.mean(atoms[:, None] <= t[None, :], axis=0)
FP = cdf(x, grid)
FQ = cdf(y, grid)
d = np.abs(FP - FQ)
integral = np.sum((d[:-1] + d[1:]) / 2 * np.diff(grid))   # trapezoidal rule
assert np.isclose(integral, W1, atol=2e-5)
print("integral |F_P - F_Q| dt    =", round(float(integral), 4))

# --- energy distance = squared MMD with kernel -|x-y| ---
def mean_abs(a, b):
    return np.mean(np.abs(a[:, None] - b[None, :]))
Exy = mean_abs(x, y)
Exx = mean_abs(x, x)
Eyy = mean_abs(y, y)
energy = 2 * Exy - Exx - Eyy
assert np.isclose(energy, 2 / 3)
print("E|X-Y| =", round(float(Exy), 4),
      " E|X-X'| =", round(float(Exx), 4),
      " E|Y-Y'| =", round(float(Eyy), 4))
print("energy distance E(P,Q)     =", round(float(energy), 4))
print("1/2 * energy (Feydy eps->inf Sinkhorn-divergence limit) =",
      round(float(energy / 2), 4))
