"""Worked example 1: the Stein kernel u_p and the empirical KSD against a
standard-Gaussian target, for a sample that fits p and a sample that does not.

Target p = N(0,1), so the score is s_p(x) = grad log p(x) = -x (the normalizer
1/sqrt(2 pi) drops out of the log-derivative). Base kernel is the RBF
k(x,x') = exp(-(x-x')^2 / (2 h^2)) with bandwidth h = 1.

The (Langevin) Stein kernel is the four-term object

  u_p(x,x') = s_p(x) s_p(x') k(x,x')
            + s_p(x) d_{x'} k(x,x')
            + s_p(x') d_x k(x,x')
            + d_x d_{x'} k(x,x'),

which we build here from the explicit derivatives of the RBF kernel, and then
cross-check against the closed form obtained by hand for this p and k,

  u_p(x,x') = [ x x' + 1 - 2 (x-x')^2 ] exp(-(x-x')^2 / 2).

The empirical KSD^2 is the double average of u_p over the sample:
  V-statistic (biased, always >= 0):  (1/n^2)      sum_{i,j}     u_p(x_i,x_j)
  U-statistic (unbiased, degenerate): (1/(n(n-1))) sum_{i != j}  u_p(x_i,x_j).
"""
import numpy as np

h = 1.0

def score(x):                       # s_p for N(0,1)
    return -x

def stein_kernel_matrix(xs):
    """u_p(x_i, x_j) built from RBF derivatives; returns the n x n matrix."""
    xs = np.asarray(xs, dtype=float)
    X = xs[:, None]
    Y = xs[None, :]
    g = X - Y                                   # x - x'
    k = np.exp(-(g ** 2) / (2 * h ** 2))        # k(x,x')
    dk_x = -(g / h ** 2) * k                    # d_x  k
    dk_y = (g / h ** 2) * k                     # d_x' k
    dk_xy = (1.0 / h ** 2 - g ** 2 / h ** 4) * k   # d_x d_x' k
    sx = score(X)
    sy = score(Y)
    u = sx * sy * k + sx * dk_y + sy * dk_x + dk_xy
    # cross-check against the hand-derived closed form for N(0,1), h=1
    closed = (X * Y + 1.0 - 2.0 * g ** 2) * np.exp(-(g ** 2) / 2.0)
    assert np.allclose(u, closed), "four-term u_p disagrees with closed form"
    return u

def ksd_stats(xs):
    U = stein_kernel_matrix(xs)
    n = len(xs)
    V = U.sum() / n ** 2
    off = U.sum() - np.trace(U)
    Ust = off / (n * (n - 1))
    return U, V, Ust

for name, xs in [("A (fits N(0,1))", [-1.0, 0.0, 1.0]),
                 ("B (shifted to +3)", [2.0, 3.0, 4.0])]:
    U, V, Ust = ksd_stats(xs)
    print(f"--- sample {name}: x = {xs} ---")
    print("Stein kernel matrix u_p(x_i,x_j) =")
    print(np.round(U, 4))
    print("diagonal u_p(x_i,x_i) =", np.round(np.diag(U), 4))
    print(f"V-statistic  KSD^2_V = {V:.4f}")
    print(f"U-statistic  KSD^2_U = {Ust:.4f}")
    print()
