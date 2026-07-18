"""Worked example: the Matern smoothness ladder and its spectral density.

The Matern kernel with length scale ell and smoothness nu is

    k_nu(r) = 2^{1-nu}/Gamma(nu) * (sqrt(2nu) r/ell)^nu * K_nu(sqrt(2nu) r/ell).

At half-integer nu = p + 1/2 it has the closed form (a polynomial of degree p in
r times a decaying exponential), so no Bessel evaluation is needed and the whole
table is pure numpy. We take ell = 1 and:

  1. tabulate k_nu(r) at nu = 1/2, 3/2, 5/2 (closed forms) and the Gaussian
     limit exp(-r^2/2), for r = 0.5 and r = 1.0, showing the smoothness ladder:
     smoother (larger nu) kernels stay more correlated at short range;
  2. drive nu up the half-integer ladder and watch k_nu -> Gaussian, reporting
     the largest gap max_r |k_nu(r) - exp(-r^2/2)| over a fine r-grid;
  3. tabulate the normalized Matern spectral density in d = 1,
        s_nu(w) = (1 + w^2 ell^2 / (2nu))^{-(nu+1/2)},
     a Student-t density with 2nu degrees of freedom (up to scale), at
     w = 1, 3, 10, against the Gaussian spectrum exp(-w^2/2), exhibiting the
     heavy polynomial (power-law) tail that the Gaussian lacks.

Pure numpy. Prints every number the worked example displays.
"""
import numpy as np
from math import factorial

ell = 1.0


def matern_half_integer(r, p):
    """Matern k_nu(r) at nu = p + 1/2, ell = 1, via the exact closed form
        k = exp(-a r) * p!/(2p)! * sum_{i=0}^p (p+i)!/(i!(p-i)!) (2 a r)^{p-i},
    with a = sqrt(2 nu) = sqrt(2p+1). Reduces to exp(-r), (1+sqrt3 r)exp(-sqrt3 r),
    (1+sqrt5 r+5r^2/3)exp(-sqrt5 r) for p = 0, 1, 2."""
    r = np.asarray(r, dtype=np.float64)
    nu = p + 0.5
    a = np.sqrt(2.0 * nu) / ell
    pref = factorial(p) / factorial(2 * p)
    s = np.zeros_like(r)
    for i in range(p + 1):
        coef = factorial(p + i) / (factorial(i) * factorial(p - i))
        s = s + coef * (2.0 * a * r) ** (p - i)
    return np.exp(-a * r) * pref * s


def gaussian(r):
    r = np.asarray(r, dtype=np.float64)
    return np.exp(-(r ** 2) / (2.0 * ell ** 2))


# --- 1. the smoothness ladder: kernel values ---
rs = [0.5, 1.0]
ps = {"1/2": 0, "3/2": 1, "5/2": 2}
print("Matern kernel k_nu(r), ell = 1  (closed forms; Gaussian = nu->infinity limit)")
print(f"{'r':>5} | {'nu=1/2':>9} {'nu=3/2':>9} {'nu=5/2':>9} {'Gaussian':>9}")
for r in rs:
    row = [matern_half_integer(r, ps[k]) for k in ["1/2", "3/2", "5/2"]]
    row.append(gaussian(r))
    print(f"{r:>5} | " + " ".join(f"{float(v):9.6f}" for v in row))

# sanity: the closed forms match the textbook polynomial forms
r = 0.5
assert abs(float(matern_half_integer(r, 0)) - np.exp(-r)) < 1e-12
assert abs(float(matern_half_integer(r, 1)) - (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)) < 1e-12
assert abs(float(matern_half_integer(r, 2)) - (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)) < 1e-12

# --- 2. convergence to the Gaussian limit up the half-integer ladder ---
rgrid = np.linspace(0.0, 3.0, 601)
print("\nApproach to the Gaussian limit, max_r |k_nu(r) - exp(-r^2/2)| on [0,3]:")
for p in [2, 10, 20]:
    nu = p + 0.5
    gap = float(np.max(np.abs(matern_half_integer(rgrid, p) - gaussian(rgrid))))
    print(f"  nu = {nu:>4}:  max gap = {gap:.4f}")

print("\nPointwise at r = 1 as nu grows (-> exp(-1/2) = %.6f):" % float(gaussian(1.0)))
for p in [0, 1, 2, 5, 10, 20]:
    nu = p + 0.5
    print(f"  nu = {nu:>4}:  k_nu(1) = {float(matern_half_integer(1.0, p)):.6f}")

# --- 3. spectral density: Student-t shape and its heavy tail ---
def matern_spec(w, nu):
    """Normalized 1-D Matern spectral density s_nu(w)/s_nu(0), ell = 1:
    a Student-t density with 2nu degrees of freedom (up to scale)."""
    w = np.asarray(w, dtype=np.float64)
    return (1.0 + w ** 2 * ell ** 2 / (2.0 * nu)) ** (-(nu + 0.5))


def fmt(v):
    """Fixed 4-dp for readable magnitudes, 3-sig scientific for tiny tails."""
    return f"{v:.4f}" if v >= 1e-2 else f"{v:.3e}"


ws = [1.0, 3.0, 10.0]
nus = [0.5, 1.5, 2.5]
print("\nNormalized spectral density s_nu(w)/s_nu(0), ell = 1  (Student-t, 2nu dof):")
print(f"{'w':>5} | {'nu=1/2':>11} {'nu=3/2':>11} {'nu=5/2':>11} {'Gaussian':>12}")
for w in ws:
    row = [float(matern_spec(w, nu)) for nu in nus]
    row.append(float(np.exp(-(w ** 2) / 2.0)))
    print(f"{w:>5.0f} | " + " ".join(f"{fmt(v):>11}" for v in row))
print("The Matern tails decay as a power law w^{-(2nu+1)}; the Gaussian tail is")
print("super-exponential, so at w = 10 it is smaller by many orders of magnitude.")
