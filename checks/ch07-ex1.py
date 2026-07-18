"""Worked example: the integral-operator minimax rate for kernel ridge regression.

Concrete polynomial spectrum lambda_i = i^{-2b} of the kernel integral operator
T (take b = 1, so lambda_i = i^{-2}; decay exponent s = 2b = 2). We:

  1. confirm T is trace class (sum_i lambda_i < infty),
  2. compute the effective dimension
        N(lambda) = sum_i lambda_i/(lambda_i + lambda) = tr(T (T + lambda I)^{-1})
     and fit its capacity exponent p (N(lambda) ~ lambda^{-p}), which the theory
     predicts to be p = 1/(2b) = 1/2,
  3. tabulate the predicted KRR rate exponent
        rho_KRR(r) = 2 min(r,1) / (2 min(r,1) + p)
     against the minimax exponent rho_min(r) = 2 r / (2 r + p), exhibiting the
     saturation of KRR at the source exponent r = 1,
  4. verify at a sample size n that bias^2 ~ lambda^{2 min(r,1)} and
     variance ~ N(lambda)/n balance at lambda_* = n^{-1/(2 min(r,1) + p)}, both of
     order n^{-rho_KRR(r)}.

Pure numpy. Prints every number the worked example displays.
"""
import numpy as np

b = 1.0
s = 2.0 * b            # eigenvalue-decay exponent, lambda_i = i^{-s}
p_theory = 1.0 / s     # capacity exponent, N(lambda) ~ lambda^{-p}
I = 2_000_000          # truncation for the spectral sums

def eff_dim(lam, I=I):
    """N(lambda) = sum_{i>=1} 1/(1 + lambda i^s), with a midpoint integral tail
    correction beyond the truncation I. For s = 2 the tail integral is
    (1/sqrt(lambda)) (pi/2 - arctan(sqrt(lambda) (I+0.5)))."""
    i = np.arange(1, I + 1, dtype=np.float64)
    head = np.sum(1.0 / (1.0 + lam * i ** s))
    sl = np.sqrt(lam)
    tail = (1.0 / sl) * (np.pi / 2.0 - np.arctan(sl * (I + 0.5)))
    return head + tail

# --- 1. trace-class check ---
i = np.arange(1, I + 1, dtype=np.float64)
trace = np.sum(i ** (-s)) + 1.0 / ((s - 1.0) * (I + 0.5) ** (s - 1.0))
print("b =", b, " decay exponent s = 2b =", s, " capacity p = 1/s =", p_theory)
print("trace sum_i lambda_i =", round(float(trace), 6), "(finite: T is trace class)")

# --- 2. effective dimension and capacity exponent ---
lams = np.array([1e-2, 1e-3, 1e-4])
Ns = np.array([eff_dim(l) for l in lams])
print("\nEffective dimension N(lambda) = tr(T (T + lambda I)^-1):")
for l, N in zip(lams, Ns):
    print(f"  lambda = {l:.0e}   N(lambda) = {N:.3f}   compare (pi/2) lambda^-1/2 = {(np.pi/2)*l**-0.5:.3f}")

lam_fit = np.array([1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
N_fit = np.array([eff_dim(l) for l in lam_fit])
slope, _ = np.polyfit(np.log(lam_fit), np.log(N_fit), 1)
print("fitted capacity exponent  p = -slope(log N vs log lambda) =",
      round(float(-slope), 4), " (theory p =", p_theory, ")")

# --- 3. rate exponents and saturation ---
p = p_theory
rs = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
print("\n   r     rho_KRR = 2min(r,1)/(2min(r,1)+p)     rho_minimax = 2r/(2r+p)")
for r in rs:
    reff = min(r, 1.0)
    rho_krr = 2 * reff / (2 * reff + p)
    rho_min = 2 * r / (2 * r + p)
    print(f"  {r:>4}            {rho_krr:.4f}                        {rho_min:.4f}")
print("KRR plateau value for r >= 1:  2/(2+p) =", round(2.0 / (2.0 + p), 4))

# --- 4. bias/variance balance at the optimal lambda ---
n = 10_000
print(f"\nBias-variance balance at n = {n}:")
for r in [0.5, 1.0, 2.0]:
    reff = min(r, 1.0)
    rho = 2 * reff / (2 * reff + p)
    lam_star = n ** (-1.0 / (2 * reff + p))
    bias2 = lam_star ** (2 * reff)
    var = eff_dim(lam_star) / n
    print(f"  r = {r}:  lambda* = {lam_star:.3e}   bias^2 = {bias2:.3e}   "
          f"var = {var:.3e}   sum = {bias2 + var:.3e}   n^-rho = {n ** -rho:.3e}")
