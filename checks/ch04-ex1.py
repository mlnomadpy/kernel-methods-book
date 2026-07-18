"""Worked example: the local-Rademacher fixed point for polynomial eigen-decay.

Take Mercer eigenvalues lambda_i = i^{-2b} of the kernel integral operator
(trace-class for b > 1/2). The local Rademacher complexity of the unit RKHS
ball intersected with the small-variance sub-ball {E f^2 <= r} is bounded by
the sub-root function

    psi(r) = sqrt( (1/n) * sum_i min(lambda_i, r) ),

whose unique positive fixed point r* = psi(r*) is the critical radius that sets
the learning rate. This script solves the fixed-point equation

    r = sqrt( (1/n) * sum_i min(lambda_i, r) )

by bisection for several sample sizes n and two decay exponents b, then reads
off the empirical log-log slope of r*(n). The claim to verify: that slope is
the FAST exponent 2b/(2b+1), strictly larger than the slow-rate exponent 1/2.
Every number printed here is the ground truth for the worked example.

Pure numpy, runs in well under a second.
"""
import numpy as np


def critical_radius(b, n, I=2_000_000):
    """Solve r = sqrt( (1/n) sum_i min(i^{-2b}, r) ) by bisection.

    Eigenvalues are summed to index I with an integral tail correction
    sum_{i>I} i^{-2b} ~ I^{1-2b}/(2b-1), so the truncation does not move r*.
    """
    i = np.arange(1, I + 1, dtype=float)
    lam = i ** (-2.0 * b)               # descending eigenvalues
    csum = np.cumsum(lam)
    tail = I ** (1.0 - 2.0 * b) / (2.0 * b - 1.0)
    S = csum[-1] + tail                 # total trace sum_i lambda_i

    def sum_min(r):
        # count of eigenvalues strictly greater than r (lam is descending)
        k = int(np.searchsorted(-lam, -r, side="left"))
        head = k * r                    # the truncated-to-r part
        tail_sum = S - (csum[k - 1] if k >= 1 else 0.0)
        return head + tail_sum, k

    def psi(r):
        s, _ = sum_min(r)
        return np.sqrt(s / n)

    lo, hi = 1e-18, 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if psi(mid) > mid:
            lo = mid
        else:
            hi = mid
    r = 0.5 * (lo + hi)
    _, k = sum_min(r)
    return r, k, S


def slope(rs, ns):
    """log-log slope of r*(n): -d log r* / d log n between consecutive n."""
    out = []
    for a in range(len(ns) - 1):
        out.append(-(np.log(rs[a + 1]) - np.log(rs[a]))
                   / (np.log(ns[a + 1]) - np.log(ns[a])))
    return out


ns = [100, 1000, 10000]

for b in (1.0, 2.0):
    fast = 2 * b / (2 * b + 1)          # predicted fast-rate exponent
    print(f"\n=== eigen-decay lambda_i = i^(-{2*b:.0f})  (b = {b:.0f}) ===")
    print(f"total trace  sum_i lambda_i = {critical_radius(b, ns[0])[2]:.6f}")
    print(f"predicted fast exponent 2b/(2b+1) = {fast:.4f} ; slow exponent = 0.5000")
    rs, ks = [], []
    for n in ns:
        r, k, _ = critical_radius(b, n)
        rs.append(r)
        ks.append(k)
        print(f"n = {n:6d} :  r* = {r:.6e}   eff.dim d(r*) = {k:3d}   "
              f"n^-1/2 = {n**-0.5:.6e}   n^-2b/(2b+1) = {n**-fast:.6e}   "
              f"r*/(n^-2b/(2b+1)) = {r / n**-fast:.4f}   r*/(n^-1/2) = {r / n**-0.5:.4f}")
    sl = slope(rs, ns)
    for a in range(len(ns) - 1):
        print(f"empirical slope  n={ns[a]}->{ns[a+1]} :  {sl[a]:.4f}  "
              f"(fast target {fast:.4f}, slow 0.5000)")
