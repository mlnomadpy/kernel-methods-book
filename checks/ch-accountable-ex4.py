"""ch-accountable-ex4: an MMD two-sample test as a drift monitor.

Deploy a model, then ask each day: is today's input distributed like training?
The kernel Maximum Mean Discrepancy answers it. With a characteristic kernel MMD=0
iff the two distributions match, and the unbiased U-statistic estimator is computed
straight from Gram blocks. A permutation test gives a finite-sample p-value. We run
it against (a) a fresh unshifted draw -- no false alarm -- and (b) a mean-shifted
draw -- drift detected.
"""
import numpy as np

rng = np.random.default_rng(4)

def rbf(A, B, g):
    d = A[:, None] - B[None, :]
    return np.exp(-g * d ** 2)

def mmd2_unbiased(X, Y, g):
    m, n = len(X), len(Y)
    Kxx = rbf(X, X, g); Kyy = rbf(Y, Y, g); Kxy = rbf(X, Y, g)
    np.fill_diagonal(Kxx, 0.0); np.fill_diagonal(Kyy, 0.0)
    return (Kxx.sum() / (m * (m - 1)) + Kyy.sum() / (n * (n - 1))
            - 2.0 * Kxy.mean())

def perm_pvalue(X, Y, g, B=2000):
    obs = mmd2_unbiased(X, Y, g)
    Z = np.concatenate([X, Y]); m = len(X); count = 0
    for _ in range(B):
        p = rng.permutation(len(Z))
        if mmd2_unbiased(Z[p[:m]], Z[p[m:]], g) >= obs:
            count += 1
    return obs, (count + 1) / (B + 1)

n = 200
ref = rng.standard_normal(n)                       # training/reference sample
same = rng.standard_normal(n)                      # a fresh unshifted draw
shift = rng.standard_normal(n) + 0.5               # a mean-shifted production window

# median-heuristic bandwidth on the reference sample
d = np.abs(ref[:, None] - ref[None, :])
med = np.median(d[d > 0])
g = 1.0 / (2 * med ** 2)
print(f"MMD drift test (RBF, median-heuristic bandwidth; n={n}, 2000 permutations)")
print(f"  median pairwise distance : {med:.3f}   -> gamma = {g:.3f}")

mmd_same, p_same = perm_pvalue(ref, same, g)
mmd_shift, p_shift = perm_pvalue(ref, shift, g)
print(f"  unshifted draw : MMD^2 = {mmd_same:+.4f}   p = {p_same:.3f}   (no false alarm)")
print(f"  shift +0.5     : MMD^2 = {mmd_shift:+.4f}   p = {p_shift:.3f}   (drift detected)")
