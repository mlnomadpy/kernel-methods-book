"""Worked example: Nystrom landmark choice, uniform versus ridge leverage scores.

Seven points on a line: a tight cluster of five near the origin, plus two
isolated points far away. Because the cluster is highly redundant, its influence
is nearly rank one, so uniform sampling wastes landmarks inside it and usually
misses the two isolated directions. We build the Gaussian Gram matrix, compute
the ridge leverage scores and the effective dimension, then compare the rank-p
Nystrom approximation error for three landmark choices: the exact optimum (top
eigenvectors), the top-p ridge-leverage points, and uniform sampling (averaged
and worst case over all subsets of size p). Prints every number the worked
example displays.
"""
import itertools
import math

import numpy as np

# a tight cluster of five near 0, plus two isolated points at 5 and 10
x = np.array([0.0, 0.15, 0.3, 0.45, 0.6, 5.0, 10.0])
n = x.size
sigma = 1.0
lam = 1e-2       # ridge parameter for leverage scores
p = 3            # number of landmarks

def kern(a, b):
    return np.exp(-(a - b) ** 2 / (2 * sigma ** 2))

K = kern(x[:, None], x[None, :])
print("n =", n, " p =", p, " sigma =", sigma, " lambda =", lam)

# eigenvalues (descending): the low-rank floor
evals = np.sort(np.linalg.eigvalsh(K))[::-1]
print("eigenvalues of K =", np.round(evals, 4))
tail_fro = np.sqrt(np.sum(evals[p:] ** 2))    # best rank-p Frobenius error
tail_tr = np.sum(evals[p:])                    # best rank-p trace (nuclear) error
print("best rank-p Frobenius error  ||K-K_p||_F =", round(tail_fro, 4))
print("best rank-p trace error       tr(K-K_p)  =", round(tail_tr, 4))

# ridge leverage scores  l_i = [K (K + lam I)^{-1}]_ii ; effective dim = their sum
S = K @ np.linalg.inv(K + lam * np.eye(n))
lev = np.diag(S).copy()
deff = np.trace(S)
print("ridge leverage scores =", np.round(lev, 4))
print("effective dimension d_eff(lambda) = sum of leverage scores =", round(float(deff), 4))

def nystrom_err(Z):
    Z = list(Z)
    Kmm = K[np.ix_(Z, Z)]
    Knm = K[:, Z]
    Ktil = Knm @ np.linalg.pinv(Kmm) @ Knm.T
    D = K - Ktil
    return np.linalg.norm(D, "fro"), np.trace(D)

# leverage-score landmarks: the p points of highest leverage
lev_Z = np.argsort(lev)[::-1][:p]
lev_Z = sorted(int(i) for i in lev_Z)
lev_fro, lev_tr = nystrom_err(lev_Z)
print("leverage-score landmarks =", lev_Z)
print("leverage Nystrom Frobenius error =", round(lev_fro, 4))
print("leverage Nystrom trace error     =", round(float(lev_tr), 4))

# uniform sampling: enumerate every subset of size p
fros, trs = [], []
worst_fro, worst_Z = -1.0, None
for Z in itertools.combinations(range(n), p):
    f, t = nystrom_err(Z)
    fros.append(f)
    trs.append(t)
    if f > worst_fro:
        worst_fro, worst_Z = f, Z
print("number of size-p subsets =", len(fros))
print("uniform Nystrom Frobenius error, mean =", round(float(np.mean(fros)), 4),
      " worst =", round(float(worst_fro), 4), " at Z =", list(worst_Z))
print("uniform Nystrom trace error, mean =", round(float(np.mean(trs)), 4))

# a representative bad uniform draw: all three landmarks inside one cluster
bad_Z = [0, 1, 2]
bad_fro, bad_tr = nystrom_err(bad_Z)
print("clustered pick Z =", bad_Z, " Frobenius error =", round(bad_fro, 4),
      " trace error =", round(float(bad_tr), 4))

# how often does a uniform draw land entirely inside the 5-point cluster?
p_all_cluster = math.comb(5, p) / math.comb(n, p)
print("P(uniform draw entirely inside the cluster) =", round(p_all_cluster, 4))
