"""ch-ranking, Example 1: pairwise difference vectors and misordered pairs.

Four items in R^2 with a linear kernel k(x,x') = <x, x'>. The true order is
item 1 > item 2 > item 3 > item 4 (item 1 most preferred). A candidate scoring
function f(x) = <w, x> is tested: for every preferred pair (i, j) with y_i > y_j
we form the difference vector d_ij = phi(x_i) - phi(x_j) and check the sign of
<w, d_ij>. A pair is misordered when <w, d_ij> <= 0. We also verify the
pair-kernel identity <d_ij, d_kl> = K_ik - K_il - K_jk + K_jl.

Every number printed here appears in the worked example.
"""
import numpy as np
from itertools import combinations

# --- setup ---
X = np.array([[4.0, 1.0],   # item 1 (best)
              [3.0, 3.0],   # item 2
              [2.0, 0.0],   # item 3
              [1.0, 2.0]])  # item 4 (worst)
# true rank labels, larger = more preferred
y = np.array([4, 3, 2, 1])
w = np.array([1.0, -1.0])   # candidate scoring direction
m = len(y)
K = X @ X.T                 # linear Gram matrix
print("X =\n", X)
print("y (rank labels) =", y)
print("w =", w)
print("linear Gram K =\n", K)

# --- scores ---
f = X @ w
print("scores f(x_i) = <w, x_i> =", f)

# --- preferred pairs (i, j): y_i > y_j ---
pairs = [(i, j) for i, j in combinations(range(m), 2) if y[i] > y[j]]
print("number of preferred pairs P =", len(pairs))

# --- difference vectors and margins ---
misordered = []
for (i, j) in pairs:
    d = X[i] - X[j]
    margin = w @ d
    kernel_expand = K[i, i] - K[i, j] - K[j, i] + K[j, j]  # <d, d>
    status = "correct" if margin > 0 else "MISORDERED"
    print(f"  pair ({i+1}>{j+1}): d = {d}, <w,d> = {margin:+.1f}  <d,d> = {kernel_expand:.1f}  {status}")
    if margin <= 0:
        misordered.append((i + 1, j + 1))

n_bad = len(misordered)
print("misordered pairs =", misordered)
print("count misordered =", n_bad)
print("ranking risk (fraction misordered) =", n_bad, "/", len(pairs),
      "=", round(n_bad / len(pairs), 4))

# --- pair-kernel identity check between two distinct pairs ---
(i, j) = (1, 2)   # pair 2>3  (0-indexed 1,2)
(k, l) = (0, 3)   # pair 1>4  (0-indexed 0,3)
d_ij = X[i] - X[j]
d_kl = X[k] - X[l]
lhs = d_ij @ d_kl
rhs = K[i, k] - K[i, l] - K[j, k] + K[j, l]
print(f"pair-kernel <d_23, d_14> direct = {lhs:.1f}, via Gram = {rhs:.1f}")
