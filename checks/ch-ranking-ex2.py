"""ch-ranking, Example 2: one ranking-perceptron update fixing a swapped pair.

Three items in R^2 with a linear kernel. True order 1 > 2 > 3. A starting
weight w0 ranks item 1 at the top but swaps the pair (2, 3). The pairwise
ranking perceptron scans the preferred pairs; on a misordered pair (i, j) with
<w, x_i - x_j> <= 0 it updates w <- w + eta (x_i - x_j) with eta = 1. We show
that the single update on the swapped pair flips its margin positive and yields
a perfect ranking.

Every number printed here appears in the worked example.
"""
import numpy as np
from itertools import combinations

# --- setup ---
X = np.array([[4.0, 4.0],   # item 1 (best)
              [3.0, 0.0],   # item 2
              [1.0, 1.0]])  # item 3 (worst)
y = np.array([3, 2, 1])     # larger = more preferred
w0 = np.array([0.0, 1.0])   # starting direction
eta = 1.0
m = len(y)
pairs = [(i, j) for i, j in combinations(range(m), 2) if y[i] > y[j]]

def report(w, tag):
    f = X @ w
    bad = []
    for (i, j) in pairs:
        if w @ (X[i] - X[j]) <= 0:
            bad.append((i + 1, j + 1))
    print(f"{tag}: w = {w}, scores = {f}, misordered = {bad} (count {len(bad)})")
    return bad

print("X =\n", X)
print("true order 1 > 2 > 3, preferred pairs =", [(i+1,j+1) for i,j in pairs])
bad0 = report(w0, "before")

# --- the swapped pair is (2, 3): i=1, j=2 (0-indexed) ---
i, j = 1, 2
d = X[i] - X[j]
margin0 = w0 @ d
print(f"swapped pair (2>3): d = x2 - x3 = {d}, margin <w0,d> = {margin0:+.1f} (<= 0)")

# --- perceptron update ---
w1 = w0 + eta * d
margin1 = w1 @ d
print(f"update w1 = w0 + eta*d = {w1}, new margin <w1,d> = {margin1:+.1f}")

bad1 = report(w1, "after")
print("misordered count before -> after:", len(bad0), "->", len(bad1))
