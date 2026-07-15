"""Check for ch-vc Example: halfplanes in R^2 shatter 3 points but not 4.

Class = { x -> sgn(w.x + b) } (oriented affine halfplanes). We test linear
separability of a labeling via an LP feasibility problem: exists (w,b) with
y_i (w.x_i + b) >= 1 for all i.  Then we confirm VC-dim = 3 for R^2 halfplanes
(= N + 1 with N = 2) and plug h = 3 into the Sauer-bounded VC risk bound.
"""
import numpy as np
from itertools import product
from scipy.optimize import linprog


def separable(X, y):
    """True iff the labeling y of points X is realizable by an affine halfplane.

    Feasibility LP: find (w in R^2, b) s.t. y_i (w.x_i + b) >= 1.
    Variables u = (w1, w2, b). Constraint -y_i [x_i, 1] . u <= -1. No objective.
    """
    n = X.shape[0]
    A = np.hstack([X, np.ones((n, 1))])          # rows [x_i, 1]
    A_ub = -(y[:, None] * A)                      # -y_i [x_i, 1]
    b_ub = -np.ones(n)
    res = linprog(c=np.zeros(3), A_ub=A_ub, b_ub=b_ub,
                  bounds=[(None, None)] * 3, method="highs")
    return res.success


def shatters(X):
    n = X.shape[0]
    got = 0
    for lab in product([-1, 1], repeat=n):
        y = np.array(lab, dtype=float)
        if separable(X, y):
            got += 1
    return got


# Three points in general position (a triangle): all 2^3 = 8 labelings realizable.
tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
g3 = shatters(tri)
print("triangle: realizable labelings =", g3, "of", 2**3, "-> shatters 3 points:", g3 == 8)

# Four points in convex position (a square): the XOR labeling is NOT separable.
sq = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
g4 = shatters(sq)
xor = np.array([1.0, -1.0, 1.0, -1.0])  # opposite corners share a label
print("square:   realizable labelings =", g4, "of", 2**4, "-> shatters 4 points:", g4 == 16)
print("XOR labeling (+,-,+,-) separable?", separable(sq, xor))

# Four points with one inside the triangle of the other three: also not shattered.
inside = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.5, 0.5]])
g4b = shatters(inside)
print("one-inside-triangle: realizable labelings =", g4b, "of", 2**4,
      "-> shatters:", g4b == 16)
print("VC-dim(halfplanes in R^2) = 3  (= N + 1, N = 2)")

# ---- Plug h = 3 into the VC bound via Sauer's lemma ----
# S(2n) <= sum_{i=0}^{h} C(2n, i);  bound: Remp + sqrt((8/n)(ln S(2n) + ln(4/delta))).
from math import comb, log, sqrt

h = 3
n = 1000
delta = 0.05
S2n_sauer = sum(comb(2 * n, i) for i in range(h + 1))
conf = sqrt((8.0 / n) * (log(S2n_sauer) + log(4.0 / delta)))
print("\nVC bound plug-in (halfplanes, VC-dim 3):")
print("  n =", n, " delta =", delta)
print("  Sauer sum_{i=0}^{3} C(2n,i) =", S2n_sauer)
print("  ln S(2n) <=", round(log(S2n_sauer), 4))
print("  ln(4/delta) =", round(log(4.0 / delta), 4))
print("  confidence term =", round(conf, 4))
