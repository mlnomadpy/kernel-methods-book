"""Worked example (ch-geom, ex2): making an RBF kernel invariant to the cyclic
rotation group Z_4 acting on R^2 by Haar (here, finite-group) averaging over the
orbit, and verifying positive-definiteness and invariance numerically.

Group G = Z_4 = { rotations by 0, 90, 180, 270 degrees } acting on R^2 through
the rotation matrices R_j. Base kernel is the RBF k(x,y) = exp(-||x-y||^2/(2 s^2))
with s = 1, which is already invariant to any orthogonal map: ||R x - R y|| =
||x - y||. The group-averaged (invariant) kernel is

    kG(x, y) = (1/|G|) sum_{j} k(x, R_j y).

Because k is jointly orthogonally invariant, this single orbit average equals the
double average (1/|G|^2) sum_{j,l} k(R_j x, R_l y) = <Phibar(x), Phibar(y)> with
Phibar(x) = (1/|G|) sum_j Phi(R_j x), so kG is positive definite. It is invariant:
kG(R x, y) = kG(x, y) for every R in G. The script builds a 3x3 Gram matrix of kG,
checks its eigenvalues are nonnegative, and confirms invariance (and that the raw
RBF is NOT invariant). Pure numpy, no training.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

# Z_4 rotation group
G = [rot(j * np.pi / 2) for j in range(4)]

s2 = 1.0  # s^2 in the RBF
def k(x, y):
    return np.exp(-np.sum((x - y) ** 2) / (2 * s2))

def kG(x, y):
    return np.mean([k(x, R @ y) for R in G])

# three generic 2-D points (no special symmetry among them)
X = [np.array([1.0, 0.4]),
     np.array([0.3, 0.6]),
     np.array([-0.7, 0.5])]
labels = ["x1", "x2", "x3"]

# --- base RBF Gram (not invariant) -----------------------------------------
Kbase = np.array([[k(a, b) for b in X] for a in X])
print("base RBF Gram K =\n", Kbase)

# --- invariant (group-averaged) Gram ---------------------------------------
KG = np.array([[kG(a, b) for b in X] for a in X])
print("\ninvariant Gram kG =\n", KG)
print("symmetric? max|kG - kG^T| =", round(float(np.max(np.abs(KG - KG.T))), 12))
evals = np.linalg.eigvalsh(KG)
print("eigenvalues of kG =", np.round(evals, 4))
print("min eigenvalue of kG =", round(float(np.min(evals)), 6), "(>= 0 => PSD)")

# --- per-orbit terms for x1,x2 (shows what the average is made of) ----------
terms = [k(X[0], R @ X[1]) for R in G]
print("\norbit terms k(x1, R_j x2), j=0..3 =", np.round(terms, 4))
print("their mean kG(x1,x2) =", round(float(np.mean(terms)), 4))

# --- invariance check: rotate an argument by a group element ---------------
R = G[1]  # 90-degree rotation
print("\nInvariance under the 90-degree rotation R:")
print("  raw RBF   k(R x1, x2) =", round(k(R @ X[0], X[1]), 4),
      "  vs k(x1, x2) =", round(k(X[0], X[1]), 4),
      "  -> differ:", round(abs(k(R @ X[0], X[1]) - k(X[0], X[1])), 4))
print("  invariant kG(R x1, x2) =", round(kG(R @ X[0], X[1]), 6),
      "  vs kG(x1, x2) =", round(kG(X[0], X[1]), 6),
      "  -> differ:", round(abs(kG(R @ X[0], X[1]) - kG(X[0], X[1])), 12))
print("  invariant kG(R x1, R x3) =", round(kG(R @ X[0], R @ X[2]), 6),
      "  vs kG(x1, x3) =", round(kG(X[0], X[2]), 6),
      "  -> differ:", round(abs(kG(R @ X[0], R @ X[2]) - kG(X[0], X[2])), 12))
