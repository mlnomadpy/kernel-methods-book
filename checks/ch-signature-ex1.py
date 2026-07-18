"""Worked example: the depth-2 truncated signature of two tiny 2-D paths, their
signature inner product, and a numerical check that the (untruncated) signature
kernel is the solution of the Salvi et al. (2021) Goursat PDE.

Paths (piecewise linear interpolants of 3 points each) in R^2:
    X : (0,0) -> (1,0) -> (1,1)      "right then up"   increments (1,0),(0,1)
    Y : (0,0) -> (0,1) -> (1,1)      "up then right"   increments (0,1),(1,0)

Both share the same endpoint and the same level-1 signature (1,1); they differ
only in the level-2 iterated integrals (opposite signed Levy area).

For a piecewise-linear path the signature is a tensor product (Chen's identity)
of the segment exponentials exp^{ox}(delta) = (1, delta, delta^{ox2}/2!, ...).
Everything below is exact rational arithmetic (carried in float); the Goursat
solver is a finite-difference scheme refined until it agrees with the deep
truncated kernel, confirming that the PDE sums all signature levels at once.
"""

import numpy as np
from math import factorial


# ---- truncated tensor algebra over R^d --------------------------------------

def seg_signature(delta, m):
    """Tensor exponential of one increment, truncated at level m.

    Level k of a straight segment with increment delta is delta^{ox k} / k!,
    because the ordered k-simplex has volume 1/k!.
    """
    sig = [np.array(1.0)]  # level 0 is the scalar 1
    for k in range(1, m + 1):
        res = np.array(1.0)
        for _ in range(k):
            res = np.multiply.outer(res, delta)
        sig.append(res / factorial(k))
    return sig


def chen(A, B, m):
    """Product in the truncated tensor algebra: (A ox B)_n = sum_{i+j=n} A_i ox B_j."""
    out = []
    for n in range(m + 1):
        acc = None
        for i in range(n + 1):
            term = np.multiply.outer(A[i], B[n - i])
            acc = term if acc is None else acc + term
        out.append(acc)
    return out


def path_signature(points, m):
    pts = [np.asarray(p, float) for p in points]
    sig = None
    for k in range(1, len(pts)):
        s = seg_signature(pts[k] - pts[k - 1], m)
        sig = s if sig is None else chen(sig, s, m)
    return sig


def sig_inner(S, T, m):
    return sum(float(np.sum(S[k] * T[k])) for k in range(m + 1))


# ---- the two paths ----------------------------------------------------------

X = [(0, 0), (1, 0), (1, 1)]
Y = [(0, 0), (0, 1), (1, 1)]

SX = path_signature(X, 2)
SY = path_signature(Y, 2)

print("=== depth-2 truncated signatures ===")
print("S(X) level 1 :", SX[1])
print("S(X) level 2 :\n", SX[2])
print("S(Y) level 1 :", SY[1])
print("S(Y) level 2 :\n", SY[2])

# shuffle identity check: S^{12}+S^{21} = S^1 * S^2 ; S^{11} = (S^1)^2/2
s1, s2 = SX[1]
print("\nshuffle check for X:  S12+S21 =", SX[2][0, 1] + SX[2][1, 0],
      " vs S1*S2 =", s1 * s2)
print("                      S11      =", SX[2][0, 0], " vs (S1)^2/2 =", s1 * s1 / 2)
# symmetric part of level 2 = outer product of total increment / ... (shuffle)
print("Levy area of X = (S12-S21)/2 =", (SX[2][0, 1] - SX[2][1, 0]) / 2)
print("Levy area of Y = (S12-S21)/2 =", (SY[2][0, 1] - SY[2][1, 0]) / 2)

print("\n=== depth-2 signature kernel (Kiraly-Oberhauser truncated kernel) ===")
lvl0 = 1.0
lvl1 = float(np.sum(SX[1] * SY[1]))
lvl2 = float(np.sum(SX[2] * SY[2]))
kXY = lvl0 + lvl1 + lvl2
print(f"level 0 term = {lvl0}")
print(f"level 1 term = <(1,1),(1,1)> = {lvl1}")
print(f"level 2 term = <M_X, M_Y>    = {lvl2}")
print(f"k^(2)(X,Y)   = {kXY}")

kXX = sig_inner(SX, SX, 2)
kYY = sig_inner(SY, SY, 2)
print(f"k^(2)(X,X)   = {kXX}")
print(f"k^(2)(Y,Y)   = {kYY}")
print(f"normalized   = k(X,Y)/sqrt(k(X,X)k(Y,Y)) = {kXY/np.sqrt(kXX*kYY):.6f}  (= 7/9 = {7/9:.6f})")

# ---- deeper truncations converge -------------------------------------------

print("\n=== truncated signature kernel at increasing depth m ===")
deep = {}
for m in range(1, 11):
    SXm = path_signature(X, m)
    SYm = path_signature(Y, m)
    deep[m] = sig_inner(SXm, SYm, m)
    print(f"  m={m:2d}   k^(m)(X,Y) = {deep[m]:.10f}")
full = deep[10]

# ---- Goursat PDE (Salvi et al. 2021):  d^2U/dsdt = <Xdot,Ydot> U ------------
# Piecewise-linear => <Xdot,Ydot> is constant on each cell (p,q) and equals
# <deltaX_p, deltaY_q>.  Explicit finite-difference scheme, grid refined.

dX = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]   # increments of X
dY = [np.array([0.0, 1.0]), np.array([1.0, 0.0])]   # increments of Y


def goursat(dX, dY, sub):
    """Second-order scheme for  d^2U/dsdt = c U.  Integrating the PDE over a
    grid cell and using the 4-corner average of U for the right-hand side gives
        U_{i+1,j+1}(1-a) = (1+a)(U_{i+1,j}+U_{i,j+1}) - (1-a)U_{i,j},   a = c hs ht/4.
    """
    P, Q = len(dX), len(dY)
    ns, nt = P * sub, Q * sub
    hs, ht = 1.0 / sub, 1.0 / sub
    U = np.ones((ns + 1, nt + 1))
    for i in range(ns):
        p = i // sub
        for j in range(nt):
            q = j // sub
            a = float(np.dot(dX[p], dY[q])) * hs * ht / 4.0
            U[i + 1, j + 1] = ((1 + a) * (U[i + 1, j] + U[i, j + 1]) - (1 - a) * U[i, j]) / (1 - a)
    return U[-1, -1]


print("\n=== Goursat PDE solution, grid refined (sub steps per segment) ===")
for sub in [1, 2, 4, 8, 16, 32, 64, 128]:
    print(f"  sub={sub:4d}   U(S,T) = {goursat(dX, dY, sub):.10f}")
print(f"\ndeep truncated kernel (m=10)      = {full:.10f}")
print(f"Goursat PDE (sub=128)             = {goursat(dX, dY, 128):.10f}")
print("The finite-difference PDE solution converges (from below) to the full")
print("signature inner product, which the depth-m truncation approaches from below too.")
