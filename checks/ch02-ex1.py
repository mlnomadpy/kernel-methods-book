"""Worked example 1 (ch02): the kernel trick computed two ways.

Homogeneous degree-2 polynomial kernel K(x,y) = (x . y)^2 on 2-D points has the
explicit 3-dimensional feature map Phi(x) = (x1^2, sqrt(2) x1 x2, x2^2), because
<Phi(x), Phi(y)> = x1^2 y1^2 + 2 x1 x2 y1 y2 + x2^2 y2^2 = (x1 y1 + x2 y2)^2.

Route A goes through the explicit feature vectors; route B stays in the kernel.
Both must give the same inner product and the same feature-space distance,
which is the whole content of the kernel trick. Prints every displayed number.
"""
import numpy as np

np.set_printoptions(suppress=True)

xa = np.array([1.0, 2.0])
xb = np.array([2.0, 1.0])


def phi(x):
    return np.array([x[0] ** 2, np.sqrt(2.0) * x[0] * x[1], x[1] ** 2])


def K(x, y):
    return float(x @ y) ** 2


# --- Route A: explicit feature map ---
pa, pb = phi(xa), phi(xb)
print("Phi(xa) =", np.round(pa, 6))
print("Phi(xb) =", np.round(pb, 6))
ipA = float(pa @ pb)
print("Route A  <Phi(xa),Phi(xb)> =", round(ipA, 6))

# --- Route B: kernel directly ---
ipB = K(xa, xb)
print("Route B  K(xa,xb) = (xa.xb)^2 =", round(ipB, 6))
print("dot xa.xb =", float(xa @ xb))
print("inner products agree:", np.isclose(ipA, ipB))

# --- Feature-space distance, two ways ---
Kaa, Kbb, Kab = K(xa, xa), K(xb, xb), K(xa, xb)
print("K(xa,xa) =", round(Kaa, 6), " K(xb,xb) =", round(Kbb, 6), " K(xa,xb) =", round(Kab, 6))
d2_kernel = Kaa + Kbb - 2 * Kab
d2_feature = float((pa - pb) @ (pa - pb))
print("d^2 via kernel  K(xa,xa)+K(xb,xb)-2K(xa,xb) =", round(d2_kernel, 6))
print("d^2 via features ||Phi(xa)-Phi(xb)||^2       =", round(d2_feature, 6))
print("distance d =", round(np.sqrt(d2_kernel), 6))
print("distances agree:", np.isclose(d2_kernel, d2_feature))
