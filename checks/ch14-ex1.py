"""Check script for ch14 worked example 1.

Arc-cosine (ReLU-NNGP) kernel between two 2-D unit vectors, and how the
normalized kernel evolves under depth via the dual-activation recursion.
Every number printed here is embedded in the worked example.

Pure linear algebra + a Monte Carlo cross-check of the closed forms.
"""
import numpy as np

np.random.seed(0)

# --- setup: two unit vectors at 60 degrees --------------------------------
z = np.array([1.0, 0.0])
zp = np.array([0.5, np.sqrt(3) / 2])          # (0.5, 0.8660254...)
c0 = float(z @ zp)                            # 0.5
theta0 = np.arccos(c0)                        # pi/3

print("zp            =", np.round(zp, 6))
print("c0 = <z,zp>   =", round(c0, 6))
print("theta0 (rad)  =", round(theta0, 6))
print("theta0 (deg)  =", round(np.degrees(theta0), 6))

# --- arc-cosine angular parts ---------------------------------------------
def J0(t):
    return np.pi - t

def J1(t):
    return np.sin(t) + (np.pi - t) * np.cos(t)

# degree-0 and degree-1 normalized kernels on the unit sphere (Cho-Saul,
# rectified-power activation with the sqrt(2) factor => k_n = J_n/pi here).
k0 = J0(theta0) / np.pi
k1 = J1(theta0) / np.pi
p_both_pos = (np.pi - theta0) / (2 * np.pi)   # P(w.z>0 and w.zp>0)

print("J0(theta0)    =", round(J0(theta0), 6))
print("J1(theta0)    =", round(J1(theta0), 6))
print("k0 = J0/pi    =", round(k0, 6))
print("k1 = J1/pi    =", round(k1, 6))
print("P(both>0)     =", round(p_both_pos, 6))

# --- Monte Carlo cross-check (w ~ N(0, I) in R^2) -------------------------
N = 4_000_000
W = np.random.randn(N, 2)
a = W @ z
b = W @ zp
relu = lambda t: np.maximum(t, 0.0)
mc_relu = float(np.mean(relu(a) * relu(b)))           # -> J1/(2 pi)
mc_step = float(np.mean((a > 0) & (b > 0)))           # -> (pi-theta)/(2 pi)
print("MC E[ReLU ReLU]         =", round(mc_relu, 4), " closed J1/(2pi) =",
      round(J1(theta0) / (2 * np.pi), 4))
print("MC k1 = 2 E[ReLU ReLU]  =", round(2 * mc_relu, 4), " closed k1 =",
      round(k1, 4))
print("MC P(both>0)            =", round(mc_step, 4), " closed =",
      round(p_both_pos, 4))

# --- depth recursion: normalized ReLU-NNGP correlation map ----------------
# c_{l} = J1(arccos c_{l-1}) / pi  (variances normalized to 1 each layer)
def T(c):
    return J1(np.arccos(np.clip(c, -1.0, 1.0))) / np.pi

cs = [c0]
for _ in range(6):
    cs.append(float(T(cs[-1])))
print("correlation by depth:", [round(c, 6) for c in cs])

# fixed point: iterate to convergence
c = 0.5
for _ in range(200):
    c = T(c)
print("fixed point c* =", round(c, 6))
