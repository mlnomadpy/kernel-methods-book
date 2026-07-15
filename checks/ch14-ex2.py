"""Check script for ch14 worked example 2.

Deep ReLU neural tangent kernel by the layerwise recursion, on the unit
sphere with He normalization (factor 2 in the dual activations), so the
diagonal stays at 1 and every quantity is a correlation in [0, 1].

  Sigma^0(x,x')      = <x,x'>                         (input correlation)
  Sigma^h(x,x')      = J1(theta_{h-1}) / pi           (NNGP, ReLU dual act.)
  Sigmadot^h(x,x')   = (pi - theta_{h-1}) / pi        (derivative kernel)
  Theta^0            = Sigma^0
  Theta^h            = Sigma^h + Theta^{h-1} * Sigmadot^h

Every number printed here is embedded in the worked example.
"""
import numpy as np

c0 = 0.5                                  # <x,x'> for two unit vectors, 60 deg


def J1(t):
    return np.sin(t) + (np.pi - t) * np.cos(t)


def sigma_map(c):                          # NNGP correlation after one layer
    return J1(np.arccos(np.clip(c, -1.0, 1.0))) / np.pi


def sigmadot_map(c):                       # derivative-kernel correlation
    return (np.pi - np.arccos(np.clip(c, -1.0, 1.0))) / np.pi


# --- forward pass of the two coupled recursions ---------------------------
L = 3
Sigma = [c0]          # Sigma^0
Theta = c0            # Theta^0
rows = []
for h in range(1, L + 1):
    c_prev = Sigma[-1]
    S = sigma_map(c_prev)          # Sigma^h
    Sd = sigmadot_map(c_prev)      # Sigmadot^h
    Theta = S + Theta * Sd         # Theta^h
    Sigma.append(S)
    rows.append((h, c_prev, S, Sd, Theta))

print("layer |  c_{h-1}  |  Sigma^h  | Sigmadot^h |   Theta^h")
for (h, cprev, S, Sd, Th) in rows:
    print(f"  {h}   | {cprev:.6f} | {S:.6f} |  {Sd:.6f}  | {Th:.6f}")

# normalized NTK (divide by its diagonal, which for x=x' equals L+1)
print("Theta^L (raw)        =", round(rows[-1][4], 6))
print("Theta diagonal (L+1) =", L + 1)
print("normalized NTK       =", round(rows[-1][4] / (L + 1), 6))
