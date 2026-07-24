"""Substring counts become coordinates; their dot products form the kernel."""
from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np

from _style import ACCENT, HEAT, INK, MUTED, RULE, apply_style, save

apply_style()

sequences = ("ABAB", "AABB", "BABA")
features = tuple("".join(pair) for pair in itertools.product("AB", repeat=2))


def counts(sequence: str) -> np.ndarray:
    windows = (sequence[i : i + 2] for i in range(len(sequence) - 1))
    histogram = {feature: 0 for feature in features}
    for window in windows:
        histogram[window] += 1
    return np.array([histogram[feature] for feature in features], dtype=float)


phi = np.stack([counts(sequence) for sequence in sequences])
gram = phi @ phi.T
if np.any(np.linalg.eigvalsh(gram) < -1e-12):
    raise RuntimeError("The explicit substring Gram matrix must be positive semidefinite.")

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.5, 2.75), gridspec_kw={"width_ratios": [1.2, 1]})
im0 = ax0.imshow(phi, cmap=HEAT, vmin=0, vmax=max(2.0, float(phi.max())), aspect="auto")
ax0.set_xticks(range(len(features)), features)
ax0.set_yticks(range(len(sequences)), sequences)
ax0.set_xlabel("length-2 substring coordinate")
ax0.set_title(r"Explicit map $\Phi_2(s)$")
for row, col in np.ndindex(phi.shape):
    ax0.text(col, row, f"{phi[row, col]:.0f}", ha="center", va="center", color=INK, fontsize=8)

im1 = ax1.imshow(gram, cmap=HEAT, vmin=0, vmax=float(gram.max()), aspect="equal")
ax1.set_xticks(range(len(sequences)), sequences, rotation=35, ha="right")
ax1.set_yticks(range(len(sequences)), sequences)
ax1.set_title(r"Kernel $K_2=\Phi_2\Phi_2^\top$")
for row, col in np.ndindex(gram.shape):
    ax1.text(col, row, f"{gram[row, col]:.0f}", ha="center", va="center", color=INK, fontsize=8)

for ax in (ax0, ax1):
    for spine in ax.spines.values():
        spine.set_color(RULE)
    ax.tick_params(length=0, colors=MUTED)
fig.text(
    0.5,
    0.01,
    "ABAB and BABA are close because both activate AB and BA; no string enumeration beyond observed windows is needed.",
    ha="center",
    color=MUTED,
    fontsize=8,
)
fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.25, wspace=0.38)
save(fig, "sequence-feature-map")
