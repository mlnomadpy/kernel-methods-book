"""Posterior averaging of latent feature maps yields a PSD observed kernel."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _style import ACCENT, HEAT, INK, MUTED, POS, RULE, apply_style, save

apply_style()

objects = ("A", "B", "C")
states = (r"$z_1$", r"$z_2$", r"$z_3$")
posterior = np.array(
    [
        [0.72, 0.23, 0.05],
        [0.56, 0.37, 0.07],
        [0.08, 0.24, 0.68],
    ]
)
if not np.allclose(posterior.sum(axis=1), 1.0):
    raise RuntimeError("Latent posterior rows must sum to one.")
gram = posterior @ posterior.T
if np.any(np.linalg.eigvalsh(gram) < -1e-12):
    raise RuntimeError("A marginalized identity-feature kernel must be PSD.")

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.6, 2.85), gridspec_kw={"width_ratios": [1.25, 1]})
x_left, x_right = 0.0, 1.0
y_objects = np.array([0.82, 0.50, 0.18])
y_states = np.array([0.82, 0.50, 0.18])
for i, (name, y) in enumerate(zip(objects, y_objects)):
    ax0.scatter(x_left, y, s=180, facecolor="white", edgecolor=INK, zorder=3)
    ax0.text(x_left, y, name, ha="center", va="center", fontsize=9, zorder=4)
    for j, state_y in enumerate(y_states):
        weight = posterior[i, j]
        ax0.plot(
            (x_left + 0.06, x_right - 0.06),
            (y, state_y),
            color=ACCENT if weight >= 0.5 else POS,
            lw=0.5 + 4.0 * weight,
            alpha=0.25 + 0.75 * weight,
            solid_capstyle="round",
        )
for state, y in zip(states, y_states):
    ax0.scatter(x_right, y, s=180, facecolor="white", edgecolor=INK, marker="s", zorder=3)
    ax0.text(x_right, y, state, ha="center", va="center", fontsize=9, zorder=4)
ax0.text(x_left, 1.02, "observed object", ha="center", color=MUTED, fontsize=8)
ax0.text(x_right, 1.02, "latent explanation", ha="center", color=MUTED, fontsize=8)
ax0.set_xlim(-0.22, 1.22)
ax0.set_ylim(0.02, 1.08)
ax0.axis("off")
ax0.set_title(r"Posterior feature $\bar\Phi(x)=\mathbb{E}[\Phi(x,Z)\mid x]$")

ax1.imshow(gram, cmap=HEAT, vmin=0, vmax=float(gram.max()), aspect="equal")
ax1.set_xticks(range(3), objects)
ax1.set_yticks(range(3), objects)
ax1.set_title(r"Marginalized kernel $\bar\Phi\bar\Phi^\top$")
for row, col in np.ndindex(gram.shape):
    ax1.text(col, row, f"{gram[row, col]:.2f}", ha="center", va="center", color=INK, fontsize=8)
ax1.tick_params(length=0, colors=MUTED)
for spine in ax1.spines.values():
    spine.set_color(RULE)
ax1.text(
    0.5,
    -0.22,
    "A and B are close because their posterior mass\nfalls on the same hidden states.",
    transform=ax1.transAxes,
    ha="center",
    va="top",
    color=MUTED,
    fontsize=8,
)

fig.subplots_adjust(left=0.03, right=0.98, top=0.84, bottom=0.24, wspace=0.3)
save(fig, "latent-marginalization")
