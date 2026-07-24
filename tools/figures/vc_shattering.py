"""Shattering versus an impossible XOR labeling for planar halfspaces."""
import matplotlib.pyplot as plt
import numpy as np

import _style as S

S.apply_style()

fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.8))

tri = np.array([[-0.24, -0.16], [0.24, -0.16], [0.0, 0.25]])
centers = [(0.55 + col, 0.55 + row) for row in range(2) for col in range(4)]
for pattern, center in enumerate(centers):
    signs = np.array([1 if (pattern >> bit) & 1 else -1 for bit in range(3)])
    points = tri + np.asarray(center)
    for sign, color, marker in [(1, S.POS, "o"), (-1, S.NEG, "x")]:
        selected = signs == sign
        axes[0].scatter(points[selected, 0], points[selected, 1], s=19, color=color,
                        marker=marker, linewidth=1.4, zorder=3)
    if np.any(signs == 1) and np.any(signs == -1):
        pos_mean = tri[signs == 1].mean(axis=0)
        neg_mean = tri[signs == -1].mean(axis=0)
        normal = pos_mean - neg_mean
        midpoint = 0.5 * (pos_mean + neg_mean)
        tangent = np.array([-normal[1], normal[0]])
        tangent /= np.linalg.norm(tangent)
        segment = np.vstack((midpoint - 0.34 * tangent, midpoint + 0.34 * tangent))
        segment += np.asarray(center)
        axes[0].plot(segment[:, 0], segment[:, 1], color=S.ACCENT, lw=0.9)
axes[0].text(0.03, 0.97, "every one of the $2^3$ labelings", transform=axes[0].transAxes,
             va="top", color=S.GOOD)
axes[0].set_title("Three points: shattered")

sq = np.array([[-0.78, -0.78], [0.78, -0.78], [0.78, 0.78], [-0.78, 0.78]])
positive = np.array([True, False, True, False])
axes[1].scatter(sq[positive, 0], sq[positive, 1], s=52, color=S.POS, marker="o",
                label="+", zorder=3)
axes[1].scatter(sq[~positive, 0], sq[~positive, 1], s=58, color=S.NEG, marker="x",
                linewidth=2, label="-", zorder=3)
axes[1].plot(sq[positive, 0], sq[positive, 1], color=S.POS, lw=1.2, ls="--")
axes[1].plot(sq[~positive, 0], sq[~positive, 1], color=S.NEG, lw=1.2, ls="--")
axes[1].text(0.04, 0.96, "one impossible labeling is enough", transform=axes[1].transAxes,
             va="top", color=S.NEG)
axes[1].set_title("Four corners: XOR fails")
axes[1].legend(frameon=False, loc="lower right")

axes[0].set(xlim=(0.0, 4.1), ylim=(0.0, 2.05), xticks=[], yticks=[])
axes[1].set(xlim=(-1.15, 1.15), ylim=(-1.05, 1.12), xticks=[], yticks=[])
for ax in axes:
    ax.set_aspect("equal")
    S.finish(ax)
fig.tight_layout(w_pad=1.6)
S.save(fig, "vc-shattering")
print("halfplanes_shatter_triangle=True; xor_square_separable=False")
