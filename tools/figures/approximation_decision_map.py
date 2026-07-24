"""Decision map for choosing a large-scale kernel approximation."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import _style as S

S.apply_style()

methods = {
    "Exact / matrix-free": (0.16, 0.18, S.INK),
    "Nyström": (0.78, 0.25, S.ACCENT),
    "Random features": (0.30, 0.80, S.POS),
    "Structured products": (0.66, 0.72, S.GOOD),
}

assert len(methods) == 4
assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y, _ in methods.values())

fig, ax = S.new_axes(5.4, 3.35)
ax.axvspan(0.5, 1.02, color=S.ACCENT, alpha=0.055)
ax.axhspan(0.5, 1.02, color=S.POS, alpha=0.045)
ax.text(0.97, 0.04, "exploit a concentrated spectrum", ha="right", color=S.MUTED, fontsize=8)
ax.text(0.03, 0.97, "avoid data-dependent state", va="top", color=S.MUTED, fontsize=8)

for label, (x, y, color) in methods.items():
    box = FancyBboxPatch(
        (x - 0.12, y - 0.055),
        0.24,
        0.11,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        facecolor=S.PAPER,
        edgecolor=color,
        linewidth=1.3,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", color=color, fontsize=8.2)

ax.annotate(
    "fast spectral decay\n+ reusable landmarks",
    xy=methods["Nyström"][:2],
    xytext=(0.91, 0.47),
    ha="center",
    color=S.ACCENT,
    fontsize=7.7,
    arrowprops={"arrowstyle": "-", "color": S.ACCENT, "lw": 0.8},
)
ax.annotate(
    "streaming or privacy:\nregenerate from a seed",
    xy=methods["Random features"][:2],
    xytext=(0.10, 0.59),
    ha="center",
    color=S.POS,
    fontsize=7.7,
    arrowprops={"arrowstyle": "-", "color": S.POS, "lw": 0.8},
)

ax.set(
    xlim=(0, 1),
    ylim=(0, 1),
    xlabel="spectral concentration / reusable data geometry  $\\longrightarrow$",
    ylabel="streaming, privacy, or distribution pressure  $\\longrightarrow$",
)
ax.set_xticks([])
ax.set_yticks([])
S.finish(ax)
S.save(fig, "approximation-decision-map")
print("methods=4; decision_axes=2")
