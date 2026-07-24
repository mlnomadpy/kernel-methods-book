"""End-to-end decision workflow for a defensible kernel project."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import _style as S

S.apply_style()

steps = [
    (0.12, 0.76, "1  Encode structure", "kernel + invariance"),
    (0.50, 0.76, "2  Normalize", "inputs, diagonal, centering"),
    (0.88, 0.76, "3  Probe geometry", "spectrum + conditioning"),
    (0.88, 0.29, "4  Choose compute", "exact / Nyström / features"),
    (0.50, 0.29, "5  Select honestly", "nested validation"),
    (0.12, 0.29, "6  Audit the decision", "shift, calibration, influence"),
]
assert len(steps) == 6

fig, ax = plt.subplots(figsize=(6.1, 3.0))
for index, (x, y, title, subtitle) in enumerate(steps):
    color = S.ACCENT if index in (0, 3) else (S.GOOD if index == 5 else S.INK)
    patch = FancyBboxPatch(
        (x - 0.14, y - 0.105),
        0.28,
        0.21,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=S.PAPER,
        edgecolor=color,
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(x, y + 0.033, title, ha="center", va="center", color=color, fontsize=8.3, weight="bold")
    ax.text(x, y - 0.043, subtitle, ha="center", va="center", color=S.MUTED, fontsize=7.4)

route = [(0.26, 0.76, 0.36, 0.76), (0.64, 0.76, 0.74, 0.76), (0.88, 0.65, 0.88, 0.40),
         (0.74, 0.29, 0.64, 0.29), (0.36, 0.29, 0.26, 0.29)]
for x0, y0, x1, y1 in route:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "->", "color": S.RULE, "lw": 1.25})

ax.annotate(
    "validation failure",
    xy=(0.36, 0.37),
    xytext=(0.23, 0.57),
    color=S.NEG,
    fontsize=7.7,
    arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=-0.35", "color": S.NEG, "lw": 0.9},
)
ax.annotate(
    "audit failure",
    xy=(0.12, 0.65),
    xytext=(0.05, 0.48),
    color=S.NEG,
    fontsize=7.7,
    arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=-0.34", "color": S.NEG, "lw": 0.9},
)
ax.text(0.50, 0.05, "A deployed result is a loop with evidence, not a one-way training pipeline.",
        ha="center", color=S.MUTED, fontsize=8)
ax.set(xlim=(-0.04, 1.04), ylim=(0, 1))
ax.axis("off")
S.save(fig, "kernel-workflow")
print("steps=6; feedback_loops=2")
