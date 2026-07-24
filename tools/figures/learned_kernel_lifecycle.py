"""Lifecycle diagram for a kernel whose representation is learned."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import _style as S

S.apply_style()

stages = [
    ("Raw object", "structure\n+ nuisance"),
    ("Prior geometry", "architecture\n+ invariance"),
    ("Feature learning", "labels move\nthe representation"),
    ("Trained kernel", "similarity\nafter training"),
    ("Shift audit", "calibration\n+ failure"),
]
colors = [S.INK, S.POS, S.ACCENT, S.GOOD, S.NEG]
assert len(stages) == len(colors) == 5

fig, ax = plt.subplots(figsize=(6.5, 2.9))
xs = [0.09, 0.295, 0.50, 0.705, 0.91]
for index, ((title, subtitle), color, x) in enumerate(zip(stages, colors, xs)):
    patch = FancyBboxPatch(
        (x - 0.087, 0.42),
        0.174,
        0.30,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=S.PAPER,
        edgecolor=color,
        linewidth=1.25,
    )
    ax.add_patch(patch)
    ax.text(x, 0.625, title, ha="center", va="center", color=color, fontsize=8.2, weight="bold")
    ax.text(x, 0.505, subtitle, ha="center", va="center", color=S.MUTED, fontsize=6.8, linespacing=1.1)
    if index < len(stages) - 1:
        ax.annotate("", xy=(xs[index + 1] - 0.09, 0.57), xytext=(x + 0.09, 0.57),
                    arrowprops={"arrowstyle": "->", "color": S.RULE, "lw": 1.2})

ax.annotate(
    "If features barely move, this stage collapses:\nthe model stays in its NTK geometry.",
    xy=(0.50, 0.42),
    xytext=(0.50, 0.12),
    ha="center",
    color=S.ACCENT,
    fontsize=8,
    arrowprops={"arrowstyle": "-", "color": S.ACCENT, "lw": 0.9},
)
ax.annotate(
        "Failure can enter at every arrow;\nthe final audit makes it visible.",
    xy=(0.91, 0.73),
    xytext=(0.74, 0.90),
    ha="center",
    color=S.NEG,
    fontsize=7.8,
    arrowprops={"arrowstyle": "-", "color": S.NEG, "lw": 0.8},
)
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")
S.save(fig, "learned-kernel-lifecycle")
print("stages=5; transitions=4")
