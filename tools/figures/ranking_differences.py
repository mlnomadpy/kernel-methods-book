"""ranking-differences: preferences become labeled difference vectors."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import _style as S

S.apply_style()


def main() -> str:
    items = np.array([[0.4, 0.7], [1.2, 1.0], [1.7, 1.8], [2.6, 1.55]])
    pairs = [(3, 2), (2, 1), (1, 0), (3, 1)]
    differences = np.array([items[i] - items[j] for i, j in pairs])
    mirrored = -differences

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.65))
    ax = axes[0]
    ax.scatter(items[:, 0], items[:, 1], s=38, color=S.POS, edgecolor=S.INK, linewidth=0.6)
    for rank, (x, y) in enumerate(items, start=1):
        ax.text(x + 0.08, y + 0.03, str(rank), fontsize=8, color=S.INK)
    for i, j in pairs[:3]:
        ax.annotate("", xy=items[i], xytext=items[j],
                    arrowprops={"arrowstyle": "->", "color": S.ACCENT, "lw": 1.1})
    ax.set_title("Original items: arrows mean preferred")
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    S.finish(ax)

    bx = axes[1]
    bx.axhline(0, color=S.RULE, lw=0.7)
    bx.axvline(0, color=S.RULE, lw=0.7)
    bx.scatter(differences[:, 0], differences[:, 1], s=34, color=S.POS,
               edgecolor=S.INK, linewidth=0.55, label=r"$x_i-x_j$ (+)")
    bx.scatter(mirrored[:, 0], mirrored[:, 1], s=34, facecolor=S.PAPER,
               edgecolor=S.NEG, linewidth=1.1, marker="s", label=r"$x_j-x_i$ (-)")
    limit = 1.75
    line = np.linspace(-limit, limit, 100)
    bx.plot(line, -0.4 * line, color=S.ACCENT, lw=1.5, ls=(0, (5, 3)),
            label=r"$w^\top z=0$")
    bx.set_xlim(-limit, limit); bx.set_ylim(-limit, limit)
    bx.set_aspect("equal", adjustable="box")
    bx.set_title("Difference space: one binary boundary")
    bx.set_xlabel(r"$z_1$"); bx.set_ylabel(r"$z_2$")
    bx.legend(frameon=False, loc="lower right", fontsize=7)
    S.finish(bx)
    fig.subplots_adjust(wspace=0.32)
    assert np.allclose(mirrored, -differences)
    return S.save(fig, "ranking-differences")


if __name__ == "__main__":
    print(main())
