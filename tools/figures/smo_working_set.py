"""smo-working-set: a two-variable SVM update is a clipped line search."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import _style as S

S.apply_style()


def main() -> str:
    c = 1.0
    old = 0.18
    unconstrained = 1.28
    clipped = np.clip(unconstrained, 0.0, c)
    alpha = np.linspace(0.0, c, 240)
    objective = 1.0 - 1.55 * (alpha - unconstrained) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.65))
    ax = axes[0]
    ax.fill_between([0, c], [0, c], color=S.RULE, alpha=0.32)
    ax.plot([0, c], [0, c], color=S.ACCENT, lw=2.0)
    ax.scatter([old, clipped], [old, clipped], s=[35, 45],
               color=[S.MUTED, S.POS], edgecolor=S.INK, linewidth=0.55, zorder=3)
    ax.annotate("", xy=(clipped, clipped), xytext=(old, old),
                arrowprops={"arrowstyle": "->", "color": S.INK, "lw": 1.0})
    ax.text(old + 0.03, old - 0.12, "old pair", fontsize=8, color=S.MUTED)
    ax.text(clipped - 0.36, clipped - 0.12, "clipped optimum", fontsize=8, color=S.POS)
    ax.set_xlim(-0.05, 1.08); ax.set_ylim(-0.05, 1.08)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\alpha_i$"); ax.set_ylabel(r"$\alpha_j$")
    ax.set_title(r"Equality constraint leaves a line, $\alpha_i=\alpha_j$")
    S.finish(ax)

    bx = axes[1]
    bx.plot(alpha, objective, color=S.INK, lw=1.8)
    bx.axvline(clipped, color=S.ACCENT, lw=1.4, ls=(0, (4, 3)))
    bx.scatter([old, clipped], [np.interp(old, alpha, objective),
                               np.interp(clipped, alpha, objective)],
               color=[S.MUTED, S.ACCENT], s=[30, 42], edgecolor=S.INK, linewidth=0.5)
    bx.annotate("unconstrained peak\nlies beyond the box", xy=(1.0, objective[-1]),
                xytext=(0.45, 0.3),
                arrowprops={"arrowstyle": "->", "color": S.ACCENT, "lw": 1.0},
                fontsize=8, color=S.ACCENT)
    bx.set_xlim(0, c)
    bx.set_xlabel("position along feasible line")
    bx.set_ylabel("dual objective")
    bx.set_title("SMO maximizes a concave parabola, then clips")
    S.finish(bx)
    fig.subplots_adjust(wspace=0.35)
    assert clipped == c
    return S.save(fig, "smo-working-set")


if __name__ == "__main__":
    print(main())
