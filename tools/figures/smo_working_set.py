"""smo-working-set: a two-variable SVM update is a clipped line search."""
from __future__ import annotations

import jax
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import _style as S
import matplotlib.pyplot as plt

S.apply_style()


def main() -> str:
    c = 1.0
    old = 0.18
    unconstrained = 1.28
    clipped = jnp.clip(unconstrained, 0.0, c)
    alpha = jnp.linspace(0.0, c, 240, dtype=jnp.float64)
    objective = 1.0 - 1.55 * (alpha - unconstrained) ** 2
    assert bool(jnp.all(jnp.isfinite(objective)))
    assert bool(jnp.all(jnp.diff(objective) > 0.0))
    marked_objective = 1.0 - 1.55 * (
        jnp.array([old, clipped], dtype=jnp.float64) - unconstrained
    ) ** 2
    alpha_h, objective_h = np.asarray(alpha), np.asarray(objective)
    clipped_h = float(clipped)
    marked_objective_h = np.asarray(marked_objective)

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.45))
    ax = axes[0]
    ax.fill_between([0, c], [0, c], color=S.RULE, alpha=0.32)
    ax.plot([0, c], [0, c], color=S.ACCENT, lw=2.0)
    ax.scatter([old, clipped_h], [old, clipped_h], s=[35, 45],
               color=[S.MUTED, S.POS], edgecolor=S.INK, linewidth=0.55, zorder=3)
    ax.annotate("", xy=(clipped_h, clipped_h), xytext=(old, old),
                arrowprops={"arrowstyle": "->", "color": S.INK, "lw": 1.0})
    ax.text(old + 0.04, old - 0.13, "start", fontsize=7.6, color=S.MUTED)
    ax.text(clipped - 0.34, clipped - 0.13, "box optimum", fontsize=7.6, color=S.POS)
    ax.set_xlim(-0.05, 1.08); ax.set_ylim(-0.05, 1.08)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\alpha_i$"); ax.set_ylabel(r"$\alpha_j$")
    S.panel_label(ax, "a")
    ax.text(
        0.02, 0.96, r"feasible line  $\alpha_i=\alpha_j$",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.7, color=S.MUTED,
    )
    S.finish(ax)

    bx = axes[1]
    bx.plot(alpha_h, objective_h, color=S.INK, lw=1.8)
    bx.axvline(clipped_h, color=S.ACCENT, lw=1.4, ls=(0, (4, 3)))
    bx.scatter([old, clipped_h], marked_objective_h,
               color=[S.MUTED, S.ACCENT], s=[30, 42], edgecolor=S.INK, linewidth=0.5)
    bx.text(
        0.98, 0.58, r"box edge $C$", transform=bx.transAxes,
        ha="right", va="top", fontsize=7.5, color=S.ACCENT,
    )
    bx.set_xlim(0, c)
    bx.set_xlabel("position along feasible line")
    bx.set_ylabel("dual objective")
    S.panel_label(bx, "b")
    bx.text(
        0.02, 0.96, "maximize, then clip",
        transform=bx.transAxes, ha="left", va="top", fontsize=7.7, color=S.MUTED,
    )
    S.finish(bx)
    fig.subplots_adjust(wspace=0.42, top=0.94)
    assert clipped_h == c
    return S.save(fig, "smo-working-set")


if __name__ == "__main__":
    print(main())
