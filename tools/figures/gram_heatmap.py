"""gram-heatmap: twelve points on a line and their Gram matrix.

Reproduces ``WIDGETS["gram-heatmap"]`` in ``public/assets/viz.js`` at its default
state: 12 points spread on [-3, 3] with a small deterministic jitter, Gaussian
kernel at bandwidth 1. Top strip shows the points; the panel below is the
12x12 Gram matrix K_ij = k(x_i, x_j), scaled by its largest entry exactly as
the widget's ``heat()`` ramp does (PAPER -> ACCENT).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()


def gaussian_gram(x: jnp.ndarray, bw: float) -> jnp.ndarray:
    d = x[:, None] - x[None, :]
    return jnp.exp(-(d ** 2) / (2.0 * bw * bw))


def main() -> str:
    # widget default: x_i = -3 + 6 i/11 + U(-0.15, 0.15); Gaussian, bw = 1.
    g = S.rng(0)
    i = np.arange(12)
    x = jnp.asarray(-3.0 + 6.0 * i / 11.0 + (g.random(12) - 0.5) * 0.3)
    K = gaussian_gram(x, bw=1.0)
    K = np.asarray(K)
    K = K / np.max(np.abs(K))

    fig = plt.figure(figsize=(4.05, 4.15))
    gs = GridSpec(2, 1, height_ratios=[0.75, 9], hspace=0.08, figure=fig)

    ax_line = fig.add_subplot(gs[0])
    ax_line.axhline(0.0, color=S.RULE, lw=0.8, zorder=1)
    ax_line.scatter(np.asarray(x), np.zeros(12), s=26, color=S.POS,
                    edgecolor=S.PAPER, linewidth=0.8, zorder=2)
    ax_line.set_xlim(-3.6, 3.6)
    ax_line.set_ylim(-0.5, 0.5)
    ax_line.axis("off")

    ax = fig.add_subplot(gs[1])
    S.matrix_image(ax, K)
    ticks = np.arange(12)
    tick_labels = [str(v + 1) if v in (0, 3, 7, 11) else "" for v in ticks]
    ax.set_xticks(ticks, tick_labels)
    ax.set_yticks(ticks, tick_labels)
    ax.set_xlabel(r"column $j$")
    ax.set_ylabel(r"row $i$")
    for spine in ax.spines.values():
        spine.set_color(S.INK)
        spine.set_linewidth(0.5)
    return S.save(fig, "gram-heatmap")


if __name__ == "__main__":
    print(main())
