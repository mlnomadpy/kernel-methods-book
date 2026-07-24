"""epsilon-tube: only residuals outside the tolerance tube pay loss."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import _style as S

S.apply_style()


def main() -> str:
    x = np.array([0.3, 0.9, 1.5, 2.1, 2.7, 3.3, 3.9, 4.5])
    prediction = 0.34 * x + 0.45
    residual = np.array([0.08, -0.22, 0.48, -0.05, -0.55, 0.18, 0.67, -0.12])
    y = prediction + residual
    epsilon = 0.25
    loss = np.maximum(np.abs(residual) - epsilon, 0.0)
    outside = loss > 0
    grid = np.linspace(0.0, 4.8, 200)
    fit = 0.34 * grid + 0.45

    fig, axes = plt.subplots(
        2, 1, figsize=(5.2, 3.55), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.12},
    )
    ax = axes[0]
    ax.fill_between(grid, fit - epsilon, fit + epsilon, color=S.RULE, alpha=0.55)
    ax.plot(grid, fit, color=S.INK, lw=1.7, label="prediction")
    ax.plot(grid, fit - epsilon, color=S.MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.plot(grid, fit + epsilon, color=S.MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.scatter(x[~outside], y[~outside], s=30, facecolor=S.PAPER, edgecolor=S.POS,
               linewidth=1.2, marker="o", label="zero loss")
    ax.scatter(x[outside], y[outside], s=36, color=S.NEG,
               linewidth=1.1, marker="x", label="support vector")
    ax.set_ylabel("target")
    ax.set_title(rf"An $\varepsilon$-tube with $\varepsilon={epsilon:.2f}$")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    S.finish(ax)

    bx = axes[1]
    colors = np.where(outside, S.ACCENT, S.RULE)
    bx.bar(x, loss, width=0.28, color=colors, edgecolor=S.INK, linewidth=0.45)
    bx.axhline(0, color=S.INK, lw=0.7)
    bx.set_ylabel(r"$L_\varepsilon$")
    bx.set_xlabel("input")
    bx.text(0.04, 0.76, "inside the tube: no charge", transform=bx.transAxes,
            color=S.MUTED, fontsize=8)
    S.finish(bx)
    assert np.all(loss[~outside] == 0)
    assert np.allclose(loss[outside], np.abs(residual[outside]) - epsilon)
    return S.save(fig, "epsilon-tube")


if __name__ == "__main__":
    print(main())
