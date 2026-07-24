"""conditioning-clinic: ridge lifts weak Gram-matrix eigendirections."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import _style as S

S.apply_style()


def main() -> str:
    eigenvalues = np.array([10.0, 1.0, 1e-2, 1e-5])
    ridge = 1e-2
    regularized = eigenvalues + ridge
    index = np.arange(eigenvalues.size)

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.65), sharey=True)
    panels = (
        (eigenvalues, "Raw Gram spectrum", S.MUTED),
        (regularized, rf"After adding $\lambda I$, $\lambda={ridge:g}$", S.ACCENT),
    )
    for ax, (values, title, color) in zip(axes, panels):
        ax.bar(index, values, width=0.62, color=color, edgecolor=S.INK, linewidth=0.55)
        ax.set_yscale("log")
        ax.set_xticks(index, [rf"$\lambda_{i + 1}$" for i in index])
        ax.set_ylim(5e-6, 20)
        ax.set_title(title)
        condition = values.max() / values.min()
        ax.text(
            0.04,
            0.06,
            rf"$\kappa_2={condition:,.0f}$",
            transform=ax.transAxes,
            color=S.INK,
            bbox={"facecolor": S.PAPER, "edgecolor": S.RULE, "pad": 2.5},
        )
        S.finish(ax)
    axes[0].set_ylabel("eigenvalue (log scale)")
    axes[1].annotate(
        "weak directions are lifted",
        xy=(3, regularized[-1]),
        xytext=(1.55, 0.11),
        arrowprops={"arrowstyle": "->", "color": S.ACCENT, "lw": 1.0},
        color=S.ACCENT,
        fontsize=8,
    )
    fig.subplots_adjust(wspace=0.18)
    assert np.isclose(eigenvalues.max() / eigenvalues.min(), 1e6)
    assert regularized.max() / regularized.min() < 1001
    return S.save(fig, "conditioning-clinic")


if __name__ == "__main__":
    print(main())
