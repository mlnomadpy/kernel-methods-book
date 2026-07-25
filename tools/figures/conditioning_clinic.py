"""conditioning-clinic: ridge lifts weak Gram-matrix eigendirections."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import _style as S

import matplotlib.pyplot as plt

S.apply_style()
jax.config.update("jax_enable_x64", True)


def main() -> str:
    eigenvalues = jnp.array([10.0, 1.0, 1e-2, 1e-5])
    ridge = 1e-2
    regularized = eigenvalues + ridge
    index = np.arange(eigenvalues.size)
    raw_condition = eigenvalues.max() / eigenvalues.min()
    regularized_condition = regularized.max() / regularized.min()
    assert bool(jnp.all(jnp.isfinite(regularized)))
    assert bool(jnp.all(regularized > eigenvalues))
    assert bool(jnp.isclose(raw_condition, 1e6))
    assert float(regularized_condition) < 1001
    eigenvalues, regularized = map(np.asarray, (eigenvalues, regularized))

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.65), sharey=True)
    panels = (
        (eigenvalues, "Raw Gram spectrum", S.MUTED),
        (regularized, rf"After adding $\lambda I$, $\lambda={ridge:g}$", S.ACCENT),
    )
    for panel_index, (ax, (values, title, color)) in enumerate(zip(axes, panels)):
        S.bars(
            ax, index, values, width=0.62,
            highlight=[int(np.argmin(values))] if panel_index == 0 else None,
        )
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
    return S.save(fig, "conditioning-clinic")


if __name__ == "__main__":
    print(main())
