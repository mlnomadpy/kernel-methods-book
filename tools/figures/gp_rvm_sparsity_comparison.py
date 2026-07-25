"""gp-rvm-sparsity-comparison: similar fits can use radically different basis counts."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
import numpy as np
import matplotlib.pyplot as plt
import _style as S

S.apply_style()
jax.config.update("jax_enable_x64", True)


def rbf(a, b, ell=.42):
    return jnp.exp(-.5 * ((a[:, None] - b[None, :]) / ell) ** 2)


def main() -> str:
    x = jnp.linspace(-2.2, 2.2, 41)
    y = jnp.sin(1.8 * x) * jnp.exp(-.08 * x**2)
    grid = jnp.linspace(-2.4, 2.4, 400)
    K = rbf(x, x)
    alpha = cho_solve((jnp.linalg.cholesky(K + .03 * jnp.eye(len(x))), True), y)
    gp = rbf(grid, x) @ alpha
    support = jnp.array([2, 8, 15, 20, 26, 33, 39])
    Ks = rbf(x, x[support])
    coef = jnp.linalg.lstsq(Ks, y, rcond=1e-6)[0]
    sparse = rbf(grid, x[support]) @ coef
    rmse = jnp.sqrt(jnp.mean((gp - sparse) ** 2))
    S.require_finite(gp=gp, sparse=sparse, coef=coef)
    assert len(support) < len(x) / 4
    assert float(rmse) < .09
    x, y, grid, gp, sparse, support = S.host(x, y, grid, gp, sparse, support)
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.55))
    axes[0].plot(grid, gp, color=S.POS, label="dense GP mean")
    axes[0].plot(grid, sparse, color=S.ACCENT, ls="--", label="sparse RVM-like fit")
    axes[0].scatter(x, y, s=9, color=S.MUTED, alpha=.55)
    axes[0].scatter(x[support], y[support], s=30, color=S.ACCENT, edgecolor=S.INK, label="7 relevance sites")
    axes[0].set(xlabel=r"$x$", ylabel="prediction")
    axes[0].legend(
        fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 1.015),
        borderaxespad=0, ncol=3, handlelength=1.5, columnspacing=0.8,
    )
    S.bars(
        axes[1], np.arange(2), [len(x), len(support)],
        labels=["GP", "sparse"], highlight=[1], value_labels=True,
    )
    axes[1].set(ylabel="active kernel centres")
    S.panel_label(axes[0], "a")
    S.panel_label(axes[1], "b")
    for ax in axes: S.finish(ax)
    fig.subplots_adjust(wspace=.3)
    return S.save(fig, "gp-rvm-sparsity-comparison")


if __name__ == "__main__":
    print(main())
