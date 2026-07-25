"""epsilon-tube: only residuals outside the tolerance tube pay loss."""
from __future__ import annotations

import _style as S
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

S.apply_style()


def main() -> str:
    x_jax = jnp.array([0.3, 0.9, 1.5, 2.1, 2.7, 3.3, 3.9, 4.5])
    residual_jax = jnp.array([0.08, -0.22, 0.48, -0.05, -0.55, 0.18, 0.67, -0.12])
    prediction_jax = 0.34 * x_jax + 0.45
    epsilon = 0.25
    loss_jax = jnp.maximum(jnp.abs(residual_jax) - epsilon, 0.0)
    outside_jax = loss_jax > 0
    grid_jax = jnp.linspace(0.0, 4.8, 200)
    fit_jax = 0.34 * grid_jax + 0.45
    assert bool(jnp.all(jnp.isfinite(jnp.concatenate((prediction_jax, loss_jax, fit_jax)))))
    assert bool(jnp.all(loss_jax[~outside_jax] == 0.0))
    assert bool(jnp.allclose(loss_jax[outside_jax], jnp.abs(residual_jax[outside_jax]) - epsilon))

    # Explicit device-to-host boundary: NumPy is used only by Matplotlib.
    x, residual, prediction, loss, outside, grid, fit = map(
        np.asarray, (x_jax, residual_jax, prediction_jax, loss_jax, outside_jax, grid_jax, fit_jax)
    )
    y = prediction + residual

    fig, axes = plt.subplots(
        2, 1, figsize=(5.2, 3.45), sharex=True,
        gridspec_kw={"height_ratios": [2.35, 0.85], "hspace": 0.12},
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
    ax.text(
        0.02, 0.96, rf"tolerance $\varepsilon={epsilon:.2f}$",
        transform=ax.transAxes, ha="left", va="top", color=S.MUTED, fontsize=7.7,
    )
    S.legend_above(ax, columns=3)
    S.finish(ax)

    bx = axes[1]
    S.lollipops(bx, x, loss, active=outside)
    bx.axhline(0, color=S.INK, lw=0.55)
    bx.set_ylabel(r"$L_\varepsilon$")
    bx.set_xlabel("input")
    bx.text(
        0.02, 0.78, "zero inside the tube",
        transform=bx.transAxes, color=S.MUTED, fontsize=7.5,
    )
    S.finish(bx)
    return S.save(fig, "epsilon-tube")


if __name__ == "__main__":
    print(main())
