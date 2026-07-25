"""pivoted-cholesky-residual: greedy pivots erase the largest diagonal residual."""
from __future__ import annotations

import _style as S
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

S.apply_style()


def main() -> str:
    x = jnp.linspace(-2.5, 2.5, 70)
    k = jnp.exp(-0.5 * ((x[:, None] - x[None, :]) / .55)**2)
    residual = k
    traces, maxima, pivots = [], [], []
    for _ in range(20):
        diagonal = jnp.diag(residual)
        pivot = int(jnp.argmax(diagonal))
        column = residual[:, pivot] / jnp.sqrt(jnp.maximum(diagonal[pivot], 1e-15))
        residual = 0.5 * ((residual - jnp.outer(column, column)) + (residual - jnp.outer(column, column)).T)
        pivots.append(pivot)
        traces.append(jnp.trace(residual))
        maxima.append(jnp.max(jnp.diag(residual)))
    traces, maxima = jnp.stack(traces), jnp.stack(maxima)
    assert bool(jnp.all(traces[1:] <= traces[:-1] + 1e-11))
    assert bool(jnp.all(maxima[1:] <= maxima[:-1] + 1e-11))
    steps, tr, ma, xx = S.host(jnp.arange(1, 21), traces / jnp.trace(k), maxima, x)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    axes[0].semilogy(steps, tr, color=S.POS, marker="o", label="relative trace residual")
    axes[0].semilogy(steps, ma, color=S.ACCENT, marker="s", label="largest diagonal residual")
    axes[0].set(xlabel="pivot count", ylabel="residual")
    axes[0].legend()
    chosen = S.host(x[jnp.array(pivots)])
    axes[1].scatter(xx, jnp.zeros_like(xx), s=10, color=S.RULE)
    axes[1].scatter(chosen, jnp.arange(1, len(chosen) + 1), c=range(len(chosen)), cmap=S.HEAT, s=25)
    axes[1].set(xlabel="input location", ylabel="selection order", title="Greedy pivot locations")
    for ax in axes: S.finish(ax)
    fig.tight_layout()
    return S.save(fig, "pivoted-cholesky-residual")


if __name__ == "__main__":
    print(main())
