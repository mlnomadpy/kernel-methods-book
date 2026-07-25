"""influence-concentration-curve: kernel locality controls how many cases explain a prediction."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()


def main() -> str:
    x = jnp.linspace(-2.5, 2.5, 90)
    y = jnp.sin(1.7 * x) + 0.2 * jnp.cos(4.2 * x)
    query = 0.35
    ridge = 0.04
    lengthscales = jnp.array([0.22, 0.55, 1.2])

    def curve(ell):
        gram = jnp.exp(-0.5 * ((x[:, None] - x[None, :]) / ell) ** 2)
        system = gram + ridge * jnp.eye(x.size)
        chol = jnp.linalg.cholesky(system)
        kq = jnp.exp(-0.5 * ((query - x) / ell) ** 2)
        weights = jax.scipy.linalg.cho_solve((chol, True), kq)
        contributions = jnp.abs(weights * y)
        ordered = jnp.sort(contributions)[::-1]
        cumulative = jnp.cumsum(ordered) / jnp.sum(ordered)
        residual = jnp.linalg.norm(system @ weights - kq)
        return cumulative, residual

    cumulative, residuals = jax.vmap(curve)(lengthscales)
    assert bool(jnp.all(jnp.isfinite(cumulative)))
    assert float(residuals.max()) < 1e-9
    assert bool(jnp.allclose(cumulative[:, -1], 1.0, atol=1e-12))
    assert bool(jnp.all(cumulative[:, 1:] >= cumulative[:, :-1]))
    count = jnp.arange(1, x.size + 1)
    count_h, curves, ell_h = S.host(count, cumulative, lengthscales)

    fig, ax = S.new_axes(5.2, 2.9)
    for curve_h, ell, color in zip(curves, ell_h, (S.ACCENT, S.POS, S.VIOLET)):
        ax.plot(count_h, curve_h, color=color, label=rf"$\ell={ell:.2g}$")
    ax.axhline(0.9, color=S.INK, ls="--", lw=1)
    ax.set(title="How many training cases explain one prediction?",
           xlabel="largest absolute contributions retained", ylabel="cumulative influence mass",
           xlim=(1, 55), ylim=(0, 1.02))
    ax.legend(title="kernel scale", fontsize=7)
    S.finish(ax)
    return S.save(fig, "influence-concentration-curve")


if __name__ == "__main__":
    print(main())
