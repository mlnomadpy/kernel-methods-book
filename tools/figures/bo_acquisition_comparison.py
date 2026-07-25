"""bo-acquisition-comparison: acquisitions disagree because they price uncertainty differently."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
import matplotlib.pyplot as plt
import _style as S

S.apply_style()
jax.config.update("jax_enable_x64", True)


def main() -> str:
    grid = jnp.linspace(0, 2, 500)
    train = jnp.array([0., .35, 1.35, 2.])
    y = jnp.sin(3 * train)
    kernel = lambda a, b: jnp.exp(-.5 * ((a[:, None] - b[None, :]) / .36) ** 2)
    K = kernel(train, train) + .01 * jnp.eye(len(train))
    factor = jnp.linalg.cholesky(K)
    cross = kernel(train, grid)
    mean = cross.T @ cho_solve((factor, True), y)
    solved = cho_solve((factor, True), cross)
    std = jnp.sqrt(jnp.maximum(1 - jnp.sum(cross * solved, axis=0), 0))
    incumbent = y.max()
    z = (mean - incumbent) / jnp.maximum(std, 1e-12)
    pdf = jnp.exp(-z**2 / 2) / jnp.sqrt(2 * jnp.pi)
    cdf = .5 * (1 + jax.scipy.special.erf(z / jnp.sqrt(2.)))
    ei = jnp.maximum((mean - incumbent) * cdf + std * pdf, 0)
    ucb = mean + 1.8 * std
    pi = cdf
    S.require_finite(mean=mean, std=std, ei=ei, ucb=ucb, pi=pi)
    assert float(std.min()) >= 0
    choices = jnp.array([jnp.argmax(ucb), jnp.argmax(ei), jnp.argmax(pi)])
    assert len(set(map(int, S.host(choices)))) >= 2
    grid, mean, std, ei, ucb, pi, choices = S.host(grid, mean, std, ei, ucb, pi, choices)
    fig, axes = plt.subplots(2, 1, figsize=(5.7, 3.9), sharex=True)
    axes[0].fill_between(grid, mean - 2 * std, mean + 2 * std, color=S.POS, alpha=.16)
    axes[0].plot(grid, mean, color=S.POS, label="posterior mean")
    axes[0].scatter(train, y, color=S.INK, zorder=3)
    axes[0].set(ylabel="latent value", title="One posterior, three next-query decisions")
    curves = [(ucb - ucb.min()) / (ucb.max() - ucb.min()), ei / ei.max(), pi]
    for curve, color, label, idx in zip(curves, (S.POS, S.ACCENT, S.VIOLET), ("UCB", "EI", "PI"), choices):
        axes[1].plot(grid, curve, color=color, label=label)
        axes[1].scatter(grid[idx], curve[idx], color=color, edgecolor=S.INK, zorder=4)
    axes[1].set(xlabel=r"$x$", ylabel="normalized score")
    axes[1].legend(ncol=3)
    for ax in axes: S.finish(ax)
    fig.subplots_adjust(hspace=.12)
    return S.save(fig, "bo-acquisition-comparison")


if __name__ == "__main__":
    print(main())
