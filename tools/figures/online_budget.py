"""online-budget: unbudgeted support grows; maintenance caps prediction cost."""
from __future__ import annotations

import matplotlib.pyplot as plt
from jax import config, lax
import jax.numpy as jnp
import numpy as np

import _style as S

config.update("jax_enable_x64", True)
S.apply_style()


def main() -> str:
    rounds = jnp.arange(1, 121, dtype=jnp.int32)
    mistakes = jnp.logical_or(rounds % 3 == 0, rounds % 11 == 0).astype(jnp.int32)

    def update_support(total: jnp.ndarray, mistake: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        updated = total + mistake
        return updated, updated

    _, unbounded = lax.scan(update_support, jnp.array(0, dtype=jnp.int32), mistakes)
    budget = 12
    bounded = jnp.minimum(unbounded, budget)
    coefficients = jnp.array([0.08, 0.72, -0.12, 0.34, -0.58, 0.05, 0.44, -0.27], dtype=jnp.float64)
    keep = jnp.abs(coefficients) >= 0.15
    assert bool(jnp.all(jnp.isfinite(coefficients)))
    assert int(jnp.max(bounded)) == budget
    assert int(unbounded[-1]) > budget
    assert bool(jnp.all(jnp.diff(unbounded) >= 0))
    rounds, unbounded, bounded, coefficients, keep = map(np.asarray, (rounds, unbounded, bounded, coefficients, keep))

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.65))
    ax = axes[0]
    ax.step(rounds, unbounded, where="post", color=S.NEG, lw=1.7,
            label="append on every mistake")
    ax.step(rounds, bounded, where="post", color=S.POS, lw=1.9,
            label=rf"hard budget $B={budget}$")
    ax.axhline(budget, color=S.MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.set_xlabel("stream round")
    ax.set_ylabel("retained support vectors")
    ax.set_title("Prediction cost follows support-set size")
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    S.finish(ax)

    bx = axes[1]
    index = np.arange(coefficients.size)
    bx.axhline(0, color=S.INK, lw=0.7)
    bx.vlines(index[keep], 0, coefficients[keep], color=S.POS, lw=2.2,
              label="retained")
    bx.scatter(index[keep], coefficients[keep], color=S.POS, s=24,
               edgecolor=S.INK, linewidth=0.45)
    bx.vlines(index[~keep], 0, coefficients[~keep], color=S.RULE, lw=1.8,
              linestyles=(0, (3, 2)), label="removed")
    bx.scatter(index[~keep], coefficients[~keep], facecolor=S.PAPER,
               edgecolor=S.MUTED, s=24, linewidth=0.8)
    bx.set_xticks(index, [rf"$x_{i+1}$" for i in index])
    bx.set_ylabel("kernel coefficient")
    bx.set_title("A cheap rule deletes small contributions")
    bx.legend(frameon=False, loc="lower right", fontsize=7)
    S.finish(bx)
    fig.subplots_adjust(wspace=0.32)
    return S.save(fig, "online-budget")


if __name__ == "__main__":
    print(main())
