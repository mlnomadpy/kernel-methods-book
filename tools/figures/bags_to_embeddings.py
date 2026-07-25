"""bags-to-embeddings: two geometries in distribution regression.

Three fixed bags are transformed into empirical Gaussian-kernel mean curves.
Their exact empirical MMD distances then define a Gaussian-on-MMD Gram matrix.
The plate shows the complete stage-one map from points, to one RKHS object per
bag, to the similarities consumed by stage-two kernel ridge regression.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import _style as S

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

S.apply_style()
jax.config.update("jax_enable_x64", True)

SIGMA = 0.65
GAMMA = 0.75
BAGS = (
    jnp.array([-0.55, -0.25, 0.05, 0.20, 0.48]),
    jnp.array([0.55, 0.78, 1.05, 1.24, 1.47]),
    jnp.array([1.55, 1.88, 2.08, 2.35, 2.60]),
)
COLORS = (S.POS, S.ACCENT, S.GOOD)
MARKERS = ("o", "s", "^")
LINESTYLES = ("-", "--", "-.")


def base_kernel(a: jax.Array, b: jax.Array) -> jax.Array:
    delta = a[:, None] - b[None, :]
    return jnp.exp(-(delta**2) / (2.0 * SIGMA**2))


def inner_products(bags: jax.Array) -> jax.Array:
    pair = lambda a, b: base_kernel(a, b).mean()
    return jax.vmap(lambda a: jax.vmap(lambda b: pair(a, b))(bags))(bags)


def main() -> str:
    grid = jnp.linspace(-1.2, 3.2, 320)
    bags = jnp.stack(BAGS)
    means = jax.vmap(lambda bag: base_kernel(grid, bag).mean(axis=1))(bags)
    inner = inner_products(bags)
    squared_mmd = (
        jnp.diag(inner)[:, None] + jnp.diag(inner)[None, :] - 2.0 * inner
    )
    squared_mmd = jnp.maximum(squared_mmd, 0.0)
    gram = jnp.exp(-squared_mmd / (2.0 * GAMMA**2))
    eigenvalues = jnp.linalg.eigvalsh(gram)

    assert bool(jnp.all(jnp.isfinite(means))) and bool(jnp.all(jnp.isfinite(gram)))
    assert bool(jnp.allclose(inner, inner.T, atol=1e-12))
    assert bool(jnp.allclose(gram, gram.T, atol=1e-12))
    assert bool(jnp.allclose(jnp.diag(gram), 1.0, atol=1e-12))
    assert float(eigenvalues.min()) > -1e-12
    assert float(gram[0, 1]) > float(gram[0, 2])
    grid, means = map(np.asarray, (grid, means))

    fig = plt.figure(figsize=(7.2, 2.75))
    layout = GridSpec(1, 3, figure=fig, width_ratios=[0.9, 1.45, 0.82], wspace=0.42)
    ax_bags = fig.add_subplot(layout[0, 0])
    ax_embed = fig.add_subplot(layout[0, 1])
    ax_gram = fig.add_subplot(layout[0, 2])

    for index, (bag, color, marker) in enumerate(zip(BAGS, COLORS, MARKERS), start=1):
        bag = np.asarray(bag)
        y = np.full_like(bag, 4 - index, dtype=float)
        ax_bags.scatter(
            bag,
            y,
            s=35,
            color=color,
            marker=marker,
            edgecolor=S.PAPER,
            linewidth=0.8,
            zorder=3,
        )
        ax_bags.hlines(y[0], bag.min(), bag.max(), color=color, linewidth=0.8, alpha=0.5)
    ax_bags.set_xlim(-1.2, 3.2)
    ax_bags.set_ylim(0.5, 3.5)
    ax_bags.set_yticks([1, 2, 3], labels=["bag 3", "bag 2", "bag 1"])
    ax_bags.set_xlabel("sample value")
    ax_bags.set_title("1  Samples inside bags", loc="left")
    S.finish(ax_bags)

    for index, (curve, color, style) in enumerate(
        zip(means, COLORS, LINESTYLES),
        start=1,
    ):
        ax_embed.plot(
            grid,
            curve,
            color=color,
            linestyle=style,
            linewidth=2.0,
            label=rf"$\widehat\mu_{index}$",
        )
    ax_embed.set_xlim(-1.2, 3.2)
    ax_embed.set_ylim(0.0, 1.05)
    ax_embed.set_xlabel("evaluation point $t$")
    ax_embed.set_ylabel(r"$\widehat\mu_i(t)$")
    ax_embed.set_title("2  One mean embedding per bag", loc="left")
    ax_embed.legend(loc="upper right", frameon=False, ncol=1)
    S.finish(ax_embed)

    gram_host = np.asarray(gram)
    image = ax_gram.imshow(gram_host, cmap=S.HEAT, vmin=0.0, vmax=1.0, interpolation="nearest")
    del image
    for row in range(gram.shape[0]):
        for col in range(gram.shape[1]):
            value = gram_host[row, col]
            text_color = S.PAPER if value > 0.72 else S.INK
            ax_gram.text(col, row, f"{value:.2f}", ha="center", va="center",
                         fontsize=8, color=text_color)
    ax_gram.set_xticks(range(3), labels=["1", "2", "3"])
    ax_gram.set_yticks(range(3), labels=["1", "2", "3"])
    ax_gram.set_xlabel("bag $j$")
    ax_gram.set_ylabel("bag $i$")
    ax_gram.set_title("3  Stage-two Gram", loc="left")
    for spine in ax_gram.spines.values():
        spine.set_color(S.RULE)

    fig.suptitle(
        "Bag samples $\\longrightarrow$ mean embeddings $\\longrightarrow$ regression geometry",
        x=0.5,
        y=1.02,
        fontsize=10,
        color=S.INK,
    )
    return S.save(fig, "bags-to-embeddings")


if __name__ == "__main__":
    print(main())
