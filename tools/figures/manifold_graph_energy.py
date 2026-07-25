"""manifold-graph-energy: harmonic extension propagates sparse labels along a graph."""
from __future__ import annotations

import matplotlib.pyplot as plt
import jax
from jax import config, lax
import jax.numpy as jnp
import numpy as np

import _style as S

config.update("jax_enable_x64", True)
S.apply_style()


def main() -> str:
    t = jnp.linspace(0.08, jnp.pi - 0.08, 34, dtype=jnp.float64)
    upper = jnp.stack((jnp.cos(t), jnp.sin(t)), axis=1)
    lower = jnp.stack((1.0 - jnp.cos(t), 0.48 - jnp.sin(t)), axis=1)
    points = jnp.concatenate((upper, lower), axis=0)
    n = points.shape[0]
    distances = jnp.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    neighbours = jnp.argsort(distances, axis=1)[:, 1:5]

    def add_neighbours(weights: jax.Array, row: jax.Array) -> tuple[jax.Array, None]:
        values = jnp.exp(-jnp.square(distances[row, neighbours[row]] / 0.36))
        return weights.at[row, neighbours[row]].set(values), None

    weights, _ = lax.scan(add_neighbours, jnp.zeros((n, n), dtype=jnp.float64), jnp.arange(n))
    weights = jnp.maximum(weights, weights.T)
    laplacian = jnp.diag(jnp.sum(weights, axis=1)) - weights
    labelled = jnp.array([4, 29, 38, 63], dtype=jnp.int32)
    labels = jnp.array([1.0, 1.0, -1.0, -1.0], dtype=jnp.float64)
    labelled_mask = jnp.zeros(n, dtype=bool).at[labelled].set(True)
    unlabelled = jnp.flatnonzero(~labelled_mask, size=n - labelled.size)
    luu = laplacian[jnp.ix_(unlabelled, unlabelled)]
    lul = laplacian[jnp.ix_(unlabelled, labelled)]
    ridge = 1e-9 * jnp.eye(unlabelled.size, dtype=jnp.float64)
    scores = jnp.zeros(n, dtype=jnp.float64).at[labelled].set(labels)
    scores = scores.at[unlabelled].set(jnp.linalg.solve(luu + ridge, -(lul @ labels)))

    assert bool(jnp.all(jnp.isfinite(points)))
    assert bool(jnp.all(jnp.isfinite(scores)))
    assert bool(jnp.allclose(weights, weights.T, atol=1e-12, rtol=0.0))
    assert float(jnp.max(jnp.abs(laplacian @ jnp.ones(n)))) < 1e-12
    assert float(jnp.linalg.norm((luu + ridge) @ scores[unlabelled] + lul @ labels)) < 1e-8
    baseline = jnp.sign(jnp.arange(n, dtype=jnp.float64) - n / 2)
    energy = scores @ laplacian @ scores
    assert float(energy) < float(baseline @ laplacian @ baseline)
    assert float(scores[10]) > 0.8 and float(scores[52]) < -0.8

    X, W, scores_np = map(np.asarray, (points, weights, scores))
    labelled_np = np.asarray(labelled)

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.7), sharex=True, sharey=True)
    for ax, show_scores, title in zip(
        axes, (False, True), ("Four labels on the data graph", "Minimum-energy extension")
    ):
        for i in range(n):
            for j in np.flatnonzero(W[i] > 0):
                if j > i:
                    ax.plot(*X[[i, j]].T, color=S.RULE, lw=0.45, zorder=0)
        if show_scores:
            colors = np.where(scores_np >= 0, S.POS, S.NEG)
            sizes = 18 + 18 * np.abs(scores_np)
            ax.scatter(X[:, 0], X[:, 1], c=colors, s=sizes, edgecolor=S.PAPER, linewidth=0.35)
        else:
            ax.scatter(X[:, 0], X[:, 1], s=16, facecolor=S.PAPER, edgecolor=S.MUTED, linewidth=0.65)
        ax.scatter(X[labelled_np[:2], 0], X[labelled_np[:2], 1], s=58, color=S.POS, marker="s", edgecolor=S.INK)
        ax.scatter(X[labelled_np[2:], 0], X[labelled_np[2:], 1], s=58, color=S.NEG, marker="s", edgecolor=S.INK)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(wspace=0.05)
    return S.save(fig, "manifold-graph-energy")


if __name__ == "__main__":
    print(main())
