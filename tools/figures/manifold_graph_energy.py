"""manifold-graph-energy: harmonic extension propagates sparse labels along a graph."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import _style as S

S.apply_style()


def main() -> str:
    t = np.linspace(0.08, np.pi - 0.08, 34)
    upper = np.column_stack((np.cos(t), np.sin(t)))
    lower = np.column_stack((1.0 - np.cos(t), 0.48 - np.sin(t)))
    X = np.vstack((upper, lower))
    n = X.shape[0]
    distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    order = np.argsort(distances, axis=1)[:, 1:5]
    W = np.zeros((n, n))
    for i in range(n):
        W[i, order[i]] = np.exp(-(distances[i, order[i]] / 0.36) ** 2)
    W = np.maximum(W, W.T)
    L = np.diag(W.sum(axis=1)) - W
    labelled = np.array([4, 29, 38, 63])
    y = np.array([1.0, 1.0, -1.0, -1.0])
    unlabelled = np.setdiff1d(np.arange(n), labelled)
    scores = np.empty(n)
    scores[labelled] = y
    scores[unlabelled] = np.linalg.solve(
        L[np.ix_(unlabelled, unlabelled)] + 1e-9 * np.eye(unlabelled.size),
        -L[np.ix_(unlabelled, labelled)] @ y,
    )

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.7), sharex=True, sharey=True)
    for ax, show_scores, title in zip(
        axes, (False, True), ("Four labels on the data graph", "Minimum-energy extension")
    ):
        for i in range(n):
            for j in np.flatnonzero(W[i] > 0):
                if j > i:
                    ax.plot(*X[[i, j]].T, color=S.RULE, lw=0.45, zorder=0)
        if show_scores:
            colors = np.where(scores >= 0, S.POS, S.NEG)
            sizes = 18 + 18 * np.abs(scores)
            ax.scatter(X[:, 0], X[:, 1], c=colors, s=sizes, edgecolor=S.PAPER, linewidth=0.35)
        else:
            ax.scatter(X[:, 0], X[:, 1], s=16, facecolor=S.PAPER, edgecolor=S.MUTED, linewidth=0.65)
        ax.scatter(X[labelled[:2], 0], X[labelled[:2], 1], s=58, color=S.POS, marker="s", edgecolor=S.INK)
        ax.scatter(X[labelled[2:], 0], X[labelled[2:], 1], s=58, color=S.NEG, marker="s", edgecolor=S.INK)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    energy = float(scores @ L @ scores)
    assert energy < float(np.sign(np.arange(n) - n / 2) @ L @ np.sign(np.arange(n) - n / 2))
    assert scores[10] > 0.8 and scores[52] < -0.8
    fig.subplots_adjust(wspace=0.05)
    return S.save(fig, "manifold-graph-energy")


if __name__ == "__main__":
    print(main())
