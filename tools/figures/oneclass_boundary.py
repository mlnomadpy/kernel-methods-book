"""oneclass-boundary: nu changes the accepted empirical mass."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import _style as S

S.apply_style()


def kde(query: np.ndarray, points: np.ndarray, bandwidth: float) -> np.ndarray:
    delta = query[:, None, :] - points[None, :, :]
    return np.exp(-np.sum(delta * delta, axis=2) / (2.0 * bandwidth**2)).mean(axis=1)


def main() -> str:
    generator = S.rng(17)
    left = generator.normal((-0.85, 0.0), (0.42, 0.5), size=(44, 2))
    right = generator.normal((0.85, 0.1), (0.38, 0.43), size=(40, 2))
    outliers = np.array([[2.1, 1.6], [-2.0, 1.45], [0.1, -1.75], [2.15, -1.25]])
    points = np.vstack([left, right, outliers])
    bandwidth = 0.55
    train_score = kde(points, points, bandwidth)
    gx = np.linspace(-2.65, 2.7, 180)
    gy = np.linspace(-2.15, 2.2, 160)
    xx, yy = np.meshgrid(gx, gy)
    query = np.column_stack([xx.ravel(), yy.ravel()])
    density = kde(query, points, bandwidth).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.65), sharex=True, sharey=True)
    for ax, nu in zip(axes, (0.10, 0.30)):
        threshold = float(np.quantile(train_score, nu))
        accepted = train_score >= threshold
        ax.contourf(xx, yy, density, levels=[threshold, density.max()],
                    colors=[S.RULE], alpha=0.45)
        ax.contour(xx, yy, density, levels=[threshold], colors=[S.ACCENT], linewidths=1.6)
        ax.scatter(points[accepted, 0], points[accepted, 1], s=12, color=S.POS,
                   edgecolor=S.INK, linewidth=0.25)
        ax.scatter(points[~accepted, 0], points[~accepted, 1], s=25, marker="x",
                   color=S.NEG, linewidth=1.1)
        fraction = accepted.mean()
        ax.set_title(rf"$\nu={nu:.1f}$: accepted {fraction:.0%}")
        ax.set_xlabel(r"$x_1$")
        S.finish(ax)
    axes[0].set_ylabel(r"$x_2$")
    axes[0].text(0.04, 0.05, "looser level set", transform=axes[0].transAxes,
                 color=S.MUTED, fontsize=8)
    axes[1].text(0.04, 0.05, "more mass may be rejected", transform=axes[1].transAxes,
                 color=S.ACCENT, fontsize=8)
    fig.subplots_adjust(wspace=0.08)
    assert np.isfinite(density).all()
    return S.save(fig, "oneclass-boundary")


if __name__ == "__main__":
    print(main())
