"""hilbert-vs-variation: quadratic and atomic norms select different coefficient geometry."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import _style as S

S.apply_style()


def main() -> str:
    x = np.linspace(-1.5, 1.5, 180)
    centers = np.linspace(-1.1, 1.1, 15)
    Phi = np.exp(-0.5 * ((x[:, None] - centers[None, :]) / 0.34) ** 2)
    target = 0.95 * np.exp(-0.5 * ((x + 0.62) / 0.34) ** 2) + 0.72 * np.exp(-0.5 * ((x - 0.55) / 0.34) ** 2)
    ridge = np.linalg.solve(Phi.T @ Phi + 0.9 * np.eye(Phi.shape[1]), Phi.T @ target)
    lipschitz = np.linalg.norm(Phi, 2) ** 2
    step, penalty = 1.0 / lipschitz, 0.22
    sparse = np.zeros(Phi.shape[1])
    for _ in range(4000):
        proposal = sparse - step * (Phi.T @ (Phi @ sparse - target))
        sparse = np.sign(proposal) * np.maximum(np.abs(proposal) - step * penalty, 0.0)
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.65))
    axes[0].plot(x, target, color=S.INK, lw=1.4, label="target")
    axes[0].plot(x, Phi @ ridge, color=S.POS, lw=1.6, label="quadratic norm")
    axes[0].plot(x, Phi @ sparse, color=S.ACCENT, lw=1.6, ls="--", label="variation norm")
    axes[0].set(xlabel=r"$x$", ylabel="function value", title="Similar fits")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].stem(centers - 0.025, ridge, linefmt=S.POS, markerfmt="o", basefmt=" ", label="quadratic")
    axes[1].stem(centers + 0.025, sparse, linefmt=S.ACCENT, markerfmt="s", basefmt=" ", label="atomic")
    axes[1].set(xlabel="atom location", ylabel="coefficient", title="Different geometry")
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        S.finish(ax)
    assert np.count_nonzero(np.abs(ridge) > 1e-3) >= 10
    assert np.count_nonzero(np.abs(sparse) > 1e-3) <= 5
    assert np.mean((Phi @ sparse - target) ** 2) < 0.01
    fig.subplots_adjust(wspace=0.26)
    return S.save(fig, "hilbert-vs-variation")


if __name__ == "__main__":
    print(main())
