"""dkl-collapse: a learned map can collapse distances and erase GP uncertainty geometry."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import _style as S

S.apply_style()


def rbf(values: np.ndarray, length: float = 0.42) -> np.ndarray:
    d = values[:, None] - values[None, :]
    return np.exp(-0.5 * (d / length) ** 2)


def main() -> str:
    x = np.linspace(-2.0, 2.0, 32)
    learned = 0.17 * np.tanh(3.0 * x)
    K_raw, K_collapsed = rbf(x), rbf(learned)
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.55))
    for ax, K, title in zip(
        axes,
        (K_raw, K_collapsed),
        ("Geometry before feature learning", "Collapsed learned geometry"),
    ):
        image = ax.imshow(K, origin="lower", cmap=S.HEAT, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("input index")
        ax.set_ylabel("input index")
        ax.tick_params(length=0)
    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.03, label="RBF similarity")
    offdiag_raw = (K_raw.sum() - np.trace(K_raw)) / (K_raw.size - len(x))
    offdiag_collapsed = (K_collapsed.sum() - np.trace(K_collapsed)) / (K_collapsed.size - len(x))
    assert offdiag_collapsed > 0.85
    assert offdiag_collapsed > 3.0 * offdiag_raw
    fig.subplots_adjust(wspace=0.24, right=0.88)
    return S.save(fig, "dkl-collapse")


if __name__ == "__main__":
    print(main())
