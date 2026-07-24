"""operator-valued-field: output coupling turns one scalar similarity into a vector response."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import _style as S

S.apply_style()


def main() -> str:
    x = np.linspace(-2.2, 2.2, 240)
    k = np.exp(-0.5 * x**2)
    couplings = (np.eye(2), np.array([[1.0, 0.72], [0.72, 1.0]]))
    titles = ("Independent outputs", "Coupled outputs")
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.6), sharex=True, sharey=True)
    for ax, B, title in zip(axes, couplings, titles):
        response = B @ np.vstack((k, np.zeros_like(k)))
        ax.plot(x, response[0], color=S.POS, lw=1.8, label=r"output 1")
        ax.plot(x, response[1], color=S.ACCENT, lw=1.6, ls="--", label=r"output 2")
        ax.scatter([0], [1], color=S.INK, s=24, marker="s", zorder=3)
        ax.set_title(title)
        ax.set_xlabel("input displacement")
        S.finish(ax)
    axes[0].set_ylabel("response to a unit impulse in output 1")
    axes[1].legend(frameon=False, loc="upper right")
    assert np.allclose(couplings[0] @ np.array([1.0, 0.0]), [1.0, 0.0])
    assert np.all(np.linalg.eigvalsh(couplings[1]) > 0)
    assert np.isclose((couplings[1] @ np.array([1.0, 0.0]))[1], 0.72)
    fig.subplots_adjust(wspace=0.16)
    return S.save(fig, "operator-valued-field")


if __name__ == "__main__":
    print(main())
