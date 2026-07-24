"""rollout-error: a small one-step bias compounds after model outputs become model inputs."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import _style as S

S.apply_style()


def main() -> str:
    horizon = np.arange(0, 41)
    truth = 1.045 ** horizon
    learned = 1.035 ** horizon
    one_step = abs(1.045 - 1.035)
    error = np.abs(truth - learned)
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.55))
    axes[0].plot(horizon, truth, color=S.INK, lw=1.8, label="true dynamics")
    axes[0].plot(horizon, learned, color=S.ACCENT, lw=1.7, ls="--", label="learned model")
    axes[0].set(xlabel="rollout step", ylabel="state", title="Repeated prediction")
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].plot(horizon, error, color=S.NEG, lw=1.8)
    axes[1].axhline(one_step, color=S.RULE, lw=1.0, ls=":")
    axes[1].annotate("one-step error", (22, one_step), (16, 0.16), arrowprops={"arrowstyle": "->", "color": S.MUTED}, fontsize=8)
    axes[1].set(xlabel="rollout step", ylabel="absolute state error", title="Compounding bias")
    for ax in axes:
        S.finish(ax)
    assert np.isclose(error[1], one_step)
    assert error[-1] > 15 * one_step
    fig.subplots_adjust(wspace=0.26)
    return S.save(fig, "rollout-error")


if __name__ == "__main__":
    print(main())
