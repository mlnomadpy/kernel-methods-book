"""spline-decomposition: an unpenalized trend and a penalized bend play different roles."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import _style as S

S.apply_style()


def main() -> str:
    x = np.linspace(0.0, 1.0, 400)
    null = 0.35 + 0.95 * x
    penalized = 0.23 * np.sin(2 * np.pi * x) - 0.08 * np.sin(4 * np.pi * x)
    total = null + penalized
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.25), sharex=True, sharey=True)
    specs = (
        (null, "null-space trend", S.POS, "-"),
        (penalized, "penalized bend", S.ACCENT, "--"),
        (total, "spline fit", S.INK, "-"),
    )
    for ax, (curve, title, color, style) in zip(axes, specs):
        ax.axhline(0, color=S.RULE, lw=0.7)
        ax.plot(x, curve, color=color, lw=1.8, ls=style)
        ax.set_title(title)
        ax.set_xlabel(r"$x$")
        S.finish(ax)
    axes[0].set_ylabel("function value")
    axes[2].plot(x, null, color=S.POS, lw=1.0, alpha=0.65)
    assert np.max(np.abs(total - null - penalized)) < 1e-14
    second_difference = np.diff(null, n=2)
    assert np.max(np.abs(second_difference)) < 1e-12
    fig.subplots_adjust(wspace=0.12)
    return S.save(fig, "spline-decomposition")


if __name__ == "__main__":
    print(main())
