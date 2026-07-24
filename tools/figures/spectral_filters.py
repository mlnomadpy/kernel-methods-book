"""spectral-filters: ridge, cutoff, and early stopping preserve different eigendirections."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import _style as S

S.apply_style()


def main() -> str:
    mu = np.logspace(-3, 1, 500)
    lam = 0.08
    eta, steps = 0.08, 28
    ridge = mu / (mu + lam)
    cutoff = (mu >= lam).astype(float)
    early = 1.0 - (1.0 - eta * mu) ** steps
    fig, ax = S.new_axes(5.5, 2.9)
    ax.semilogx(mu, ridge, color=S.POS, lw=1.8, label="ridge")
    ax.semilogx(mu, cutoff, color=S.ACCENT, lw=1.5, ls="--", label="spectral cutoff")
    ax.semilogx(mu, early, color=S.GOOD, lw=1.5, ls="-.", label=f"early stopping ({steps} steps)")
    ax.axvline(lam, color=S.RULE, lw=1.0)
    ax.text(lam * 1.08, 0.08, r"nominal scale $\lambda$", color=S.MUTED, fontsize=8)
    ax.set(xlabel="operator eigenvalue", ylabel="retained fraction", ylim=(-0.04, 1.05))
    ax.legend(frameon=False, loc="lower right")
    S.finish(ax)
    assert np.all((ridge >= 0) & (ridge <= 1))
    assert np.all((early >= 0) & (early <= 1 + 1e-12))
    assert ridge[0] < 0.02 and early[-1] > 0.999
    return S.save(fig, "spectral-filters")


if __name__ == "__main__":
    print(main())
