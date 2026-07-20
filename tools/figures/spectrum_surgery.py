"""spectrum-surgery: an indefinite tanh similarity, repaired live.

Reproduces ``WIDGETS["spectrum-surgery"]`` in
``public/assets/viz-spectrum-surgery.js`` at its default state: the 10x10 tanh
similarity S_ij = tanh(a x_i x_j + B) on ten fixed 1-D points, steepness
a = 1.0, offset B = 0.1. The widget eigendecomposes S live (cyclic Jacobi in
the browser; jnp.linalg.eigh here) and marks the negative eigenvalues as the
Krein part. Three repairs recompose a PSD matrix K' from surgically altered
eigenvalues -- clip: max(lam, 0); flip: |lam|; shift: lam - lam_min -- and the
Frobenius readout ||S - K'||_F compares all three. Because only the spectrum
changes, ||S - K'||_F is a function of the eigenvalue changes alone:

    clip  = sqrt( sum_{lam<0} lam^2 )
    flip  = sqrt( sum_{lam<0} (2 lam)^2 ) = 2 * clip
    shift = |lam_min| * sqrt(N)

Clip changes the matrix least: it is the nearest PSD matrix in Frobenius norm.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()

# ten fixed 1-D points in two groups plus a straggler (widget default)
X = np.array([-1.9, -1.6, -1.4, -1.1, -0.05, 1.0, 1.3, 1.55, 1.8, 2.1])
N = X.size
B = 0.1   # tanh offset
A = 1.0   # default steepness a


def tanh_similarity(x: jnp.ndarray, a: float) -> jnp.ndarray:
    # S_ij = tanh(a * x_i * x_j + B) -- symmetric, indefinite in general.
    return jnp.tanh(a * x[:, None] * x[None, :] + B)


def frobenius_repairs(lam: jnp.ndarray) -> dict:
    # ||S - K'||_F from the spectral change alone (eigenvectors untouched).
    neg = jnp.minimum(lam, 0.0)                      # negative part of the spectrum
    clip = jnp.sqrt(jnp.sum(neg ** 2))               # clip: drop lam<0 -> sum lam^2
    flip = jnp.sqrt(jnp.sum((2.0 * neg) ** 2))       # flip: lam -> |lam|, delta = 2|lam|
    lmin = jnp.min(lam)
    c = jnp.where(lmin < 0, -lmin, 0.0)
    shift = jnp.sqrt(c * c * N)                       # shift: diagonal +c on all N modes
    return {"clip": float(clip), "flip": float(flip), "shift": float(shift)}


def main() -> str:
    x = jnp.asarray(X)
    Smat = tanh_similarity(x, A)
    lam, _ = jnp.linalg.eigh(Smat)                    # symmetric eigendecomposition
    lam = np.asarray(lam)[::-1]                        # sort descending, as the widget draws
    lmin = float(lam.min())
    n_neg = int(np.sum(lam < -1e-12))
    frob = frobenius_repairs(jnp.asarray(lam))

    fig = plt.figure(figsize=(5.4, 3.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.15, 1.0], wspace=0.42)

    # LEFT: sorted spectrum, positive bars (POS) vs negative Krein bars (NEG).
    ax = fig.add_subplot(gs[0])
    idx = np.arange(N)
    colors = [S.POS if v >= 0 else S.NEG for v in lam]
    ax.bar(idx, lam, width=0.72, color=colors, edgecolor=S.PAPER, linewidth=0.6, zorder=3)
    ax.axhline(0.0, color=S.RULE, lw=0.9, zorder=2)
    ax.set_xticks([])
    ax.set_xlabel("eigenvalue index (sorted)")
    ax.set_ylabel(r"$\lambda_i$")
    ax.set_title(r"tanh similarity spectrum, $S_{ij}=\tanh(a\,x_i x_j+b)$, $a=1$",
                 color=S.INK, pad=6)
    ax.annotate(f"Krein part: {n_neg} negative\n" + r"$\lambda_{\min}=$" + f"{lmin:.3f}",
                xy=(N - 1.4, lmin), xytext=(N - 4.4, lmin - 0.25 * abs(lam).max()),
                color=S.NEG, fontsize=8,
                arrowprops=dict(arrowstyle="-", color=S.NEG, lw=0.8))
    S.finish(ax)

    # RIGHT: Frobenius distance of each repair to S -- clip is smallest.
    axb = fig.add_subplot(gs[1])
    names = ["clip", "flip", "shift"]
    vals = [frob[k] for k in names]
    bcolors = [S.GOOD, S.MUTED, S.MUTED]              # clip highlighted (nearest PSD)
    axb.bar(np.arange(3), vals, width=0.66, color=bcolors,
            edgecolor=S.PAPER, linewidth=0.6, zorder=3)
    for k, v in enumerate(vals):
        axb.text(k, v + 0.02 * max(vals), f"{v:.3f}", ha="center", va="bottom",
                 fontsize=7.5, color=S.INK)
    axb.set_xticks(np.arange(3))
    axb.set_xticklabels(names)
    axb.set_ylim(0, max(vals) * 1.22)
    axb.set_ylabel(r"$\|S-K'\|_F$")
    axb.set_title("repair cost", color=S.INK, pad=6)
    S.finish(axb)

    return S.save(fig, "spectrum-surgery")


if __name__ == "__main__":
    print(main())
