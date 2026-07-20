"""mmd-twosample: maximum mean discrepancy between two Gaussian clouds.

Reproduces ``WIDGETS["mmd-twosample"]`` in ``public/assets/viz.js`` at its default
state: 60 samples each from P ~ N((-1.2, 0), 0.7^2 I) (blue) and
Q ~ N((1.2, 0), 0.7^2 I) (red), Gaussian kernel at bandwidth 1.2. The biased
V-statistic estimate (the widget's ``mmd2``, whose sums run over all index pairs
including i = j) is

    MMD^2 = mean k(X, X') + mean k(Y, Y') - 2 mean k(X, Y),

computed here in JAX. We scatter both clouds, mark the two means, and annotate
the computed MMD^2 and MMD.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()

MU_P = np.array([-1.2, 0.0])
MU_Q = np.array([1.2, 0.0])
STD = 0.7
BW = 1.2
N = 60


def sample():
    """Widget default draw: 60 points per cloud, std 0.7 about each mean."""
    g = S.rng(0)
    SP = MU_P + g.standard_normal((N, 2)) * STD
    SQ = MU_Q + g.standard_normal((N, 2)) * STD
    return jnp.asarray(SP), jnp.asarray(SQ)


def gram(A: jnp.ndarray, B: jnp.ndarray, bw: float) -> jnp.ndarray:
    d2 = jnp.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
    return jnp.exp(-d2 / (2.0 * bw * bw))


def mmd2(SP: jnp.ndarray, SQ: jnp.ndarray, bw: float) -> float:
    # Biased V-statistic: means over ALL pairs, diagonal included (matches JS).
    pp = jnp.mean(gram(SP, SP, bw))
    qq = jnp.mean(gram(SQ, SQ, bw))
    pq = jnp.mean(gram(SP, SQ, bw))
    return float(pp + qq - 2.0 * pq)


def main() -> str:
    SP, SQ = sample()
    m2 = mmd2(SP, SQ, BW)
    m = float(np.sqrt(max(0.0, m2)))
    SP, SQ = np.asarray(SP), np.asarray(SQ)

    fig, ax = S.new_axes(5.4, 3.4)
    ax.scatter(SP[:, 0], SP[:, 1], s=16, color=S.POS, alpha=0.75,
               edgecolor="none", zorder=2, label="$P$")
    ax.scatter(SQ[:, 0], SQ[:, 1], s=16, color=S.NEG, alpha=0.75,
               edgecolor="none", zorder=2, label="$Q$")
    # Cloud means.
    ax.scatter(*MU_P, s=90, color=S.POS, edgecolor=S.PAPER,
               linewidth=1.4, zorder=4, marker="D")
    ax.scatter(*MU_Q, s=90, color=S.NEG, edgecolor=S.PAPER,
               linewidth=1.4, zorder=4, marker="D")

    ax.set_xlim(-4, 4); ax.set_ylim(-3, 3)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("Maximum mean discrepancy, Gaussian kernel (bandwidth $1.2$)",
                 color=S.INK)
    ax.annotate(rf"$\mathrm{{MMD}}^2 = {m2:.3f}$" + "\n" + rf"$\mathrm{{MMD}} = {m:.3f}$",
                xy=(0.5, 0.04), xycoords="axes fraction", ha="center", va="bottom",
                color=S.INK, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc=S.PAPER, ec=S.RULE, lw=0.8))
    ax.legend(loc="upper right", frameon=False)
    S.finish(ax)
    return S.save(fig, "mmd-twosample")


if __name__ == "__main__":
    print(main())
