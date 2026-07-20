"""herding-greedy: greedy kernel herding as a quadrature rule.

Reproduces ``public/assets/viz-herding-greedy.js`` at its default state.
Target ``P = N(0,1)`` with the unit-bandwidth Gaussian kernel
``k(x,y) = exp(-(x-y)^2/2)``; the closed forms match the widget exactly:
    mean embedding  mu_P(x) = E k(x,Y) = e^{-x^2/4}/sqrt(2)
    constant        C = E k(Y,Y') = 1/sqrt(3).
Each step selects x_{n+1} = argmax_x [ mu_P(x) - (1/(n+1)) sum_i k(x, x_i) ]
on the same dense candidate grid the widget scans (600 cells on [-4, 4],
strict ``>`` so ties break to the leftmost point).

Top panel: mu_P(x) with the faint N(0,1) density for context and the herded
node locations (ticks, first few numbered). Bottom panel: the true worst-case
integration error vs number of nodes on a log axis --- uniform-weight herding
e_n^2 = C - (2/n) sum_i mu_P(x_i) + (1/n^2) sum_{ij} k(x_i,x_j), the optimally
reweighted same nodes e*^2 = C - z^T K^{-1} z (z_i = mu_P(x_i)), and the exact
Monte Carlo expectation E e_MC^2 = (1 - C)/n.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()

# Widget constants (viz-herding-greedy.js).
XMIN, XMAX, G = -4.0, 4.0, 600
NSTEPS = 18                       # ~15-20 deterministic greedy steps
C = 1.0 / jnp.sqrt(3.0)           # C = E k(Y,Y') for P = N(0,1)


def mu_P(x):                      # closed-form mean embedding
    return jnp.exp(-x * x / 4.0) / jnp.sqrt(2.0)


def kf(a, b):                     # unit-bandwidth Gaussian kernel
    d = a - b
    return jnp.exp(-d * d / 2.0)


def herd(nsteps: int):
    """Greedy herding: return node locations and the two error curves."""
    # candidate grid: cell centres, exactly as the widget builds it.
    g = jnp.arange(G)
    grid = XMIN + (XMAX - XMIN) * (g + 0.5) / G
    muG = mu_P(grid)

    ksum = jnp.zeros(G)           # running sum_i k(grid, x_i)
    nodes = []
    errU = np.zeros(nsteps)
    errW = np.zeros(nsteps)

    for n in range(nsteps):
        crit = muG - ksum / (n + 1)            # herding acquisition
        bi = int(jnp.argmax(crit))             # strict argmax -> leftmost tie
        x = grid[bi]
        nodes.append(float(x))
        ksum = ksum + kf(grid, x)

        xs = jnp.asarray(nodes)
        z = mu_P(xs)                            # z_i = mu_P(x_i)
        K = kf(xs[:, None], xs[None, :])        # Gram matrix
        # uniform weights 1/n
        m = n + 1
        eU = C - (2.0 / m) * jnp.sum(z) + jnp.sum(K) / (m * m)
        errU[n] = float(jnp.sqrt(jnp.maximum(0.0, eU)))
        # optimal weights: e*^2 = C - z^T K^{-1} z
        w = jnp.linalg.solve(K + 1e-10 * jnp.eye(m), z)
        eW = C - jnp.dot(z, w)
        errW[n] = float(jnp.sqrt(jnp.maximum(0.0, eW)))

    return np.asarray(nodes), np.asarray(grid), np.asarray(muG), errU, errW


def main() -> str:
    nodes, grid, muG, errU, errW = herd(NSTEPS)
    ns = np.arange(1, NSTEPS + 1)
    Cf = float(C)
    e_mc = np.sqrt((1.0 - Cf) / ns)             # exact Monte Carlo expectation
    dens = np.exp(-grid ** 2 / 2.0) / np.sqrt(2.0 * np.pi)

    fig = plt.figure(figsize=(5.2, 4.6))
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.0], hspace=0.42, figure=fig)

    # -- top: embedding mu_P and herded nodes -------------------------------
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(grid, dens, color=S.RULE, lw=1.1,
             label=r"$\mathcal{N}(0,1)$ density")
    ax0.plot(grid, muG, color=S.ACCENT, lw=2.0,
             label=r"$\mu_P(x)=e^{-x^2/4}/\sqrt{2}$")
    ymax = float(muG.max())
    for i, xn in enumerate(nodes):
        ax0.plot([xn, xn], [0.0, -0.045 * ymax], color=S.INK,
                 lw=1.6 if i == len(nodes) - 1 else 1.0, zorder=3)
        if i < 6:
            ax0.text(xn, -0.11 * ymax, str(i + 1), color=S.MUTED,
                     ha="center", va="top", fontsize=7)
    ax0.set_xlim(XMIN, XMAX)
    ax0.set_ylim(-0.16 * ymax, ymax * 1.12)
    ax0.set_xlabel("$x$")
    ax0.set_title("Greedy kernel herding: embedding and placed nodes",
                  color=S.INK)
    ax0.legend(loc="upper right", frameon=False)
    S.finish(ax0)

    # -- bottom: worst-case error vs number of nodes (log y) ----------------
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(ns, e_mc, color=S.MUTED, lw=1.2, ls=(0, (4, 3)),
             label=r"$\sqrt{\mathbb{E}\,e_{\mathrm{MC}}^2}=\sqrt{(1-C)/n}$")
    ax1.plot(ns, errU, color=S.POS, lw=2.0, marker="o", ms=3.2,
             label="herding, uniform weights")
    ax1.plot(ns, errW, color=S.GOOD, lw=1.8, marker="s", ms=3.0,
             label="same nodes, optimal weights")
    ax1.set_yscale("log")
    ax1.set_xlim(1, NSTEPS)
    ax1.set_xlabel("number of nodes $n$")
    ax1.set_ylabel("worst-case error $e_n$")
    ax1.set_title("Integration error vs. $n$", color=S.INK)
    ax1.legend(loc="lower left", frameon=False)
    S.finish(ax1)

    return S.save(fig, "herding-greedy")


if __name__ == "__main__":
    print(main())
