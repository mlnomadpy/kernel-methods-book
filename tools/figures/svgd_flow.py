"""svgd-flow: Stein variational gradient descent to a two-mode target.

Reproduces ``public/assets/viz-svgd-flow.js`` at its default state.  Eighty
particles start in a clump at ``x = -5`` (init draw ``x_i = -5 + 0.3 g``,
``g`` standard normal) and follow the exact SVGD update of the chapter,

    x_i <- x_i + eps * (1/n) sum_j [ k(x_j,x_i) s_p(x_j) + d/dx_j k(x_j,x_i) ],

toward the two-mode target ``p = 0.5 N(-2, 0.6^2) + 0.5 N(2, 0.8^2)`` whose
analytic score ``s_p = (log p)'`` is evaluated in log space.  The kernel is the
RBF ``k(a,b) = exp(-(a-b)^2 / (2 h^2))``, so ``d/dx_j k(x_j,x_i) =
((x_i-x_j)/h^2) k``.  The default bandwidth is the median heuristic (median
pairwise distance, recomputed every 20 steps).  The default step is
``eps = 10^(-0.7)``.  The readout's empirical KSD^2 is the V-statistic
``(1/n^2) sum_ij u_p(x_i,x_j)`` with the Stein kernel of the same RBF,

    u_p(x,y) = k [ s(x)s(y) + ((x-y)/h^2)(s(x)-s(y)) + 1/h^2 - (x-y)^2/h^4 ].

All maths is in JAX (float64); the only randomness is the one-time init draw,
made deterministic here with ``S.rng(0)``.  We run the deterministic flow to a
converged state where both modes are populated, then plot the target density,
the converged particle histogram / rug, and the KSD^2 decay.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()

# ---- target p(x) = 0.5 N(-2, 0.6^2) + 0.5 N(2, 0.8^2) ----------------------
M1, S1, M2, S2 = -2.0, 0.6, 2.0, 0.8
SQRT2PI = np.sqrt(2.0 * np.pi)

N = 80                       # particles
EPS = 10.0 ** (-0.7)         # default control value eps = -0.7
XR = (-6.8, 5.2)             # widget's plot range
NB = 44                      # widget's histogram bins
N_STEPS = 20000              # run to the converged (balanced) state
MEDIAN_EVERY = 20            # widget recomputes h every 20 steps


def dens(v):
    """Target density p(v)."""
    z1 = (v - M1) / S1
    z2 = (v - M2) / S2
    return 0.5 * (jnp.exp(-0.5 * z1 * z1) / (S1 * SQRT2PI)
                  + jnp.exp(-0.5 * z2 * z2) / (S2 * SQRT2PI))


def score(v):
    """Analytic score s_p(v) = (log p)'(v), responsibilities in log space."""
    z1 = (v - M1) / S1
    z2 = (v - M2) / S2
    e1 = -0.5 * z1 * z1 - jnp.log(S1)
    e2 = -0.5 * z2 * z2 - jnp.log(S2)
    m = jnp.maximum(e1, e2)
    w1 = jnp.exp(e1 - m)
    w2 = jnp.exp(e2 - m)
    r1 = w1 / (w1 + w2)
    return -(r1 * z1 / S1 + (1.0 - r1) * z2 / S2)


def median_h(x):
    """Median-heuristic bandwidth: median of the pairwise distances |x_i-x_j|."""
    d = jnp.abs(x[:, None] - x[None, :])
    iu = jnp.triu_indices(N, k=1)              # the N(N-1)/2 upper-triangle pairs
    med = jnp.median(d[iu])
    return jnp.maximum(med, 1e-4)


@jax.jit
def svgd_step(x, h):
    """One exact SVGD step; h held fixed within the step (as in the widget)."""
    ih2 = 1.0 / (h * h)
    sc = score(x)                              # (N,)
    d = x[:, None] - x[None, :]                # d[i,j] = x_i - x_j
    kv = jnp.exp(-0.5 * d * d * ih2)
    # phi_i = (1/N) sum_j k(x_j,x_i) [ s_p(x_j) + (x_i-x_j)/h^2 ]
    phi = jnp.mean(kv * (sc[None, :] + d * ih2), axis=1)
    x_new = jnp.clip(x + EPS * phi, -12.0, 12.0)
    return x_new


@jax.jit
def svgd_block(x, h):
    """MEDIAN_EVERY exact SVGD steps at fixed bandwidth (h fixed within block)."""
    def body(_, xc):
        return svgd_step(xc, h)
    return jax.lax.fori_loop(0, MEDIAN_EVERY, body, x)


@jax.jit
def ksd2(x, h):
    """Empirical KSD^2, the V-statistic (1/N^2) sum_ij u_p(x_i,x_j)."""
    ih2 = 1.0 / (h * h)
    sc = score(x)
    d = x[:, None] - x[None, :]
    kv = jnp.exp(-0.5 * d * d * ih2)
    u = kv * (sc[:, None] * sc[None, :]
              + d * ih2 * (sc[:, None] - sc[None, :])
              + ih2 - d * d * ih2 * ih2)
    return jnp.maximum(0.0, jnp.sum(u) / (N * N))


def run_flow():
    """Deterministic SVGD flow from the clump at x=-5 to a converged state.

    Bandwidth is refreshed by the median heuristic every ``MEDIAN_EVERY`` steps
    and held fixed within each block, exactly as the widget does; the KSD^2 is
    recorded at every block boundary to trace its decay.
    """
    g = S.rng(0)
    x = jnp.asarray(-5.0 + 0.3 * g.standard_normal(N))   # one-time init draw
    h = median_h(x)                                      # updateH() at init
    hist_k = [(0, float(ksd2(x, h)))]
    for blk in range(N_STEPS // MEDIAN_EVERY):
        h = median_h(x)                                  # refresh every 20 steps
        x = svgd_block(x, h)
        hist_k.append(((blk + 1) * MEDIAN_EVERY, float(ksd2(x, h))))
    return np.asarray(x), float(h), np.asarray(hist_k)


def main() -> str:
    x, h, hk = run_flow()

    grid = np.linspace(XR[0], XR[1], 400)
    p = np.asarray(dens(jnp.asarray(grid)))

    # empirical histogram density of the converged particles (widget's bins)
    edges = np.linspace(XR[0], XR[1], NB + 1)
    counts, _ = np.histogram(x, bins=edges)
    w_bin = (XR[1] - XR[0]) / NB
    emp = counts / (N * w_bin)

    fig, ax = S.new_axes(5.6, 3.2)

    # faint filled target density + its outline
    ax.fill_between(grid, p, color=S.ACCENT, alpha=0.10, zorder=1)
    ax.plot(grid, p, color=S.INK, lw=2.0, zorder=3,
            label=r"target $p=\frac{1}{2}\mathcal{N}(-2,0.6^2)+\frac{1}{2}\mathcal{N}(2,0.8^2)$")

    # converged particle histogram, on the density scale
    ax.bar(edges[:-1], emp, width=w_bin, align="edge",
           color=S.ACCENT, alpha=0.22, edgecolor="none", zorder=2,
           label="converged particles (empirical density)")

    # beeswarm rug of the final positions below the axis (rank mod 6 levels)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(N, dtype=int)
    ranks[order] = np.arange(N)
    ax.axhline(0.0, color=S.RULE, lw=0.8, zorder=2)
    y_bee = -0.03 - (ranks % 6) * 0.017
    ax.scatter(x, y_bee, s=12, color=S.POS, edgecolor=S.PAPER,
               linewidth=0.4, zorder=4)

    ax.set_xlim(XR)
    ax.set_ylim(-0.16, max(p.max(), emp.max()) * 1.12)
    ax.set_xlabel("$x$")
    ax.set_ylabel("density")
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.set_title("Stein variational gradient descent: 80 particles reach both modes",
                 color=S.INK)
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    S.finish(ax)

    # inset: empirical KSD^2 falling over iterations (log-y)
    axi = ax.inset_axes([0.66, 0.52, 0.31, 0.42])
    axi.semilogy(hk[:, 0], hk[:, 1], color=S.NEG, lw=1.4)
    axi.set_title(r"$\widehat{\mathrm{KSD}}^2$", color=S.INK, fontsize=7.5, pad=2)
    axi.set_xlabel("iteration", fontsize=6.5, labelpad=1)
    axi.tick_params(labelsize=6, length=2, color=S.RULE)
    for side in ("top", "right"):
        axi.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        axi.spines[side].set_color(S.RULE)

    return S.save(fig, "svgd-flow")


if __name__ == "__main__":
    print(main())
