"""sinkhorn-plan: converged entropic optimal-transport plan.

Reproduces ``WIDGETS["sinkhorn-plan"]`` in ``public/assets/viz-sinkhorn-plan.js``
at its default converged state. A two-bump source ``a`` is transported to a wide
single-bump target ``b`` on ``N = 28`` shared grid points in [0, 1] with squared
cost ``C_ij = (x_i - y_j)^2`` and Gibbs kernel ``K = exp(-C / eps)``. Sinkhorn runs
the exact live update, ``u <- a / (K v)``, ``v <- b / (K^T u)`` per tick, and the
plan is ``pi = diag(u) K diag(v)``. Iteration halts when the row-marginal
violation ``||pi 1 - a||_1`` falls below ``1e-6`` (the widget's threshold).

The slider default is ``leps = -1.7``, i.e. ``eps = 10^{-1.7} ~= 0.02`` -- the
same value the widget converges at on load. The heatmap is the converged plan
(HEAT ramp), with ``a`` as a bar profile on the left and ``b`` on the bottom,
matching the widget layout; the annotation carries the converged cost <C, pi>.
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

N = 28
EPS = 10.0 ** -1.7   # slider default leps = -1.7  ->  eps = 10^{-1.7} ~= 0.0200
TAU = 1e-6           # widget stopping threshold on ||pi 1 - a||_1


def problem():
    """Fixed source a, target b, and squared cost C -- exactly the JS setup."""
    xs = jnp.arange(N) / (N - 1)                       # xs[i] = i / (N - 1)
    bump = lambda z, m, s: jnp.exp(-((z - m) ** 2) / (2.0 * s * s))
    mu = 0.55 * bump(xs, 0.22, 0.06) + 0.45 * bump(xs, 0.72, 0.05)  # two bumps
    nu = bump(xs, 0.5, 0.16)                            # one wide bump
    a = mu / jnp.sum(mu)                                # normalize to sum 1
    b = nu / jnp.sum(nu)
    C = (xs[:, None] - xs[None, :]) ** 2                # C_ij = (x_i - y_j)^2
    return a, b, C


@jax.jit
def sinkhorn(a, b, C, eps):
    """Run Sinkhorn until ||pi 1 - a||_1 < TAU; return plan and cost."""
    K = jnp.exp(-C / eps)                               # Gibbs kernel
    u = jnp.ones(N)
    v = jnp.ones(N)

    def viol(u, v):
        pi = u[:, None] * K * v[None, :]
        return jnp.sum(jnp.abs(jnp.sum(pi, axis=1) - a))

    def cond(state):
        u, v, it = state
        return (viol(u, v) >= TAU) & (it < 200_000)

    def body(state):
        u, v, it = state
        u = a / (K @ v)                                 # u <- a / (K v)
        v = b / (K.T @ u)                               # v <- b / (K^T u)
        return u, v, it + 1

    u, v, it = jax.lax.while_loop(cond, body, (u, v, 0))
    pi = u[:, None] * K * v[None, :]                    # pi = diag(u) K diag(v)
    cost = jnp.sum(pi * C)                              # <C, pi>
    return pi, cost, it


def main() -> str:
    a, b, C = problem()
    pi, cost, it = sinkhorn(a, b, C, EPS)
    pi = np.asarray(pi); a = np.asarray(a); b = np.asarray(b)
    cost = float(cost); it = int(it)
    pi_n = pi / pi.max()                                # shade relative to max entry

    fig = plt.figure(figsize=(4.6, 4.6))
    gs = GridSpec(2, 2, width_ratios=[1, 6], height_ratios=[6, 1],
                  wspace=0.06, hspace=0.06, figure=fig)

    # main panel: the converged transport plan as a heatmap
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(pi_n, cmap=S.HEAT, vmin=0, vmax=1, interpolation="nearest",
              aspect="auto", extent=(-0.5, N - 0.5, N - 0.5, -0.5))
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(S.RULE)
    ax.set_title(r"Entropic plan $\pi=\operatorname{diag}(u)\,K\,\operatorname{diag}(v)$,"
                 " $\\varepsilon=10^{-1.7}$", color=S.INK, pad=6)

    # left panel: source marginal a (bars grow toward the heatmap)
    axl = fig.add_subplot(gs[0, 0], sharey=ax)
    axl.barh(np.arange(N), a, height=0.9, color=S.POS, alpha=0.6, linewidth=0)
    axl.set_ylim(N - 0.5, -0.5)
    axl.invert_xaxis()
    axl.axis("off")
    axl.set_title("$a$", color=S.MUTED, fontsize=9, pad=4)

    # bottom panel: target marginal b (bars hang down from the heatmap edge)
    axb = fig.add_subplot(gs[1, 1], sharex=ax)
    axb.bar(np.arange(N), b, width=0.9, color=S.POS, alpha=0.6, linewidth=0)
    axb.set_xlim(-0.5, N - 0.5)
    axb.invert_yaxis()
    axb.axis("off")
    axb.text(-0.02, 0.5, "$b$", transform=axb.transAxes,
             ha="right", va="center", color=S.MUTED, fontsize=9)

    # converged diagnostics, matching the widget readout
    fig.text(0.5, 0.02,
             rf"converged in {it} iters $\cdot$ "
             rf"$\langle C,\pi\rangle = {cost:.4f}$",
             ha="center", va="bottom", color=S.MUTED, fontsize=8.5)

    return S.save(fig, "sinkhorn-plan")


if __name__ == "__main__":
    print(main())
