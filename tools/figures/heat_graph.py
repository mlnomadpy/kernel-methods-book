"""heat-graph: the graph heat kernel from real Laplacian eigenpairs.

Reproduces ``public/assets/viz-heat-graph.js`` at its default state. A fixed
24-node graph (two dense 8-node clusters joined by a three-edge bridge, plus
pendants) has Laplacian L = D - W. It is eigendecomposed once (the widget uses
a cyclic-Jacobi routine; a symmetric eigensolver gives the identical kernel,
since the filter is applied and reassembled). The displayed number on each node
is the exact spectral-filter correlation

    K_t = sum_l e^{-t lambda_l} v_l v_l^T ,   r(i,j) = K(i,j)/sqrt(K(i,i)K(j,j))

for the heat kernel (the widget's default ``kernel = heat``) at the default
source node ``src = 2`` and default time ``t = 10^{-0.3}``. The right panel
raises t to the slider's maximum ``10^{0.9}`` to show similarity diffusing
across the bridge into the far cluster.
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

N = 24

# ---- fixed graph: clusters A (0-7), B (8-15), bridge (16-18), pendants -------
def _edges() -> list[tuple[int, int]]:
    E: list[tuple[int, int]] = []

    def ring(ids, extra):
        for i in range(len(ids)):
            E.append((ids[i], ids[(i + 1) % len(ids)]))
        E.extend(extra)

    ring([0, 1, 2, 3, 4, 5, 6, 7], [(0, 2), (1, 3), (4, 6), (5, 7), (0, 4)])
    ring([8, 9, 10, 11, 12, 13, 14, 15], [(8, 10), (9, 11), (12, 14), (13, 15), (8, 12)])
    E += [(3, 16), (16, 17), (17, 18), (18, 11)]           # the bridge
    E += [(6, 19), (14, 20), (9, 21), (1, 22), (13, 23)]   # pendants
    return E


EDGES = _edges()

# node layout in [0,1]^2 (canvas coords: y grows downward)
LAY = []
for i in range(8):
    a = (i / 8) * 2 * np.pi
    LAY.append([0.19 + 0.115 * np.cos(a), 0.5 + 0.3 * np.sin(a)])
for i in range(8):
    a = (i / 8) * 2 * np.pi
    LAY.append([0.81 + 0.115 * np.cos(a), 0.5 + 0.3 * np.sin(a)])
LAY += [[0.4, 0.42], [0.5, 0.5], [0.6, 0.42]]
LAY += [[0.045, 0.86], [0.955, 0.86], [0.9, 0.1], [0.1, 0.1], [0.72, 0.95]]
LAY = np.asarray(LAY)

SRC = 2  # widget default source node


def laplacian() -> jnp.ndarray:
    L = np.zeros((N, N))
    for i, j in EDGES:
        L[i, j] -= 1.0
        L[j, i] -= 1.0
        L[i, i] += 1.0
        L[j, j] += 1.0
    return jnp.asarray(L)


def heat_corr(t: float):
    """Correlation r(SRC, j) of the heat kernel K_t = sum_l e^{-t lam_l} v v^T."""
    L = laplacian()
    lam, vec = jnp.linalg.eigh(L)          # vec[:, l] is eigenvector l
    filt = jnp.exp(-t * lam)               # spectral filter Phi(lam)=e^{-t lam}
    K = (vec * filt) @ vec.T               # sum_l filt_l v_l v_l^T
    d = jnp.sqrt(jnp.diag(K))
    r = K[SRC] / (d[SRC] * d)              # r(SRC, j)
    return np.asarray(r)


def draw_panel(ax, r: np.ndarray, title: str) -> None:
    xy = np.column_stack([LAY[:, 0], 1.0 - LAY[:, 1]])   # flip y to upright
    for i, j in EDGES:                                   # thin edges
        ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                color=S.RULE, lw=0.9, zorder=1)
    t = np.clip(r, 0.0, 1.0)
    for i in range(N):
        marked = i == SRC
        ax.scatter(xy[i, 0], xy[i, 1], s=190 if marked else 120,
                   c=[S.HEAT(t[i])],
                   edgecolor=S.INK if marked else S.MUTED,
                   linewidth=1.6 if marked else 0.9, zorder=3)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, color=S.INK, pad=4)


def main() -> str:
    t_default = 10.0 ** (-0.3)   # widget default slider value
    t_diffuse = 10.0 ** (0.9)    # slider maximum: diffusion across the bridge
    r0 = heat_corr(t_default)
    r1 = heat_corr(t_diffuse)

    fig = plt.figure(figsize=(6.4, 3.4))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.08, figure=fig)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    draw_panel(ax0, r0, rf"$t=10^{{-0.3}}\approx{t_default:.2f}$ (localized)")
    draw_panel(ax1, r1, rf"$t=10^{{0.9}}\approx{t_diffuse:.1f}$ (diffused)")

    cax = fig.add_subplot(gs[2])
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=S.HEAT)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(r"$r(\mathrm{src},j)$", color=S.INK)
    cb.outline.set_edgecolor(S.RULE)
    cb.ax.tick_params(color=S.RULE)

    fig.suptitle(r"Graph heat kernel $K_t=\sum_\ell e^{-t\lambda_\ell}v_\ell v_\ell^\top$,"
                 r" source node $2$", color=S.INK, y=0.99)
    return S.save(fig, "heat-graph")


if __name__ == "__main__":
    print(main())
