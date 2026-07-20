"""kernel-lab: kernel ridge regression on the widget's default six points.

Reproduces ``WIDGETS["kernel-lab"]`` in ``public/assets/viz.js`` at its default
state: six fixed points, Gaussian kernel at bandwidth 1, ridge lambda = 10^-2.
The curve is the exact kernel ridge solution f = sum_i alpha_i k(., x_i) with
alpha = (K + lambda n I)^{-1} y, the same linear solve the widget runs live.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()

PTS = np.array([[-3.5, -1.2], [-2, 0.6], [-0.5, -0.4],
                [0.6, 1.3], [2, -0.3], [3.4, 1.1]])


def gaussian(a: jnp.ndarray, b: jnp.ndarray, bw: float) -> jnp.ndarray:
    d = a[:, None] - b[None, :]
    return jnp.exp(-(d ** 2) / (2.0 * bw * bw))


def krr_curve(xs, ys, grid, bw: float, lam: float):
    n = xs.shape[0]
    K = gaussian(xs, xs, bw)
    alpha = jnp.linalg.solve(K + lam * n * jnp.eye(n), ys)      # (K + lambda n I)^{-1} y
    Kg = gaussian(grid, xs, bw)
    return Kg @ alpha


def main() -> str:
    xs = jnp.asarray(PTS[:, 0]); ys = jnp.asarray(PTS[:, 1])
    grid = jnp.linspace(-5, 5, 400)
    f = np.asarray(krr_curve(xs, ys, grid, bw=1.0, lam=1e-2))

    fig, ax = S.new_axes(5.2, 3.0)
    ax.axhline(0.0, color=S.RULE, lw=0.8, ls=(0, (3, 3)))
    ax.plot(np.asarray(grid), f, color=S.ACCENT, lw=2.2,
            label=r"$f=\sum_i\alpha_i\,k(\cdot,x_i)$")
    ax.scatter(PTS[:, 0], PTS[:, 1], s=32, color=S.POS,
               edgecolor=S.PAPER, linewidth=0.9, zorder=3)
    ax.set_xlim(-5, 5); ax.set_ylim(-2.4, 2.4)
    ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$")
    ax.set_title(r"Kernel ridge, Gaussian kernel, bandwidth $1$, $\lambda=10^{-2}$",
                 color=S.INK)
    ax.legend(loc="upper left", frameon=False)
    S.finish(ax)
    return S.save(fig, "kernel-lab")


if __name__ == "__main__":
    print(main())
