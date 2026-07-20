"""svm-margin: the maximum-margin boundary for the widget's eight default points.

Reproduces ``WIDGETS["svm-margin"]`` in ``public/assets/viz.js`` at its default
state: eight fixed points (four per class), Gaussian kernel at bandwidth 1.2, and
C = 10. The soft-margin SVM dual is solved by the same exact two-coordinate
ascent as ``public/assets/smo.js`` (choose, at each pass, the feasible pair
d_i = y_i, d_j = -y_j preserving y^T alpha = 0 with the largest objective gain),
reimplemented here in NumPy on a Gram matrix computed in JAX.

The decision function is f(x) = b + sum_j alpha_j y_j k(x, x_j). We draw the
boundary f = 0 and the two margins f = +/-1, faintly shade the decision regions,
and circle the support vectors (alpha_i > 1e-4).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()

# Widget defaults: [x1, x2, y].
P = np.array([[-2.0, 1.0, 1], [-1.4, 2.0, 1], [-2.6, -0.4, 1], [-0.8, 0.6, 1],
              [1.8, -1.0, -1], [2.4, 0.4, -1], [1.0, -1.8, -1], [2.6, -1.6, -1]])
BW = 1.2
C = 10.0
XR = (-4.0, 4.0)
YR = (-3.0, 3.0)


def gaussian_gram(A: jnp.ndarray, B: jnp.ndarray, bw: float) -> jnp.ndarray:
    d2 = jnp.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
    return jnp.exp(-d2 / (2.0 * bw * bw))


def smo_solve(K: np.ndarray, y: np.ndarray, C: float,
              tol: float = 1e-4, max_passes: int = 2000):
    """Exact two-coordinate ascent on the soft-margin dual (mirrors smo.js)."""
    n = len(y)
    alpha = np.zeros(n)

    def bounds(a, d):
        return (-a, C - a) if d > 0 else (a - C, a)

    for _ in range(max_passes):
        raw = K @ (alpha * y)                 # rawDecisionAt(i) = sum_j a_j y_j K_ji
        grad = 1.0 - y * raw
        best = None
        for i in range(n):
            for j in range(i + 1, n):
                di, dj = y[i], -y[j]
                bi, bj = bounds(alpha[i], di), bounds(alpha[j], dj)
                lo = max(bi[0], bj[0]); hi = min(bi[1], bj[1])
                if hi - lo < 1e-14:
                    continue
                deriv = grad[i] * di + grad[j] * dj
                curv = max(1e-12, K[i, i] + K[j, j] - 2 * K[i, j])
                step = max(lo, min(hi, deriv / curv))
                gain = deriv * step - 0.5 * curv * step * step
                if best is None or gain > best[0]:
                    best = (gain, i, j, di, dj, step)
        if best is None or best[0] < tol * tol:
            break
        _, i, j, di, dj, step = best
        alpha[i] += step * di
        alpha[j] += step * dj

    # Intercept from free support vectors (else midpoint of KKT interval).
    raw = K @ (alpha * y)
    free = (alpha > tol) & (alpha < C - tol)
    if free.any():
        bias = float(np.mean(y[free] - raw[free]))
    else:
        lower, upper = -np.inf, np.inf
        for i in range(n):
            b = y[i] - raw[i]
            if (y[i] > 0 and alpha[i] <= tol) or (y[i] < 0 and alpha[i] >= C - tol):
                lower = max(lower, b)
            if (y[i] < 0 and alpha[i] <= tol) or (y[i] > 0 and alpha[i] >= C - tol):
                upper = min(upper, b)
        bias = (lower + upper) / 2 if np.isfinite(lower) and np.isfinite(upper) \
            else lower if np.isfinite(lower) else upper if np.isfinite(upper) else 0.0
    return alpha, bias


def main() -> str:
    X = jnp.asarray(P[:, :2]); y = P[:, 2].astype(float)
    K = np.asarray(gaussian_gram(X, X, BW))
    alpha, bias = smo_solve(K, y, C)
    sv = alpha > 1e-4

    # Decision field on a grid, f(x) = b + sum_j a_j y_j k(x, x_j), in JAX.
    gx = jnp.linspace(*XR, 220)
    gy = jnp.linspace(*YR, 180)
    GX, GY = jnp.meshgrid(gx, gy)
    grid = jnp.stack([GX.ravel(), GY.ravel()], axis=1)
    Kg = gaussian_gram(grid, X, BW)                     # (Ngrid, 8)
    F = np.asarray(Kg @ (jnp.asarray(alpha) * jnp.asarray(y)) + bias).reshape(GX.shape)
    GXn, GYn = np.asarray(GX), np.asarray(GY)

    fig, ax = S.new_axes(5.4, 3.4)

    # Faint decision regions (blue = +1, red = -1).
    ax.contourf(GXn, GYn, F, levels=[-1e9, 0.0, 1e9],
                colors=[S.NEG, S.POS], alpha=0.10, zorder=0)
    # Margins f = +/-1 (faint) and boundary f = 0 (ink).
    ax.contour(GXn, GYn, F, levels=[-1.0, 1.0], colors=[S.MUTED],
               linewidths=1.0, linestyles="dashed", zorder=1)
    ax.contour(GXn, GYn, F, levels=[0.0], colors=[S.INK],
               linewidths=2.0, zorder=2)

    # Points, coloured by class; circle the support vectors.
    for cls, col in ((1, S.POS), (-1, S.NEG)):
        m = P[:, 2] == cls
        ax.scatter(P[m, 0], P[m, 1], s=44, color=col, edgecolor=S.PAPER,
                   linewidth=0.9, zorder=4)
    ax.scatter(P[sv, 0], P[sv, 1], s=150, facecolors="none",
               edgecolors=S.INK, linewidths=1.6, zorder=3,
               label="support vectors")

    ax.set_xlim(*XR); ax.set_ylim(*YR)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(rf"Max-margin SVM, Gaussian kernel (bw $1.2$), $C={C:.0f}$ "
                 rf"-- {int(sv.sum())} support vectors", color=S.INK)
    ax.legend(loc="upper right", frameon=False)
    S.finish(ax)
    return S.save(fig, "svm-margin")


if __name__ == "__main__":
    print(main())
