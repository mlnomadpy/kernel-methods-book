"""active-variance: GP uncertainty sampling versus random, reproduced in JAX.

Reproduces ``WIDGETS["active-variance"]`` in ``public/assets/viz-active-variance.js``
at its default state. A Gaussian process over the target
``f(x) = (x^2 - 1)^2 + 0.3 sin(4x)`` on ``[-2, 2]`` is conditioned on three seed
points ``{-2, 0, 2}``. The active learner repeatedly queries the argmax of the
posterior standard deviation ``sigma(x)`` (uncertainty sampling: no mean term),
evaluates the true ``f`` there, and refits -- exactly the widget's ``stepOnce``
loop. A random learner spends the same budget for comparison.

The GP matches the widget's engine: Gaussian kernel ``k(a,b)=exp(-(a-b)^2/(2 l^2))``
with lengthscale ``l = 0.35`` and jitter ``sigma_n^2 = 1e-4``; because ``k(x,x)=1``
the prior variance is 1 and ``sigma^2(x) = 1 - k(x)^T (K + sigma_n^2 I)^{-1} k(x)``.
Everything below is that solve in ``jax.numpy`` (x64), the same linear algebra the
browser runs live via its Cholesky routine.

Left plate: the posterior mean +/- 2 sigma after several variance-driven queries,
with the next query marked at the argmax of sigma (before its label is known).
Right plate: RMSE-against-truth learning curves. The active learner reaches
RMSE <= 0.05 at n = 19 points; a random learner (mean over 160 seeds from S.rng(0),
budget 48) needs ~30 on average and is still ~6x worse at n = 19.
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

# --- widget constants (viz-active-variance.js) ---------------------------------
XR = (-2.0, 2.0)      # input range
G = 220               # posterior grid resolution
ELL = 0.35            # Gaussian-kernel lengthscale
NOISE = 1e-4          # sigma_n^2 jitter added to K's diagonal
SEED_X = (-2.0, 0.0, 2.0)     # three initial observations
TARGET = 0.05         # RMSE target for the points-to-target comparison

GX = jnp.linspace(XR[0], XR[1], G)


def f(x):
    """True target: a double-well plus a wiggle (widget's f)."""
    a = x * x - 1.0
    return a * a + 0.3 * jnp.sin(4.0 * x)


GY = f(GX)


def kf(a, b):
    """Gaussian kernel k(a,b) = exp(-(a-b)^2 / (2 l^2)); k(x,x)=1."""
    d = a[:, None] - b[None, :]
    return jnp.exp(-(d ** 2) / (2.0 * ELL * ELL))


def fit(xs, ys):
    """GP posterior mean, sd on the grid, and RMSE vs truth (widget's fit()).

    mu(x)     = k(x)^T (K + sigma_n^2 I)^{-1} y
    sigma^2(x)= k(x,x) - k(x)^T (K + sigma_n^2 I)^{-1} k(x)   with k(x,x)=1
    """
    n = xs.shape[0]
    K = kf(xs, xs) + NOISE * jnp.eye(n)
    L = jnp.linalg.cholesky(K)
    alpha = jax.scipy.linalg.cho_solve((L, True), ys)          # (K+noise I)^{-1} y
    Kg = kf(GX, xs)                                            # (G, n)
    mu = Kg @ alpha
    v = jax.scipy.linalg.solve_triangular(L, Kg.T, lower=True)  # L^{-1} k(x)
    q = jnp.sum(v * v, axis=0)                                  # k^T (K+..)^{-1} k
    sd = jnp.sqrt(jnp.maximum(1.0 - q, 0.0))
    rmse = jnp.sqrt(jnp.mean((mu - GY) ** 2))
    return mu, sd, rmse


def active_run(n_max):
    """Sequential uncertainty sampling: query argmax sigma, refit, repeat."""
    xs = list(SEED_X)
    ys = [float(f(jnp.asarray(x))) for x in xs]
    curve = []
    mu = sd = None
    while True:
        xa = jnp.asarray(xs)
        ya = jnp.asarray(ys)
        mu, sd, rmse = fit(xa, ya)
        curve.append((len(xs), float(rmse)))
        if len(xs) >= n_max:
            break
        bi = int(jnp.argmax(sd))                # next query = argmax posterior sd
        xs.append(float(GX[bi]))
        ys.append(float(f(GX[bi])))
    return np.asarray(xs), np.asarray(ys), np.asarray(mu), np.asarray(sd), curve


def random_curve(rng, n_max):
    """One random-sampling run to n_max; RMSE after each added point."""
    xs = list(SEED_X)
    ys = [float(f(jnp.asarray(x))) for x in xs]
    curve = []
    while True:
        _, _, rmse = fit(jnp.asarray(xs), jnp.asarray(ys))
        curve.append(float(rmse))
        if len(xs) >= n_max:
            break
        xr = XR[0] + (XR[1] - XR[0]) * float(rng.random())
        xs.append(xr)
        ys.append(float(f(jnp.asarray(xr))))
    return np.asarray(curve)


def pts_to_target(curve_rmse, ns):
    idx = np.where(curve_rmse <= TARGET)[0]
    return ns[idx[0]] if len(idx) else np.nan


def main() -> str:
    # ---- active learner: run to a small display state, then to the target ----
    N_SHOW = 8                     # posterior panel: 5 active queries (n = 8)
    xs_s, ys_s, mu_s, sd_s, _ = active_run(N_SHOW)
    bi = int(np.argmax(sd_s))      # next query the widget would place
    x_next, sd_next = float(GX[bi]), float(sd_s[bi])

    # full active curve out to a generous budget for the learning panel
    N_MAX = 48
    _, _, _, _, a_curve = active_run(N_MAX)
    a_n = np.array([c[0] for c in a_curve])
    a_rmse = np.array([c[1] for c in a_curve])
    a_hit = int(pts_to_target(a_rmse, a_n))                    # = 19

    # ---- random ensemble: mean +/- band, all draws from one S.rng(0) stream --
    rng = S.rng(0)
    R = 160
    r_curves = np.stack([random_curve(rng, N_MAX) for _ in range(R)])
    r_mean = r_curves.mean(axis=0)
    r_lo = np.percentile(r_curves, 10, axis=0)
    r_hi = np.percentile(r_curves, 90, axis=0)
    r_hits = np.array([pts_to_target(c, a_n) for c in r_curves])
    r_hit_mean = np.nanmean(r_hits)                            # ~30
    r_at_active = float(r_mean[a_hit - a_n[0]])                # random RMSE at n=19

    # ================================ figure ================================
    fig = plt.figure(figsize=(7.4, 3.2))
    gs = GridSpec(1, 2, width_ratios=[1.35, 1.0], wspace=0.28, figure=fig)

    # -- Panel A: GP posterior after several variance-driven queries ----------
    axA = fig.add_subplot(gs[0])
    YR = (-1.2, 3.4)
    gx = np.asarray(GX)
    axA.fill_between(gx, np.clip(mu_s - 2 * sd_s, *YR),
                     np.clip(mu_s + 2 * sd_s, *YR),
                     color=S.ACCENT, alpha=0.14, lw=0, label=r"GP mean $\pm\,2\sigma$")
    axA.plot(gx, np.asarray(GY), color=S.MUTED, lw=1.1, ls=(0, (4, 3)), label="truth $f$")
    axA.plot(gx, mu_s, color=S.ACCENT, lw=1.9, label="GP mean")
    axA.scatter(xs_s, ys_s, s=26, color=S.ACCENT, edgecolor=S.PAPER,
                linewidth=0.9, zorder=4, label="queries")
    # next query at argmax sigma (marked before its label is known)
    axA.axvline(x_next, color=S.NEG, lw=0.9, ls=(0, (2, 2)), zorder=2)
    axA.scatter([x_next], [YR[1] - 0.18], marker="v", s=44, color=S.NEG,
                edgecolor=S.PAPER, linewidth=0.7, zorder=6)
    axA.annotate(r"next query (max $\sigma$)", (x_next, YR[1] - 0.18),
                 xytext=(8, -1), textcoords="offset points",
                 color=S.NEG, fontsize=7.5, ha="left", va="top")
    axA.set_xlim(*XR); axA.set_ylim(*YR)
    axA.set_xlabel("$x$"); axA.set_ylabel("$f(x)$")
    axA.set_title(f"Posterior after {N_SHOW - 3} active queries "
                  f"($n={N_SHOW}$)", color=S.INK)
    axA.legend(loc="lower center", frameon=False, ncol=2,
               fontsize=7.2, handlelength=1.4, columnspacing=1.1)
    S.finish(axA)

    # -- Panel B: learning curves, active vs random ---------------------------
    axB = fig.add_subplot(gs[1])
    ns = a_n
    axB.fill_between(ns, r_lo, r_hi, color=S.POS, alpha=0.13, lw=0)
    axB.plot(ns, r_mean, color=S.POS, lw=1.7, label="random (mean)")
    axB.plot(a_n, a_rmse, color=S.ACCENT, lw=2.0, label="uncertainty sampling")
    axB.axhline(TARGET, color=S.MUTED, lw=0.8, ls=(0, (3, 3)))
    axB.annotate(f"target {TARGET:g}", (ns[-1], TARGET), xytext=(-2, 3),
                 textcoords="offset points", color=S.MUTED, fontsize=7, ha="right")
    # points-to-target markers
    axB.axvline(a_hit, color=S.ACCENT, lw=0.8, ls=(0, (2, 2)))
    axB.scatter([a_hit], [TARGET], s=22, color=S.ACCENT, zorder=5,
                edgecolor=S.PAPER, linewidth=0.7)
    axB.annotate(f"{a_hit}", (a_hit, TARGET), xytext=(3, 6),
                 textcoords="offset points", color=S.ACCENT, fontsize=8)
    axB.annotate(rf"random $\approx${r_hit_mean:.0f}", (r_hit_mean, TARGET),
                 xytext=(2, 8), textcoords="offset points", color=S.POS, fontsize=8)
    axB.set_yscale("log")
    axB.set_xlim(ns[0], ns[-1])
    axB.set_xlabel("labelled points $n$")
    axB.set_ylabel("RMSE vs truth")
    axB.set_title("Labels to a target error", color=S.INK)
    axB.legend(loc="lower left", frameon=False, fontsize=7.2)
    S.finish(axB)

    out = S.save(fig, "active-variance")
    print(f"active: n={a_hit} to RMSE<= {TARGET}; "
          f"random mean={r_hit_mean:.1f} (RMSE at n={a_hit} is {r_at_active:.3f})")
    return out


if __name__ == "__main__":
    print(main())
