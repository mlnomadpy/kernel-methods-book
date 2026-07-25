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

The learning curves are evaluated with fixed-size masked systems.  JAX compiles
one ``lax.scan`` over sample size and one ``vmap`` over random trials, replacing
thousands of separately dispatched, repeatedly traced factorizations while
leaving every active set and every GP posterior mathematically unchanged.

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
N_SHOW = 8            # posterior panel: 5 active queries (n = 8)
N_MAX = 48            # learning-curve budget
R = 160               # random-learning trials
VAR_TOL = 2e-10       # only roundoff-sized negative posterior variances allowed

GX = jnp.linspace(XR[0], XR[1], G)
SEED_N = len(SEED_X)


def f(x):
    """True target: a double-well plus a wiggle (widget's f)."""
    a = x * x - 1.0
    return a * a + 0.3 * jnp.sin(4.0 * x)


GY = f(GX)


def kf(a, b):
    """Gaussian kernel k(a,b) = exp(-(a-b)^2 / (2 l^2)); k(x,x)=1."""
    d = a[:, None] - b[None, :]
    return jnp.exp(-(d ** 2) / (2.0 * ELL * ELL))


def _masked_system(xs, ys, mask):
    """Cholesky factors for one fixed-size, prefix-masked GP system.

    Inactive rows are an identity block with zero right-hand side and zero
    cross-covariance.  The active principal block is therefore exactly
    ``K(X_n, X_n) + NOISE I``.
    """
    mask = mask.astype(xs.dtype)
    active = mask[:, None] * mask[None, :]
    K = kf(xs, xs) * active
    K = K + jnp.diag(NOISE * mask + (1.0 - mask))
    L = jnp.linalg.cholesky(K)
    return L, ys * mask, mask


def fit_padded(xs, ys, mask):
    """GP posterior mean, variance, and RMSE for one masked prefix.

    mu(x)     = k(x)^T (K + sigma_n^2 I)^{-1} y
    sigma^2(x)= k(x,x) - k(x)^T (K + sigma_n^2 I)^{-1} k(x)   with k(x,x)=1
    """
    L, rhs, mask = _masked_system(xs, ys, mask)
    alpha = jax.scipy.linalg.cho_solve((L, True), rhs)
    Kg = kf(GX, xs) * mask[None, :]
    mu = Kg @ alpha
    v = jax.scipy.linalg.solve_triangular(L, Kg.T, lower=True)
    q = jnp.sum(v * v, axis=0)
    var = 1.0 - q
    rmse = jnp.sqrt(jnp.mean((mu - GY) ** 2))
    return mu, var, rmse


def rmse_padded(xs, ys, mask):
    """RMSE-only version used by the random ensemble."""
    L, rhs, mask = _masked_system(xs, ys, mask)
    alpha = jax.scipy.linalg.cho_solve((L, True), rhs)
    mu = (kf(GX, xs) * mask[None, :]) @ alpha
    return jnp.sqrt(jnp.mean((mu - GY) ** 2))


@jax.jit
def active_trajectory():
    """All active-learning states in one compiled scan.

    The carry stores the eventual length-``N_MAX`` design.  At step ``n`` the
    mask exposes only its first ``n`` entries, so the posterior is exactly the
    one obtained by factoring the growing ``n`` by ``n`` Gram matrix.
    """
    xs = jnp.zeros(N_MAX, dtype=GX.dtype)
    xs = xs.at[:SEED_N].set(jnp.asarray(SEED_X))
    ys = jnp.zeros(N_MAX, dtype=GX.dtype)
    ys = ys.at[:SEED_N].set(f(xs[:SEED_N]))
    positions = jnp.arange(N_MAX)

    def add_query(carry, n):
        x_now, y_now = carry
        mask = positions < n
        mu, var, rmse = fit_padded(x_now, y_now, mask)
        query = GX[jnp.argmax(var)]
        x_now = x_now.at[n].set(query)
        y_now = y_now.at[n].set(f(query))
        return (x_now, y_now), (mu, var, rmse)

    (xs, ys), (mus, variances, rmses) = jax.lax.scan(
        add_query, (xs, ys), jnp.arange(SEED_N, N_MAX)
    )
    mu_last, var_last, rmse_last = fit_padded(
        xs, ys, jnp.ones(N_MAX, dtype=bool)
    )
    return (
        xs,
        ys,
        jnp.concatenate([mus, mu_last[None, :]], axis=0),
        jnp.concatenate([variances, var_last[None, :]], axis=0),
        jnp.concatenate([rmses, rmse_last[None]], axis=0),
    )


@jax.jit
def random_ensemble_curves(random_x):
    """Evaluate every random trial and every prefix in one scan/vmap program."""
    trials = random_x.shape[0]
    seeds = jnp.broadcast_to(jnp.asarray(SEED_X), (trials, SEED_N))
    xs = jnp.concatenate([seeds, random_x], axis=1)
    ys = f(xs)
    positions = jnp.arange(N_MAX)

    def evaluate_prefix(_, n):
        mask = positions < n
        rmses = jax.vmap(rmse_padded, in_axes=(0, 0, None))(xs, ys, mask)
        return None, rmses

    _, curves = jax.lax.scan(
        evaluate_prefix, None, jnp.arange(SEED_N, N_MAX + 1)
    )
    return curves.T


def pts_to_target(curve_rmse, ns):
    idx = np.where(curve_rmse <= TARGET)[0]
    return ns[idx[0]] if len(idx) else np.nan


def main() -> str:
    # ---- active learner: one trajectory supplies display and learning panels --
    xs_all, ys_all, mus, variances, a_rmse = active_trajectory()
    xs_all, ys_all, mus, variances, a_rmse = map(
        np.asarray, (xs_all, ys_all, mus, variances, a_rmse)
    )
    if not all(np.isfinite(a).all() for a in
               (xs_all, ys_all, mus, variances, a_rmse)):
        raise FloatingPointError("active GP trajectory contains NaN or infinity")
    min_var = float(variances.min())
    if min_var < -VAR_TOL:
        raise FloatingPointError(
            f"posterior variance {min_var:.3e} is below roundoff tolerance"
        )

    show_i = N_SHOW - SEED_N
    xs_s, ys_s = xs_all[:N_SHOW], ys_all[:N_SHOW]
    mu_s = mus[show_i]
    sd_s = np.sqrt(np.maximum(variances[show_i], 0.0))
    bi = int(np.argmax(sd_s))      # next query the widget would place
    x_next, sd_next = float(GX[bi]), float(sd_s[bi])

    a_n = np.arange(SEED_N, N_MAX + 1)
    a_hit = int(pts_to_target(a_rmse, a_n))                    # = 19

    # ---- random ensemble: mean +/- band, all draws from one S.rng(0) stream --
    rng = S.rng(0)
    random_x = XR[0] + (XR[1] - XR[0]) * rng.random(
        (R, N_MAX - SEED_N)
    )
    r_curves = np.asarray(random_ensemble_curves(jnp.asarray(random_x)))
    if not np.isfinite(r_curves).all():
        raise FloatingPointError("random GP ensemble contains NaN or infinity")
    r_mean = r_curves.mean(axis=0)
    r_lo = np.percentile(r_curves, 10, axis=0)
    r_hi = np.percentile(r_curves, 90, axis=0)
    r_hits = np.array([pts_to_target(c, a_n) for c in r_curves])
    r_hit_mean = np.nanmean(r_hits)
    r_at_active = float(r_mean[a_hit - a_n[0]])                # random RMSE at n=19
    # The exact random baseline depends on the documented PRNG algorithm; the
    # invariant that matters is the active policy's substantial sample saving.
    if a_hit != 19 or not (a_hit + 8.0 <= r_hit_mean <= N_MAX):
        raise AssertionError(
            "active-learning benchmark changed: "
            f"active hit={a_hit}, random mean hit={r_hit_mean:.3f}"
        )

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
