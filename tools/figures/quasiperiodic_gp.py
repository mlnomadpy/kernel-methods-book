"""quasiperiodic-gp: recover a stellar rotation period from a synthetic light curve.

Reproduces ``public/assets/viz-quasiperiodic-gp.js`` (and the worked example in
``checks/ch-highstakes-ex1.py``). The kernel is the quasi-periodic covariance of
Angus et al. (2018) -- an exp-sine-squared periodic term times an RBF decay,

    k(t, t') = exp( -(t-t')^2 / (2 l^2) - Gamma sin^2( pi |t-t'| / P ) ),

with rotation period ``P``, spot-coherence length ``l``, and modulation ``Gamma``.
The data are one fixed draw from this GP at the true hyperparameters (P = 10 d,
l = 30 d, Gamma = 2), plus white noise sigma = 0.15, exactly as the widget seeds
its light curve; here the browser LCG is replaced by ``S.rng(0)`` for a byte-stable
plate (120 irregular samples over 60 days, matching the manuscript example).

We then do the accountable step the widget invites by hand: sweep the GP log
marginal likelihood over P, form the flat-prior posterior, and read off the MAP
period with a 68% credible interval. The plate shows the noisy observations, the
GP posterior mean with a +/-2 sigma band conditioned at the recovered period, and
the period posterior in an inset -- the period and its uncertainty together.
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

# --- widget / worked-example constants (viz-quasiperiodic-gp.js) --------------
N = 120            # irregular brightness samples (manuscript worked example)
TMAX = 60.0        # observing baseline (days)
SIG = 0.15         # white-noise standard deviation
P_TRUE = 10.0      # true rotation period (days)
ELL = 30.0         # spot-coherence length l (days) -- fixed at its default
GAM = 2.0          # modulation depth Gamma
JIT = 1e-8         # jitter for the noise-free data draw


def qp(a: jnp.ndarray, b: jnp.ndarray, P: float) -> jnp.ndarray:
    """Quasi-periodic kernel: exp-sine-squared periodic term x RBF decay."""
    tau = a[:, None] - b[None, :]
    s = jnp.sin(jnp.pi * jnp.abs(tau) / P)
    return jnp.exp(-(tau ** 2) / (2.0 * ELL * ELL) - GAM * s * s)


def synth_curve(g):
    """One fixed GP draw at the truth + white noise (widget's seeded light curve)."""
    t = jnp.asarray(np.sort(g.uniform(0.0, TMAX, N)))
    K0 = qp(t, t, P_TRUE) + JIT * jnp.eye(N)
    L0 = jnp.linalg.cholesky(K0)
    z = jnp.asarray(g.standard_normal(N))
    noise = jnp.asarray(g.standard_normal(N))
    y = L0 @ z + SIG * noise
    return t, y


@jax.jit
def log_marg_like(t, y, P):
    """GP log marginal likelihood: -1/2 y^T K^-1 y - sum log L_ii - n/2 log 2pi."""
    K = qp(t, t, P) + SIG * SIG * jnp.eye(N)
    L = jnp.linalg.cholesky(K)
    alpha = jax.scipy.linalg.cho_solve((L, True), y)
    return -0.5 * (y @ alpha) - jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * N * jnp.log(2 * jnp.pi)


def posterior(t, y, P, grid):
    """Posterior mean and predictive sd on a grid, conditioned at period P."""
    K = qp(t, t, P) + SIG * SIG * jnp.eye(N)
    L = jnp.linalg.cholesky(K)
    alpha = jax.scipy.linalg.cho_solve((L, True), y)
    Ksg = qp(grid, t, P)                                # (G, N)
    mu = Ksg @ alpha
    v = jax.scipy.linalg.solve_triangular(L, Ksg.T, lower=True)   # (N, G)
    q = jnp.sum(v * v, axis=0)
    sd = jnp.sqrt(jnp.clip(1.0 - q, 0.0, None) + SIG * SIG)       # k(t,t)=1 on the diagonal
    return mu, sd


def main() -> str:
    g = S.rng(0)
    t, y = synth_curve(g)

    # --- period recovery: sweep the marginal likelihood, flat prior on P ------
    Pgrid = jnp.linspace(5.0, 20.0, 601)
    dP = float(Pgrid[1] - Pgrid[0])
    ll = jax.vmap(lambda P: log_marg_like(t, y, P))(Pgrid)
    ll = np.asarray(ll)
    post = np.exp(ll - ll.max())
    post /= post.sum() * dP
    Pg = np.asarray(Pgrid)
    P_map = float(Pg[np.argmax(ll)])
    P_mean = float(np.sum(Pg * post) * dP)
    cdf = np.cumsum(post) * dP
    lo = float(Pg[np.searchsorted(cdf, 0.16)])
    hi = float(Pg[np.searchsorted(cdf, 0.84)])

    # --- GP fit at the recovered period ---------------------------------------
    grid = jnp.linspace(0.0, TMAX, 240)
    mu, sd = posterior(t, y, P_map, grid)
    grid = np.asarray(grid); mu = np.asarray(mu); sd = np.asarray(sd)
    tt = np.asarray(t); yy = np.asarray(y)

    # ---- plate: light curve on top, period posterior as a bottom strip -------
    fig = plt.figure(figsize=(5.4, 3.8))
    gs = GridSpec(2, 1, height_ratios=[3.0, 1.15], hspace=0.55, figure=fig)

    # top panel: the conditioned quasi-periodic GP
    ax = fig.add_subplot(gs[0])
    ax.axhline(0.0, color=S.RULE, lw=0.8, ls=(0, (3, 3)), zorder=0)
    ax.fill_between(grid, mu - 2 * sd, mu + 2 * sd, color=S.ACCENT, alpha=0.14,
                    lw=0, zorder=1, label=r"posterior $\pm2\sigma$")
    ax.plot(grid, mu, color=S.ACCENT, lw=1.8, zorder=3, label="posterior mean")
    ax.scatter(tt, yy, s=9, color=S.INK, zorder=4, label="observations")

    ax.set_xlim(0, TMAX)
    ymax = max(2.6, np.max(mu + 2 * sd) + 0.2, yy.max() + 0.2)
    ymin = min(-2.6, np.min(mu - 2 * sd) - 0.2, yy.min() - 0.2)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("time (days)", labelpad=2)
    ax.set_ylabel("relative brightness")
    ax.set_title("Quasi-periodic GP recovers a rotation period", color=S.INK)
    ax.legend(loc="lower center", frameon=False, ncol=3, handlelength=1.4,
              columnspacing=1.1, borderaxespad=0.2, bbox_to_anchor=(0.5, -0.02))
    S.finish(ax)

    # bottom panel: marginal-likelihood posterior on P, with 68% credible interval
    axp = fig.add_subplot(gs[1])
    m = (Pg >= 9.0) & (Pg <= 11.0)
    axp.fill_between(Pg[m], 0, post[m], where=(Pg[m] >= lo) & (Pg[m] <= hi),
                     color=S.ACCENT, alpha=0.22, lw=0,
                     label=f"68% CI $[{lo:.2f},\\,{hi:.2f}]$")
    axp.plot(Pg[m], post[m], color=S.ACCENT, lw=1.5)
    axp.axvline(P_map, color=S.ACCENT, lw=1.0)
    axp.axvline(P_TRUE, color=S.MUTED, lw=0.9, ls=(0, (2, 2)), label="truth $10.0$ d")
    axp.set_xlim(9, 11); axp.set_ylim(0, post.max() * 1.18)
    axp.set_yticks([])
    axp.set_xticks([9.0, 9.5, 10.0, 10.5, 11.0])
    axp.set_xlabel("period $P$ (days)", labelpad=2)
    axp.set_title(f"period posterior (flat prior): MAP $= {P_map:.2f}$ d",
                  color=S.MUTED, fontsize=8.5, pad=3)
    axp.legend(loc="upper right", frameon=False, fontsize=7.5, handlelength=1.2,
               borderaxespad=0.2)
    for side in ("top", "right", "left"):
        axp.spines[side].set_visible(False)
    axp.spines["bottom"].set_color(S.RULE)
    axp.tick_params(length=3, color=S.RULE)

    print(f"recovered P: MAP={P_map:.2f} mean={P_mean:.2f} 68% CI=[{lo:.2f},{hi:.2f}] d")
    return S.save(fig, "quasiperiodic-gp")


if __name__ == "__main__":
    print(main())
