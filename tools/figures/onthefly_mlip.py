"""onthefly-mlip: an on-the-fly machine-learned interatomic potential in miniature.

Reproduces ``WIDGETS["onthefly-mlip"]`` in ``public/assets/viz-onthefly-mlip.js``
at its default state. A walker is driven across a 1-D tilted double-well
potential-energy surface. A Gaussian process (RBF length scale ELL = 0.45,
prior variance 1) is trained on the configurations "computed" so far; at every
step it predicts the energy and its posterior standard deviation sigma(x). While
sigma stays below the trigger threshold THRESH = 0.15 the cheap GP prediction is
accepted; when the walker enters unseen territory sigma spikes, a "DFT call"
fires (the true energy E(walker) is evaluated), the point is added to the
training set, and the GP refits by Cholesky. The final state -- true curve, GP
posterior mean +/- 2 sigma, the on-the-fly training points, sigma(x) with the
trigger threshold, and the recovered equilibrium bond length -- is drawn here.

The JS trajectory is fully deterministic (a driven oscillation, no RNG), so the
figure is exactly the widget's math; S.rng(0) is instantiated per the shared
determinism contract.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()

# --- widget constants (viz-onthefly-mlip.js) --------------------------------
XR = (-1.9, 1.9)      # coordinate range
G = 200               # posterior grid resolution
ELL = 0.45            # RBF length scale
JIT = 1e-6            # Cholesky jitter
THRESH = 0.15         # DFT trigger: fire when sigma(walker) exceeds this
NMAX = 30             # cap on training-set size
YR = (-0.6, 2.6)      # energy axis (matches the widget)


def E(x):
    """Tilted double well:  E(x) = (x^2 - 1)^2 + 0.12 x."""
    a = x * x - 1.0
    return a * a + 0.12 * x


def kf(a, b):
    """RBF kernel with unit prior variance:  exp(-(a-b)^2 / (2 ELL^2))."""
    d = a[:, None] - b[None, :]
    return jnp.exp(-(d ** 2) / (2.0 * ELL * ELL))


GX = jnp.linspace(XR[0], XR[1], G)          # posterior grid
GE = np.asarray(E(GX))                       # true energy on the grid


def fit(xs: jnp.ndarray, ys: jnp.ndarray):
    """GP posterior mean and sigma on GX, exactly as the widget's fit().

    alpha = (K + JIT I)^{-1} y  via Cholesky; mu = k_*^T alpha; and the
    predictive variance k(x,x) - k_*^T K^{-1} k_* = 1 - |L^{-1} k_*|^2.
    """
    n = xs.shape[0]
    K = kf(xs, xs) + JIT * jnp.eye(n)
    L = jnp.linalg.cholesky(K)
    alpha = solve_triangular(L.T, solve_triangular(L, ys, lower=True), lower=False)
    Kg = kf(GX, xs)                          # (G, n) cross-covariances
    mu = Kg @ alpha
    Vt = solve_triangular(L, Kg.T, lower=True)   # (n, G):  L^{-1} k_*
    q = jnp.sum(Vt ** 2, axis=0)
    sd = jnp.sqrt(jnp.clip(1.0 - q, 0.0, None))
    return np.asarray(mu), np.asarray(sd)


def walker_at(t: float) -> float:
    """Driven oscillation that slowly explores outward and over the barrier."""
    w = -1.0 + 0.55 * np.sin(1.7 * t) + min(2.6, 0.09 * t) * (0.5 - 0.5 * np.cos(0.23 * t))
    return float(np.clip(w, XR[0], XR[1]))


def simulate():
    """Run the widget's on-the-fly loop to completion (t > 90)."""
    _ = S.rng(0)                             # deterministic contract; trajectory is RNG-free
    gx = np.asarray(GX)
    xs = [-1.0]                              # one seed configuration in the left well
    ys = [float(E(jnp.asarray([-1.0]))[0])]
    triggers = [-1.0]                        # coordinates where a DFT call fired (seed included)
    mu, sd = fit(jnp.asarray(xs), jnp.asarray(ys))
    t = 0.0
    while t <= 90.0:
        t += 0.045
        walker = walker_at(t)
        gi = int(round((walker - XR[0]) / (XR[1] - XR[0]) * (G - 1)))
        gi = max(0, min(G - 1, gi))
        if sd[gi] > THRESH and len(xs) < NMAX:
            xs.append(walker)
            ys.append(float(E(jnp.asarray([walker]))[0]))
            triggers.append(walker)
            mu, sd = fit(jnp.asarray(xs), jnp.asarray(ys))
    return np.asarray(xs), np.asarray(ys), np.asarray(triggers), mu, sd


def main() -> str:
    xs, ys, triggers, mu, sd = simulate()
    gx = np.asarray(GX)

    # Recovered equilibrium bond length = argmin of the GP posterior mean,
    # compared against the true minimum of the tilted double well.
    r_hat = gx[int(np.argmin(mu))]
    r_true = gx[int(np.argmin(GE))]

    fig = plt.figure(figsize=(5.4, 4.2))
    gs = GridSpec(2, 1, height_ratios=[3, 1.35], hspace=0.28, figure=fig)

    # ---- upper: energy surface, GP posterior, training set -----------------
    ax = fig.add_subplot(gs[0])
    ax.fill_between(gx, np.clip(mu - 2 * sd, YR[0], YR[1]),
                    np.clip(mu + 2 * sd, YR[0], YR[1]),
                    color=S.ACCENT, alpha=0.13, lw=0,
                    label=r"GP posterior $\mu\pm2\sigma$")
    ax.plot(gx, GE, color=S.MUTED, lw=1.1, label="true PES")
    ax.plot(gx, mu, color=S.ACCENT, lw=1.8, label="GP mean")
    ax.scatter(xs, ys, s=26, color=S.INK, edgecolor=S.PAPER, linewidth=0.8,
               zorder=4, label="DFT-computed points")
    ax.axvline(r_hat, color=S.GOOD, lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.annotate(rf"recovered $r_{{\mathrm{{eq}}}}={r_hat:+.3f}$",
                xy=(r_hat, float(np.min(mu))), xytext=(r_hat + 0.14, -0.42),
                color=S.GOOD, fontsize=8, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=S.GOOD, lw=0.8))
    ax.set_xlim(*XR); ax.set_ylim(*YR)
    ax.set_ylabel("energy  $E(x)$")
    ax.set_title(r"On-the-fly ML potential: variance-triggered DFT calls "
                 rf"(RBF $\ell={ELL}$, {len(xs)} points)", color=S.INK)
    ax.legend(loc="upper center", frameon=False, ncol=2, fontsize=7.5,
              handlelength=1.4, columnspacing=1.2)
    S.finish(ax)

    # ---- lower: posterior sigma(x) and the DFT trigger threshold -----------
    axs = fig.add_subplot(gs[1])
    axs.axhline(THRESH, color=S.MUTED, lw=0.9, ls=(0, (4, 3)),
                label=rf"threshold ${THRESH}$")
    axs.plot(gx, sd, color=S.POS, lw=1.4, label=r"$\sigma(x)$")
    axs.scatter(triggers, np.full_like(triggers, -0.03), s=16, marker="^",
                color=S.NEG, clip_on=False, zorder=4, label="DFT triggered")
    axs.set_xlim(*XR); axs.set_ylim(0.0, 1.02)
    axs.set_xlabel("reaction coordinate  $x$")
    axs.set_ylabel(r"$\sigma(x)$")
    axs.legend(loc="upper center", frameon=False, ncol=3, fontsize=7.5,
               handlelength=1.4, columnspacing=1.1)
    S.finish(axs)

    return S.save(fig, "onthefly-mlip")


if __name__ == "__main__":
    print(main())
