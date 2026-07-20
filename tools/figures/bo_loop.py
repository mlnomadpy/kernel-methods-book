"""bo-loop: one static plate of the Bayesian optimization loop.

Reproduces ``WIDGETS["bo-loop"]`` in ``public/assets/viz-bo-loop.js`` at its
default state. A hidden 1-D objective is queried one point at a time; after
every observation an RBF Gaussian-process posterior is refit through a Cholesky
solve, the acquisition (default GP-UCB) is scored on a 200-point grid over
[0, 1], and its argmax becomes the next query.

Everything here matches the widget exactly:

* objective   f(x) = sin(7x)(1 - x) + 0.6 exp(-((x-0.8)/0.08)^2)
* kernel      RBF, lengthscale ell = 0.08, unit prior variance
* GP noise    NOISE = 1e-3 added to the Gram diagonal (the Cholesky solve)
* obs. noise  y_i = f(x_i) + 0.02 * eps_i
* acquisition GP-UCB, alpha(x) = mu(x) + sqrt(beta) * sigma(x), sqrt(beta) = 2

Determinism: the widget seeds its own PRNG at ``reset`` (``seed = 20260719``)
and replays the identical run every time, so faithful reproduction means
replaying that exact seeded stream of observation noise rather than drawing
fresh randomness -- we reimplement its mulberry32 generator below. The result
is byte-stable across builds. (The initial design is the widget's fixed
(0.15, 0.55); we then run six BO steps to a representative 8-query state.)

The GP posterior mean mu +/- 2 sigma band (top) and the acquisition with its
argmax = pending next query (bottom) are the same quantities the widget draws.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()

# ---- widget constants (viz-bo-loop.js) ------------------------------------
G = 200        # posterior evaluation grid on [0, 1]
NOISE = 1e-3   # GP noise variance added to the Gram diagonal
OBS = 0.02     # observation-noise std on each query
ELL = 0.08     # RBF lengthscale (default slider value)
SQRT_BETA = 2.0  # GP-UCB exploration weight sqrt(beta) (default slider value)
SEED = 20260719  # widget's reset seed
INIT = (0.15, 0.55)  # fixed initial design
N_STEPS = 6    # BO steps after the initial design -> 8 total queries


def f_obj(x):
    """Hidden objective: a sine ridge plus a narrow bump near x = 0.8."""
    u = (x - 0.8) / 0.08
    return jnp.sin(7.0 * x) * (1.0 - x) + 0.6 * jnp.exp(-u * u)


# ---- widget PRNG (mulberry32), replayed exactly ---------------------------
# The browser widget uses this seeded stream for its observation noise so a
# reset replays the identical run; we mirror the 32-bit integer arithmetic.
_M = 0xFFFFFFFF


def _u32(x):
    return x & _M


def _i32(x):
    x &= _M
    return x - 0x100000000 if x & 0x80000000 else x


def _imul(a, b):
    return _i32((_u32(a) * _u32(b)) & _M)


class WidgetRNG:
    """Deterministic reimplementation of viz-bo-loop.js's rnd()/nrand()."""

    def __init__(self, seed: int):
        self.s = _i32(seed)

    def rnd(self) -> float:
        s = _i32(_u32(self.s) + 0x6D2B79F5)
        self.s = s
        su = _u32(s)
        t = _imul(su ^ (su >> 15), 1 | su)
        tu = _u32(t)
        inner = _imul(tu ^ (tu >> 7), _i32(61 | tu))
        a = _i32(t + inner)
        t2 = _i32(_u32(a) ^ _u32(t))
        tu2 = _u32(t2)
        return ((tu2 ^ (tu2 >> 14)) & _M) / 4294967296.0

    def nrand(self) -> float:
        u = self.rnd() or 1e-12
        v = self.rnd()
        return math.sqrt(-2.0 * math.log(u)) * math.cos(6.2831853 * v)


# ---- exact GP posterior on the grid ---------------------------------------
def _rbf(a: jnp.ndarray, b: jnp.ndarray, ell: float) -> jnp.ndarray:
    d = a[:, None] - b[None, :]
    return jnp.exp(-(d ** 2) / (2.0 * ell * ell))


def gp_posterior(xs, ys, grid, ell, noise):
    """Posterior mean/std on the grid via one Cholesky factorization.

    mu(x)   = k(x, X) (K + noise I)^{-1} y
    var(x)  = k(x, x) - k(x, X) (K + noise I)^{-1} k(X, x),   k(x, x) = 1
    """
    n = xs.shape[0]
    K = _rbf(xs, xs, ell) + noise * jnp.eye(n)
    L = jnp.linalg.cholesky(K)
    alpha = jax.scipy.linalg.cho_solve((L, True), ys)          # (K+noise I)^{-1} y
    Ks = _rbf(grid, xs, ell)                                    # (G, n)
    mu = Ks @ alpha
    W = jax.scipy.linalg.cho_solve((L, True), Ks.T)             # (n, G)
    q = jnp.sum(Ks.T * W, axis=0)                               # (G,)
    var = jnp.clip(1.0 - q, 1e-12, None)
    return mu, jnp.sqrt(var)


def run_loop():
    """Replay the widget's default run to an 8-query state; return the state."""
    grid = jnp.linspace(0.0, 1.0, G)
    rng = WidgetRNG(SEED)

    xs, ys = [], []

    def add_obs(x: float) -> None:
        xs.append(x)
        ys.append(float(f_obj(jnp.asarray(x))) + OBS * rng.nrand())

    for x0 in INIT:                     # fixed initial design
        add_obs(x0)

    # each BO step: refit, take the acquisition argmax, query it
    next_idx = 0
    for _ in range(N_STEPS + 1):        # +1 so the final state also has a pending query
        xa = jnp.asarray(xs)
        ya = jnp.asarray(ys)
        mu, sd = gp_posterior(xa, ya, grid, ELL, NOISE)
        acq = mu + SQRT_BETA * sd       # GP-UCB
        next_idx = int(jnp.argmax(acq))
        if len(xs) >= len(INIT) + N_STEPS:
            break                        # keep the pending query; stop querying
        add_obs(float(grid[next_idx]))

    return {
        "grid": np.asarray(grid),
        "xs": np.asarray(xs),
        "ys": np.asarray(ys),
        "mu": np.asarray(mu),
        "sd": np.asarray(sd),
        "acq": np.asarray(acq),
        "next_x": float(grid[next_idx]),
        "ftrue": np.asarray(f_obj(grid)),
    }


def main() -> str:
    st = run_loop()
    grid = st["grid"]
    mu, sd = st["mu"], st["sd"]
    lo, hi = mu - 2.0 * sd, mu + 2.0 * sd

    # incumbent (best observed) query
    best = int(np.argmax(st["ys"]))

    fig = plt.figure(figsize=(5.4, 3.9))
    gs = GridSpec(2, 1, height_ratios=[2.5, 1.0], hspace=0.32, figure=fig)
    ax = fig.add_subplot(gs[0])
    axa = fig.add_subplot(gs[1], sharex=ax)

    # --- top panel: objective (faint), GP posterior mu +/- 2 sigma, queries ---
    ax.axhline(0.0, color=S.RULE, lw=0.8, ls=(0, (3, 3)))
    ax.plot(grid, st["ftrue"], color=S.MUTED, lw=1.3, alpha=0.45,
            label="objective $f$")
    ax.fill_between(grid, lo, hi, color=S.ACCENT, alpha=0.15, lw=0,
                    label=r"$\mu\pm2\sigma$")
    ax.plot(grid, mu, color=S.ACCENT, lw=2.2, label=r"GP mean $\mu$")
    ax.scatter(st["xs"], st["ys"], s=34, color=S.INK,
               edgecolor=S.PAPER, linewidth=0.9, zorder=4)
    ax.scatter([st["xs"][best]], [st["ys"][best]], s=120, facecolors="none",
               edgecolor=S.GOOD, linewidth=1.7, zorder=5, label="incumbent best")
    ax.axvline(st["next_x"], color=S.ACCENT, lw=1.4, ls=(0, (5, 4)), zorder=2)
    ax.set_ylim(-1.5, 1.5)
    ax.set_ylabel("$f(x)$")
    ax.set_title("The Bayesian optimization loop: GP posterior after 8 queries",
                 color=S.INK)
    ax.legend(loc="lower left", frameon=False, ncol=2, fontsize=7.5,
              handlelength=1.4, columnspacing=1.2)
    ax.tick_params(labelbottom=False)
    S.finish(ax)

    # --- bottom panel: acquisition with its argmax = next query ---------------
    axa.plot(grid, st["acq"], color=S.GOOD, lw=1.9)
    axa.axvline(st["next_x"], color=S.ACCENT, lw=1.4, ls=(0, (5, 4)),
                label=f"next query $x={st['next_x']:.3f}$")
    axa.scatter([st["next_x"]], [st["acq"][int(np.argmax(st["acq"]))]],
                s=30, color=S.ACCENT, edgecolor=S.PAPER, linewidth=0.8, zorder=5)
    axa.set_xlim(0.0, 1.0)
    axa.set_xlabel("$x$")
    axa.set_ylabel(r"$\mu+\sqrt{\beta}\,\sigma$")
    axa.set_title(r"GP-UCB acquisition, $\sqrt{\beta}=2$", color=S.MUTED,
                  fontsize=8.5, pad=4)
    axa.legend(loc="upper left", frameon=False, fontsize=7.5, handlelength=1.4)
    S.finish(axa)

    return S.save(fig, "bo-loop")


if __name__ == "__main__":
    print(main())
