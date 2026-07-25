"""confounding-intervention: association can reverse an intervention effect.

The structural model is

    U ~ N(0, 1)
    X = U + eps_x
    Y = beta X + gamma U + eps_y

with beta = -0.8 and gamma = 2.0.  Hidden ``U`` drives both treatment and
outcome, so the observational regression slopes upward even though the
interventional response E[Y | do(X=x)] = beta x slopes downward.  The seed,
sample size, and all coefficients are fixed.  Assertions below fail the build
if the intended sign reversal disappears.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import _style as S

import matplotlib.pyplot as plt

S.apply_style()
jax.config.update("jax_enable_x64", True)

N = 260
BETA = -0.8
GAMMA = 2.0
SIGMA_X = 0.55
SIGMA_Y = 0.55


def make_sample() -> tuple[jax.Array, jax.Array, jax.Array]:
    keys = jax.random.split(jax.random.PRNGKey(35), 3)
    u = jax.random.normal(keys[0], (N,))
    x = u + SIGMA_X * jax.random.normal(keys[1], (N,))
    y = BETA * x + GAMMA * u + SIGMA_Y * jax.random.normal(keys[2], (N,))
    return u, x, y


def main() -> str:
    _, x, y = make_sample()
    design = jnp.column_stack([jnp.ones_like(x), x])
    intercept, observed_slope = jnp.linalg.lstsq(design, y, rcond=None)[0]

    # The exact Gaussian conditional regression slope.
    population_observed_slope = BETA + GAMMA / (1.0 + SIGMA_X**2)
    assert BETA < 0.0
    assert population_observed_slope > 0.0
    assert observed_slope > 0.0
    assert abs(observed_slope - population_observed_slope) < 0.18

    grid = jnp.linspace(-3.0, 3.0, 240)
    observational = population_observed_slope * grid
    interventional = BETA * grid

    assert bool(jnp.all(jnp.isfinite(jnp.concatenate((x, y, observational, interventional)))))
    x, y, grid, observational, interventional = map(
        np.asarray, (x, y, grid, observational, interventional)
    )
    intercept, observed_slope = float(intercept), float(observed_slope)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(6.8, 2.8),
        gridspec_kw={"width_ratios": [1.08, 0.92]},
    )
    ax, ax2 = axes

    ax.scatter(
        x,
        y,
        s=10,
        color=S.MUTED,
        alpha=0.34,
        edgecolor="none",
        rasterized=True,
    )
    ax.plot(
        grid,
        intercept + observed_slope * grid,
        color=S.POS,
        linewidth=2.0,
        label=rf"observational fit, slope ${observed_slope:.2f}$",
    )
    ax.plot(
        grid,
        BETA * grid,
        color=S.ACCENT,
        linewidth=2.0,
        linestyle="--",
        label=rf"causal response, slope ${BETA:.1f}$",
    )
    ax.axhline(0.0, color=S.RULE, linewidth=0.7, zorder=0)
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel("treatment $X$")
    ax.set_ylabel("outcome $Y$")
    ax.set_title("Observed data mix effect and confounding")
    ax.legend(loc="upper left", frameon=False, handlelength=2.4)
    S.finish(ax)

    ax2.plot(
        grid,
        observational,
        color=S.POS,
        linewidth=2.2,
        marker="o",
        markevery=[36, 120, 203],
        markersize=3.8,
        label=r"$\mathbb{E}[Y\mid X=x]$",
    )
    ax2.plot(
        grid,
        interventional,
        color=S.ACCENT,
        linewidth=2.2,
        linestyle="--",
        marker="s",
        markevery=[36, 120, 203],
        markersize=3.6,
        label=r"$\mathbb{E}[Y\mid\mathrm{do}(X=x)]$",
    )
    ax2.axhline(0.0, color=S.RULE, linewidth=0.7, zorder=0)
    ax2.axvline(0.0, color=S.RULE, linewidth=0.7, zorder=0)
    ax2.set_xlim(-3.0, 3.0)
    ax2.set_ylim(-3.0, 3.0)
    ax2.set_xlabel("queried value $x$")
    ax2.set_ylabel("expected outcome")
    ax2.set_title("Conditioning is not intervention")
    ax2.legend(loc="upper center", frameon=False, handlelength=2.4)
    ax2.annotate(
        "opposite signs",
        xy=(2.2, observational[np.searchsorted(grid, 2.2)]),
        xytext=(0.75, 0.55),
        arrowprops={"arrowstyle": "->", "color": S.MUTED, "lw": 0.8},
        color=S.MUTED,
        fontsize=8,
    )
    S.finish(ax2)

    fig.tight_layout(w_pad=1.7)
    return S.save(fig, "confounding-intervention")


if __name__ == "__main__":
    print(main())
