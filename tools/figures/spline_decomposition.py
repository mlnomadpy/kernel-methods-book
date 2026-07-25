"""spline-decomposition: an unpenalized trend and a penalized bend play different roles."""
from __future__ import annotations

import jax
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import _style as S
import matplotlib.pyplot as plt

S.apply_style()


def main() -> str:
    x = jnp.linspace(0.0, 1.0, 400, dtype=jnp.float64)
    null = 0.35 + 0.95 * x
    penalized = 0.23 * jnp.sin(2 * jnp.pi * x) - 0.08 * jnp.sin(4 * jnp.pi * x)
    total = null + penalized
    assert bool(jnp.all(jnp.isfinite(jnp.stack((null, penalized, total)))))
    assert float(jnp.max(jnp.abs(total - null - penalized))) < 1e-14
    assert float(jnp.max(jnp.abs(jnp.diff(null, n=2)))) < 1e-12
    x_h, null_h, penalized_h, total_h = map(np.asarray, (x, null, penalized, total))
    fig, axes = plt.subplots(1, 3, figsize=(6.2, 2.25), sharex=True, sharey=True)
    specs = (
        (null_h, "null-space trend", S.POS, "-"),
        (penalized_h, "penalized bend", S.ACCENT, "--"),
        (total_h, "spline fit", S.INK, "-"),
    )
    for ax, (curve, title, color, style) in zip(axes, specs):
        ax.axhline(0, color=S.RULE, lw=0.7)
        ax.plot(x_h, curve, color=color, lw=1.8, ls=style)
        ax.set_title(title)
        ax.set_xlabel(r"$x$")
        S.finish(ax)
    axes[0].set_ylabel("function value")
    axes[2].plot(x_h, null_h, color=S.POS, lw=1.0, alpha=0.65)
    fig.subplots_adjust(wspace=0.12)
    return S.save(fig, "spline-decomposition")


if __name__ == "__main__":
    print(main())
