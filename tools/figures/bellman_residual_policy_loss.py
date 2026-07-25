"""bellman-residual-policy-loss: residual-to-value conversion amplifies near gamma one."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import _style as S

S.apply_style()
jax.config.update("jax_enable_x64", True)


def main() -> str:
    gammas = jnp.linspace(.05, .98, 180)
    residuals = jnp.array([.01, .03, .1])
    bounds = residuals[:, None] / (1 - gammas[None, :])
    S.require_finite(gammas=gammas, bounds=bounds)
    assert bool(jnp.all(jnp.diff(bounds, axis=1) > 0))
    assert float(bounds[0, -1]) > 40 * float(bounds[0, 0])
    gammas, bounds = S.host(gammas, bounds)
    fig, ax = S.new_axes()
    for curve, color, label in zip(bounds, (S.GOOD, S.ACCENT, S.NEG),
                                   (r"$\varepsilon=.01$", r"$\varepsilon=.03$", r"$\varepsilon=.10$")):
        ax.plot(gammas, curve, color=color, label=label)
    ax.set_yscale("log")
    ax.set(xlabel=r"discount $\gamma$", ylabel=r"value-error certificate $\varepsilon/(1-\gamma)$",
           title="A small Bellman residual is not an absolute guarantee")
    ax.legend(title="residual")
    S.finish(ax)
    return S.save(fig, "bellman-residual-policy-loss")


if __name__ == "__main__":
    print(main())
