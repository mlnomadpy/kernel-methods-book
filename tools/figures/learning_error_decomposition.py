"""learning-error-decomposition: approximation, estimation, and optimization."""
from __future__ import annotations

import _style as S
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

S.apply_style()


def main() -> str:
    budget = jnp.geomspace(0.12, 12.0, 500)
    approximation = 0.62 / (1.0 + 1.8 * budget)
    estimation = 0.035 * budget**0.72
    optimization = 0.17 * jnp.exp(-0.55 * budget) + 0.012
    total = approximation + estimation + optimization
    optimum = jnp.argmin(total)
    assert int(optimum) not in (0, budget.size - 1)
    assert bool(jnp.all(approximation[1:] < approximation[:-1]))
    assert bool(jnp.all(estimation[1:] > estimation[:-1]))
    x, a, e, o, t = S.host(budget, approximation, estimation, optimization, total)
    fig, ax = S.new_axes(5.8, 3.2)
    ax.semilogx(x, a, color=S.POS, label="approximation")
    ax.semilogx(x, e, color=S.ACCENT, label="estimation")
    ax.semilogx(x, o, color=S.VIOLET, label="optimization")
    ax.semilogx(x, t, color=S.INK, lw=2.2, label="total")
    i = int(optimum)
    ax.scatter([x[i]], [t[i]], color=S.GOOD, zorder=5)
    ax.annotate("balanced operating point", xy=(x[i], t[i]), xytext=(x[i] * 1.7, t[i] + .15),
                arrowprops={"arrowstyle": "->", "color": S.GOOD}, color=S.GOOD, fontsize=8)
    ax.set(xlabel="effective model and computation budget", ylabel="excess-risk contribution", ylim=(0, .75))
    ax.legend(ncol=2)
    S.finish(ax)
    return S.save(fig, "learning-error-decomposition")


if __name__ == "__main__":
    print(main())
