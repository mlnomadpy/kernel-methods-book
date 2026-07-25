"""svm-kkt-convergence: objective progress alone is not a stopping certificate."""
from __future__ import annotations

import _style as S
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

S.apply_style()


def main() -> str:
    sweep = jnp.arange(0, 61, dtype=jnp.float64)
    dual_gap = 1.8 * jnp.exp(-0.145 * sweep) + .012 * jnp.exp(-.025 * sweep)
    kkt = 1.25 * jnp.exp(-0.105 * sweep) * (1 + .08 * jnp.cos(.7 * sweep))
    kkt = jnp.maximum.accumulate(kkt[::-1])[::-1]
    objective = 1.0 - dual_gap / dual_gap[0]
    tolerance = 1e-2
    assert bool(jnp.all(jnp.diff(objective) >= -1e-12))
    assert bool(jnp.all(jnp.diff(kkt) <= 1e-12))
    assert float(kkt[-1]) < tolerance
    x, obj, gap, residual = S.host(sweep, objective, dual_gap, kkt)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9))
    axes[0].plot(x, obj, color=S.POS)
    axes[0].set(xlabel="solver sweep", ylabel="normalized dual objective", ylim=(-.03, 1.04))
    axes[0].annotate("looks flat early", xy=(27, obj[27]), xytext=(10, .72),
                     arrowprops={"arrowstyle": "->", "color": S.MUTED}, color=S.MUTED, fontsize=8)
    axes[1].semilogy(x, gap, color=S.ACCENT, label="duality gap")
    axes[1].semilogy(x, residual, color=S.POS, label="maximum KKT violation")
    axes[1].axhline(tolerance, color=S.INK, ls=":", label=r"declared tolerance $\tau$")
    axes[1].set(xlabel="solver sweep", ylabel="optimality certificate")
    axes[1].legend()
    for ax in axes: S.finish(ax)
    fig.tight_layout()
    return S.save(fig, "svm-kkt-convergence")


if __name__ == "__main__":
    print(main())
