"""rkbs-sparsity-path: atomic regularization trades fit for support size."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()


def main() -> str:
    x = jnp.linspace(-1.0, 1.0, 72)
    centers = jnp.linspace(-1.15, 1.15, 31)
    design = jnp.exp(-0.5 * ((x[:, None] - centers[None, :]) / 0.16) ** 2)
    truth = 1.1 * design[:, 7] - 0.75 * design[:, 22] + 0.45 * design[:, 15]
    lipschitz = jnp.linalg.eigvalsh(design.T @ design)[-1]
    step = 0.95 / lipschitz
    penalties = jnp.geomspace(0.015, 1.2, 28)

    def solve(penalty):
        def update(coef, _):
            proposal = coef - step * design.T @ (design @ coef - truth)
            coef = jnp.sign(proposal) * jnp.maximum(jnp.abs(proposal) - step * penalty, 0)
            return coef, None
        coef, _ = jax.lax.scan(update, jnp.zeros(centers.size), None, length=2200)
        residual = jnp.sqrt(jnp.mean((design @ coef - truth) ** 2))
        support = jnp.sum(jnp.abs(coef) > 2e-3)
        return residual, support

    residual, support = jax.vmap(solve)(penalties)
    assert bool(jnp.all(jnp.isfinite(jnp.concatenate((penalties, residual)))))
    assert int(support[0]) > int(support[-1])
    assert float(residual[0]) < float(residual[-1])
    assert bool(jnp.all(support[1:] <= support[:-1] + 1))
    lam, err, active = S.host(penalties, residual, support)

    fig, ax = S.new_axes(5.2, 2.9)
    ax.plot(active, err, color=S.POS, marker="o", ms=3)
    for index in (0, 9, 18, 27):
        ax.annotate(rf"$\lambda={lam[index]:.2g}$", (active[index], err[index]),
                    xytext=(4, 5), textcoords="offset points", fontsize=7, color=S.MUTED)
    ax.invert_xaxis()
    ax.set(title="Atomic regularization traces a fit–sparsity frontier",
           xlabel="active atoms", ylabel="training RMSE")
    S.finish(ax)
    return S.save(fig, "rkbs-sparsity-path")


if __name__ == "__main__":
    print(main())
