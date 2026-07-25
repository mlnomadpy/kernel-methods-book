"""uncertainty-decomposition-highstakes: total uncertainty changes currency out of support."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()


def main() -> str:
    train = jnp.linspace(-1.4, 1.4, 20)
    grid = jnp.linspace(-3.0, 3.0, 280)
    ell, noise = 0.48, 0.12
    kernel = lambda a, b: jnp.exp(-0.5 * ((a[:, None] - b[None, :]) / ell) ** 2)
    gram = kernel(train, train) + noise**2 * jnp.eye(train.size)
    chol = jnp.linalg.cholesky(gram)
    cross = kernel(grid, train)
    solved = jax.scipy.linalg.solve_triangular(chol, cross.T, lower=True)
    epistemic = jnp.maximum(1.0 - jnp.sum(solved**2, axis=0), 0)
    aleatoric = jnp.full_like(grid, noise**2)
    distance = jnp.maximum(jnp.abs(grid) - 1.4, 0)
    shift = 0.22 * distance**2
    total = epistemic + aleatoric + shift
    assert bool(jnp.all(jnp.isfinite(jnp.concatenate((epistemic, aleatoric, shift, total)))))
    assert float(epistemic.min()) >= 0
    assert bool(jnp.allclose(total, epistemic + aleatoric + shift))
    center = jnp.argmin(jnp.abs(grid))
    edge = jnp.argmax(grid)
    assert float(total[edge]) > 8 * float(total[center])
    gx, epi, alea, drift, total_h = S.host(grid, epistemic, aleatoric, shift, total)

    fig, ax = S.new_axes(5.6, 3.0)
    ax.stackplot(gx, alea, epi, drift, colors=(S.MUTED, S.POS, S.ACCENT),
                 alpha=0.68, labels=("aleatoric", "posterior epistemic", "shift allowance"))
    ax.plot(gx, total_h, color=S.INK, lw=1.5, label="declared total")
    ax.axvspan(-1.4, 1.4, color=S.GOOD, alpha=0.07)
    ax.text(0, total_h.max() * 0.08, "training support", ha="center", color=S.GOOD, fontsize=8)
    ax.set(title="Outside support, model variance is not the whole uncertainty budget",
           xlabel="operating condition", ylabel="variance currency", ylim=(0, float(total_h.max()) * 1.04))
    ax.legend(ncol=2, fontsize=7, loc="upper center")
    S.finish(ax)
    return S.save(fig, "uncertainty-decomposition-highstakes")


if __name__ == "__main__":
    print(main())
