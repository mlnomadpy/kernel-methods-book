"""hilbert-vs-variation: quadratic and atomic norms select different coefficient geometry."""
from __future__ import annotations

import _style as S
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

S.apply_style()


def main() -> str:
    x_jax = jnp.linspace(-1.5, 1.5, 180)
    centers_jax = jnp.linspace(-1.1, 1.1, 15)
    phi = jnp.exp(-0.5 * ((x_jax[:, None] - centers_jax[None, :]) / 0.34) ** 2)
    target_jax = (
        0.95 * jnp.exp(-0.5 * ((x_jax + 0.62) / 0.34) ** 2)
        + 0.72 * jnp.exp(-0.5 * ((x_jax - 0.55) / 0.34) ** 2)
    )
    system = phi.T @ phi + 0.9 * jnp.eye(phi.shape[1])
    chol = jnp.linalg.cholesky(system)
    ridge_jax = jax.scipy.linalg.cho_solve((chol, True), phi.T @ target_jax)
    lipschitz = jnp.linalg.norm(phi, 2) ** 2
    step, penalty = 1.0 / lipschitz, 0.22

    def ista_step(coef, _):
        proposal = coef - step * (phi.T @ (phi @ coef - target_jax))
        updated = jnp.sign(proposal) * jnp.maximum(jnp.abs(proposal) - step * penalty, 0.0)
        return updated, None

    sparse_jax, _ = jax.lax.scan(ista_step, jnp.zeros(phi.shape[1]), xs=None, length=4000)
    ridge_fit, sparse_fit = phi @ ridge_jax, phi @ sparse_jax
    assert bool(jnp.all(jnp.isfinite(jnp.concatenate((ridge_jax, sparse_jax, ridge_fit, sparse_fit)))))
    assert float(jnp.linalg.norm(system @ ridge_jax - phi.T @ target_jax)) < 1e-9
    assert int(jnp.count_nonzero(jnp.abs(ridge_jax) > 1e-3)) >= 10
    assert int(jnp.count_nonzero(jnp.abs(sparse_jax) > 1e-3)) <= 5
    assert float(jnp.mean((sparse_fit - target_jax) ** 2)) < 0.01
    x, centers, Phi, target, ridge, sparse, ridge_fit, sparse_fit = map(
        np.asarray,
        (x_jax, centers_jax, phi, target_jax, ridge_jax, sparse_jax, ridge_fit, sparse_fit),
    )
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.65))
    axes[0].plot(x, target, color=S.INK, lw=1.4, label="target")
    axes[0].plot(x, ridge_fit, color=S.POS, lw=1.6, label="quadratic norm")
    axes[0].plot(x, sparse_fit, color=S.ACCENT, lw=1.6, ls="--", label="variation norm")
    axes[0].set(xlabel=r"$x$", ylabel="function value", title="Similar fits")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].stem(centers - 0.025, ridge, linefmt=S.POS, markerfmt="o", basefmt=" ", label="quadratic")
    axes[1].stem(centers + 0.025, sparse, linefmt=S.ACCENT, markerfmt="s", basefmt=" ", label="atomic")
    axes[1].set(xlabel="atom location", ylabel="coefficient", title="Different geometry")
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        S.finish(ax)
    fig.subplots_adjust(wspace=0.26)
    return S.save(fig, "hilbert-vs-variation")


if __name__ == "__main__":
    print(main())
