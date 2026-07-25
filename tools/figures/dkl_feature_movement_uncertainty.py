"""dkl-feature-movement-uncertainty: evidence can compress features and OOD variance."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()


def main() -> str:
    x = jnp.linspace(-1.0, 1.0, 28)
    y = jnp.ones_like(x)
    noise = 0.08

    def diagnostics(log_scale):
        scale = jnp.exp(log_scale)
        z = scale * x
        gram = jnp.exp(-0.5 * (z[:, None] - z[None, :]) ** 2)
        cov = gram + noise**2 * jnp.eye(x.size)
        chol = jnp.linalg.cholesky(cov)
        alpha = jax.scipy.linalg.cho_solve((chol, True), y)
        nll = 0.5 * y @ alpha + jnp.sum(jnp.log(jnp.diag(chol)))
        z_star = scale * 2.5
        k_star = jnp.exp(-0.5 * (z_star - z) ** 2)
        variance = 1.0 - k_star @ jax.scipy.linalg.cho_solve((chol, True), k_star)
        return nll, variance, scale

    def step(log_scale, _):
        gradient = jax.grad(lambda q: diagnostics(q)[0])(log_scale)
        return jnp.clip(log_scale - 0.035 * gradient, -3.0, 1.0), diagnostics(log_scale)

    final, history = jax.lax.scan(step, jnp.array(0.8), None, length=90)
    nll, variance, scale = history
    assert bool(jnp.all(jnp.isfinite(jnp.concatenate((nll, variance, scale)))))
    assert float(nll[-1]) < float(nll[0])
    assert float(scale[-1]) < 0.35 * float(scale[0])
    assert float(variance[-1]) < float(variance[0])
    assert float(variance.min()) >= -1e-10
    epochs, nll_h, var_h, scale_h = S.host(
        jnp.arange(nll.size), nll - nll.min(), variance, scale
    )

    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.55))
    axes[0].plot(epochs, scale_h, color=S.POS)
    axes[0].set(title="Learned feature scale contracts", xlabel="optimization step",
                ylabel="feature scale")
    axes[1].plot(epochs, var_h, color=S.ACCENT, label="OOD posterior variance")
    axes[1].plot(epochs, nll_h / max(float(nll_h.max()), 1e-12),
                 color=S.MUTED, ls="--", label="normalized NLL gap")
    axes[1].set(title="OOD variance contracts with it", xlabel="optimization step",
                ylabel="relative diagnostic")
    axes[1].legend()
    for ax in axes:
        S.finish(ax)
    fig.subplots_adjust(wspace=0.28)
    return S.save(fig, "dkl-feature-movement-uncertainty")


if __name__ == "__main__":
    print(main())
