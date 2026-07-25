"""Kernel Bayes weights are signed regularized coordinates, not probabilities."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import _style as S
import matplotlib.pyplot as plt

S.apply_style()
x = jnp.linspace(-2.8, 2.8, 55)
K = jnp.exp(-0.5 * ((x[:, None] - x[None, :]) / 0.65) ** 2)
prior = jnp.exp(-0.5 * ((x + 0.4) / 1.1) ** 2)
likelihood = 0.25 + jnp.exp(-0.5 * ((x - 1.15) / 0.42) ** 2)
target = prior * likelihood
target = target / jnp.max(target)
lams = jnp.logspace(-7, -0.4, 75)

def weights(lam):
    A = K + x.size * lam * jnp.eye(x.size)
    L = jnp.linalg.cholesky(A)
    w = jax.scipy.linalg.cho_solve((L, True), target)
    return jnp.sum(jnp.maximum(-w, 0)), jnp.sum(w), jnp.linalg.norm(w)

negative, sums, norms = jax.vmap(weights)(lams)
assert bool(jnp.all(jnp.isfinite(jnp.stack([negative, sums, norms]))))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
axes[0].semilogx(S.host(lams), S.host(negative), color=S.ACCENT, label="negative mass")
axes[0].semilogx(S.host(lams), S.host(jnp.abs(sums - 1)), color=S.POS, label=r"$|\sum_iw_i-1|$")
axes[0].set(xlabel=r"ridge $\lambda$", ylabel="departure from probability weights",
            title="Coordinates need not form a simplex")
axes[0].legend()
for lam, color, label in [(1e-6, S.ACCENT, r"$10^{-6}$"), (1e-2, S.POS, r"$10^{-2}$")]:
    A = K + x.size * lam * jnp.eye(x.size)
    L = jnp.linalg.cholesky(A)
    w = jax.scipy.linalg.cho_solve((L, True), target)
    axes[1].plot(S.host(x), S.host(w), color=color, label=label)
axes[1].axhline(0, color=S.RULE, lw=.8)
axes[1].set(xlabel="training location", ylabel="empirical weight",
            title="Small ridge exposes signed oscillations")
axes[1].legend(title=r"$\lambda$")
for ax in axes: S.finish(ax)
S.save(fig, "kernel-bayes-weight-path")
