"""Regularization controls the stability of empirical CME weights."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import _style as S
import matplotlib.pyplot as plt

S.apply_style()
x = jnp.linspace(-2.5, 2.5, 60)
xstar = 0.35
K = jnp.exp(-0.5 * ((x[:, None] - x[None, :]) / 0.55) ** 2)
kx = jnp.exp(-0.5 * ((x - xstar) / 0.55) ** 2)
lams = jnp.logspace(-7, -0.2, 90)

def solve(lam):
    A = K + x.size * lam * jnp.eye(x.size)
    L = jnp.linalg.cholesky(A)
    beta = jax.scipy.linalg.cho_solve((L, True), kx)
    return jnp.linalg.norm(beta), jnp.sum(jnp.maximum(-beta, 0)), jnp.linalg.cond(A)

norms, neg, cond = jax.vmap(solve)(lams)
assert bool(jnp.all(jnp.isfinite(jnp.stack([norms, neg, cond]))))
assert float(cond[0]) > float(cond[-1])

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
axes[0].loglog(S.host(lams), S.host(norms), color=S.POS, label=r"$\|\beta_\lambda\|_2$")
axes[0].loglog(S.host(lams), S.host(neg + 1e-12), color=S.ACCENT, label="negative mass")
axes[0].set(xlabel=r"ridge $\lambda$", ylabel="weight diagnostic", title="Regularization suppresses cancellation")
axes[0].legend()
axes[1].loglog(S.host(lams), S.host(cond), color=S.INK)
axes[1].axhline(1e6, color=S.ACCENT, ls=":", lw=1)
axes[1].set(xlabel=r"ridge $\lambda$", ylabel=r"$\mathrm{cond}(K+n\lambda I)$",
            title="The spectral floor stabilizes the solve")
for ax in axes: S.finish(ax)
S.save(fig, "cme-regularization-path")
