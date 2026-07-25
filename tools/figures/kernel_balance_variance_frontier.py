"""Kernel balancing trades RKHS imbalance against unstable weights."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import _style as S
import matplotlib.pyplot as plt

S.apply_style()
x = jnp.linspace(-2.5, 1.1, 65)
z = jnp.linspace(-1.2, 2.5, 65)
K = jnp.exp(-0.5 * ((x[:, None] - x[None, :]) / .75) ** 2)
kxz = jnp.exp(-0.5 * ((x[:, None] - z[None, :]) / .75) ** 2)
mu = jnp.mean(kxz, axis=1)
lams = jnp.logspace(-6, 1, 90)

def solve(lam):
    A = K + lam * jnp.eye(x.size)
    L = jnp.linalg.cholesky(A)
    w = jax.scipy.linalg.cho_solve((L, True), mu)
    w = w / jnp.sum(w)
    imbalance2 = 1.0 - 2 * w @ mu + w @ K @ w
    ess = 1 / jnp.sum(w**2)
    return jnp.sqrt(jnp.maximum(imbalance2, 0)), ess, jnp.max(jnp.abs(w))

imb, ess, maxw = jax.vmap(solve)(lams)
assert bool(jnp.all(jnp.isfinite(jnp.stack([imb, ess, maxw]))))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
axes[0].plot(S.host(imb), S.host(ess), color=S.POS)
axes[0].scatter(S.host(imb[::14]), S.host(ess[::14]), c=S.host(jnp.log10(lams[::14])),
                cmap=S.HEAT, s=24, zorder=3)
axes[0].set(xlabel="RKHS imbalance", ylabel="effective sample size",
            title="Exact balance is purchased with variance")
axes[1].loglog(S.host(lams), S.host(maxw), color=S.ACCENT, label=r"$\max_i|w_i|$")
axes[1].loglog(S.host(lams), S.host(1 / ess), color=S.POS, label="inverse ESS")
axes[1].set(xlabel="weight penalty", ylabel="instability diagnostic",
            title="Regularization controls weight concentration")
axes[1].legend()
for ax in axes: S.finish(ax)
S.save(fig, "kernel-balance-variance-frontier")
