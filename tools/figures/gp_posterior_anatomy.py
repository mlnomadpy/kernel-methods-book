"""Posterior mean and uncertainty of a one-dimensional Gaussian process."""
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import _style as S

S.apply_style()

x_train_jax = jnp.array([-1.55, -0.45, 0.35, 1.45])
y_train_jax = jnp.array([-0.72, 0.18, 0.64, 0.05])
x_jax = jnp.linspace(-2.5, 2.5, 360)
lengthscale = 0.72
noise = 0.09


def kernel(a, b):
    return jnp.exp(-0.5 * ((a[:, None] - b[None, :]) / lengthscale) ** 2)


k_xx = kernel(x_train_jax, x_train_jax) + noise**2 * jnp.eye(x_train_jax.size)
chol = jnp.linalg.cholesky(k_xx)
alpha = jax.scipy.linalg.cho_solve((chol, True), y_train_jax)
k_s = kernel(x_jax, x_train_jax)
mean = k_s @ alpha
v = jax.scipy.linalg.solve_triangular(chol, k_s.T, lower=True)
variance = jnp.maximum(1.0 - jnp.sum(v * v, axis=0), 0.0)
std = jnp.sqrt(variance)

near = jnp.min(jnp.abs(x_jax[:, None] - x_train_jax[None, :]), axis=1) < 0.08
far = jnp.abs(x_jax) > 2.25
residual = k_xx @ alpha - y_train_jax
assert bool(jnp.all(jnp.isfinite(mean))) and bool(jnp.all(jnp.isfinite(std)))
assert float(jnp.linalg.norm(residual)) < 1e-12
assert float(std[far].mean()) > 2.0 * float(std[near].mean())
x_train, y_train, x, mean, std, near, far = map(
    np.asarray, (x_train_jax, y_train_jax, x_jax, mean, std, near, far)
)

fig, ax = S.new_axes(5.4, 3.15)
ax.fill_between(x, -1.96, 1.96, color=S.RULE, alpha=0.22, label="prior 95% band")
ax.fill_between(x, mean - 1.96 * std, mean + 1.96 * std, color=S.POS, alpha=0.18, label="posterior 95% band")
ax.plot(x, mean, color=S.ACCENT, lw=2.2, label="posterior mean")
ax.scatter(x_train, y_train, s=32, facecolor=S.PAPER, edgecolor=S.INK, zorder=4, label="observations")
ax.annotate(
    "data shrink variance",
    xy=(0.35, mean[np.argmin(np.abs(x - 0.35))] + 0.12),
    xytext=(0.82, 1.35),
    color=S.ACCENT,
    fontsize=8,
    arrowprops={"arrowstyle": "-", "color": S.ACCENT, "lw": 0.8},
)
ax.annotate(
    "mean reverts;\nuncertainty returns",
    xy=(2.35, mean[-12] + 1.25 * std[-12]),
    xytext=(1.35, -1.38),
    color=S.MUTED,
    fontsize=8,
    arrowprops={"arrowstyle": "-", "color": S.MUTED, "lw": 0.8},
)
ax.set(xlabel="input $x$", ylabel="latent function $f(x)$", ylim=(-2.15, 2.15))
ax.legend(frameon=False, ncol=2, loc="lower center")
S.finish(ax)
S.save(fig, "gp-posterior-anatomy")
print(f"near_std={std[near].mean():.4f}; far_std={std[far].mean():.4f}")
