"""Paired views before and after regularized linear CCA."""
import jax
import jax.numpy as jnp
import numpy as np

import _style as S

import matplotlib.pyplot as plt

S.apply_style()
jax.config.update("jax_enable_x64", True)
n = 70
keys = jax.random.split(jax.random.PRNGKey(19), 5)
latent = jax.random.normal(keys[0], (n,))
X = jnp.column_stack((latent + 0.20 * jax.random.normal(keys[1], (n,)), 2.5 * jax.random.normal(keys[2], (n,))))
Y = jnp.column_stack((0.85 * latent + 0.20 * jax.random.normal(keys[3], (n,)), 2.2 * jax.random.normal(keys[4], (n,))))
X = X - X.mean(axis=0)
Y = Y - Y.mean(axis=0)

ridge = 0.18
Cxx = X.T @ X / n + ridge * jnp.eye(2)
Cyy = Y.T @ Y / n + ridge * jnp.eye(2)
Cxy = X.T @ Y / n

def invsqrt(matrix: jax.Array) -> jax.Array:
    values, vectors = jnp.linalg.eigh(matrix)
    assert float(values.min()) > 0.0
    return (vectors * values**-0.5) @ vectors.T

Wx, Wy = invsqrt(Cxx), invsqrt(Cyy)
left, _, right_t = jnp.linalg.svd(Wx @ Cxy @ Wy, full_matrices=False)
wx = Wx @ left[:, 0]
wy = Wy @ right_t.T[:, 0]
sx, sy = X @ wx, Y @ wy
correlation = lambda a, b: jnp.vdot(a - a.mean(), b - b.mean()) / (jnp.linalg.norm(a - a.mean()) * jnp.linalg.norm(b - b.mean()))
sy = jnp.where(correlation(sx, sy) < 0.0, -sy, sy)
corr = float(correlation(sx, sy))
assert bool(jnp.all(jnp.isfinite(jnp.concatenate((X.ravel(), Y.ravel(), sx, sy)))))
assert corr > 0.9

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
X, Y, sx, sy, latent = map(np.asarray, (X, Y, sx, sy, latent))
color = np.where(latent >= 0, S.POS, S.NEG)
axes[0].scatter(X[:, 0], X[:, 1], c=color, s=15, alpha=0.85)
axes[1].scatter(Y[:, 0], Y[:, 1], c=color, s=15, alpha=0.85, marker="^")
axes[0].set(title="View A", xlabel="$x_1$", ylabel="$x_2$")
axes[1].set(title="View B", xlabel="$y_1$", ylabel="$y_2$")
axes[2].scatter(sx, sy, c=color, s=16, alpha=0.9)
limit = 1.05 * max(np.max(np.abs(sx)), np.max(np.abs(sy)))
axes[2].plot([-limit, limit], [-limit, limit], color=S.ACCENT, lw=1.2, ls="--")
axes[2].set(title=f"Canonical scores ($r={corr:.2f}$)", xlabel="score from A", ylabel="score from B",
            xlim=(-limit, limit), ylim=(-limit, limit))
for ax in axes:
    S.finish(ax)
fig.tight_layout(w_pad=1.0)
S.save(fig, "cca-paired-projections")
print(f"regularized_score_correlation={corr:.6f}")
