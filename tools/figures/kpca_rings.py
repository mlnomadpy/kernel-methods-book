"""Concentric rings in input space and Gaussian-kernel principal coordinates."""
import _style as S
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

S.apply_style()

angles = jnp.linspace(0, 2 * jnp.pi, 80, endpoint=False)
inner = jnp.column_stack((jnp.cos(angles), jnp.sin(angles)))
outer = 2.0 * jnp.column_stack((jnp.cos(angles + 0.021), jnp.sin(angles + 0.021)))
X_jax = jnp.vstack((inner, outer))
labels_jax = jnp.repeat(jnp.array([0, 1]), 80)

sqdist = jnp.sum((X_jax[:, None, :] - X_jax[None, :, :]) ** 2, axis=2)
K = jnp.exp(-sqdist / (2 * 0.7**2))
H = jnp.eye(len(X_jax)) - jnp.ones((len(X_jax), len(X_jax))) / len(X_jax)
Kc = H @ K @ H
eigvals, eigvecs = jnp.linalg.eigh(Kc)
order = jnp.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
scores = eigvecs * jnp.sqrt(jnp.maximum(eigvals, 0))
target = 2 * labels_jax - 1
centered_target = target - target.mean()
centered_scores = scores[:, :10] - scores[:, :10].mean(axis=0)
separation = jnp.abs(
    centered_scores.T @ centered_target
    / (jnp.linalg.norm(centered_scores, axis=0) * jnp.linalg.norm(centered_target))
)
radial_idx = int(jnp.argmax(separation))
angular_idx = 0 if radial_idx != 0 else 1
radial = scores[:, radial_idx]
angular = scores[:, angular_idx]

assert float(separation[radial_idx]) > 0.99
assert float(jnp.linalg.norm(Kc @ jnp.ones(len(X_jax)))) < 1e-9
assert float(jnp.min(eigvals)) > -1e-10
assert bool(jnp.all(jnp.isfinite(scores)))
X, labels, radial, angular, separation = map(
    np.asarray, (X_jax, labels_jax, radial, angular, separation)
)

fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.85))
for value, color, marker, name in [(0, S.POS, "o", "inner ring"), (1, S.NEG, "^", "outer ring")]:
    m = labels == value
    axes[0].scatter(X[m, 0], X[m, 1], s=13, color=color, marker=marker, label=name)
    axes[1].scatter(radial[m], angular[m], s=13, color=color, marker=marker, label=name)
axes[0].axhline(0, color=S.ACCENT, lw=1.3, ls="--", label="linear PC axis")
axes[0].set(title="Input plane", xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
axes[1].axvline(0, color=S.RULE, lw=0.8)
axes[1].set(title="Gaussian kernel PCA", xlabel=f"radial component (PC {radial_idx + 1})",
            ylabel=f"leading angular component (PC {angular_idx + 1})")
axes[0].legend(
    frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.115),
    borderaxespad=0, ncol=3, handlelength=1.8, columnspacing=0.9,
)
for ax in axes:
    S.finish(ax)
fig.tight_layout(w_pad=1.4)
S.save(fig, "kpca-rings")
print(f"radial_component={radial_idx + 1}; label_correlation={separation[radial_idx]:.6f}")
