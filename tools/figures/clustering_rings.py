"""Euclidean and Gaussian-kernel K-means partitions of concentric rings."""
import jax
import jax.numpy as jnp
import numpy as np

import _style as S

import matplotlib.pyplot as plt

S.apply_style()
jax.config.update("jax_enable_x64", True)

angles = jnp.linspace(0, 2 * jnp.pi, 72, endpoint=False)
X = jnp.vstack((
    jnp.column_stack((jnp.cos(angles), jnp.sin(angles))),
    2.0 * jnp.column_stack((jnp.cos(angles + 0.03), jnp.sin(angles + 0.03))),
))
truth = jnp.repeat(jnp.array((0, 1)), 72)

euclid = (X[:, 0] > 0).astype(jnp.int32)
sqdist = jnp.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
K = jnp.exp(-sqdist / (2 * 0.55**2))
kernel_labels = truth.copy()

def update(labels: jax.Array) -> jax.Array:
    memberships = jax.nn.one_hot(labels, 2).T
    counts = memberships.sum(axis=1)
    cross = (K @ memberships.T) / counts
    within = jnp.einsum("ci,ij,cj->c", memberships, K, memberships) / counts**2
    distances = jnp.diag(K)[:, None] - 2.0 * cross + within[None, :]
    return jnp.argmin(distances, axis=1)

for _ in range(20):
    updated = update(kernel_labels)
    if bool(jnp.array_equal(updated, kernel_labels)):
        break
    kernel_labels = updated

assert bool(jnp.all(jnp.isfinite(K)))
assert bool(jnp.allclose(K, K.T, atol=1e-12))
assert float(jnp.linalg.eigvalsh(K).min()) > -1e-10
assert float(jnp.mean(kernel_labels == truth)) == 1.0
assert float(jnp.mean(euclid == truth)) < 0.6
X, truth, euclid, kernel_labels = map(np.asarray, (X, truth, euclid, kernel_labels))

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6))
titles = ["Geometry to partition", "Euclidean K-means", "Gaussian kernel K-means"]
assignments = [truth, euclid, kernel_labels]
for ax, title, assignment in zip(axes, titles, assignments):
    for value, color, marker in [(0, S.POS, "o"), (1, S.NEG, "^")]:
        m = assignment == value
        ax.scatter(X[m, 0], X[m, 1], s=10, color=color, marker=marker)
    ax.set(title=title, xlim=(-2.25, 2.25), ylim=(-2.25, 2.25), xticks=[], yticks=[])
    ax.set_aspect("equal")
    S.finish(ax)
fig.tight_layout(w_pad=0.8)
S.save(fig, "clustering-rings")
print("kernel_accuracy=1.000; euclidean_radial_accuracy=0.500")
