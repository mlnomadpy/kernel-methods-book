"""Distance-to-Gram-to-coordinate pipeline for classical MDS."""
import matplotlib.pyplot as plt
from jax import config
import jax.numpy as jnp
import numpy as np

import _style as S

config.update("jax_enable_x64", True)
S.apply_style()
X = jnp.array([[-2.0, -1.5], [2.0, -1.5], [2.0, 1.5], [-2.0, 1.5]], dtype=jnp.float64)
D2 = jnp.sum(jnp.square(X[:, None, :] - X[None, :, :]), axis=2)
J = jnp.eye(4, dtype=jnp.float64) - jnp.ones((4, 4), dtype=jnp.float64) / 4
row_centered = J @ D2
B = -0.5 * row_centered @ J
values, vectors = jnp.linalg.eigh(B)
order = jnp.argsort(values)[::-1]
values, vectors = values[order], vectors[:, order]
coords = vectors[:, :2] * jnp.sqrt(jnp.maximum(values[:2], 0.0))

recovered_d2 = jnp.sum(jnp.square(coords[:, None, :] - coords[None, :, :]), axis=2)
error = jnp.max(jnp.abs(D2 - recovered_d2))
assert bool(jnp.all(jnp.isfinite(coords)))
assert bool(jnp.allclose(B, B.T, atol=1e-12, rtol=0.0))
assert float(error) < 1e-10
assert float(jnp.max(jnp.abs(B @ jnp.ones(4)))) < 1e-10
D2, row_centered, B, coords = map(np.asarray, (D2, row_centered, B, coords))

fig, axes = plt.subplots(1, 4, figsize=(7.5, 2.25))
images = [
    (D2, "$D^{(2)}$\nsquared distances", S.HEAT),
    (row_centered, "$JD^{(2)}$\nremove row means", S.DIVERGING),
    (B, "$-\\frac{1}{2} JD^{(2)}J$\ncentered Gram", S.DIVERGING),
]
for ax, (matrix, title, cmap) in zip(axes[:3], images):
    ax.imshow(matrix, cmap=cmap, aspect="equal")
    ax.set(title=title, xticks=range(4), yticks=range(4))
    ax.tick_params(length=0)
axes[3].scatter(coords[:, 0], coords[:, 1], s=35, color=S.POS)
for i, point in enumerate(coords):
    axes[3].text(point[0] + 0.12, point[1] + 0.08, str(i + 1), color=S.MUTED)
axes[3].set(title="positive eigenpairs\nrecover coordinates", xlabel="axis 1", ylabel="axis 2")
axes[3].set_aspect("equal")
S.finish(axes[3])
fig.tight_layout(w_pad=0.7)
S.save(fig, "mds-double-centering")
print(f"distance_reconstruction_error={float(error):.3e}")
