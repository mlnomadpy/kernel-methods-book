"""A low-variance label direction missed by the first principal component."""
import jax
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import _style as S
import matplotlib.pyplot as plt

S.apply_style()
key = jax.random.PRNGKey(7)
key0, key1, key2, key3 = jax.random.split(key, 4)
n = 65
x0 = jnp.column_stack((2.0 * jax.random.normal(key0, (n,), dtype=jnp.float64),
                       -0.34 + 0.12 * jax.random.normal(key1, (n,), dtype=jnp.float64)))
x1 = jnp.column_stack((2.0 * jax.random.normal(key2, (n,), dtype=jnp.float64),
                       0.34 + 0.12 * jax.random.normal(key3, (n,), dtype=jnp.float64)))
X = jnp.vstack((x0, x1))
cov = jnp.cov(X.T)
values, vectors = jnp.linalg.eigh(cov)
pc = vectors[:, jnp.argmax(values)]
pc = jnp.where(pc[0] < 0, -pc, pc)
mean_gap = jnp.mean(x1, axis=0) - jnp.mean(x0, axis=0)
sw = jnp.cov(x0.T) + jnp.cov(x1.T) + 1e-6 * jnp.eye(2, dtype=jnp.float64)
chol = jnp.linalg.cholesky(sw)
fisher = jax.scipy.linalg.cho_solve((chol, True), mean_gap)
fisher /= jnp.linalg.norm(fisher)
assert bool(jnp.all(jnp.isfinite(jnp.concatenate((values, pc, fisher)))))
assert bool(jnp.all(values >= -1e-12))
assert float(jnp.linalg.norm(sw @ fisher - mean_gap / jnp.linalg.norm(jax.scipy.linalg.cho_solve((chol, True), mean_gap)))) < 1e-10
assert float(jnp.abs(pc[0])) > 0.98
assert float(jnp.abs(fisher[1])) > 0.98
x0_h, x1_h, pc_h, fisher_h = map(np.asarray, (x0, x1, pc, fisher))

fig, ax = S.new_axes(5.5, 3.0)
ax.scatter(x0_h[:, 0], x0_h[:, 1], s=18, color=S.NEG, marker="x", label="class -")
ax.scatter(x1_h[:, 0], x1_h[:, 1], s=18, color=S.POS, marker="o", label="class +")
scale = 1.55
ax.annotate("", xy=scale * pc_h, xytext=-scale * pc_h,
            arrowprops={"arrowstyle": "<->", "lw": 2.0, "color": S.ACCENT})
ax.text(1.65, -0.10, "PC1: variance", color=S.ACCENT)
ax.annotate("", xy=0.75 * fisher_h, xytext=-0.75 * fisher_h,
            arrowprops={"arrowstyle": "<->", "lw": 2.0, "color": S.GOOD})
ax.text(0.10, 0.62, "Fisher: relevance", color=S.GOOD)
ax.set(xlabel="$x_1$", ylabel="$x_2$", ylim=(-0.9, 0.9))
ax.legend(frameon=False, loc="lower left")
S.finish(ax)
S.save(fig, "variance-vs-relevance")
print(f"pc_horizontal={float(jnp.abs(pc[0])):.5f}; fisher_vertical={float(jnp.abs(fisher[1])):.5f}")
