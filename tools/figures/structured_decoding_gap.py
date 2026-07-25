"""Score, loss augmentation, and globally feasible structured decoding."""
from __future__ import annotations

import jax
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import _style as S
import matplotlib.pyplot as plt

S.apply_style()

labels = ["A", "B", "A", "B", "B"]
raw = jnp.array([0.82, 0.58, 0.71, 0.62, 0.76], dtype=jnp.float64)
loss = jnp.array([0.0, 0.20, 0.0, 0.20, 0.0], dtype=jnp.float64)
augmented = raw + loss
decoded = ["A", "B", "B", "B", "B"]
feasible = jnp.array([0.86, 0.80, 0.76, 0.82, 0.88], dtype=jnp.float64)
series = jnp.stack((raw, augmented, feasible))
assert bool(jnp.all(jnp.isfinite(series)))
assert bool(jnp.allclose(augmented - raw, loss, rtol=0.0, atol=1e-14))
assert bool(jnp.all((series >= 0.0) & (series <= 1.0)))
series_h = np.asarray(series)

fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.85), sharey=True)
titles = ["1  local score", "2  add task loss", "3  enforce structure"]
for panel_index, (ax, title, values) in enumerate(zip(axes, titles, series_h)):
    S.bars(
        ax, np.arange(5), values, width=0.58,
        highlight=[int(np.argmax(values))] if panel_index == 2 else None,
    )
    ax.set_title(title, pad=10)
    ax.set_xticks(range(5), [f"$y_{i+1}$" for i in range(5)])
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.5, 1.0])
    S.finish(ax)

for i, label in enumerate(labels):
    axes[0].text(i, series_h[0, i] + 0.035, label, ha="center", color=S.INK, fontsize=8)
    axes[1].text(i, series_h[1, i] + 0.035, label, ha="center", color=S.INK, fontsize=8)
for i, label in enumerate(decoded):
    axes[2].text(i, series_h[2, i] + 0.035, label, ha="center", color=S.INK, fontsize=8)

axes[0].set_ylabel("candidate score")
axes[2].plot(range(5), series_h[2], color=S.DEEP, lw=1.0, marker="o",
             markersize=2.8, zorder=3)

fig.subplots_adjust(wspace=0.20)
S.save(fig, "structured-decoding-gap")
print("positions=5; columns=3; feasible_paths=1")
