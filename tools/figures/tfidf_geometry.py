"""TF-idf removes corpus-wide terms and changes document angles."""
from __future__ import annotations

import jax
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import _style as S
from _style import ACCENT, HEAT, INK, MUTED, POS, RULE, apply_style, save
import matplotlib.pyplot as plt

apply_style()

documents = ("D1", "D2", "D3", "D4")
terms = ("the", "kernel", "graph", "method", "trick", "walk")
counts = jnp.array(
    [
        [1, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 1],
    ],
    dtype=float,
)


@jax.jit
def cosine_gram(matrix: jax.Array) -> jax.Array:
    norms = jnp.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / norms
    return normalized @ normalized.T


document_frequency = jnp.count_nonzero(counts, axis=0)
assert bool(jnp.all(document_frequency > 0))
idf = jnp.log(jnp.asarray(len(documents), dtype=jnp.float64) / document_frequency)
raw = cosine_gram(counts)
weighted = cosine_gram(counts * idf)
assert bool(jnp.all(jnp.isfinite(jnp.stack((raw, weighted)))))
assert bool(jnp.allclose(jnp.diag(weighted), 1.0, rtol=1e-12, atol=1e-12))
assert bool(jnp.allclose(raw, raw.T, rtol=0.0, atol=1e-14))
assert bool(jnp.all(jnp.linalg.eigvalsh(weighted) >= -1e-12))
raw_h, weighted_h, idf_h = map(np.asarray, (raw, weighted, idf))

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.65), gridspec_kw={"width_ratios": [1, 1, 0.82]})
for ax, matrix, title in zip(axes[:2], (raw_h, weighted_h), ("Raw-count cosine", "Tf-idf cosine")):
    ax.imshow(matrix, cmap=HEAT, vmin=0, vmax=1)
    ax.set_xticks(range(4), documents)
    ax.set_yticks(range(4), documents)
    ax.set_title(title)
    for row, col in np.ndindex(matrix.shape):
        ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color=INK, fontsize=7.5)
    ax.tick_params(length=0, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(RULE)

S.bars(
    axes[2], np.arange(len(terms)), idf_h, orientation="horizontal",
    labels=terms, highlight=np.flatnonzero(idf_h > np.median(idf_h)),
)
axes[2].invert_yaxis()
axes[2].set_xlabel("idf weight")
axes[2].set_title("Corpus decides what counts")
axes[2].axvline(0, color=RULE, lw=0.8)
for side in ("top", "right", "left"):
    axes[2].spines[side].set_visible(False)
axes[2].spines["bottom"].set_color(RULE)
axes[2].tick_params(axis="y", length=0)
axes[2].text(0.02, 0.98, '"the" appears everywhere\nand receives weight 0', transform=axes[2].transAxes,
             va="top", color=MUTED, fontsize=7.5)

fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.16, wspace=0.42)
save(fig, "tfidf-geometry")
