"""TF-idf removes corpus-wide terms and changes document angles."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _style import DIVERGING, HEAT, INK, MUTED, RULE, apply_style, save

apply_style()

documents = ("D1", "D2", "D3", "D4")
terms = ("the", "kernel", "graph", "method", "trick", "walk")
counts = np.array(
    [
        [1, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0, 1],
    ],
    dtype=float,
)


def cosine_gram(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise RuntimeError("Every document must retain at least one weighted term.")
    normalized = matrix / norms
    return normalized @ normalized.T


document_frequency = np.count_nonzero(counts, axis=0)
idf = np.log(len(documents) / document_frequency)
raw = cosine_gram(counts)
weighted = cosine_gram(counts * idf)
if not np.allclose(np.diag(weighted), 1.0):
    raise RuntimeError("Normalized tf-idf vectors must have unit self-similarity.")

fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.65), gridspec_kw={"width_ratios": [1, 1, 0.82]})
for ax, matrix, title in zip(axes[:2], (raw, weighted), ("Raw-count cosine", "Tf-idf cosine")):
    ax.imshow(matrix, cmap=HEAT, vmin=0, vmax=1)
    ax.set_xticks(range(4), documents)
    ax.set_yticks(range(4), documents)
    ax.set_title(title)
    for row, col in np.ndindex(matrix.shape):
        ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", color=INK, fontsize=7.5)
    ax.tick_params(length=0, colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(RULE)

axes[2].barh(np.arange(len(terms)), idf, color=[RULE, "#8a4c1f", "#8a4c1f", "#3f6c9e", "#59616b", "#59616b"])
axes[2].set_yticks(np.arange(len(terms)), terms)
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
