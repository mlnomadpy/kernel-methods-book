"""Distance-to-Gram-to-coordinate pipeline for classical MDS."""
import matplotlib.pyplot as plt
import numpy as np

import _style as S

S.apply_style()
X = np.array([[-2.0, -1.5], [2.0, -1.5], [2.0, 1.5], [-2.0, 1.5]])
D2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
J = np.eye(4) - np.ones((4, 4)) / 4
row_centered = J @ D2
B = -0.5 * row_centered @ J
values, vectors = np.linalg.eigh(B)
order = np.argsort(values)[::-1]
values, vectors = values[order], vectors[:, order]
coords = vectors[:, :2] * np.sqrt(values[:2])

recovered_d2 = np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=2)
assert np.allclose(D2, recovered_d2, atol=1e-10)
assert np.allclose(B @ np.ones(4), 0, atol=1e-10)

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
print(f"distance_reconstruction_error={np.max(np.abs(D2 - recovered_d2)):.3e}")
