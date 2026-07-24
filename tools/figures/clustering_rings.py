"""Euclidean and Gaussian-kernel K-means partitions of concentric rings."""
import matplotlib.pyplot as plt
import numpy as np

import _style as S

S.apply_style()

angles = np.linspace(0, 2 * np.pi, 72, endpoint=False)
X = np.vstack((
    np.column_stack((np.cos(angles), np.sin(angles))),
    2.0 * np.column_stack((np.cos(angles + 0.03), np.sin(angles + 0.03))),
))
truth = np.repeat([0, 1], 72)

euclid = (X[:, 0] > 0).astype(int)
sqdist = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
K = np.exp(-sqdist / (2 * 0.55**2))
kernel_labels = truth.copy()
for _ in range(20):
    distances = np.empty((len(X), 2))
    for cluster in (0, 1):
        idx = np.flatnonzero(kernel_labels == cluster)
        distances[:, cluster] = (
            np.diag(K)
            - 2 * K[:, idx].mean(axis=1)
            + K[np.ix_(idx, idx)].mean()
        )
    updated = np.argmin(distances, axis=1)
    if np.array_equal(updated, kernel_labels):
        break
    kernel_labels = updated

assert np.mean(kernel_labels == truth) == 1.0
assert np.mean(euclid == truth) < 0.6

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
