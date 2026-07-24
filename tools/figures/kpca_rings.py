"""Concentric rings in input space and Gaussian-kernel principal coordinates."""
import matplotlib.pyplot as plt
import numpy as np

import _style as S

S.apply_style()

angles = np.linspace(0, 2 * np.pi, 80, endpoint=False)
inner = np.column_stack((np.cos(angles), np.sin(angles)))
outer = 2.0 * np.column_stack((np.cos(angles + 0.021), np.sin(angles + 0.021)))
X = np.vstack((inner, outer))
labels = np.repeat([0, 1], 80)

sqdist = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
K = np.exp(-sqdist / (2 * 0.7**2))
H = np.eye(len(X)) - np.ones((len(X), len(X))) / len(X)
Kc = H @ K @ H
eigvals, eigvecs = np.linalg.eigh(Kc)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]
scores = eigvecs * np.sqrt(np.maximum(eigvals, 0))
target = 2 * labels - 1
separation = np.array([abs(np.corrcoef(scores[:, j], target)[0, 1]) for j in range(10)])
radial_idx = int(np.argmax(separation))
angular_idx = 0 if radial_idx != 0 else 1
radial = scores[:, radial_idx]
angular = scores[:, angular_idx]

assert separation[radial_idx] > 0.99
assert np.linalg.norm(Kc @ np.ones(len(X))) < 1e-9

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
axes[0].legend(frameon=False, loc="upper right")
for ax in axes:
    S.finish(ax)
fig.tight_layout(w_pad=1.4)
S.save(fig, "kpca-rings")
print(f"radial_component={radial_idx + 1}; label_correlation={separation[radial_idx]:.6f}")
