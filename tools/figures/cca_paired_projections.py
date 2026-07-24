"""Paired views before and after regularized linear CCA."""
import matplotlib.pyplot as plt
import numpy as np

import _style as S

S.apply_style()
gen = S.rng(19)
n = 70
latent = gen.normal(size=n)
X = np.column_stack((latent + 0.20 * gen.normal(size=n), 2.5 * gen.normal(size=n)))
Y = np.column_stack((0.85 * latent + 0.20 * gen.normal(size=n), 2.2 * gen.normal(size=n)))
X -= X.mean(axis=0)
Y -= Y.mean(axis=0)

ridge = 0.18
Cxx = X.T @ X / n + ridge * np.eye(2)
Cyy = Y.T @ Y / n + ridge * np.eye(2)
Cxy = X.T @ Y / n

def invsqrt(matrix):
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * values**-0.5) @ vectors.T

Wx, Wy = invsqrt(Cxx), invsqrt(Cyy)
left, _, right_t = np.linalg.svd(Wx @ Cxy @ Wy)
wx = Wx @ left[:, 0]
wy = Wy @ right_t.T[:, 0]
sx, sy = X @ wx, Y @ wy
if np.corrcoef(sx, sy)[0, 1] < 0:
    sy = -sy
corr = float(np.corrcoef(sx, sy)[0, 1])
assert corr > 0.9

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
color = np.where(latent >= 0, S.POS, S.NEG)
axes[0].scatter(X[:, 0], X[:, 1], c=color, s=15, alpha=0.85)
axes[1].scatter(Y[:, 0], Y[:, 1], c=color, s=15, alpha=0.85, marker="^")
axes[0].set(title="View A", xlabel="$x_1$", ylabel="$x_2$")
axes[1].set(title="View B", xlabel="$y_1$", ylabel="$y_2$")
axes[2].scatter(sx, sy, c=color, s=16, alpha=0.9)
limit = 1.05 * max(np.max(np.abs(sx)), np.max(np.abs(sy)))
axes[2].plot([-limit, limit], [-limit, limit], color=S.ACCENT, lw=1.2, ls="--")
axes[2].set(title=f"Canonical scores ($r={corr:.2f}$)", xlabel="score from A", ylabel="score from B",
            xlim=(-limit, limit), ylim=(-limit, limit))
for ax in axes:
    S.finish(ax)
fig.tight_layout(w_pad=1.0)
S.save(fig, "cca-paired-projections")
print(f"regularized_score_correlation={corr:.6f}")
