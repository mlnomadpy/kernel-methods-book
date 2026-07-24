"""Posterior mean and uncertainty of a one-dimensional Gaussian process."""
import numpy as np

import _style as S

S.apply_style()

x_train = np.array([-1.55, -0.45, 0.35, 1.45])
y_train = np.array([-0.72, 0.18, 0.64, 0.05])
x = np.linspace(-2.5, 2.5, 360)
lengthscale = 0.72
noise = 0.09


def kernel(a, b):
    return np.exp(-0.5 * ((a[:, None] - b[None, :]) / lengthscale) ** 2)


k_xx = kernel(x_train, x_train) + noise**2 * np.eye(x_train.size)
chol = np.linalg.cholesky(k_xx)
alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, y_train))
k_s = kernel(x, x_train)
mean = k_s @ alpha
v = np.linalg.solve(chol, k_s.T)
variance = np.maximum(1.0 - np.sum(v * v, axis=0), 0.0)
std = np.sqrt(variance)

near = np.min(np.abs(x[:, None] - x_train[None, :]), axis=1) < 0.08
far = np.abs(x) > 2.25
assert np.all(np.isfinite(mean)) and np.all(np.isfinite(std))
assert std[far].mean() > 2.0 * std[near].mean()

fig, ax = S.new_axes(5.4, 3.15)
ax.fill_between(x, -1.96, 1.96, color=S.RULE, alpha=0.22, label="prior 95% band")
ax.fill_between(x, mean - 1.96 * std, mean + 1.96 * std, color=S.POS, alpha=0.18, label="posterior 95% band")
ax.plot(x, mean, color=S.ACCENT, lw=2.2, label="posterior mean")
ax.scatter(x_train, y_train, s=32, facecolor=S.PAPER, edgecolor=S.INK, zorder=4, label="observations")
ax.annotate(
    "data shrink variance",
    xy=(0.35, mean[np.argmin(np.abs(x - 0.35))] + 0.12),
    xytext=(0.82, 1.35),
    color=S.ACCENT,
    fontsize=8,
    arrowprops={"arrowstyle": "-", "color": S.ACCENT, "lw": 0.8},
)
ax.annotate(
    "mean reverts;\nuncertainty returns",
    xy=(2.35, mean[-12] + 1.25 * std[-12]),
    xytext=(1.35, -1.38),
    color=S.MUTED,
    fontsize=8,
    arrowprops={"arrowstyle": "-", "color": S.MUTED, "lw": 0.8},
)
ax.set(xlabel="input $x$", ylabel="latent function $f(x)$", ylim=(-2.15, 2.15))
ax.legend(frameon=False, ncol=2, loc="lower center")
S.finish(ax)
S.save(fig, "gp-posterior-anatomy")
print(f"near_std={std[near].mean():.4f}; far_std={std[far].mean():.4f}")
