"""Finite-sample balance between universal approximation and capacity."""
import matplotlib.pyplot as plt
import numpy as np

import _style as S

S.apply_style()
radius = np.linspace(0.08, 5.0, 260)
approximation = 0.90 * np.exp(-1.05 * radius) + 0.018
effective_dimension = 1.0 + 1.7 * radius**1.45
sample_size = 180
estimation = 0.62 * np.sqrt(effective_dimension / sample_size)
total = approximation + estimation
best = int(np.argmin(total))

assert approximation[-1] < 0.03
assert np.all(np.diff(estimation) > 0)
assert 0 < best < len(radius) - 1

fig, ax = S.new_axes(5.5, 3.0)
ax.plot(radius, approximation, color=S.POS, lw=2, label="approximation")
ax.plot(radius, estimation, color=S.NEG, lw=2, ls="--", label="estimation")
ax.plot(radius, total, color=S.INK, lw=2.3, label="finite-sample total")
ax.scatter(radius[best], total[best], s=38, color=S.ACCENT, zorder=4)
ax.annotate("best accessible ball", (radius[best], total[best]), xytext=(16, 18),
            textcoords="offset points", color=S.ACCENT,
            arrowprops={"arrowstyle": "-", "lw": 0.8, "color": S.ACCENT})
ax.set(xlabel="accessible RKHS radius", ylabel="error contribution", ylim=(0, 0.92))
ax.legend(frameon=False, ncol=3, loc="upper center")
S.finish(ax)
S.save(fig, "universality-capacity")
print(f"optimal_radius={radius[best]:.3f}; effective_dimension={effective_dimension[best]:.3f}")
