"""Approximation-capacity tradeoff for the learning-theory chapter."""
import matplotlib.pyplot as plt
import numpy as np

import _style as S

S.apply_style()

radius = np.linspace(0.15, 4.2, 240)
approximation = 0.82 * np.exp(-1.15 * radius) + 0.035
capacity = 0.035 + 0.030 * radius**1.65
total = approximation + capacity
best = int(np.argmin(total))

assert 0 < best < radius.size - 1
assert np.all(np.diff(approximation) < 0)
assert np.all(np.diff(capacity) > 0)

fig, ax = S.new_axes(5.4, 3.0)
ax.plot(radius, approximation, color=S.POS, lw=2.0, label="approximation error")
ax.plot(radius, capacity, color=S.NEG, lw=2.0, ls="--", label="capacity price")
ax.plot(radius, total, color=S.INK, lw=2.3, label="sum")
ax.scatter(radius[best], total[best], s=38, color=S.ACCENT, zorder=4)
ax.axvline(radius[best], color=S.ACCENT, lw=1.0, ls=":")
ax.annotate(
    "balanced radius",
    (radius[best], total[best]),
    xytext=(13, 18),
    textcoords="offset points",
    color=S.ACCENT,
    arrowprops={"arrowstyle": "-", "color": S.ACCENT, "lw": 0.8},
)
ax.set(xlabel="accessible RKHS radius $B$", ylabel="error / bound contribution")
ax.set_ylim(0, 0.78)
ax.legend(frameon=False, ncol=3, loc="upper center")
S.finish(ax)
S.save(fig, "complexity-tradeoff")
print(f"balanced_radius={radius[best]:.3f}; total={total[best]:.3f}")
