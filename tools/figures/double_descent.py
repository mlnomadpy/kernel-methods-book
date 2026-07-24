"""Ridgeless test risk on both sides of the interpolation threshold."""
import numpy as np

import _style as S

S.apply_style()

noise = 0.18
signal = 0.58
left = np.linspace(0.08, 0.94, 190)
right = np.linspace(1.06, 3.0, 250)
risk_left = noise * left / (1.0 - left)
risk_right = signal * (1.0 - 1.0 / right) + noise / (right - 1.0)

assert np.all(np.diff(risk_left) > 0)
assert risk_right[0] > risk_right[-1]
assert np.all(np.isfinite(risk_left)) and np.all(np.isfinite(risk_right))

fig, ax = S.new_axes(5.35, 3.05)
ax.plot(left, risk_left, color=S.INK, lw=2.2)
ax.plot(right, risk_right, color=S.ACCENT, lw=2.2, label="minimum-norm interpolant")
ax.axvline(1.0, color=S.NEG, lw=1.1, ls="--")
ax.text(1.03, 2.25, "interpolation\nthreshold", color=S.NEG, fontsize=8, va="top")
ax.annotate(
    "too few directions:\nvariance explodes",
    xy=(0.90, risk_left[np.argmin(np.abs(left - 0.90))]),
    xytext=(0.25, 1.72),
    color=S.MUTED,
    fontsize=8,
    arrowprops={"arrowstyle": "-", "color": S.MUTED, "lw": 0.8},
)
ax.annotate(
    "extra directions spread the\nminimum-norm solution",
    xy=(2.25, risk_right[np.argmin(np.abs(right - 2.25))]),
    xytext=(1.65, 1.30),
    color=S.ACCENT,
    fontsize=8,
    arrowprops={"arrowstyle": "-", "color": S.ACCENT, "lw": 0.8},
)
ax.set(xlim=(0, 3.02), ylim=(0, 2.4), xlabel="aspect ratio $\\gamma=p/n$", ylabel="test excess risk")
ax.legend(frameon=False, loc="upper right")
S.finish(ax)
S.save(fig, "double-descent")
print(f"left_peak={risk_left[-1]:.3f}; right_tail={risk_right[-1]:.3f}")
