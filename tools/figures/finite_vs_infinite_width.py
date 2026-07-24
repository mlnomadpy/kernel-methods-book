"""Finite-width deviation from an infinite-width kernel limit."""
import numpy as np

import _style as S

S.apply_style()

width = np.array([32, 64, 128, 256, 512, 1024, 2048, 4096], dtype=float)
sampling_error = 0.82 / np.sqrt(width)
feature_motion = 1.55 / np.sqrt(width)
tolerance = 0.04

assert np.all(np.diff(sampling_error) < 0)
assert np.all(np.diff(feature_motion) < 0)
assert np.isclose(sampling_error[2] / sampling_error[6], 4.0)

fig, ax = S.new_axes(5.35, 3.05)
ax.loglog(width, sampling_error, "o-", color=S.POS, lw=2, ms=4.5, label="kernel sampling error")
ax.loglog(width, feature_motion, "s--", color=S.ACCENT, lw=2, ms=4.2, label="feature movement")
ax.axhline(tolerance, color=S.INK, lw=1.0, ls=":", label="chosen tolerance")
cross = width[np.flatnonzero(feature_motion < tolerance)[0]]
ax.annotate(
    f"kernel-like beyond\nwidth $\\approx$ {int(cross)}",
    xy=(cross, feature_motion[width == cross][0]),
    xytext=(740, 0.12),
    color=S.ACCENT,
    fontsize=8,
    arrowprops={"arrowstyle": "-", "color": S.ACCENT, "lw": 0.8},
)
ax.set(xlabel="network width", ylabel="deviation from infinite-width limit")
ax.legend(frameon=False, loc="lower left")
S.finish(ax)
S.save(fig, "finite-vs-infinite-width")
print(f"tolerance={tolerance:.3f}; crossover_width={int(cross)}")
