"""Group averaging sends every point on one rotation orbit to one feature."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _style import ACCENT, INK, MUTED, POS, RULE, apply_style, save

apply_style()


def quadratic_feature(points: np.ndarray) -> np.ndarray:
    """Degree-two feature map whose rotation average retains squared radius."""
    x, y = points.T
    return np.column_stack((x * x, y * y, np.sqrt(2.0) * x * y))


angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
unit_orbit = np.column_stack((np.cos(angles), np.sin(angles)))
radii = np.array([0.72, 1.18])
orbits = radii[:, None, None] * unit_orbit[None, :, :]
averages = np.stack([quadratic_feature(orbit).mean(axis=0) for orbit in orbits])
expected = np.column_stack((radii**2 / 2, radii**2 / 2, np.zeros_like(radii)))
if not np.allclose(averages, expected, atol=1e-12):
    raise RuntimeError("Discrete rotation average does not match the invariant feature.")

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.4, 2.8), gridspec_kw={"width_ratios": [1.2, 1]})
colors = (POS, ACCENT)
for orbit, radius, color in zip(orbits, radii, colors):
    ax0.plot(orbit[:, 0], orbit[:, 1], color=color, lw=1.0, alpha=0.5)
    ax0.scatter(
        orbit[:, 0],
        orbit[:, 1],
        s=24,
        facecolor="none",
        edgecolor=color,
        linewidth=1.0,
        label=fr"orbit, $r={radius:.2f}$",
    )
    ax0.scatter(orbit[2, 0], orbit[2, 1], s=32, color=color, marker="s", zorder=3)
ax0.axhline(0, color=RULE, lw=0.8)
ax0.axvline(0, color=RULE, lw=0.8)
ax0.set_aspect("equal")
ax0.set_xlim(-1.35, 1.35)
ax0.set_ylim(-1.35, 1.35)
ax0.set_title("Input: many rotated representatives")
ax0.set_xlabel(r"$x_1$")
ax0.set_ylabel(r"$x_2$")
ax0.legend(frameon=False, loc="lower left")

invariant_coordinate = averages[:, 0] + averages[:, 1]
ax1.axhline(0, color=RULE, lw=0.8)
for y, radius, value, color in zip((0.35, -0.35), radii, invariant_coordinate, colors):
    ax1.scatter(value, y, s=70, color=color, marker="D", zorder=3)
    ax1.annotate(
        fr"all rotations with $r={radius:.2f}$",
        xy=(value, y),
        xytext=(4, 10 if y > 0 else -15),
        textcoords="offset points",
        color=INK,
        fontsize=8,
    )
ax1.text(
    0.04,
    0.03,
    r"$\overline{\Phi}(x)=(r^2/2,r^2/2,0)$",
    transform=ax1.transAxes,
    color=MUTED,
    fontsize=8,
)
ax1.set_xlim(0.25, 1.65)
ax1.set_ylim(-0.75, 0.75)
ax1.set_yticks([])
ax1.set_xlabel(r"invariant coordinate $r^2$")
ax1.set_title("Feature space: one point per orbit")
for side in ("top", "right", "left"):
    ax1.spines[side].set_visible(False)
ax1.spines["bottom"].set_color(RULE)
ax1.tick_params(axis="x", color=RULE)

fig.subplots_adjust(wspace=0.32)
save(fig, "orbit-averaging")
