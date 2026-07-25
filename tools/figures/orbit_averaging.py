"""Group averaging sends every point on one rotation orbit to one feature."""
from __future__ import annotations

import matplotlib.pyplot as plt
import jax
from jax import config
import jax.numpy as jnp
import numpy as np

from _style import ACCENT, INK, MUTED, POS, RULE, apply_style, save

config.update("jax_enable_x64", True)
apply_style()


def quadratic_feature(points: jax.Array) -> jax.Array:
    """Degree-two feature map whose rotation average retains squared radius."""
    x, y = points.T
    return jnp.stack((x * x, y * y, jnp.sqrt(2.0) * x * y), axis=1)


angles = jnp.linspace(0.0, 2.0 * jnp.pi, 12, endpoint=False, dtype=jnp.float64)
unit_orbit = jnp.stack((jnp.cos(angles), jnp.sin(angles)), axis=1)
radii = jnp.array([0.72, 1.18], dtype=jnp.float64)
orbits = radii[:, None, None] * unit_orbit[None, :, :]
averages = jax.vmap(lambda orbit: jnp.mean(quadratic_feature(orbit), axis=0))(orbits)
expected = jnp.stack((radii**2 / 2, radii**2 / 2, jnp.zeros_like(radii)), axis=1)
if not bool(jnp.allclose(averages, expected, atol=1e-12, rtol=0.0)):
    raise RuntimeError("Discrete rotation average does not match the invariant feature.")
assert bool(jnp.all(jnp.isfinite(averages)))
assert float(jnp.max(jnp.abs(jnp.linalg.norm(orbits, axis=2) - radii[:, None]))) < 1e-12
orbits, radii, averages = map(np.asarray, (orbits, radii, averages))

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
