"""Finite-width deviation from an infinite-width kernel limit."""
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import _style as S

S.apply_style()

width_jax = jnp.array([32, 64, 128, 256, 512, 1024, 2048, 4096], dtype=jnp.float64)
sampling_error_jax = 0.82 / jnp.sqrt(width_jax)
feature_motion_jax = 1.55 / jnp.sqrt(width_jax)
tolerance = 0.04

assert bool(jnp.all(jnp.diff(sampling_error_jax) < 0))
assert bool(jnp.all(jnp.diff(feature_motion_jax) < 0))
assert bool(jnp.isclose(sampling_error_jax[2] / sampling_error_jax[6], 4.0))
assert bool(jnp.all(jnp.isfinite(jnp.concatenate((sampling_error_jax, feature_motion_jax)))))
cross_index = int(jnp.flatnonzero(feature_motion_jax < tolerance, size=1)[0])
width, sampling_error, feature_motion = map(
    np.asarray, (width_jax, sampling_error_jax, feature_motion_jax)
)

fig, ax = S.new_axes(5.35, 3.05)
ax.loglog(width, sampling_error, "o-", color=S.POS, lw=2, ms=4.5, label="kernel sampling error")
ax.loglog(width, feature_motion, "s--", color=S.ACCENT, lw=2, ms=4.2, label="feature movement")
ax.axhline(tolerance, color=S.INK, lw=1.0, ls=":", label="chosen tolerance")
cross = width[cross_index]
ax.annotate(
    f"kernel-like beyond\nwidth $\\approx$ {int(cross)}",
    xy=(cross, feature_motion[cross_index]),
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
