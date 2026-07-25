"""Sparse multiple-kernel weight path under increasing selection pressure."""
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import _style as S

S.apply_style()

pressure_jax = jnp.linspace(0.0, 0.92, 220)
scores_jax = jnp.array([1.00, 0.72, 0.38, 0.14])
raw_jax = jnp.maximum(scores_jax[:, None] - pressure_jax[None, :], 0.0)
weights_jax = raw_jax / jnp.maximum(raw_jax.sum(axis=0, keepdims=True), 1e-12)
active_jax = (raw_jax > 0).sum(axis=0)

assert bool(jnp.allclose(weights_jax[:, raw_jax.sum(axis=0) > 0].sum(axis=0), 1.0))
assert bool(jnp.all(jnp.diff(active_jax) <= 0))
assert int(active_jax[0]) == 4 and int(active_jax[-1]) == 1
assert bool(jnp.all(jnp.isfinite(weights_jax)))
pressure, scores, weights, active = map(
    np.asarray, (pressure_jax, scores_jax, weights_jax, active_jax)
)

fig, ax = S.new_axes(5.35, 3.05)
styles = [
    (S.ACCENT, "-", "sequence"),
    (S.POS, "--", "structure"),
    (S.GOOD, "-.", "expression"),
    (S.NEG, ":", "control"),
]
for row, (color, linestyle, label) in zip(weights, styles):
    ax.plot(pressure, row, color=color, ls=linestyle, lw=2.0, label=label)

for threshold in scores[1:]:
    ax.axvline(threshold, color=S.RULE, lw=0.8)
ax.annotate(
    "irrelevant blocks\nleave the model",
    xy=(0.39, weights[1, np.searchsorted(pressure, 0.39)]),
    xytext=(0.53, 0.33),
    color=S.MUTED,
    fontsize=8,
    arrowprops={"arrowstyle": "-", "color": S.MUTED, "lw": 0.8},
)
ax.set(
    xlim=(0, 0.92),
    ylim=(0, 1.03),
    xlabel="selection pressure",
    ylabel="normalized kernel weight",
)
ax.legend(frameon=False, ncol=2, loc="upper left")
S.finish(ax)
S.save(fig, "kernel-weight-path")
print(f"active_start={active[0]}; active_end={active[-1]}")
