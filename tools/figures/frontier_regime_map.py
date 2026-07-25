"""Map modern kernel regimes by representation movement."""
import _style as S
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

S.apply_style()

fig, ax = S.new_axes(6.15, 3.55)
ax.axhspan(0.0, 0.34, color=S.POS, alpha=0.055)
ax.axhspan(0.34, 1.0, color=S.ACCENT, alpha=0.045)
ax.axvline(0.44, color=S.RULE, lw=0.8)
ax.axhline(0.34, color=S.RULE, lw=0.8)

coordinates_jax = jnp.array([[0.20, 0.17], [0.71, 0.20], [0.62, 0.73]])
assert bool(jnp.all((coordinates_jax >= 0.0) & (coordinates_jax <= 1.0)))
assert bool(jnp.all(jnp.isfinite(coordinates_jax)))
coordinates = np.asarray(coordinates_jax)
regime_labels = [
    ("Frozen NTK\nor encoder", "coefficients move", S.POS),
    ("Tuned prompt,\nadapter, circuit", "feature geometry moves", S.ACCENT),
    ("Mean-field\nnetwork", "neuron measure flows", S.GOOD),
]
for (x, y), (title, subtitle, color) in zip(coordinates, regime_labels):
    ax.scatter([x], [y], s=76, color=color, edgecolor=S.PAPER, linewidth=1.2, zorder=4)
    ax.text(x, y + 0.095, title, ha="center", va="bottom", color=color,
            fontsize=8.8, weight="semibold", linespacing=1.05)
    ax.text(x, y - 0.085, subtitle, ha="center", va="top", color=S.MUTED, fontsize=7.4)

annotations = [
    ("fixed-kernel estimation", (0.20, 0.17), (0.06, 0.48), S.POS, "left"),
    ("data-dependent kernel selection", (0.71, 0.20), (0.59, 0.47), S.ACCENT, "left"),
    ("nonlinear measure flow", (0.62, 0.73), (0.86, 0.66), S.GOOD, "center"),
]
for text, target, origin, color, align in annotations:
    ax.annotate(text, xy=target, xytext=origin, color=color, fontsize=7.7, ha=align,
                arrowprops={"arrowstyle": "-", "color": color, "lw": 0.85})

ax.set(
    xlim=(0, 1), ylim=(0, 1),
    xlabel="relative parameter movement  $\\longrightarrow$",
    ylabel="representation movement  $\\longrightarrow$",
)
ax.set_xticks([])
ax.set_yticks([])
S.finish(ax)
S.save(fig, "frontier-regime-map")
print("regimes=3; decisive_axis=representation_movement")
