"""feature-lift: lifting two inseparable rings into a separable feature space.

Reproduces ``WIDGETS["feature-lift"]`` in ``public/assets/viz.js`` at its default
state. Forty inner points (radius ``0.9 + U(0, 0.35)``) and forty outer points
(radius ``2.1 + U(0, 0.4)``) form two concentric rings that no line separates in
the plane. The feature map ``phi(x) = (x1, x2, x1^2 + x2^2)`` lifts them so that a
flat plane -- a horizontal threshold on the third coordinate z = x1^2 + x2^2 --
slices the inner ring off cleanly.

Left panel: the two rings in the plane (not linearly separable).
Right panel: the lifted view, z = x1^2 + x2^2 against x1, with the separating
plane drawn as a horizontal green band.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()


def make_rings():
    """Widget default: 40 inner + 40 outer points on evenly spaced angles."""
    g = S.rng(0)
    i = np.arange(40)
    a = i / 40.0 * 2.0 * np.pi
    r_inner = 0.9 + g.random(40) * 0.35          # inner radius in [0.9, 1.25]
    r_outer = 2.1 + g.random(40) * 0.40          # outer radius in [2.1, 2.5]
    inner = jnp.asarray(np.stack([r_inner * np.cos(a), r_inner * np.sin(a)], 1))
    outer = jnp.asarray(np.stack([r_outer * np.cos(a), r_outer * np.sin(a)], 1))
    return inner, outer


def lift(p: jnp.ndarray) -> jnp.ndarray:
    """phi(x) third coordinate: z = x1^2 + x2^2."""
    return p[:, 0] ** 2 + p[:, 1] ** 2


def main() -> str:
    inner, outer = make_rings()
    z_in = np.asarray(lift(inner))
    z_out = np.asarray(lift(outer))
    inner = np.asarray(inner); outer = np.asarray(outer)

    # A plane between the two z-bands separates the classes after lifting.
    z_sep = 0.5 * (z_in.max() + z_out.min())

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.6, 3.1))

    # --- Left: the plane, no line separates the rings -------------------------
    axL.scatter(outer[:, 0], outer[:, 1], s=22, color=S.NEG,
                edgecolor=S.PAPER, linewidth=0.6, zorder=2, label="outer class")
    axL.scatter(inner[:, 0], inner[:, 1], s=22, color=S.POS,
                edgecolor=S.PAPER, linewidth=0.6, zorder=3, label="inner class")
    axL.set_aspect("equal")
    axL.set_xlim(-3.0, 3.0); axL.set_ylim(-3.0, 3.0)
    axL.set_xlabel("$x_1$"); axL.set_ylabel("$x_2$")
    axL.set_title("In the plane: no line separates them", color=S.INK)
    axL.legend(loc="upper right", frameon=False)
    S.finish(axL)

    # --- Right: lifted, z = x1^2 + x2^2, a plane slices cleanly ---------------
    axR.axhspan(z_sep - 0.12, z_sep + 0.12, color=S.GOOD, alpha=0.25, zorder=1)
    axR.axhline(z_sep, color=S.GOOD, lw=1.6, ls=(0, (6, 4)), zorder=2,
                label="separating plane")
    axR.scatter(outer[:, 0], z_out, s=22, color=S.NEG,
                edgecolor=S.PAPER, linewidth=0.6, zorder=3)
    axR.scatter(inner[:, 0], z_in, s=22, color=S.POS,
                edgecolor=S.PAPER, linewidth=0.6, zorder=4)
    axR.set_xlim(-3.0, 3.0); axR.set_ylim(0.0, 7.0)
    axR.set_xlabel("$x_1$"); axR.set_ylabel("$z = x_1^2 + x_2^2$")
    axR.set_title(r"Lifted by $\phi(x)=(x_1,x_2,x_1^2{+}x_2^2)$", color=S.INK)
    axR.legend(loc="upper right", frameon=False)
    S.finish(axR)

    fig.tight_layout()
    return S.save(fig, "feature-lift")


if __name__ == "__main__":
    print(main())
