"""operator-valued-field: output coupling turns one scalar similarity into a vector response."""
from __future__ import annotations

import matplotlib.pyplot as plt
from jax import config, vmap
import jax.numpy as jnp
import numpy as np

import _style as S

config.update("jax_enable_x64", True)
S.apply_style()


def main() -> str:
    x = jnp.linspace(-2.2, 2.2, 240, dtype=jnp.float64)
    kernel_section = jnp.exp(-0.5 * jnp.square(x))
    couplings = jnp.stack((
        jnp.eye(2, dtype=jnp.float64),
        jnp.array([[1.0, 0.72], [0.72, 1.0]], dtype=jnp.float64),
    ))
    impulse = jnp.stack((kernel_section, jnp.zeros_like(kernel_section)))
    responses = vmap(lambda coupling: coupling @ impulse)(couplings)
    assert bool(jnp.all(jnp.isfinite(responses)))
    assert bool(jnp.allclose(couplings, jnp.swapaxes(couplings, 1, 2), atol=1e-12))
    assert bool(jnp.all(jnp.linalg.eigvalsh(couplings) > 0.0))
    assert bool(jnp.allclose(couplings[0] @ jnp.array([1.0, 0.0]), jnp.array([1.0, 0.0])))
    assert bool(jnp.isclose((couplings[1] @ jnp.array([1.0, 0.0]))[1], 0.72))
    x_np, responses_np = map(np.asarray, (x, responses))
    titles = ("Independent outputs", "Coupled outputs")
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.6), sharex=True, sharey=True)
    for ax, response, title in zip(axes, responses_np, titles):
        ax.plot(x_np, response[0], color=S.POS, lw=1.8, label=r"output 1")
        ax.plot(x_np, response[1], color=S.ACCENT, lw=1.6, ls="--", label=r"output 2")
        ax.scatter([0], [1], color=S.INK, s=24, marker="s", zorder=3)
        ax.set_title(title)
        ax.set_xlabel("input displacement")
        S.finish(ax)
    axes[0].set_ylabel("response to a unit impulse in output 1")
    axes[1].legend(frameon=False, loc="upper right")
    fig.subplots_adjust(wspace=0.16)
    return S.save(fig, "operator-valued-field")


if __name__ == "__main__":
    print(main())
