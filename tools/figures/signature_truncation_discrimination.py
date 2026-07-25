"""signature-truncation-discrimination: depth reveals path order beyond displacement."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import _style as S

S.apply_style()
jax.config.update("jax_enable_x64", True)


def level_two(increments: jax.Array) -> jax.Array:
    outer = jnp.einsum("ni,nj->nij", increments, increments) / 2
    cross = jnp.einsum("ni,mj->nmij", increments, increments)
    mask = jnp.triu(jnp.ones((len(increments), len(increments))), 1)
    return outer.sum(0) + (cross * mask[:, :, None, None]).sum((0, 1))


def main() -> str:
    paths = jnp.array([[[0., 0.], [1., 0.], [1., 1.]],
                       [[0., 0.], [0., 1.], [1., 1.]],
                       [[0., 0.], [.5, .5], [1., 1.]]])
    inc = jnp.diff(paths, axis=1)
    s1 = inc.sum(1)
    s2 = jax.vmap(level_two)(inc)
    f1 = s1
    f2 = jnp.concatenate((s1, s2.reshape(3, -1)), axis=1)
    d1 = jnp.linalg.norm(f1[:, None] - f1[None, :], axis=-1)
    d2 = jnp.linalg.norm(f2[:, None] - f2[None, :], axis=-1)
    S.require_finite(d1=d1, d2=d2)
    assert float(d1.max()) < 1e-12
    assert float(d2[0, 1]) > 1.0
    d1, d2 = S.host(d1, d2)
    fig, axes = plt.subplots(1, 2, figsize=(5.7, 2.45))
    labels = ["right then up", "up then right", "diagonal"]
    for ax, matrix, title in zip(axes, (d1, d2), ("Depth 1: same endpoint", "Depth 2: order appears")):
        image = ax.imshow(matrix, cmap=S.HEAT, vmin=0, vmax=max(1.4, d2.max()))
        ax.set_xticks(range(3), labels, rotation=28, ha="right")
        ax.set_yticks(range(3), labels)
        ax.set_title(title)
        ax.tick_params(length=0)
    fig.colorbar(image, ax=axes, fraction=.035, pad=.03, label="signature distance")
    fig.subplots_adjust(wspace=.2, right=.88)
    return S.save(fig, "signature-truncation-discrimination")


if __name__ == "__main__":
    print(main())
