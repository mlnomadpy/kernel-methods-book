"""dkl-collapse: a learned map can collapse distances and erase GP uncertainty geometry."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import _style as S

import matplotlib.pyplot as plt

S.apply_style()
jax.config.update("jax_enable_x64", True)


def rbf(values: jax.Array, length: float = 0.42) -> jax.Array:
    d = values[:, None] - values[None, :]
    return jnp.exp(-0.5 * (d / length) ** 2)


def main() -> str:
    x = jnp.linspace(-2.0, 2.0, 32)
    learned = 0.17 * jnp.tanh(3.0 * x)
    K_raw, K_collapsed = rbf(x), rbf(learned)
    for gram in (K_raw, K_collapsed):
        assert bool(jnp.all(jnp.isfinite(gram)))
        assert bool(jnp.allclose(gram, gram.T, atol=1e-12))
        assert bool(jnp.allclose(jnp.diag(gram), 1.0, atol=1e-12))
        assert float(jnp.linalg.eigvalsh(gram).min()) > -1e-10
    offdiag_raw = (K_raw.sum() - jnp.trace(K_raw)) / (K_raw.size - len(x))
    offdiag_collapsed = (K_collapsed.sum() - jnp.trace(K_collapsed)) / (K_collapsed.size - len(x))
    assert float(offdiag_collapsed) > 0.85
    assert float(offdiag_collapsed) > 3.0 * float(offdiag_raw)
    K_raw, K_collapsed = map(np.asarray, (K_raw, K_collapsed))
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.55))
    for ax, K, title in zip(
        axes,
        (K_raw, K_collapsed),
        ("Geometry before feature learning", "Collapsed learned geometry"),
    ):
        image = ax.imshow(K, origin="lower", cmap=S.HEAT, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("input index")
        ax.set_ylabel("input index")
        ax.tick_params(length=0)
    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.03, label="RBF similarity")
    fig.subplots_adjust(wspace=0.24, right=0.88)
    return S.save(fig, "dkl-collapse")


if __name__ == "__main__":
    print(main())
