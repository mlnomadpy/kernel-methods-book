"""krein-positive-negative-decomposition: an indefinite geometry is a difference of PSD parts."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import _style as S

S.apply_style()
jax.config.update("jax_enable_x64", True)


def main() -> str:
    K = jnp.array([[1., .8, -.3, .1], [.8, 1., .5, -.7],
                   [-.3, .5, 1., .6], [.1, -.7, .6, 1.]])
    K = (K + K.T) / 2
    values, vectors = jnp.linalg.eigh(K)
    pos, neg = jnp.maximum(values, 0), jnp.maximum(-values, 0)
    Kp = (vectors * pos) @ vectors.T
    Km = (vectors * neg) @ vectors.T
    residual = jnp.linalg.norm(K - (Kp - Km))
    S.require_finite(values=values, Kp=Kp, Km=Km)
    assert float(values.min()) < 0 < float(values.max())
    assert float(jnp.linalg.eigvalsh(Kp).min()) > -1e-11
    assert float(jnp.linalg.eigvalsh(Km).min()) > -1e-11
    assert float(residual) < 1e-11
    values, K, Kp, Km = S.host(values, K, Kp, Km)
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.15), gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})
    axes[0].bar(range(4), values, color=[S.NEG if v < 0 else S.POS for v in values])
    axes[0].axhline(0, color=S.INK, lw=.7)
    axes[0].set(title="Signed spectrum", xlabel="eigen-index", ylabel="eigenvalue")
    S.finish(axes[0])
    for ax, matrix, title in zip(axes[1:], (K, Kp, Km), (r"$K$", r"$K_+$", r"$K_-$")):
        ax.imshow(matrix, cmap=S.DIVERGING, vmin=-1.35, vmax=1.35)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(wspace=.3)
    return S.save(fig, "krein-positive-negative-decomposition")


if __name__ == "__main__":
    print(main())
