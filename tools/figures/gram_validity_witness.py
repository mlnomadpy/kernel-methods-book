"""gram-validity-witness: entrywise plausibility does not certify a kernel."""
from __future__ import annotations

import _style as S
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

S.apply_style()


def main() -> str:
    x = jnp.linspace(-1.0, 1.0, 7)
    distance = jnp.abs(x[:, None] - x[None, :])
    valid = jnp.exp(-2.2 * distance**2)
    invalid = 1.0 / (1.0 + 0.15 * distance) + 0.28 * (distance > 1.0)
    invalid = invalid.at[jnp.diag_indices(x.size)].set(1.0)
    ev_valid = jnp.linalg.eigvalsh(valid)
    ev_invalid = jnp.linalg.eigvalsh(invalid)
    assert bool(jnp.allclose(valid, valid.T))
    assert bool(jnp.allclose(invalid, invalid.T))
    assert bool(jnp.all(valid > 0)) and bool(jnp.all(invalid > 0))
    assert float(ev_valid.min()) > -1e-12
    assert float(ev_invalid.min()) < -1e-2
    a, b, ea, eb = S.host(valid, invalid, ev_valid, ev_invalid)

    fig, axes = plt.subplots(2, 2, figsize=(6.1, 4.6), height_ratios=(1.25, 0.75))
    for ax, matrix, title in zip(axes[0], (a, b), ("Gaussian kernel: PSD", "Positive-looking similarity: invalid")):
        ax.imshow(matrix, cmap=S.HEAT, vmin=0, vmax=1)
        ax.set_title(title)
        ax.set(xticks=[], yticks=[])
    for ax, values, color in zip(axes[1], (ea, eb), (S.POS, S.NEG)):
        ax.axhline(0, color=S.INK, lw=0.8)
        ax.bar(range(values.size), values, color=[color if v < 0 or color == S.POS else S.MUTED for v in values])
        ax.set_yscale("symlog", linthresh=0.05)
        ax.set(xlabel="eigenvalue index", ylabel="Gram eigenvalue")
        S.finish(ax)
    axes[1, 1].text(0.04, 0.88, rf"negative witness: $\lambda_{{\min}}={eb[0]:.3f}$",
                    transform=axes[1, 1].transAxes, color=S.NEG, fontsize=8)
    fig.tight_layout()
    return S.save(fig, "gram-validity-witness")


if __name__ == "__main__":
    print(main())
