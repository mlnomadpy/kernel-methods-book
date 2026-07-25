"""mercer-spectrum-atlas: finite-rank, Matérn-like, and Gaussian regimes."""
from __future__ import annotations

import _style as S
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

S.apply_style()


def main() -> str:
    j = jnp.arange(1, 61, dtype=jnp.float64)
    finite = jnp.where(j <= 10, jnp.exp(-0.16 * (j - 1)), 1e-10)
    matern = j**-2.4
    gaussian = jnp.exp(-0.115 * (j - 1)**1.45)
    spectra = jnp.stack((finite, matern, gaussian))
    lam = jnp.geomspace(1e-4, 1.0, 220)
    neff = jax.vmap(lambda s: jax.vmap(lambda z: jnp.sum(s / (s + z)))(lam))(spectra)
    assert bool(jnp.all(jnp.diff(spectra, axis=1) <= 1e-12))
    assert bool(jnp.all(jnp.diff(neff, axis=1) < 0))
    jj, ss, ll, nn = S.host(j, spectra, lam, neff)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    labels = ("finite rank", "Matérn / Sobolev", "Gaussian")
    colors = (S.ACCENT, S.POS, S.GOOD)
    for s, n, label, color in zip(ss, nn, labels, colors):
        axes[0].semilogy(jj, s, color=color, label=label)
        axes[1].semilogx(ll, n, color=color, label=label)
    axes[0].set(xlabel="Mercer index", ylabel=r"eigenvalue $\lambda_j$", ylim=(1e-8, 1.4))
    axes[1].set(xlabel=r"regularization $\lambda$", ylabel=r"effective dimension $\mathcal{N}(\lambda)$")
    axes[0].legend()
    for ax in axes: S.finish(ax)
    fig.tight_layout()
    return S.save(fig, "mercer-spectrum-atlas")


if __name__ == "__main__":
    print(main())
