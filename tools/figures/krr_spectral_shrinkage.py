"""krr-spectral-shrinkage: ridge retains strong Gram eigendirections."""
from __future__ import annotations

import _style as S
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

S.apply_style()


def main() -> str:
    mu = jnp.geomspace(12.0, 0.015, 14)
    target = jnp.array([1.0, .72, .58, .44, .38, .30, .28, .24, .22, .20, .19, .18, .17, .16])
    lambdas = jnp.array([0.08, 0.8, 4.0])
    factors = mu[None, :] / (mu[None, :] + lambdas[:, None])
    retained = factors * target[None, :]
    assert bool(jnp.all((factors >= 0) & (factors <= 1)))
    assert bool(jnp.all(jnp.diff(factors, axis=0) < 0))
    m, t, fac, ret = S.host(mu, target, factors, retained)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    colors = (S.GOOD, S.POS, S.ACCENT)
    for lam, row, color in zip(lambdas, fac, colors):
        axes[0].semilogx(m, row, color=color, label=rf"$n\lambda={float(lam):g}$")
    axes[0].set(xlabel=r"Gram eigenvalue $\mu_j$", ylabel=r"filter $\mu_j/(\mu_j+n\lambda)$", ylim=(-.03, 1.03))
    axes[0].legend()
    idx = jnp.arange(mu.size)
    axes[1].plot(S.host(idx), t, color=S.INK, marker="o", label="target coefficient")
    for lam, row, color in zip(lambdas, ret, colors):
        axes[1].plot(S.host(idx), row, color=color, label=rf"retained, $n\lambda={float(lam):g}$")
    axes[1].set(xlabel="eigendirection (strong to weak)", ylabel="coefficient magnitude")
    axes[1].legend(ncol=1)
    for ax in axes: S.finish(ax)
    fig.tight_layout()
    return S.save(fig, "krr-spectral-shrinkage")


if __name__ == "__main__":
    print(main())
