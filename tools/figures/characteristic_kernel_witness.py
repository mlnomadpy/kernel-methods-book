"""characteristic-kernel-witness: equal linear means, unequal Gaussian embeddings."""
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import _style as S
import matplotlib.pyplot as plt

S.apply_style()

def mmd2(x, y, ell):
    k = lambda a, b: jnp.exp(-(a[:, None] - b[None, :])**2 / (2*ell**2))
    return jnp.mean(k(x, x)) + jnp.mean(k(y, y)) - 2*jnp.mean(k(x, y))

def main() -> str:
    x = jnp.array([-2., 2.])
    y = jnp.array([-1., 1.])
    t = jnp.linspace(-4., 4., 600)
    ell = .65
    ex = jnp.mean(jnp.exp(-(t[:, None]-x[None, :])**2/(2*ell**2)), axis=1)
    ey = jnp.mean(jnp.exp(-(t[:, None]-y[None, :])**2/(2*ell**2)), axis=1)
    linear = (jnp.mean(x)-jnp.mean(y))**2
    gaussian = mmd2(x, y, ell)
    assert float(linear) == 0.
    assert float(gaussian) > .5
    th, exh, eyh = S.host(t, ex, ey)
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.75))
    axes[0].scatter(S.host(x), [0, 0], color=S.POS, label="$P$")
    axes[0].scatter(S.host(y), [.08, .08], color=S.ACCENT, marker="s", label="$Q$")
    axes[0].axvline(0, color=S.RULE)
    axes[0].set(title="linear kernel: same mean", xlabel="$x$", yticks=[], ylim=(-.03, .20))
    axes[0].text(-3.3, .15, r"$\mathrm{MMD}^2=0$", color=S.MUTED)
    axes[1].plot(th, exh, color=S.POS, label="$\\mu_P(t)$")
    axes[1].plot(th, eyh, color=S.ACCENT, label="$\\mu_Q(t)$")
    axes[1].fill_between(th, exh, eyh, color=S.VIOLET, alpha=.12)
    axes[1].set(title="Gaussian kernel: distinct embeddings", xlabel="$t$")
    for ax in axes: ax.legend(); S.finish(ax)
    fig.tight_layout()
    return S.save(fig, "characteristic-kernel-witness")

if __name__ == "__main__":
    print(main())
