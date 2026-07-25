"""spectral-qualification-saturation: bias rates stop improving past qualification."""
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import _style as S
import matplotlib.pyplot as plt

S.apply_style()

def main() -> str:
    lam = jnp.logspace(-4, -0.2, 300)
    nus = (0.5, 1.0, 2.0)
    ridge = jnp.stack([lam ** jnp.minimum(nu, 1.) for nu in nus])
    cutoff = jnp.stack([lam ** nu for nu in nus])
    assert bool(jnp.all(jnp.diff(ridge, axis=1) > 0))
    lh, rh, ch = S.host(lam, ridge, cutoff)
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.7), sharey=True)
    colors = (S.POS, S.ACCENT, S.VIOLET)
    for i, nu in enumerate(nus):
        axes[0].loglog(lh, rh[i], color=colors[i], label=fr"$\nu={nu:g}$")
        axes[1].loglog(lh, ch[i], color=colors[i], label=fr"$\nu={nu:g}$")
    axes[0].set_title("Tikhonov: qualification $1$")
    axes[1].set_title("spectral cutoff: no finite saturation")
    axes[0].set_ylabel("worst-case bias scale")
    for ax in axes:
        ax.set_xlabel("regularization $\\lambda$")
        S.finish(ax)
    axes[0].annotate("smoothness beyond $\\nu=1$\nno longer changes the slope",
                     (2e-3, 2e-3), xytext=(12, -52), textcoords="offset points",
                     color=S.MUTED, arrowprops={"arrowstyle": "-", "color": S.RULE})
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    return S.save(fig, "spectral-qualification-saturation")

if __name__ == "__main__":
    print(main())
