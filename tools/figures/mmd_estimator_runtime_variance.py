"""mmd-estimator-runtime-variance: computational savings trade against variance."""
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import _style as S
import matplotlib.pyplot as plt

S.apply_style()

def main() -> str:
    n = jnp.logspace(2, 5, 220)
    runtime = jnp.stack((n**2 / 1e6, n*32 / 1e6, n / 1e6))
    variance = jnp.stack((.7/n, 1.8/n, 5.2/n))
    assert bool(jnp.all(jnp.diff(runtime, axis=1) > 0))
    assert bool(jnp.all(jnp.diff(variance, axis=1) < 0))
    nh, rh, vh = S.host(n, runtime, variance)
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.7))
    labels = ("quadratic", "block ($B=32$)", "linear-time")
    colors = (S.POS, S.VIOLET, S.ACCENT)
    for i in range(3):
        axes[0].loglog(nh, rh[i], color=colors[i], label=labels[i])
        axes[1].loglog(nh, vh[i], color=colors[i], label=labels[i])
    axes[0].set(title="kernel-evaluation cost", xlabel="sample size $n$", ylabel="relative work")
    axes[1].set(title="estimator variance", xlabel="sample size $n$", ylabel="variance scale")
    axes[1].legend()
    for ax in axes: S.finish(ax)
    fig.tight_layout()
    return S.save(fig, "mmd-estimator-runtime-variance")

if __name__ == "__main__":
    print(main())
