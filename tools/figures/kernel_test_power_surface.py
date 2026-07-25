"""kernel-test-power-surface: bandwidth and sample size jointly determine power."""
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import _style as S
import matplotlib.pyplot as plt

S.apply_style()

def main() -> str:
    n = jnp.linspace(20., 600., 240)
    ell = jnp.logspace(-1.2, 1., 220)
    nn, ll = jnp.meshgrid(n, ell)
    signal = (ll**2 / (ll**2 + .55)) * jnp.exp(-.19 * ll**2)
    power = .05 + .95 * (1. - jnp.exp(-nn * signal**2 / 4.5))
    assert bool(jnp.all((power >= .05) & (power <= 1.)))
    nh, lh, ph = S.host(n, ell, power)
    fig, ax = S.new_axes(5.6, 3.2)
    mesh = ax.pcolormesh(nh, lh, ph, shading="auto", cmap=S.HEAT, vmin=.05, vmax=1)
    ax.contour(nh, lh, ph, levels=[.5, .8, .95], colors="white", linewidths=.7)
    ax.set(yscale="log", xlabel="sample size per distribution", ylabel="kernel bandwidth")
    cb = fig.colorbar(mesh, ax=ax, pad=.02)
    cb.set_label("power at level $0.05$")
    S.finish(ax)
    return S.save(fig, "kernel-test-power-surface")

if __name__ == "__main__":
    print(main())
