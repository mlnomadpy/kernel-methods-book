"""bochner-kernel-spectrum-atlas: spatial profiles paired with nonnegative spectra."""
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import _style as S
import matplotlib.pyplot as plt

S.apply_style()

def main() -> str:
    t = jnp.linspace(-4., 4., 500)
    w = jnp.linspace(-7., 7., 500)
    profiles = (jnp.exp(-t**2 / 2), jnp.exp(-jnp.abs(t)), 1. / (1. + t**2))
    spectra = (jnp.exp(-w**2 / 2), 1. / (1. + w**2), jnp.exp(-jnp.abs(w)))
    names = ("Gaussian", "Laplace", "Cauchy")
    assert all(bool(jnp.all(x >= 0)) for x in spectra)
    assert all(abs(float(x[250]) - 1.) < .02 for x in profiles)
    th, wh = S.host(t, w)
    ph = [S.host(x) for x in profiles]
    sh = [S.host(x / jnp.max(x)) for x in spectra]
    fig, axes = plt.subplots(3, 2, figsize=(5.8, 5.1), sharex="col")
    for i, name in enumerate(names):
        axes[i, 0].plot(th, ph[i], color=S.POS)
        axes[i, 1].plot(wh, sh[i], color=S.ACCENT)
        axes[i, 0].set_ylabel(name)
        for ax in axes[i]: S.finish(ax)
    axes[0, 0].set_title("spatial profile $\\varphi(t)$")
    axes[0, 1].set_title("Bochner density $p(\\omega)$")
    axes[-1, 0].set_xlabel("separation $t$")
    axes[-1, 1].set_xlabel("frequency $\\omega$")
    fig.tight_layout(h_pad=.45, w_pad=.55)
    return S.save(fig, "bochner-kernel-spectrum-atlas")

if __name__ == "__main__":
    print(main())
