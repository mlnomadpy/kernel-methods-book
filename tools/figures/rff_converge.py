"""rff-converge: random Fourier features approximate the Gaussian kernel.

Reproduces ``WIDGETS["rff-converge"]`` in ``public/assets/viz.js`` at its default
state: the target Gaussian kernel k(x, x') = exp(-(x-x')^2 / 2), drawn as a
function of the gap d = x - x' over [-4, 4] (solid neutral curve), against its
random Fourier feature estimate for the default D = 20 features,

    k-hat(d) = (2/D) sum_j cos(w_j d + b_j) cos(b_j),   w_j ~ N(0,1), b_j ~ U(0, 2pi),

computed in JAX with a fixed seed. The RMSE to the true kernel (evaluated on the
same 201-point grid the widget scans) is reported in the title.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()

D = 20
XR = (-4.0, 4.0)


def features():
    """Default resample: w ~ N(0,1), b ~ U(0, 2pi). Only the first D are used."""
    g = S.rng(0)
    w = jnp.asarray(g.standard_normal(D))
    b = jnp.asarray(g.random(D) * 2.0 * np.pi)
    return w, b


def rff_estimate(d: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    # k-hat(d) = (2/D) sum_j cos(w_j d + b_j) cos(b_j)  (feature approx of k(x,0)).
    terms = jnp.cos(w[None, :] * d[:, None] + b[None, :]) * jnp.cos(b[None, :])
    return 2.0 * jnp.sum(terms, axis=1) / D


def main() -> str:
    w, b = features()
    d = jnp.linspace(*XR, 201)                       # widget scans i = 0..200
    k_true = jnp.exp(-(d ** 2) / 2.0)
    k_hat = rff_estimate(d, w, b)
    rmse = float(jnp.sqrt(jnp.mean((k_hat - k_true) ** 2)))

    dn = np.asarray(d)
    fig, ax = S.new_axes(5.2, 3.1)
    ax.axhline(0.0, color=S.RULE, lw=0.8, ls=(0, (3, 3)))
    ax.plot(dn, np.asarray(k_true), color=S.INK, lw=2.0,
            label=r"$k(d)=e^{-d^2/2}$")
    ax.plot(dn, np.asarray(k_hat), color=S.ACCENT, lw=2.0,
            label=r"$\hat k(d)=\frac{2}{D}\sum_j\cos(w_j d+b_j)\cos b_j$")
    ax.set_xlim(*XR); ax.set_ylim(-0.35, 1.05)
    ax.set_xlabel(r"gap $d = x - x'$"); ax.set_ylabel(r"$k(d)$")
    ax.set_title(rf"Random Fourier features, $D={D}$ -- RMSE $= {rmse:.3f}$",
                 color=S.INK)
    ax.legend(loc="upper right", frameon=False)
    S.finish(ax)
    return S.save(fig, "rff-converge")


if __name__ == "__main__":
    print(main())
