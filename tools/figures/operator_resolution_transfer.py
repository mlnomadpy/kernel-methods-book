"""operator-resolution-transfer: finer evaluation cannot recover omitted Fourier modes."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import _style as S

S.apply_style()
jax.config.update("jax_enable_x64", True)


def field(x):
    return jnp.sin(2 * jnp.pi * x) + .35 * jnp.sin(14 * jnp.pi * x)


def low_mode(x):
    return jnp.sin(2 * jnp.pi * x)


def main() -> str:
    resolutions = jnp.array([16, 32, 64, 128, 256, 512])

    def error(n):
        x = jnp.arange(n) / n
        return jnp.sqrt(jnp.mean((field(x) - low_mode(x)) ** 2))

    truncation = jnp.stack([error(int(n)) for n in resolutions.tolist()])
    resolved = jnp.stack([
        jnp.sqrt(jnp.mean((field(jnp.arange(int(n)) / int(n)) -
                           field(jnp.arange(int(n)) / int(n))) ** 2))
        for n in resolutions.tolist()
    ])
    S.require_finite(truncation=truncation, resolved=resolved)
    assert float(jnp.max(jnp.abs(truncation[2:] - truncation[-1]))) < 1e-10
    assert float(resolved.max()) == 0
    resolutions, truncation, resolved = S.host(resolutions, truncation, resolved)
    fig, ax = S.new_axes()
    ax.semilogx(resolutions, truncation, marker="o", color=S.NEG,
                label="fixed low-mode operator")
    ax.semilogx(resolutions, resolved + 1e-5, marker="s", color=S.GOOD,
                label="operator with the missing mode")
    ax.axvspan(64, 512, color=S.POS, alpha=.08, label="grid already resolves target")
    ax.set(xlabel="evaluation grid points", ylabel=r"output $L^2$ error",
           title="Resolution transfer is not approximation improvement")
    ax.legend(fontsize=7)
    S.finish(ax)
    return S.save(fig, "operator-resolution-transfer")


if __name__ == "__main__":
    print(main())
