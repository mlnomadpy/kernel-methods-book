"""anisotropic-covariance: covariance contours reveal range, direction, and local geometry."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import _style as S

import matplotlib.pyplot as plt

S.apply_style()
jax.config.update("jax_enable_x64", True)


def sigma(p: jax.Array) -> jax.Array:
    angle = 0.45 * jnp.tanh(p[0])
    c, s = jnp.cos(angle), jnp.sin(angle)
    rotation = jnp.array(((c, -s), (s, c)))
    scales = jnp.diag(jnp.array((0.22 + 0.16 * jax.nn.sigmoid(p[0]), 0.75)))
    return rotation @ scales @ rotation.T


def ps_cov(x: jax.Array, y: jax.Array) -> jax.Array:
    sx, sy = sigma(x), sigma(y)
    avg = 0.5 * (sx + sy)
    prefactor = (
        jnp.linalg.det(sx) ** 0.25
        * jnp.linalg.det(sy) ** 0.25
        / jnp.sqrt(jnp.linalg.det(avg))
    )
    delta = x - y
    return prefactor * jnp.exp(-0.5 * delta @ jnp.linalg.solve(avg, delta))


def main() -> str:
    grid = jnp.linspace(-1.8, 1.8, 150)
    X, Y = jnp.meshgrid(grid, grid)
    iso = jnp.exp(-jnp.sqrt(X**2 + Y**2) / 0.7)
    aniso = jnp.exp(-jnp.sqrt((X / 1.25) ** 2 + (Y / 0.38) ** 2))
    points = jnp.column_stack((X.ravel(), Y.ravel()))
    anchor = jnp.array((-0.75, 0.0))
    nonstat = jax.vmap(ps_cov, in_axes=(0, None))(points, anchor).reshape(X.shape)
    assert bool(jnp.all(jnp.isfinite(nonstat)))
    X, Y, iso, aniso, nonstat = map(np.asarray, (X, Y, iso, aniso, nonstat))
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.35), sharex=True, sharey=True)
    for ax, field, title, marker in zip(
        axes,
        (iso, aniso, nonstat),
        ("short isotropic range", "directional anisotropy", "location-dependent range"),
        ((0, 0), (0, 0), np.asarray(anchor)),
    ):
        ax.contour(X, Y, field, levels=(0.2, 0.4, 0.6, 0.8), colors=(S.RULE, S.MUTED, S.POS, S.ACCENT), linewidths=1.0)
        ax.scatter(*marker, color=S.INK, marker="x", s=28)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    sample = jnp.array(((-1.3, -0.4), (-0.7, 0.2), (0.1, -0.2), (0.9, 0.5)))
    K = jax.vmap(lambda a: jax.vmap(lambda b: ps_cov(a, b))(sample))(sample)
    assert bool(jnp.allclose(K, K.T, atol=1e-12))
    assert float(jnp.linalg.eigvalsh(K).min()) > -1e-10
    assert bool(jnp.isclose(iso[75, 75], 1.0, atol=0.03))
    fig.subplots_adjust(wspace=0.04)
    return S.save(fig, "anisotropic-covariance")


if __name__ == "__main__":
    print(main())
