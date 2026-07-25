"""collocation-residual: PDE and boundary functionals constrain gaps between observations."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
import numpy as np

import _style as S

import matplotlib.pyplot as plt

S.apply_style()
jax.config.update("jax_enable_x64", True)


def basis(x: jax.Array, centers: jax.Array, gamma: float) -> jax.Array:
    return jnp.exp(-gamma * (x[:, None] - centers[None, :]) ** 2)


def minus_second_basis(x: jax.Array, centers: jax.Array, gamma: float) -> jax.Array:
    delta = x[:, None] - centers[None, :]
    K = jnp.exp(-gamma * delta**2)
    return (2 * gamma - 4 * gamma**2 * delta**2) * K


def ridge_solve(design: jax.Array, target: jax.Array, ridge: float) -> jax.Array:
    normal = design.T @ design + ridge * jnp.eye(design.shape[1])
    factor = jnp.linalg.cholesky(normal)
    return cho_solve((factor, True), design.T @ target)


def main() -> str:
    centers = jnp.linspace(0.0, 1.0, 18)
    gamma = 28.0
    obs_x = jnp.array([0.18, 0.53, 0.84])
    obs_y = jnp.sin(jnp.pi * obs_x)
    colloc = jnp.linspace(0.06, 0.94, 15)
    rhs = jnp.pi**2 * jnp.sin(jnp.pi * colloc)
    A_data = basis(obs_x, centers, gamma)
    coef_data = ridge_solve(A_data, obs_y, 2e-3)
    A_phys = jnp.vstack((A_data, 0.22 * minus_second_basis(colloc, centers, gamma), 8.0 * basis(jnp.array([0.0, 1.0]), centers, gamma)))
    b_phys = jnp.concatenate((obs_y, 0.22 * rhs, jnp.zeros(2)))
    coef_phys = ridge_solve(A_phys, b_phys, 2e-3)
    x = jnp.linspace(0.0, 1.0, 400)
    true = jnp.sin(jnp.pi * x)
    fit_data = basis(x, centers, gamma) @ coef_data
    fit_phys = basis(x, centers, gamma) @ coef_phys
    res_data = minus_second_basis(colloc, centers, gamma) @ coef_data - rhs
    res_phys = minus_second_basis(colloc, centers, gamma) @ coef_phys - rhs
    assert bool(jnp.all(jnp.isfinite(jnp.concatenate((coef_data, coef_phys, res_data, res_phys)))))
    assert float(jnp.linalg.norm(res_phys)) < 0.12 * float(jnp.linalg.norm(res_data))
    assert float(jnp.max(jnp.abs(fit_phys[jnp.array((0, -1))]))) < 0.01
    x, true, fit_data, fit_phys, obs_x, obs_y, colloc, res_data, res_phys = map(
        np.asarray, (x, true, fit_data, fit_phys, obs_x, obs_y, colloc, res_data, res_phys)
    )
    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.6))
    axes[0].plot(x, true, color=S.INK, lw=1.4, label="exact solution")
    axes[0].plot(x, fit_data, color=S.MUTED, lw=1.4, ls=":", label="values only")
    axes[0].plot(x, fit_phys, color=S.ACCENT, lw=1.8, label="values + PDE + boundary")
    axes[0].scatter(obs_x, obs_y, color=S.POS, edgecolor=S.INK, s=28, zorder=3)
    axes[0].set(xlabel=r"$x$", ylabel=r"$u(x)$", title="One function, three information types")
    axes[0].legend(frameon=False, fontsize=7, loc="lower center")
    axes[1].plot(colloc, np.abs(res_data), color=S.MUTED, lw=1.4, ls=":", marker="o", ms=3, label="values only")
    axes[1].plot(colloc, np.abs(res_phys), color=S.ACCENT, lw=1.7, marker="s", ms=3, label="with equation")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="independent residual site", ylabel=r"$|{-}\widehat u''-\pi^2\sin(\pi x)|$", title="Equation residual")
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        S.finish(ax)
    fig.subplots_adjust(wspace=0.28)
    return S.save(fig, "collocation-residual")


if __name__ == "__main__":
    print(main())
