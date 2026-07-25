"""scientific-collocation-convergence: residual and solution error converge at different rates."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
import matplotlib.pyplot as plt
import _style as S

S.apply_style()
jax.config.update("jax_enable_x64", True)


def basis(x, c, gamma):
    d = x[:, None] - c[None, :]
    return jnp.exp(-gamma * d**2)


def minus_second(x, c, gamma):
    d = x[:, None] - c[None, :]
    return (2 * gamma - 4 * gamma**2 * d**2) * jnp.exp(-gamma * d**2)


def trial(n):
    c = jnp.linspace(0, 1, n)
    interior = c[1:-1]
    gamma = 20.0
    A = jnp.vstack((minus_second(interior, c, gamma),
                    10 * basis(jnp.array([0., 1.]), c, gamma)))
    b = jnp.concatenate((jnp.pi**2 * jnp.sin(jnp.pi * interior), jnp.zeros(2)))
    normal = A.T @ A + 1e-8 * jnp.eye(n)
    coef = cho_solve((jnp.linalg.cholesky(normal), True), A.T @ b)
    grid = jnp.linspace(0, 1, 600)
    error = jnp.sqrt(jnp.mean((basis(grid, c, gamma) @ coef - jnp.sin(jnp.pi * grid)) ** 2))
    residual = jnp.sqrt(jnp.mean((minus_second(grid, c, gamma) @ coef -
                                  jnp.pi**2 * jnp.sin(jnp.pi * grid)) ** 2))
    return error, residual


def main() -> str:
    ns = jnp.array([8, 12, 18, 26])
    results = jnp.stack([jnp.asarray(trial(int(n))) for n in ns.tolist()])
    error, residual = results[:, 0], results[:, 1]
    S.require_finite(error=error, residual=residual)
    assert float(error[-1]) < float(error[0])
    assert float(residual[-1]) < float(residual[0])
    ns, error, residual = S.host(ns, error, residual)
    fig, ax = S.new_axes()
    ax.loglog(ns, residual, color=S.ACCENT, marker="s", label="independent PDE residual")
    ax.loglog(ns, error, color=S.POS, marker="o", label=r"solution $L^2$ error")
    ax.set(xlabel="collocation sites", ylabel="root-mean-square error",
           title="More equations reduce two different errors")
    ax.legend()
    S.finish(ax)
    return S.save(fig, "scientific-collocation-convergence")


if __name__ == "__main__":
    print(main())
