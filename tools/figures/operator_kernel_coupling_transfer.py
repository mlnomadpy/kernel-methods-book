"""operator-kernel-coupling-transfer: coupling helps aligned tasks and hurts opposed tasks."""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
import matplotlib.pyplot as plt
import _style as S

S.apply_style()
jax.config.update("jax_enable_x64", True)


def main() -> str:
    x = jnp.linspace(-2, 2, 45)
    K = jnp.exp(-0.5 * ((x[:, None] - x[None, :]) / .55) ** 2)
    base = jnp.sin(1.7 * x)
    rhos = jnp.linspace(0, .9, 10)

    def error(rho, sign):
        B = jnp.array([[1., rho], [rho, 1.]])
        block = jnp.kron(B, K) + .08 * jnp.eye(2 * len(x))
        y = jnp.concatenate((base, sign * base))
        factor = jnp.linalg.cholesky(block)
        fit = jnp.kron(B, K) @ cho_solve((factor, True), y)
        return jnp.mean((fit[len(x):] - sign * base) ** 2)

    aligned = jax.vmap(lambda r: error(r, 1.))(rhos)
    opposed = jax.vmap(lambda r: error(r, -1.))(rhos)
    S.require_finite(aligned=aligned, opposed=opposed)
    assert float(aligned[-1]) < float(aligned[0])
    assert float(opposed[-1]) > 3 * float(opposed[0])
    rhos, aligned, opposed = S.host(rhos, aligned, opposed)
    fig, ax = S.new_axes()
    ax.plot(rhos, aligned, color=S.GOOD, marker="o", label="aligned outputs")
    ax.plot(rhos, opposed, color=S.NEG, marker="s", label="opposed outputs")
    ax.set(xlabel=r"output coupling $\rho$", ylabel="task-2 mean squared error",
           title="Coupling transfers assumptions, not only information")
    ax.legend()
    S.finish(ax)
    return S.save(fig, "operator-kernel-coupling-transfer")


if __name__ == "__main__":
    print(main())
