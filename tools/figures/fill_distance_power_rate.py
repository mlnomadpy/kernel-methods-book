"""fill-distance-power-rate: site geometry controls the power-function envelope."""
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsp
import _style as S
import matplotlib.pyplot as plt

S.apply_style()

def kernel(x, y, ell=.13):
    return jnp.exp(-((x[:, None] - y[None, :]) ** 2) / (2 * ell**2))

def power(nodes, grid):
    kxx = kernel(nodes, nodes) + 1e-11 * jnp.eye(nodes.size)
    kx = kernel(nodes, grid)
    c = jnp.linalg.cholesky(kxx)
    solved = jsp.solve_triangular(c, kx, lower=True)
    return jnp.sqrt(jnp.maximum(0., 1. - jnp.sum(solved**2, axis=0)))

def main() -> str:
    grid = jnp.linspace(0., 1., 600)
    quasi = jnp.linspace(0., 1., 11)
    clustered = jnp.concatenate((jnp.linspace(0., .34, 6), jnp.linspace(.72, 1., 5)))
    pq, pc = power(quasi, grid), power(clustered, grid)
    assert float(jnp.max(power(quasi, quasi))) < 2e-5
    assert float(jnp.max(pc)) > 3 * float(jnp.max(pq))
    gh, qh, ch, pqh, pch = S.host(grid, quasi, clustered, pq, pc)
    fig, ax = S.new_axes(5.6, 3.0)
    ax.plot(gh, pqh, color=S.POS, label="quasi-uniform sites")
    ax.plot(gh, pch, color=S.ACCENT, label="clustered sites")
    ax.scatter(qh, jnp.zeros_like(qh), color=S.POS, s=17)
    ax.scatter(ch, -.035*jnp.ones_like(ch), color=S.ACCENT, marker="|", s=55)
    ax.set(xlabel="domain location $x$", ylabel="power function $P_X(x)$", ylim=(-.07, 1.03))
    ax.legend()
    S.finish(ax)
    return S.save(fig, "fill-distance-power-rate")

if __name__ == "__main__":
    print(main())
