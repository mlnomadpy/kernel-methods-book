"""Minimum-kernel interpolant and its sharp power-function envelope."""
import matplotlib.pyplot as plt
from jax import config
import jax.numpy as jnp
import numpy as np

import _style as S

config.update("jax_enable_x64", True)
S.apply_style()
sites = jnp.array([0.25, 0.50, 1.00], dtype=jnp.float64)
values = sites**2
K = jnp.minimum(sites[:, None], sites[None, :])
chol = jnp.linalg.cholesky(K)


def chol_solve(rhs: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.solve(chol.T, jnp.linalg.solve(chol, rhs))


coefficients = chol_solve(values)
grid = jnp.linspace(0, 1, 401, dtype=jnp.float64)
kx = jnp.minimum(grid[:, None], sites[None, :])
interpolant = kx @ coefficients
solved_sections = chol_solve(kx.T).T
power2 = grid - jnp.sum(kx * solved_sections, axis=1)
assert float(jnp.min(power2)) > -1e-12
power = jnp.sqrt(jnp.maximum(power2, 0))
native_norm = 1.0

site_power2 = sites - jnp.sum(K * chol_solve(K.T).T, axis=1)
assert float(jnp.min(site_power2)) > -1e-12
site_power = jnp.sqrt(jnp.maximum(site_power2, 0))
assert bool(jnp.all(jnp.isfinite(interpolant)))
assert float(jnp.max(site_power)) < 1e-7
assert bool(jnp.allclose(interpolant[jnp.array([100, 200, 400])], values, atol=1e-12))
assert float(jnp.linalg.norm(K @ coefficients - values)) < 1e-12
grid, interpolant, power, sites, values = map(np.asarray, (grid, interpolant, power, sites, values))

fig, ax = S.new_axes(5.6, 3.0)
ax.fill_between(grid, interpolant - native_norm * power, interpolant + native_norm * power,
                color=S.POS, alpha=0.18, label="$s_f(x)\\pm P_X(x)$")
ax.plot(grid, grid**2, color=S.INK, lw=1.8, label="target $f(x)=x^2$")
ax.plot(grid, interpolant, color=S.ACCENT, lw=2.1, ls="--", label="minimum-kernel interpolant")
ax.scatter(sites, values, color=S.POS, s=34, zorder=4, label="sites")
ax.set(xlabel="$x$", ylabel="function value", xlim=(0, 1), ylim=(-0.08, 1.08))
ax.legend(frameon=False, loc="upper left")
S.finish(ax)
S.save(fig, "power-function")
print(f"max_power={power.max():.6f}; max_site_power={float(jnp.max(site_power)):.3e}")
