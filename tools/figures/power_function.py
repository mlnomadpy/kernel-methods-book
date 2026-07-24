"""Minimum-kernel interpolant and its sharp power-function envelope."""
import matplotlib.pyplot as plt
import numpy as np

import _style as S

S.apply_style()
sites = np.array([0.25, 0.50, 1.00])
values = sites**2
K = np.minimum(sites[:, None], sites[None, :])
coefficients = np.linalg.solve(K, values)
grid = np.linspace(0, 1, 401)
kx = np.minimum(grid[:, None], sites[None, :])
interpolant = kx @ coefficients
power2 = grid - np.einsum("ij,jk,ik->i", kx, np.linalg.inv(K), kx)
power = np.sqrt(np.maximum(power2, 0))
native_norm = 1.0

site_power = np.sqrt(np.maximum(
    sites - np.einsum("ij,jk,ik->i", K, np.linalg.inv(K), K), 0
))
assert np.max(site_power) < 1e-7
assert np.allclose(interpolant[[100, 200, 400]], values)

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
print(f"max_power={power.max():.6f}; max_site_power={site_power.max():.3e}")
