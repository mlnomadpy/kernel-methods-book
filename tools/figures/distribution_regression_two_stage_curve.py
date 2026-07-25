"""Separate finite-bag error from the labelled-bag learning error."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import _style as S
import matplotlib.pyplot as plt

S.apply_style()
N = jnp.logspace(1, 4, 100)
bag_sizes = jnp.array([8., 32., 128., 512.])
irreducible = 0.025
curves = jax.vmap(lambda m: irreducible + 0.72 / jnp.sqrt(N) + 0.46 / jnp.sqrt(m))(bag_sizes)
assert bool(jnp.all(jnp.diff(curves, axis=1) < 0))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
colors = [S.ACCENT, S.VIOLET, S.POS, S.GOOD]
for y, m, c in zip(S.host(curves), S.host(bag_sizes), colors):
    axes[0].loglog(S.host(N), y, color=c, label=f"bag size {int(m)}")
axes[0].set(xlabel="number of labelled bags", ylabel="prediction error",
            title="More bags cannot remove finite-bag error")
axes[0].legend()
m = jnp.logspace(0.7, 3.3, 100)
for total, c in [(2e3, S.ACCENT), (2e4, S.POS), (2e5, S.GOOD)]:
    bags = total / m
    err = irreducible + .72 / jnp.sqrt(bags) + .46 / jnp.sqrt(m)
    axes[1].loglog(S.host(m), S.host(err), color=c, label=f"budget {total:.0e}")
axes[1].set(xlabel="samples per bag", ylabel="prediction error",
            title="A fixed budget has an interior allocation")
axes[1].legend()
for ax in axes: S.finish(ax)
S.save(fig, "distribution-regression-two-stage-curve")
