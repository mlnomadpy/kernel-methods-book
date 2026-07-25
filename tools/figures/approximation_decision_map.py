"""Compile the approximation decision map as an exact TikZ plate."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from _tikz import compile_tikz

coordinates = jnp.array(((0.16, 0.18), (0.78, 0.25), (0.30, 0.80), (0.66, 0.72)))
assert coordinates.shape == (4, 2)
assert bool(jnp.all((coordinates >= 0) & (coordinates <= 1)))
print(compile_tikz("approximation-decision-map"))
