"""Compile the exact kernel project workflow as a TikZ process diagram."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from _tikz import compile_tikz

route = jnp.arange(6)
assert route.shape == (6,) and bool(jnp.all(jnp.diff(route) == 1))
print(compile_tikz("kernel-workflow"))
