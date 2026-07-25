"""Compile shattering and the XOR obstruction as an exact TikZ plate."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from _tikz import compile_tikz

patterns = jnp.arange(8)
signs = jnp.where(((patterns[:,None] >> jnp.arange(3)) & 1) == 1, 1, -1)
assert jnp.unique(signs, axis=0).shape[0] == 8
square = jnp.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
assert bool(jnp.allclose(square[jnp.array([0,2])].mean(0), square[jnp.array([1,3])].mean(0)))
print(compile_tikz("vc-shattering"))
