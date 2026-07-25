"""Compile the exact ranking-to-difference-space construction in TikZ."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from _tikz import compile_tikz

items = jnp.array([[.4,.7],[1.2,1.0],[1.7,1.8],[2.6,1.55]])
pairs = jnp.array([[3,2],[2,1],[1,0],[3,1]])
differences = items[pairs[:,0]] - items[pairs[:,1]]
assert bool(jnp.all(differences @ jnp.array([.4,1.0]) > 0))
assert bool(jnp.allclose(-differences, items[pairs[:,1]] - items[pairs[:,0]]))
print(compile_tikz("ranking-differences"))
