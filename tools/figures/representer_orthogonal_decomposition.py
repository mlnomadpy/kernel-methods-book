"""Compile the exact representer decomposition as a TikZ teaching plate."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from _tikz import compile_tikz

p = jnp.array([2.4, 1.0])
q = jnp.array([-0.65, 1.56])
assert bool(jnp.isclose(jnp.vdot(p, q), 0.0))
assert bool(jnp.isclose(jnp.vdot(p + q, p + q), jnp.vdot(p, p) + jnp.vdot(q, q)))
print(compile_tikz("representer-orthogonal-decomposition"))
