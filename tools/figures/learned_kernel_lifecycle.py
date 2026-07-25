"""Compile the learned-kernel lifecycle as a TikZ process diagram."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from _tikz import compile_tikz

stages = jnp.arange(5)
assert stages.shape == (5,) and bool(jnp.all(jnp.diff(stages) == 1))
print(compile_tikz("learned-kernel-lifecycle"))
