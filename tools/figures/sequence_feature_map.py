"""Compile the exact substring feature map and Gram matrix in TikZ."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from _tikz import compile_tikz

encoded = jnp.array([[0,1,0,1],[0,0,1,1],[1,0,1,0]])
def counts(row):
    ids = 2 * row[:-1] + row[1:]
    return jnp.sum(jax.nn.one_hot(ids, 4, dtype=jnp.float64), axis=0)
phi = jax.vmap(counts)(encoded)
gram = phi @ phi.T
assert bool(jnp.allclose(phi, jnp.array([[0,2,1,0],[1,1,0,1],[0,1,2,0]])))
assert bool(jnp.all(jnp.linalg.eigvalsh(gram) >= -1e-12))
print(compile_tikz("sequence-feature-map"))
