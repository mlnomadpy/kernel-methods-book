"""Compile posterior averaging and its marginalized Gram matrix in TikZ."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from _tikz import compile_tikz

posterior = jnp.array([[.72,.23,.05],[.56,.37,.07],[.08,.24,.68]])
gram = posterior @ posterior.T
assert bool(jnp.allclose(posterior.sum(1), 1))
assert bool(jnp.all(jnp.linalg.eigvalsh(gram) >= -1e-12))
print(compile_tikz("latent-marginalization"))
