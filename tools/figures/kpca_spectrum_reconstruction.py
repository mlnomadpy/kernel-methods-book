"""KPCA spectrum and feature-space reconstruction are the same accounting."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
import _style as S
import matplotlib.pyplot as plt

S.apply_style()
key = jr.key(17)
n = 180
theta = jr.uniform(key, (n,), minval=0, maxval=2*jnp.pi)
rad = jnp.where(jnp.arange(n) % 2 == 0, 1.0, 2.0)
X = jnp.c_[rad*jnp.cos(theta), rad*jnp.sin(theta)]
D2 = jnp.sum((X[:, None, :] - X[None, :, :])**2, axis=-1)
K = jnp.exp(-D2 / (2*.55**2))
H = jnp.eye(n) - jnp.ones((n, n))/n
Kc = H @ K @ H
eig = jnp.linalg.eigvalsh((Kc + Kc.T)/2)[::-1]
eig = jnp.maximum(eig, 0)
resid = (jnp.sum(eig) - jnp.cumsum(eig)) / jnp.sum(eig)
assert float(jnp.min(jnp.linalg.eigvalsh((Kc+Kc.T)/2))) > -1e-9
assert bool(jnp.all(jnp.diff(resid) <= 1e-12))
q = jnp.arange(1, 31)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
axes[0].semilogy(S.host(q), S.host(eig[:30]/eig[0]), color=S.POS, marker="o", markevery=3)
axes[0].set(xlabel="kernel principal component", ylabel="relative eigenvalue",
            title="Centered Gram spectrum")
axes[1].plot(S.host(q), S.host(resid[:30]), color=S.ACCENT)
axes[1].fill_between(S.host(q), 0, S.host(resid[:30]), color=S.ACCENT, alpha=.12)
axes[1].set(xlabel="retained components", ylabel="unexplained feature variance",
            title="Spectral tail is reconstruction loss")
for ax in axes: S.finish(ax)
S.save(fig, "kpca-spectrum-reconstruction")
