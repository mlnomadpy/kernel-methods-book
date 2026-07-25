"""Bandwidth changes both the Laplacian eigengap and the recovered partition."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import _style as S
import matplotlib.pyplot as plt

S.apply_style()
n = 120
t = jnp.linspace(0, 2*jnp.pi, n//2, endpoint=False)
X = jnp.concatenate([jnp.c_[jnp.cos(t), jnp.sin(t)],
                     jnp.c_[2*jnp.cos(t+.03), 2*jnp.sin(t+.03)]], axis=0)
y = jnp.concatenate([-jnp.ones(n//2), jnp.ones(n//2)])
D2 = jnp.sum((X[:, None, :] - X[None, :, :])**2, axis=-1)
sigmas = jnp.logspace(-1.1, .45, 70)

def score(s):
    W = jnp.exp(-D2/(2*s*s)) - jnp.eye(n)
    d = jnp.sum(W, axis=1)
    Dm = 1/jnp.sqrt(jnp.maximum(d, 1e-12))
    L = jnp.eye(n) - Dm[:, None]*W*Dm[None, :]
    vals, vecs = jnp.linalg.eigh((L+L.T)/2)
    v = vecs[:, 1]
    acc = jnp.maximum(jnp.mean(jnp.sign(v)==y), jnp.mean(jnp.sign(-v)==y))
    return vals[2]-vals[1], acc

gap, acc = jax.vmap(score)(sigmas)
assert bool(jnp.all(jnp.isfinite(jnp.stack([gap, acc]))))
fig, ax = S.new_axes(5.8, 3.0)
gap_scaled = gap / jnp.max(gap)
ax.semilogx(S.host(sigmas), S.host(gap_scaled), color=S.POS, label=r"normalized eigengap")
ax.semilogx(S.host(sigmas), S.host(acc), color=S.ACCENT, label="partition accuracy")
best = int(jnp.argmax(gap))
ax.axvline(float(sigmas[best]), color=S.RULE, ls=":", lw=1)
ax.set(xlabel="graph bandwidth", ylabel="spectral diagnostic",
       title="The useful graph scale opens an eigengap")
ax.legend()
S.finish(ax)
S.save(fig, "spectral-clustering-eigengap")
