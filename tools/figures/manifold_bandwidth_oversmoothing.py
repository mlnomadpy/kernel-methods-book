"""Graph bandwidth passes from disconnection through propagation to oversmoothing."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import _style as S
import matplotlib.pyplot as plt

S.apply_style()
n = 90
t = jnp.linspace(0, jnp.pi, n//2)
X = jnp.concatenate([jnp.c_[jnp.cos(t), jnp.sin(t)],
                     jnp.c_[1-jnp.cos(t), .48-jnp.sin(t)]], axis=0)
truth = jnp.concatenate([jnp.ones(n//2), -jnp.ones(n//2)])
label_idx = jnp.array([8, 36, 53, 81])
unlab = jnp.setdiff1d(jnp.arange(n), label_idx)
sigmas = jnp.logspace(-2.0, -.05, 70)

def harmonic(s):
    D2 = jnp.sum((X[:, None, :]-X[None, :, :])**2, axis=-1)
    W = jnp.exp(-D2/(2*s*s)) * (D2 > 0)
    L = jnp.diag(jnp.sum(W, axis=1)) - W
    Luu = L[jnp.ix_(unlab, unlab)] + 1e-8*jnp.eye(unlab.size)
    Lul = L[jnp.ix_(unlab, label_idx)]
    fu = jnp.linalg.solve(Luu, -Lul @ truth[label_idx])
    f = jnp.zeros(n).at[label_idx].set(truth[label_idx]).at[unlab].set(fu)
    return jnp.mean(jnp.sign(f)==truth), jnp.abs(jnp.mean(f[:n//2])-jnp.mean(f[n//2:]))

acc, contrast = jax.vmap(harmonic)(sigmas)
assert bool(jnp.all(jnp.isfinite(jnp.stack([acc, contrast]))))
fig, ax = S.new_axes(5.8, 3.0)
ax.semilogx(S.host(sigmas), S.host(acc), color=S.POS, label="label accuracy")
ax.semilogx(S.host(sigmas), S.host(contrast/2), color=S.ACCENT, label="between-manifold contrast")
ax.set(xlabel="graph bandwidth", ylabel="normalized diagnostic",
       title="Too local disconnects; too global erases contrast")
ax.legend()
S.finish(ax)
S.save(fig, "manifold-bandwidth-oversmoothing")
