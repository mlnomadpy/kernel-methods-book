"""A small discrepancy value is not automatically a tightness certificate."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import _style as S
import matplotlib.pyplot as plt

S.apply_style()
r = jnp.linspace(1.0, 10.0, 90)

def empirical(scale):
    x = scale * jnp.arange(1., 6.)
    g = x[:, None] - x[None, :]
    # Standard-normal target and unit-bandwidth RBF Stein kernel.
    U = (x[:, None] * x[None, :] + 1 - 2*g**2) * jnp.exp(-0.5*g**2)
    u = (jnp.sum(U) - jnp.trace(U)) / (x.size * (x.size - 1))
    v = jnp.mean(U)
    return jnp.abs(u), v, jnp.min(x)

ustat, vstat, support = jax.vmap(empirical)(r)
assert bool(jnp.all(jnp.isfinite(jnp.stack([ustat, vstat, support]))))
assert float(ustat[-1]) < 1e-16
assert float(vstat[-1]) > float(vstat[0])

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
axes[0].semilogy(S.host(r), S.host(ustat + 1e-30), color=S.ACCENT, label=r"$|\widehat{\mathrm{KSD}}_U^2|$")
axes[0].semilogy(S.host(r), S.host(vstat), color=S.POS, label=r"$\widehat{\mathrm{KSD}}_V^2$")
axes[0].set(xlabel="point separation and escape scale", ylabel="empirical diagnostic",
            title="The diagonal-free statistic goes blind")
axes[0].legend()
axes[1].plot(S.host(r), S.host(support), color=S.INK)
axes[1].fill_between(S.host(r), 0, S.host(support), color=S.POS, alpha=.12)
axes[1].set(xlabel="point separation and escape scale", ylabel="nearest support point",
            title="Every point moves into the tail")
for ax in axes: S.finish(ax)
S.save(fig, "ksd-escape-failure")
