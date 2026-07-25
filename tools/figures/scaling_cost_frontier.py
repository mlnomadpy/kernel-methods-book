"""When does exact kernel storage lose to finite approximations?"""
from __future__ import annotations

import jax
import jax.numpy as jnp

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()

spec = S.PlotSpec(
    name="scaling-cost-frontier",
    question="How quickly do exact and approximate kernel representations consume memory?",
    format="wide",
    x_label="training observations $n$",
    y_label="representation memory (GiB)",
)

n = jnp.logspace(3.0, 6.0, 240)
rank = 1024.0
features = 4096.0
bytes_per_number = 8.0
gib = 2.0**30

exact = bytes_per_number * n**2 / gib
nystrom = bytes_per_number * n * rank / gib
rff = bytes_per_number * n * features / gib

S.require_finite(n=n, exact=exact, nystrom=nystrom, rff=rff)
assert bool(jnp.all(jnp.diff(exact) > 0))
assert bool(jnp.all(jnp.diff(nystrom) > 0))
assert bool(jnp.all(jnp.diff(rff) > 0))
assert bool(jnp.allclose(exact / nystrom, n / rank, rtol=1e-12, atol=0.0))

n_h, exact_h, nystrom_h, rff_h = S.host(n, exact, nystrom, rff)
fig, ax = S.plate(spec)
ax.plot(n_h, exact_h, label="exact Gram matrix", **S.role("error"))
ax.plot(n_h, nystrom_h, label=r"Nyström, $m=1024$", **S.role("geometry"))
ax.plot(n_h, rff_h, label=r"random features, $D=4096$", **S.role("decision"))
S.reference_line(ax, 64.0, label="64 GiB workstation")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(float(n[0]), float(n[-1]))
ax.set_ylim(5e-3, 1e4)
S.annotate(
    ax,
    "quadratic storage\ncrosses the machine",
    xy=(1e5, float(8.0 * (1e5**2) / gib)),
    xytext=(1.8e4, 7e2),
    role_name="error",
)
S.legend(ax, location="upper left")
S.finish(ax)
fig.subplots_adjust(left=0.13, right=0.98, bottom=0.19, top=0.97)
S.save(fig, spec.name)
