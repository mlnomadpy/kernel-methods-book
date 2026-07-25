"""nystrom-sampling-comparison: uniform and ridge-leverage landmarks."""
from __future__ import annotations

import _style as S
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import matplotlib.pyplot as plt

S.apply_style()


def _approximation(k, indices):
    c = k[:, indices]
    w = k[jnp.ix_(indices, indices)] + 1e-10 * jnp.eye(indices.size)
    factor = jnp.linalg.cholesky(w)
    return c @ jsl.cho_solve((factor, True), c.T)


def main() -> str:
    x = jnp.concatenate((jnp.linspace(-2.5, -.5, 42), jnp.linspace(.2, .8, 8), jnp.linspace(1.8, 2.3, 10)))
    k = jnp.exp(-0.5 * ((x[:, None] - x[None, :]) / .42)**2)
    ridge = .08
    factor = jnp.linalg.cholesky(k + ridge * jnp.eye(x.size))
    leverage = jnp.diag(k @ jsl.cho_solve((factor, True), jnp.eye(x.size)))
    ranks = jnp.array([4, 6, 8, 10, 12, 16])
    y = jnp.sin(1.7 * x) + .2 * jnp.cos(4.0 * x)
    exact = jsl.cho_solve((factor, True), y)

    def errors(indices):
        khat = _approximation(k, indices)
        fro = jnp.linalg.norm(k - khat) / jnp.linalg.norm(k)
        fh = jnp.linalg.cholesky(khat + ridge * jnp.eye(x.size))
        pred = khat @ jsl.cho_solve((fh, True), y)
        truth = k @ exact
        return fro, jnp.linalg.norm(pred - truth) / jnp.linalg.norm(truth)

    def uniform_indices(m):
        return jnp.linspace(0, x.size - 1, m).round().astype(jnp.int32)

    def leverage_indices(m):
        cdf = jnp.cumsum(leverage) / jnp.sum(leverage)
        quantiles = (jnp.arange(m, dtype=jnp.float64) + .5) / m
        return jnp.searchsorted(cdf, quantiles)

    uniform = jnp.stack([jnp.array(errors(uniform_indices(int(m)))) for m in ranks])
    lev = jnp.stack([jnp.array(errors(leverage_indices(int(m)))) for m in ranks])
    assert bool(jnp.all(jnp.isfinite(jnp.stack((uniform, lev)))))
    assert float(lev[-1, 0]) < float(uniform[-1, 0])
    rr, u, l = S.host(ranks, uniform, lev)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
    for ax, col, ylabel in zip(axes, (0, 1), ("relative matrix error", "relative predictor error")):
        ax.semilogy(rr, u[:, col], marker="o", color=S.ACCENT, label="uniform landmarks")
        ax.semilogy(rr, l[:, col], marker="o", color=S.POS, label="ridge-leverage landmarks")
        ax.set(xlabel="landmarks", ylabel=ylabel)
        S.finish(ax)
    axes[0].legend()
    fig.tight_layout()
    return S.save(fig, "nystrom-sampling-comparison")


if __name__ == "__main__":
    print(main())
