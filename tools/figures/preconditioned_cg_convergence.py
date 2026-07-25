"""preconditioned-cg-convergence: spectral clustering accelerates CG."""
from __future__ import annotations

import _style as S
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt

S.apply_style()


def _pcg(a, b, preconditioner, steps=45):
    x = jnp.zeros_like(b)
    r = b - a @ x
    z = preconditioner @ r
    p = z
    rz = r @ z
    history = [jnp.linalg.norm(r) / jnp.linalg.norm(b)]
    for _ in range(steps):
        ap = a @ p
        alpha = rz / (p @ ap)
        x = x + alpha * p
        r = r - alpha * ap
        z = preconditioner @ r
        rz_new = r @ z
        p = z + (rz_new / rz) * p
        rz = rz_new
        history.append(jnp.linalg.norm(r) / jnp.linalg.norm(b))
    return jnp.stack(history)


def main() -> str:
    n = 64
    q, _ = jnp.linalg.qr(jnp.sin((jnp.arange(n)[:, None] + 1) * (jnp.arange(n)[None, :] + .5)))
    spectrum = jnp.geomspace(1e-4, 1.0, n)
    a = (q * spectrum) @ q.T + 2e-4 * jnp.eye(n)
    b = jnp.cos(jnp.linspace(0, 5, n))
    identity = jnp.eye(n)
    diagonal = jnp.diag(1.0 / jnp.diag(a))
    top = spectrum > .04
    inverse_eigs = jnp.where(top, 1.0 / (spectrum + 2e-4), 1.0 / .04)
    low_rank = (q * inverse_eigs) @ q.T
    histories = jnp.stack((_pcg(a, b, identity), _pcg(a, b, diagonal), _pcg(a, b, low_rank)))
    assert bool(jnp.all(jnp.isfinite(histories)))
    assert float(histories[2, 20]) < float(histories[0, 20])
    h = S.host(histories)
    fig, ax = S.new_axes(5.7, 3.15)
    for values, label, color in zip(h, ("CG", "diagonal PCG", "spectral PCG"), (S.ACCENT, S.VIOLET, S.POS)):
        ax.semilogy(range(values.size), values, color=color, label=label)
    ax.axhline(1e-6, color=S.RULE, ls=":", lw=1)
    ax.set(xlabel="iteration", ylabel=r"relative residual $\|r_t\|/\|b\|$", ylim=(1e-10, 2))
    ax.legend()
    S.finish(ax)
    return S.save(fig, "preconditioned-cg-convergence")


if __name__ == "__main__":
    print(main())
