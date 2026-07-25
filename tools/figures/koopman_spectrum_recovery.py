"""koopman-spectrum-recovery: observable closure controls spectral recovery."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import _style as S

S.apply_style()
jax.config.update("jax_enable_x64", True)


def main() -> str:
    theta, contraction = .42, .93
    A = contraction * jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                                 [jnp.sin(theta), jnp.cos(theta)]])
    x0 = jnp.array([1., .25])
    states = jax.lax.scan(lambda x, _: (A @ x, x), x0, None, length=80)[1]
    X, Y = states[:-1].T, states[1:].T
    recovered = Y @ jnp.linalg.pinv(X, rtol=1e-10)
    exact_ev = jnp.linalg.eigvals(A)
    recovered_ev = jnp.linalg.eigvals(recovered)
    scalar = (states[1:, 0] @ states[:-1, 0]) / (states[:-1, 0] @ states[:-1, 0])
    scalar_res = jnp.linalg.norm(states[1:, 0] - scalar * states[:-1, 0])
    full_res = jnp.linalg.norm(Y - recovered @ X)
    S.require_finite(recovered=recovered, exact_ev=exact_ev, recovered_ev=recovered_ev)
    assert float(jnp.max(jnp.abs(jnp.sort_complex(exact_ev) - jnp.sort_complex(recovered_ev)))) < 1e-10
    assert float(full_res) < 1e-10
    assert float(scalar_res) > .1
    exact_ev, recovered_ev, scalar = S.host(exact_ev, recovered_ev, scalar)
    fig, ax = S.new_axes(width=4.5, height=3.1)
    circle = plt.Circle((0, 0), 1, fill=False, color=S.RULE, lw=.8)
    ax.add_patch(circle)
    ax.scatter(exact_ev.real, exact_ev.imag, s=58, facecolor="none", edgecolor=S.INK, label="true pair")
    ax.scatter(recovered_ev.real, recovered_ev.imag, marker="x", s=45, color=S.GOOD, label="closed 2D observable")
    ax.scatter([scalar], [0], marker="s", s=40, color=S.NEG, label="single coordinate")
    ax.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect="equal",
           xlabel="real part", ylabel="imaginary part", title="Closure decides which dynamics the spectrum can see")
    ax.legend(fontsize=7, loc="lower left")
    S.finish(ax)
    return S.save(fig, "koopman-spectrum-recovery")


if __name__ == "__main__":
    print(main())
