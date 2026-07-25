"""rollout-error: a small one-step bias compounds after model outputs become model inputs."""
from __future__ import annotations

import matplotlib.pyplot as plt
from jax import config, lax
import jax.numpy as jnp
import numpy as np

import _style as S

config.update("jax_enable_x64", True)
S.apply_style()


def main() -> str:
    horizon = jnp.arange(0, 41, dtype=jnp.int32)

    def rollout(rate: float) -> jnp.ndarray:
        def step(state: jnp.ndarray, _: None) -> tuple[jnp.ndarray, jnp.ndarray]:
            next_state = rate * state
            return next_state, next_state

        _, tail = lax.scan(step, jnp.array(1.0, dtype=jnp.float64), xs=None, length=horizon.size - 1)
        return jnp.concatenate((jnp.ones(1, dtype=jnp.float64), tail))

    truth = rollout(1.045)
    learned = rollout(1.035)
    one_step = abs(1.045 - 1.035)
    error = jnp.abs(truth - learned)
    assert bool(jnp.all(jnp.isfinite(error)))
    assert bool(jnp.isclose(error[1], one_step, atol=1e-14))
    assert float(error[-1]) > 15 * one_step
    assert bool(jnp.all(jnp.diff(error) >= 0.0))
    horizon, truth, learned, error = map(np.asarray, (horizon, truth, learned, error))
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.55))
    axes[0].plot(horizon, truth, color=S.INK, lw=1.8, label="true dynamics")
    axes[0].plot(horizon, learned, color=S.ACCENT, lw=1.7, ls="--", label="learned model")
    axes[0].set(xlabel="rollout step", ylabel="state", title="Repeated prediction")
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].plot(horizon, error, color=S.NEG, lw=1.8)
    axes[1].axhline(one_step, color=S.RULE, lw=1.0, ls=":")
    axes[1].annotate("one-step error", (22, one_step), (16, 0.16), arrowprops={"arrowstyle": "->", "color": S.MUTED}, fontsize=8)
    axes[1].set(xlabel="rollout step", ylabel="absolute state error", title="Compounding bias")
    for ax in axes:
        S.finish(ax)
    fig.subplots_adjust(wspace=0.26)
    return S.save(fig, "rollout-error")


if __name__ == "__main__":
    print(main())
