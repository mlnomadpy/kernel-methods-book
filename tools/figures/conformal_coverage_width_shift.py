"""conformal-coverage-width-shift: coverage under noise shift requires wider sets."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()


def main() -> str:
    generator = S.rng(31)
    alpha = 0.1
    calibration = jnp.abs(generator.normal(size=3000))
    base_q = jnp.quantile(calibration, 1 - alpha)
    scales = jnp.linspace(0.7, 2.0, 30)
    test_abs = jnp.abs(generator.normal(size=40000))
    fixed_coverage = jax.vmap(lambda s: jnp.mean(s * test_abs <= base_q))(scales)
    adapted_q = scales * jnp.quantile(test_abs, 1 - alpha)
    adapted_coverage = jax.vmap(lambda q, s: jnp.mean(s * test_abs <= q))(adapted_q, scales)
    assert bool(jnp.all(jnp.isfinite(jnp.concatenate((fixed_coverage, adapted_coverage, adapted_q)))))
    assert float(fixed_coverage[-1]) < 0.65
    assert float(jnp.max(jnp.abs(adapted_coverage - (1-alpha)))) < 0.005
    assert bool(jnp.all(adapted_q[1:] >= adapted_q[:-1]))
    scale_h, fixed_h, adapted_h, width_h, base_h = S.host(
        scales, fixed_coverage, adapted_coverage, 2 * adapted_q, 2 * base_q
    )

    fig, axes = plt.subplots(1, 2, figsize=(5.9, 2.55))
    axes[0].plot(scale_h, fixed_h, color=S.NEG, label="frozen calibration")
    axes[0].plot(scale_h, adapted_h, color=S.GOOD, label="shift-aware recalibration")
    axes[0].axhline(0.9, color=S.INK, ls="--", lw=1, label="90% target")
    axes[0].set(title="Coverage breaks when noise changes", xlabel="test / calibration noise",
                ylabel="empirical coverage", ylim=(0.45, 1.0))
    axes[0].legend(fontsize=7)
    axes[1].plot(scale_h, jnp.full_like(scales, base_h), color=S.NEG, label="frozen width")
    axes[1].plot(scale_h, width_h, color=S.ACCENT, label="width needed for 90%")
    axes[1].set(title="Restoring coverage has a visible price", xlabel="test / calibration noise",
                ylabel="mean interval width")
    axes[1].legend(fontsize=7)
    for ax in axes:
        S.finish(ax)
    fig.subplots_adjust(wspace=0.29)
    return S.save(fig, "conformal-coverage-width-shift")


if __name__ == "__main__":
    print(main())
