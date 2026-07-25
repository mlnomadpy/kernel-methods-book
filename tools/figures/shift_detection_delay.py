"""shift-detection-delay: larger distribution changes cross an MMD alarm sooner."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()


def mmd2(x, y, ell=0.8):
    kernel = lambda a, b: jnp.exp(-0.5 * ((a[:, None] - b[None, :]) / ell) ** 2)
    return kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()


def main() -> str:
    generator = S.rng(17)
    reference = generator.normal(size=90)
    stream_noise = generator.normal(size=220)
    shift_time, window = 80, 36
    magnitudes = jnp.array([0.25, 0.5, 0.8, 1.1])
    times = jnp.arange(window, stream_noise.size + 1)

    def trace(delta):
        stream = stream_noise + jnp.where(jnp.arange(stream_noise.size) >= shift_time, delta, 0)
        return jax.vmap(lambda t: mmd2(reference, jax.lax.dynamic_slice(stream, (t-window,), (window,))))(times)

    traces = jax.vmap(trace)(magnitudes)
    threshold = 1.6 * jnp.quantile(traces[:, times <= shift_time], 0.98)

    def first_delay(values):
        after = times >= shift_time
        hits = after & (values > threshold)
        first = jnp.argmax(hits)
        return jnp.where(jnp.any(hits), times[first] - shift_time, times[-1] - shift_time + 1)

    delays = jax.vmap(first_delay)(traces)
    assert bool(jnp.all(jnp.isfinite(traces)))
    assert float(threshold) > 0
    assert int(delays[-1]) < int(delays[0])
    assert int(delays[-1]) <= 24
    t, curves, mags, delay_h, threshold_h = S.host(times, traces, magnitudes, delays, threshold)

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.65), gridspec_kw={"width_ratios": [1.65, 1]})
    colors = [S.MUTED, S.POS, S.VIOLET, S.ACCENT]
    for curve, delta, color in zip(curves, mags, colors):
        axes[0].plot(t, curve, color=color, label=rf"$\Delta={delta:.2g}$")
    axes[0].axvline(shift_time, color=S.INK, ls=":", lw=1)
    axes[0].axhline(threshold_h, color=S.NEG, ls="--", lw=1, label="alarm threshold")
    axes[0].set(title="Sequential MMD after a mean shift", xlabel="stream time", ylabel=r"$\widehat{\mathrm{MMD}}^2$")
    axes[0].legend(ncol=2, fontsize=7)
    axes[1].plot(mags, delay_h, color=S.ACCENT, marker="o")
    axes[1].set(title="Subtle shifts cost time", xlabel="mean-shift magnitude", ylabel="detection delay")
    for ax in axes:
        S.finish(ax)
    fig.subplots_adjust(wspace=0.3)
    return S.save(fig, "shift-detection-delay")


if __name__ == "__main__":
    print(main())
