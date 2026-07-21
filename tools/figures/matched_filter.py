"""matched-filter: detection as projection onto a template.

Reproduces ``WIDGETS["matched-filter"]`` in ``public/assets/viz-matched-filter.js``
at its default injection. A Newtonian-style chirp h (rising frequency under a
Gaussian envelope) is injected into a seeded white-noise stream at the true
arrival lag T0. The noise level sigma = ||h|| / SNR fixes an optimal SNR of 8,
so the chirp is invisible by eye. The matched-filter statistic

    rho(L) = <d, h_L> / (sigma ||h||)

is the noise-weighted inner product (a kernel: the whitened-Hilbert-space dot
product) slid over every lag. At the true frequency it spikes at the true lag
to SNR + a standard-normal draw (~7.37, matching checks/ch-highstakes-ex2.py),
towering over the loudest pure-noise excursion elsewhere.

The browser's LCG noise is replaced by a fixed-seed generator (S.rng(0)) so the
committed plate is byte-stable; all other numbers match the JS exactly.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()

# widget defaults (viz-matched-filter.js)
N = 360        # template length
PAD = 360      # zero-lag padding on each side of the stream
SNR = 8.0      # optimal signal-to-noise ratio ||h|| / sigma
F_TRUE = 14.0  # true chirp centre frequency
T0 = 130       # true arrival lag (samples)


def template(f0: float) -> jnp.ndarray:
    """Chirp under a Gaussian envelope, mean-removed (matches mkTemplate)."""
    t = jnp.arange(N) / N
    env = jnp.exp(-((t - 0.5) ** 2) / (2 * 0.14 * 0.14))
    h = env * jnp.sin(2 * jnp.pi * f0 * (0.6 * t + 0.4 * t * t))
    return h - jnp.mean(h)


def main() -> str:
    # --- template, noise level, injected stream (exact JS construction) ---------
    h_true = template(F_TRUE)
    h_norm = jnp.linalg.norm(h_true)
    sigma = h_norm / SNR

    g = S.rng(0)                                     # replaces the browser LCG
    data = jnp.asarray(sigma * g.standard_normal(N + 2 * PAD))
    data = data.at[PAD + T0:PAD + T0 + N].add(h_true)   # inject at the true lag

    # --- matched filter: slide the (matched) template over every lag ------------
    h = template(F_TRUE)
    hn = jnp.linalg.norm(h)
    lags = jnp.arange(-PAD, PAD + 1)                 # Lg with Lg + N <= N + PAD
    offs = PAD + lags
    windows = data[offs[:, None] + jnp.arange(N)[None, :]]
    rho = windows @ h / (sigma * hn)                 # rho(L) = <d, h_L>/(sigma||h||)

    lags = np.asarray(lags)
    rho = np.asarray(rho)
    peak_i = int(np.argmax(rho))
    peak_val = float(rho[peak_i])
    peak_lag = int(lags[peak_i])
    # loudest pure-noise excursion, away from the true arrival
    off_signal = np.abs(lags - T0) > N // 4
    noise_max = float(rho[off_signal].max())

    # --- plate: raw stream (chirp buried) above, statistic vs lag below ----------
    fig = plt.figure(figsize=(5.4, 3.9))
    gs = GridSpec(2, 1, height_ratios=[3, 5], hspace=0.42, figure=fig)

    data_np = np.asarray(data)
    idx = np.arange(data_np.size)
    ax0 = fig.add_subplot(gs[0])
    ax0.axhline(0.0, color=S.RULE, lw=0.8, zorder=1)
    ax0.plot(idx, data_np, color=S.MUTED, lw=0.5, zorder=2)
    inj = np.arange(PAD + T0, PAD + T0 + N)
    ax0.plot(inj, np.asarray(h_true), color=S.ACCENT, lw=1.1, zorder=3,
             label="injected chirp $h$")
    ax0.set_xlim(0, data_np.size - 1)
    ymax = 1.05 * np.max(np.abs(data_np))
    ax0.set_ylim(-ymax, ymax)
    ax0.set_yticks([])
    ax0.set_xlabel("stream sample", labelpad=2)
    ax0.set_title("raw stream (the chirp is buried in the noise)", color=S.INK)
    ax0.legend(loc="upper right", frameon=False, handlelength=1.4)
    S.finish(ax0)

    ax1 = fig.add_subplot(gs[1])
    ax1.axhline(0.0, color=S.RULE, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax1.axhline(SNR, color=S.MUTED, lw=0.8, ls=(0, (2, 3)), zorder=1)
    ax1.text(lags[0], SNR, r" optimal SNR $=8$", color=S.MUTED,
             va="bottom", ha="left", fontsize=7.5)
    ax1.axvline(T0, color=S.RULE, lw=0.9, ls=(0, (1, 3)), zorder=1)
    ax1.plot(lags, rho, color=S.GOOD, lw=1.4, zorder=3)
    ax1.plot([peak_lag], [peak_val], "o", color=S.GOOD, ms=5,
             markeredgecolor=S.PAPER, markeredgewidth=0.8, zorder=4)
    ax1.annotate(rf"peak $\rho={peak_val:.2f}$ at true lag ${peak_lag}$",
                 xy=(peak_lag, peak_val), xytext=(-330, 9.6),
                 color=S.INK, fontsize=8, va="center", ha="left",
                 arrowprops=dict(arrowstyle="-", color=S.INK, lw=0.7,
                                 shrinkA=2, shrinkB=3))
    ax1.text(T0 - 8, -3.6, "true arrival", color=S.MUTED, ha="right", fontsize=7.5)
    ax1.set_xlim(lags[0], lags[-1])
    ax1.set_ylim(-4, 10.5)
    ax1.set_xlabel("template lag $L$")
    ax1.set_ylabel(r"$\rho(L)=\langle d,h_L\rangle/(\sigma\|h\|)$")
    ax1.set_title("matched-filter statistic: projection onto the template",
                  color=S.INK)
    S.finish(ax1)

    print(f"peak rho = {peak_val:.3f} at lag {peak_lag} (true {T0}); "
          f"||h|| = {float(h_norm):.3f}, sigma = {float(sigma):.4f}; "
          f"loudest noise = {noise_max:.2f}")
    return S.save(fig, "matched-filter")


if __name__ == "__main__":
    print(main())
