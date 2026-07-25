"""dp-fill: the gap-weighted subsequence kernel's dynamic program, filled.

Reproduces ``WIDGETS["dp-fill"]`` in ``public/assets/viz-dp-fill.js`` at the
worked example from the chapter figcaption (``manuscript/chapters/ch-strings2.md``):
strings ``s = "cat"`` and ``t = "car"`` with decay ``lambda = 0.5`` and
subsequence length ``p = 2``.

The widget fills, level by level, the exact recursion of the chapter. For each
level ``l = 1..p`` (with ``DP_0 = 1``):

    DPS_l[i][j] = [ s_i == t_j ] * lambda^2 * DP_{l-1}[i-1][j-1]
    DP_l[i][j]  = DPS_l[i][j] + lambda DP_l[i-1][j]
                  + lambda DP_l[i][j-1] - lambda^2 DP_l[i-1][j-1]
    K_l(s, t)   = sum_{i,j} DPS_l[i][j]

The epsilon row/column (i = 0 or j = 0) are zero. The animated widget renders
``DP_l`` (``levels[viewLevel-1]``) with matched letters (s_i == t_j) in accent
and each cell built from its upper, left, and diagonal neighbours. Here we draw
the two finished tables DP_1 and DP_2 as heatmap grids in the same
rows = s / cols = t layout, then report

    K_2(cat, car) = lambda^4 = 0.0625,
    normalized    = K_2 / sqrt(K_2(s,s) K_2(t,t)) = 1/(2 + lambda^2) = 0.4444.
"""
from __future__ import annotations

import _style as S
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

S.apply_style()

LAM = 0.5   # decay lambda
P = 2       # subsequence length
S_STR = "cat"
T_STR = "car"


def compute_all(s: str, t: str, p: int, lam: float):
    """The widget's ``computeAll``: DP tables per level plus the level kernels.

    Returns ``(levels, Ks)`` where ``levels[l-1]`` is the (n+1)x(m+1) table
    ``DP_l`` and ``Ks[l-1] = sum_{ij} DPS_l[i][j]`` is the level-l kernel.
    """
    n, m = len(s), len(t)
    matches = jnp.asarray(
        [[left == right for right in t] for left in s], dtype=jnp.float64
    )
    levels, Ks = [], []
    dp_prev = jnp.zeros((n + 1, m + 1), dtype=jnp.float64)
    for l in range(1, p + 1):
        predecessor = jnp.ones((n, m)) if l == 1 else dp_prev[:-1, :-1]
        dps = jnp.zeros((n + 1, m + 1), dtype=jnp.float64)
        dps = dps.at[1:, 1:].set(matches * lam**2 * predecessor)
        dp = jnp.zeros((n + 1, m + 1), dtype=jnp.float64)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                value = (
                    dps[i, j]
                    + lam * dp[i - 1, j]
                    + lam * dp[i, j - 1]
                    - lam**2 * dp[i - 1, j - 1]
                )
                dp = dp.at[i, j].set(value)
        kl = jnp.sum(dps)
        assert bool(jnp.all(jnp.isfinite(dp))) and bool(jnp.all(dp >= -1e-14))
        levels.append(dp)
        Ks.append(kl)
        dp_prev = dp
    return levels, Ks


def kernel_only(s: str, t: str, p: int, lam: float) -> float:
    return compute_all(s, t, p, lam)[1][p - 1]


def draw_table(ax, dp: np.ndarray, s: str, t: str, title: str) -> None:
    """Render one DP_l table as a heatmap grid, matching the widget layout:
    rows = letters of s (with epsilon), cols = letters of t (with epsilon)."""
    n, m = len(s), len(t)
    rows = ["ε"] + list(s)   # epsilon then s down the side
    cols = ["ε"] + list(t)   # epsilon then t across the top

    vmax = max(dp.max(), 1e-12)
    ax.imshow(dp, cmap=S.HEAT, vmin=0.0, vmax=vmax, interpolation="nearest")

    ax.set_xticks(range(m + 1)); ax.set_yticks(range(n + 1))
    ax.set_xticklabels(cols, color=S.INK); ax.set_yticklabels(rows, color=S.INK)
    ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color(S.RULE)

    # hairline cell borders
    for k in range(n + 2):
        ax.axhline(k - 0.5, color=S.RULE, lw=0.6)
    for k in range(m + 2):
        ax.axvline(k - 0.5, color=S.RULE, lw=0.6)

    for i in range(n + 1):
        for j in range(m + 1):
            v = dp[i, j]
            matched = i > 0 and j > 0 and s[i - 1] == t[j - 1]
            if matched:  # accent border marks matched letters, as the widget tints matches
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor=S.ACCENT, lw=1.8, zorder=3))
            # light text once the HEAT fill goes dark so every value stays legible
            frac = v / vmax
            colr = S.PAPER if frac > 0.5 else (S.MUTED if v == 0 else S.INK)
            txt = "0" if v == 0 else f"{v:.4g}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color=colr, zorder=4)

    ax.set_title(title, color=S.INK, pad=18)


def main() -> str:
    levels_jax, Ks = compute_all(S_STR, T_STR, P, LAM)
    K = Ks[P - 1]
    self_s = kernel_only(S_STR, S_STR, P, LAM)
    self_t = kernel_only(T_STR, T_STR, P, LAM)
    norm = K / jnp.sqrt(self_s * self_t)
    assert bool(jnp.isclose(K, LAM**4))
    assert bool(jnp.isclose(norm, 1.0 / (2.0 + LAM**2)))
    levels = [np.asarray(level) for level in levels_jax]

    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.1))
    draw_table(axes[0], levels[0], S_STR, T_STR, r"$\mathrm{DP}_1$")
    draw_table(axes[1], levels[1], S_STR, T_STR, r"$\mathrm{DP}_2$")

    fig.suptitle(r"Gap-weighted subsequence DP: $s=$cat, $t=$car "
                 r"($\lambda=0.5$, $p=2$)", color=S.INK, y=1.02)
    fig.text(0.5, -0.06,
             rf"$K_2(\mathrm{{cat}},\mathrm{{car}})=\lambda^4={float(K):.4g}$"
             rf"$\quad$normalized $=K_2/\sqrt{{K_2(s,s)\,K_2(t,t)}}={float(norm):.4f}$",
             ha="center", color=S.INK, fontsize=9)

    fig.subplots_adjust(wspace=0.35)
    return S.save(fig, "dp-fill")


if __name__ == "__main__":
    print(main())
