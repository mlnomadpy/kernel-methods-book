"""permutation-null: the permutation null of the unbiased MMD^2 U-statistic.

Reproduces the ``permutation-null`` widget in
``public/assets/viz-permutation-null.js`` at its default state: two 1-D samples
of size ``N = 25``, ``X ~ N(0, 1)`` and ``Y ~ N(delta, 1)`` with ``delta = 0.6``,
drawn from the widget's fixed LCG stream (seed ``0xc6ef3620``) so the observed
statistic and the median bandwidth match the browser exactly. The RBF kernel
uses the median heuristic on the pooled 50 pairwise-distance sample.

The observed statistic ``T_0`` is the unbiased MMD^2 U-statistic on the true
labels; the null is built by ``B = 2000`` relabelings of the pooled sample, each
resplitting the 50 points into two groups of 25 and re-summing the one
precomputed 50x50 kernel matrix (zero kernel re-evaluations), exactly as the
widget does. The p-value counts the observed labeling in,

    p_hat = (1 + #{T_pi >= T_0}) / (1 + B).

Kernel, statistic, and median heuristic are computed in JAX (x64). The 2000
relabelings are drawn with ``S.rng(0)`` for a reproducible, byte-stable plate.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import _style as S

jax.config.update("jax_enable_x64", True)
S.apply_style()

N = 25            # points per group
M = 2 * N         # pooled sample size
B = 2000          # permutations
DELTA = 0.6       # widget default separation
ALPHA = 0.05
SEED = 0xC6EF3620  # widget's fixed LCG seed


# ---- exact replica of the widget's deterministic LCG + Box-Muller stream ----
class LCG:
    """The widget's numeric-recipes LCG with cached Box-Muller spare."""

    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFF
        self.spare = 0.0
        self.has_spare = False

    def rnd(self) -> float:  # uniform in (0, 1), never 0
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return (self.s + 0.5) / 4294967296.0

    def rgauss(self) -> float:  # standard normal via Box-Muller on the LCG
        if self.has_spare:
            self.has_spare = False
            return self.spare
        u, v = self.rnd(), self.rnd()
        r = np.sqrt(-2.0 * np.log(u))
        t = 2.0 * np.pi * v
        self.spare = r * np.sin(t)
        self.has_spare = True
        return r * np.cos(t)


def draw_pooled() -> np.ndarray:
    """Pooled sample Z: X in [0, N) ~ N(0,1), Y in [N, M) ~ N(delta, 1)."""
    g = LCG(SEED)
    Z = np.empty(M)
    for i in range(N):
        Z[i] = g.rgauss()
    for j in range(N):
        Z[N + j] = g.rgauss() + DELTA
    return Z


# ---- kernel + statistic in JAX ---------------------------------------------
def build_kernel(Z: jnp.ndarray):
    """Median-heuristic RBF Gram matrix on the pooled sample (widget's math)."""
    d = Z[:, None] - Z[None, :]
    iu = jnp.triu_indices(M, 1)
    med = jnp.median(jnp.abs(d[iu]))         # median of the 1225 pairwise dists
    sigma = med
    K = jnp.exp(-(d ** 2) / (2.0 * sigma * sigma))
    Ksum = jnp.sum(K) - jnp.trace(K)         # total off-diagonal sum
    return K, Ksum, sigma


def stat_of(perm: jnp.ndarray, K: jnp.ndarray, Ksum: jnp.ndarray) -> jnp.ndarray:
    """Unbiased MMD^2 U-statistic of the split first-N vs rest of ``perm``.

    aa, bb are the within-group unordered-pair sums; the cross sum follows from
    Ksum = 2 aa + 2 bb + 2 ab, so ab = Ksum/2 - aa - bb (widget's shortcut).
    """
    a, b = perm[:N], perm[N:]
    aa = (K[a][:, a].sum() - N) / 2.0        # diag entries are 1, so subtract N
    bb = (K[b][:, b].sum() - N) / 2.0
    ab = 0.5 * Ksum - aa - bb
    return 2 * aa / (N * (N - 1)) + 2 * bb / (N * (N - 1)) - 2 * ab / (N * N)


def main() -> str:
    Z = jnp.asarray(draw_pooled())
    K, Ksum, sigma = build_kernel(Z)

    # Observed statistic on the true labels (identity relabeling).
    obs = float(stat_of(jnp.arange(M), K, Ksum))

    # Permutation null: B relabelings of the pooled indices, drawn reproducibly.
    g = S.rng(0)
    perms = jnp.asarray(np.stack([g.permutation(M) for _ in range(B)]))
    null = np.asarray(jax.vmap(lambda p: stat_of(p, K, Ksum))(perms))

    # p-value (observed labeling counted in) and level-alpha critical value.
    ge = int(np.sum(null >= obs))
    p_hat = (1 + ge) / (1 + B)
    c_alpha = float(np.quantile(null, 1.0 - ALPHA))
    reject = p_hat <= ALPHA

    # ---- render -------------------------------------------------------------
    fig, ax = S.new_axes(5.4, 3.2)

    lo = min(float(null.min()), obs)
    hi = max(float(null.max()), obs)
    span = hi - lo
    bins = np.linspace(lo - 0.04 * span, hi + 0.04 * span, 73)  # 72 bins, as widget
    ax.hist(null, bins=bins, color=S.MUTED, alpha=0.55,
            edgecolor=S.PAPER, linewidth=0.3, label="permutation null", zorder=2)

    # Level-alpha critical value: the 0.95 quantile of the null.
    ax.axvline(c_alpha, color=S.INK, lw=1.2, ls=(0, (4, 3)), zorder=3,
               label=rf"$c_\alpha$ ($\alpha={ALPHA:.2f}$)")
    # Observed statistic on the true labels.
    ax.axvline(obs, color=S.ACCENT, lw=2.2, zorder=4, label=r"observed $T_0$")

    ax.set_xlim(lo - 0.08 * span, hi + 0.12 * span)
    ax.set_xlabel(r"$\widehat{\mathrm{MMD}}^2_U$ under relabelings of the pooled sample")
    ax.set_ylabel("count")
    ax.set_title("The permutation null", color=S.INK)

    ax.annotate(
        rf"$T_0 = {obs:.4f}$" + "\n"
        rf"$\hat p = {p_hat:.4f}$" + "\n"
        + (r"reject" if reject else r"retain") + rf" $H_0$ at $\alpha = {ALPHA:.2f}$",
        xy=(0.97, 0.55), xycoords="axes fraction", ha="right", va="top",
        color=S.INK, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc=S.PAPER, ec=S.RULE, lw=0.8))

    ax.legend(loc="upper right", frameon=False, bbox_to_anchor=(1.0, 1.0))
    S.finish(ax)
    return S.save(fig, "permutation-null")


if __name__ == "__main__":
    print(main())
