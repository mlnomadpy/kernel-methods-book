"""drift-mmd: a kernel two-sample drift monitor (MMD with a permutation null).

Reproduces ``WIDGETS["drift-mmd"]`` in ``public/assets/viz-drift-mmd.js`` at its
default state and one step of its slider. A fixed reference sample and a
production window are compared with the unbiased MMD^2 U-statistic under an RBF
kernel at the median-heuristic bandwidth; a fast permutation null gives a
finite-sample p-value, exactly as the widget recomputes live:

    MMD^2 = (1/n(n-1)) sum_{i!=j} k(a_i,a_j)
          + (1/n(n-1)) sum_{i!=j} k(b_i,b_j)
          - (2/n^2)     sum_{i,j}  k(a_i,b_j)

with gamma = 1/(2 med^2) from the median pairwise distance of the reference
(widget: sorted[len>>1]; N=80, 200 permutations, k(x,x')=exp(-gamma(x-x')^2)).

The plate contrasts a *matched* window (widget default, covariate shift 0: no
drift, high p) with a *drifted* window (mean shift, low p -> alarm). Numbers
echo checks/ch-accountable-ex4.py (matched p ~ 0.98, drifted p ~ 0.001).
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

N = 80            # sample size per window (widget: N = 80)
NPERM = 200       # permutation-null replicates (widget: NPERM = 200)
SHIFT = 0.75      # covariate shift applied to the drifted window (slider step)


def rbf(a: jnp.ndarray, b: jnp.ndarray, g: float) -> jnp.ndarray:
    d = a[:, None] - b[None, :]
    return jnp.exp(-g * d * d)


def mmd2_unbiased(A: jnp.ndarray, B: jnp.ndarray, g: float) -> jnp.ndarray:
    """Unbiased MMD^2 U-statistic: diagonal dropped in xx/yy, full n^2 in xy."""
    n = A.shape[0]
    Kxx = rbf(A, A, g); Kyy = rbf(B, B, g); Kxy = rbf(A, B, g)
    Kxx = Kxx - jnp.diag(jnp.diag(Kxx))
    Kyy = Kyy - jnp.diag(jnp.diag(Kyy))
    return (Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (n * (n - 1))
            - 2.0 * Kxy.sum() / (n * n))


def median_gamma(ref: jnp.ndarray) -> float:
    """Median-heuristic gamma = 1/(2 med^2); med = sorted pairwise |x_i-x_j|[len>>1]."""
    iu = jnp.triu_indices(N, k=1)
    ds = jnp.sort(jnp.abs(ref[iu[0]] - ref[iu[1]]))
    med = ds[ds.shape[0] >> 1]                     # widget: ds[ds.length >> 1]
    return float(1.0 / (2.0 * med * med))


def permutation_test(ref, win, g, gen):
    """Observed MMD^2, its permutation null, 95% threshold and p-value."""
    obs = float(mmd2_unbiased(ref, win, g))
    pool = np.concatenate([np.asarray(ref), np.asarray(win)])
    null = np.empty(NPERM)
    for p in range(NPERM):
        idx = gen.permutation(2 * N)               # shuffle the pooled sample
        A = jnp.asarray(pool[idx[:N]]); B = jnp.asarray(pool[idx[N:]])
        null[p] = float(mmd2_unbiased(A, B, g))
    thresh = float(np.sort(null)[int(np.floor(0.95 * NPERM))])
    pval = (np.count_nonzero(null >= obs) + 1) / (NPERM + 1)   # (ge + 1)/(NPERM + 1)
    return obs, null, thresh, pval


def _strip(ax, ref, win):
    """Reference (blue, above) vs production window (orange, below) on one axis."""
    ax.axhline(0.0, color=S.RULE, lw=0.8, zorder=1)
    ax.scatter(ref, np.full(N, 0.55), s=10, color=S.POS, alpha=0.85,
               edgecolor="none", zorder=2)
    ax.scatter(win, np.full(N, -0.55), s=10, color=S.NEG, alpha=0.85,
               edgecolor="none", zorder=2)
    ax.set_xlim(-3.6, 4.6); ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([]); ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _null_panel(ax, null, obs, thresh, pval):
    alarm = pval < 0.05
    lo = min(float(null.min()), 0.0)
    hi = max(float(null.max()), obs) * 1.05
    ax.hist(null, bins=26, range=(lo, hi), color=S.MUTED, alpha=0.35,
            edgecolor="none", zorder=1)
    ax.axvline(thresh, color=S.MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    obs_c = S.NEG if alarm else S.GOOD
    ax.axvline(obs, color=obs_c, lw=2.2, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_yticks([])
    ax.set_xlabel(r"$\mathrm{MMD}^2$")
    # Annotate on the side away from the observed bar so nothing overlaps it.
    on_right = (obs - lo) / (hi - lo) > 0.5
    tx, ha = (0.03, "left") if on_right else (0.97, "right")
    tag = "DRIFT DETECTED" if alarm else "no alarm"
    ax.text(tx, 0.94, tag, transform=ax.transAxes, ha=ha, va="top",
            fontsize=8, color=obs_c, weight="bold")
    ax.text(tx, 0.80,
            rf"$\mathrm{{MMD}}^2={obs:+.4f}$" + "\n" + rf"$p={pval:.3f}$",
            transform=ax.transAxes, ha=ha, va="top", fontsize=8.5,
            color=obs_c)
    S.finish(ax)


def main() -> str:
    gen = S.rng(0)
    ref = jnp.asarray(gen.standard_normal(N))       # fixed reference sample
    base = jnp.asarray(gen.standard_normal(N))       # production window at shift 0
    g = median_gamma(ref)

    matched = base                                   # slider default: shift = 0
    drifted = base + SHIFT                            # slider advanced: mean shift

    obs_m, null_m, thr_m, p_m = permutation_test(ref, matched, g, S.rng(1))
    obs_d, null_d, thr_d, p_d = permutation_test(ref, drifted, g, S.rng(1))
    print(f"median-heuristic gamma = {g:.3f}")
    print(f"matched (shift 0.00) : MMD^2 = {obs_m:+.4f}  p = {p_m:.3f}")
    print(f"drifted (shift {SHIFT:.2f}) : MMD^2 = {obs_d:+.4f}  p = {p_d:.3f}")

    fig = plt.figure(figsize=(6.4, 3.7))
    gs = GridSpec(2, 2, height_ratios=[1, 3.1], hspace=0.28, wspace=0.16,
                  figure=fig)

    titles = (rf"matched window  (shift $=0$)",
              rf"drifted window  (shift $={SHIFT:g}$)")
    for col, (win, null, obs, thr, pval, title) in enumerate([
            (matched, null_m, obs_m, thr_m, p_m, titles[0]),
            (drifted, null_d, obs_d, thr_d, p_d, titles[1])]):
        ax_s = fig.add_subplot(gs[0, col])
        _strip(ax_s, np.asarray(ref), np.asarray(win))
        ax_s.set_title(title, color=S.INK, fontsize=9.5, pad=4)
        ax_n = fig.add_subplot(gs[1, col])
        _null_panel(ax_n, null, obs, thr, pval)

    fig.suptitle(r"Kernel MMD drift monitor: statistic vs. permutation null "
                 rf"(RBF, $\gamma={g:.2f}$, $n={N}$, {NPERM} perms)",
                 color=S.INK, fontsize=10, y=1.0)
    # Legend: reference vs window on the strips, threshold/observed on the null.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", ls="none", color=S.POS, label="reference"),
        Line2D([0], [0], marker="o", ls="none", color=S.NEG, label="window"),
        Line2D([0], [0], color=S.MUTED, ls=(0, (4, 3)), label="95% threshold"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.03))
    return S.save(fig, "drift-mmd")


if __name__ == "__main__":
    print(main())
