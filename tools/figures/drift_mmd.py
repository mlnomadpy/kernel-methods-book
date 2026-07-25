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

Each condition now forms its pooled Gram matrix once.  All permutation
statistics are then evaluated together from membership masks, so the exact
U-statistic is preserved without rebuilding three kernel blocks 200 times.
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


def mmd2_from_gram(K: jnp.ndarray, group_a: jnp.ndarray) -> jnp.ndarray:
    """Balanced-sample unbiased MMD^2 from one pooled Gram matrix.

    ``group_a`` may be one mask ``(2N,)`` or a batch ``(..., 2N)``.  The
    complement defines group B.  Subtracting the selected diagonal exactly
    reproduces the two U-statistic within-sample sums.
    """
    a = group_a.astype(K.dtype)
    b = 1.0 - a
    diag = jnp.diag(K)
    aK = a @ K
    bK = b @ K
    within_a = jnp.sum(aK * a, axis=-1) - jnp.sum(a * diag, axis=-1)
    within_b = jnp.sum(bK * b, axis=-1) - jnp.sum(b * diag, axis=-1)
    cross = jnp.sum(aK * b, axis=-1)
    return (
        within_a / (N * (N - 1))
        + within_b / (N * (N - 1))
        - 2.0 * cross / (N * N)
    )


@jax.jit
def batched_permutation_statistics(
    pooled_grams: jnp.ndarray, masks: jnp.ndarray
) -> jnp.ndarray:
    """MMD^2 for every condition and permutation in one device dispatch."""
    return jax.vmap(mmd2_from_gram, in_axes=(0, None))(pooled_grams, masks)


def median_gamma(ref: jnp.ndarray) -> float:
    """Median-heuristic gamma = 1/(2 med^2); med = sorted pairwise |x_i-x_j|[len>>1]."""
    iu = jnp.triu_indices(N, k=1)
    ds = jnp.sort(jnp.abs(ref[iu[0]] - ref[iu[1]]))
    med = ds[ds.shape[0] >> 1]                     # widget: ds[ds.length >> 1]
    med_host = float(med)
    if not np.isfinite(med_host) or med_host <= 0.0:
        raise FloatingPointError(
            f"median heuristic requires a positive finite distance, got {med_host}"
        )
    return 1.0 / (2.0 * med_host * med_host)


def permutation_masks(gen) -> np.ndarray:
    """Group-A masks for the widget's deterministic permutation stream."""
    permutations = np.stack([gen.permutation(2 * N) for _ in range(NPERM)])
    masks = np.zeros((NPERM, 2 * N), dtype=np.float64)
    rows = np.arange(NPERM)[:, None]
    masks[rows, permutations[:, :N]] = 1.0
    return masks


def summarize_test(obs: float, null: np.ndarray):
    """Observed statistic, permutation null, 95% threshold and p-value."""
    if not np.isfinite(obs) or not np.isfinite(null).all():
        raise FloatingPointError("MMD permutation statistics contain NaN or infinity")
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

    pools = jnp.stack([
        jnp.concatenate([ref, matched]),
        jnp.concatenate([ref, drifted]),
    ])
    pooled_grams = jax.vmap(lambda pool: rbf(pool, pool, g))(pools)
    masks = permutation_masks(S.rng(1))
    observed_mask = np.concatenate([np.ones(N), np.zeros(N)])
    all_masks = jnp.asarray(np.vstack([observed_mask, masks]))
    statistics = np.asarray(
        batched_permutation_statistics(pooled_grams, all_masks)
    )

    grams_host = np.asarray(pooled_grams)
    if not np.isfinite(grams_host).all():
        raise FloatingPointError("pooled RBF Gram matrix contains NaN or infinity")
    if not np.allclose(grams_host, grams_host.transpose(0, 2, 1), atol=1e-13):
        raise AssertionError("pooled RBF Gram matrix lost symmetry")
    if not np.allclose(np.diagonal(grams_host, axis1=1, axis2=2), 1.0,
                       atol=1e-13):
        raise AssertionError("RBF Gram diagonal must equal one")

    obs_m, null_m, thr_m, p_m = summarize_test(
        float(statistics[0, 0]), statistics[0, 1:]
    )
    obs_d, null_d, thr_d, p_d = summarize_test(
        float(statistics[1, 0]), statistics[1, 1:]
    )
    direct = np.array([
        float(mmd2_unbiased(ref, matched, g)),
        float(mmd2_unbiased(ref, drifted, g)),
    ])
    if not np.allclose(statistics[:, 0], direct, rtol=0.0, atol=2e-14):
        raise AssertionError("pooled-Gram and direct MMD U-statistics disagree")
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
