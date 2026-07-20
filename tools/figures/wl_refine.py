"""wl-refine: 1-WL color refinement, round by round, on the pendant pair.

Reproduces ``WIDGETS["wl-refine"]`` in ``public/assets/viz-wl-refine.js``. The
widget refines two graphs on their *disjoint union* with one shared color
dictionary -- exactly the multiset-hash update of Algorithm 24.1 and the input
to the WL subtree kernel. Of the widget's three graph pairs we render the one
whose refinement actually splits colors and separates the two graphs: the
"pend" pair (a 4-cycle carrying two pendant vertices, attached at *adjacent*
cycle vertices in A and at *opposite* ones in B). Both start from the uniform
coloring c0 == 0, both have degree sequence (3,3,2,2,1,1), and 1-WL tells them
apart at round 2 -- the worked Example 24.4 in ch10.

The refinement here is byte-identical to the JS ``refineOnce``: for each vertex
we form the pair (own color, sorted multiset of neighbor colors) and assign a
fresh integer id to each distinct pair in first-appearance order over the
concatenation A-then-B, so both graphs share one dictionary.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import _style as S

S.apply_style()

# --- fixed graph pair, copied verbatim from viz-wl-refine.js (PAIRS.pend) -----
# adjacency lists ...
A_ADJ = [[1, 3, 4], [0, 2, 5], [1, 3], [0, 2], [0], [1]]
B_ADJ = [[1, 3, 4], [0, 2], [1, 3, 5], [0, 2], [0], [2]]
# ... and the hand layouts in [0,1]^2 (layA / layB in the JS).
A_LAY = [[0.32, 0.35], [0.68, 0.35], [0.68, 0.78], [0.32, 0.78], [0.08, 0.1], [0.92, 0.1]]
B_LAY = [[0.32, 0.35], [0.68, 0.35], [0.68, 0.78], [0.32, 0.78], [0.08, 0.1], [0.94, 0.95]]

# WL color id -> book fill. The JS FILLS ramp is a set of fixed hues; here we
# use the book's qualitative palette, one entry per distinct refined class.
PAL = [S.MUTED, S.POS, S.NEG, S.GOOD, S.ACCENT]


def refine_once(colA, colB):
    """One exact 1-WL round on the disjoint union A (+) B, shared dictionary."""
    nA = len(A_ADJ)
    adj = A_ADJ + [[j + nA for j in nb] for nb in B_ADJ]
    col = list(colA) + list(colB)
    sig = [(col[i], tuple(sorted(col[j] for j in adj[i]))) for i in range(len(adj))]
    table = {}
    nc = []
    for s in sig:
        table.setdefault(s, len(table))
        nc.append(table[s])
    return nc[:nA], nc[nA:]


def draw_graph(ax, adj, lay, col):
    lay = np.asarray(lay)
    xs, ys = lay[:, 0], 1.0 - lay[:, 1]      # canvas y grows downward; flip it
    for i, nb in enumerate(adj):             # edges (each once, undirected)
        for j in nb:
            if j > i:
                ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                        color=S.RULE, lw=1.3, zorder=1)
    ax.scatter(xs, ys, s=230, c=[PAL[c % len(PAL)] for c in col],
               edgecolors=S.PAPER, linewidths=1.3, zorder=3)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> str:
    rounds = 3                                # show rounds 0, 1, 2
    colA, colB = [0] * len(A_ADJ), [0] * len(B_ADJ)
    hist = [(list(colA), list(colB))]
    for _ in range(rounds - 1):
        colA, colB = refine_once(colA, colB)
        hist.append((list(colA), list(colB)))

    fig, axes = plt.subplots(2, rounds, figsize=(6.4, 4.5))
    col_titles = ["round 0 (uniform)", "round 1 (split by degree)",
                  "round 2 (separates)"]
    for r in range(rounds):
        cA, cB = hist[r]
        draw_graph(axes[0, r], A_ADJ, A_LAY, cA)
        draw_graph(axes[1, r], B_ADJ, B_LAY, cB)
        axes[0, r].set_title(col_titles[r], color=S.INK, fontsize=9, pad=4)
    axes[0, 0].text(-0.12, 0.5, "graph $A$", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="center", color=S.MUTED)
    axes[1, 0].text(-0.12, 0.5, "graph $B$", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="center", color=S.MUTED)

    fig.suptitle(r"1-WL color refinement: "
                 r"$c_{t+1}(v)=\mathrm{HASH}(c_t(v),\{c_t(u):u\sim v\})$",
                 color=S.INK, fontsize=10, y=0.99)
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.96))
    return S.save(fig, "wl-refine")


if __name__ == "__main__":
    print(main())
