"""Worked example: 1-WL color refinement, its blind spot, and the WL subtree kernel.

We run one-dimensional Weisfeiler-Leman (1-WL) color refinement on four unlabeled
6-vertex graphs and print every number the worked example displays.

  Pair 1 (1-WL is BLIND):
     C6   = the 6-cycle                         (2-regular, connected)
     2C3  = two disjoint triangles              (2-regular, disconnected)
     Both are 2-regular, so refinement never breaks the uniform coloring:
     their per-iteration color histograms are identical at every step, hence
     the WL subtree kernel cannot separate them (Hilbert distance 0), and by the
     GNN<=1-WL theorem no message-passing GNN can either.

  Pair 2 (1-WL SUCCEEDS):
     A = 4-cycle with two pendants on ADJACENT cycle vertices
     B = 4-cycle with two pendants on OPPOSITE cycle vertices
     Same degree sequence (3,3,2,2,1,1); refinement agrees at iterations 0 and 1
     but diverges at iteration 2 (whether the two degree-3 vertices are adjacent),
     so the histograms differ and the WL subtree kernel separates them.

1-WL update:  c_{t+1}(v) = HASH( c_t(v), multiset{ c_t(u) : u ~ v } ).
The HASH is shared across the two graphs being compared (equivalently, run on
their disjoint union) so the integer color ids are directly comparable.

Pure Python plus numpy for the kernel dot products. Runs in well under a second.
"""

from collections import Counter
import numpy as np


def wl_refine(graphs, num_iters):
    """Run 1-WL jointly on a list of graphs (each an adjacency dict v -> set).

    Returns, for each graph, a list over iterations 0..num_iters of the color
    histogram (a Counter mapping the shared integer color id -> node count).
    The hash is shared across graphs so histograms are comparable.
    """
    # iteration 0: every vertex gets color 0 (unlabeled graphs).
    colors = [{v: 0 for v in g} for g in graphs]
    hists = [[Counter(c.values())] for c in colors]

    next_id = 1
    for _ in range(num_iters):
        signature_to_id = {}
        new_colors = []
        for g, c in zip(graphs, colors):
            nc = {}
            for v in g:
                sig = (c[v], tuple(sorted(c[u] for u in g[v])))
                if sig not in signature_to_id:
                    signature_to_id[sig] = next_id
                    next_id += 1
                nc[v] = signature_to_id[sig]
            new_colors.append(nc)
        colors = new_colors
        for i, c in enumerate(colors):
            hists[i].append(Counter(c.values()))
    return hists


def wl_features(hist_list, vocab):
    """Concatenate per-iteration histograms into one count vector over `vocab`."""
    return np.array([hist_list[t].get(col, 0)
                     for t in range(len(hist_list)) for col in vocab],
                    dtype=float)


def report(name1, g1, name2, g2, num_iters):
    hists = wl_refine([g1, g2], num_iters)
    h1, h2 = hists
    print(f"\n=== {name1}  vs  {name2} ===")
    for t in range(num_iters + 1):
        d1 = dict(sorted(h1[t].items()))
        d2 = dict(sorted(h2[t].items()))
        same = "identical" if d1 == d2 else "DIFFER"
        print(f"  iter {t}: {name1} colors {d1}   {name2} colors {d2}   -> {same}")
    # shared color vocabulary across both graphs and all iterations
    vocab = sorted({col for h in (h1, h2) for ct in h for col in ct})
    f1 = wl_features(h1, vocab)
    f2 = wl_features(h2, vocab)
    k11 = float(f1 @ f1)
    k22 = float(f2 @ f2)
    k12 = float(f1 @ f2)
    d2 = k11 + k22 - 2 * k12
    print(f"  WL subtree kernel: K11={k11:.0f}  K22={k22:.0f}  K12={k12:.0f}")
    print(f"  Hilbert distance^2 d_K^2 = {d2:.0f}"
          f"   -> {'CANNOT distinguish' if d2 == 0 else 'distinguishes'}")


def undirected(edges):
    g = {}
    for a, b in edges:
        g.setdefault(a, set()).add(b)
        g.setdefault(b, set()).add(a)
    return g


# --- Pair 1: two 2-regular graphs 1-WL cannot tell apart ---
C6 = undirected([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)])
C3C3 = undirected([(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])

# --- Pair 2: same degree sequence, 1-WL DOES tell apart ---
# 4-cycle 0-1-2-3-0 with pendants; deg-3 vertices adjacent (A) vs opposite (B).
A = undirected([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5)])
B = undirected([(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (2, 5)])

print("degree sequences:")
print("  C6 ", sorted((len(v) for v in C6.values()), reverse=True))
print("  2C3", sorted((len(v) for v in C3C3.values()), reverse=True))
print("  A  ", sorted((len(v) for v in A.values()), reverse=True))
print("  B  ", sorted((len(v) for v in B.values()), reverse=True))

report("C6", C6, "2C3", C3C3, num_iters=3)
report("A", A, "B", B, num_iters=3)
