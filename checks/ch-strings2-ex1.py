"""Worked example: the all-(non-contiguous)-subsequences kernel by dynamic
programming (Shawe-Taylor and Cristianini 2004, Algorithm 11.20).

The feature map counts, for every string u, how many times u occurs as a
(possibly gapped) subsequence of s. The kernel k(s,t) is then the number of
common subsequences, the empty subsequence included. We fill the table

    D[i][j] = k(s(1:i), t(1:j))

with the recursion  D[i][j] = D[i-1][j] + sum_{k<=j : t_k = s_i} D[i-1][k-1],
base D[0][j] = D[i][0] = 1. Reproduces the two values of Example 11.18:
k("gatt","cata") = 7 and k("gatta","cata") = 14.
"""

def all_subseq_table(s, t):
    n, m = len(s), len(t)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        D[0][j] = 1
    for i in range(1, n + 1):
        D[i][0] = 1
        for j in range(1, m + 1):
            val = D[i - 1][j]
            for k in range(1, j + 1):
                if t[k - 1] == s[i - 1]:
                    val += D[i - 1][k - 1]
            D[i][j] = val
    return D


def brute_common_subsequences(s, t):
    """Count common subsequences (including empty) by direct enumeration."""
    from itertools import combinations
    def subseqs(x):
        bag = {}
        for r in range(len(x) + 1):
            for idx in combinations(range(len(x)), r):
                u = "".join(x[i] for i in idx)
                bag[u] = bag.get(u, 0) + 1
        return bag
    bs, bt = subseqs(s), subseqs(t)
    return sum(bs[u] * bt[u] for u in bs if u in bt)


s, t = "gatta", "cata"
D = all_subseq_table(s, t)

print("rows index prefixes of s =", s, " cols index prefixes of t =", t)
header = "     eps " + "  ".join(t)
print(header)
labels = ["eps"] + list(s)
for i, row in enumerate(D):
    print(f"{labels[i]:>4}", "  ".join(f"{v:2d}" for v in row))

print()
print("k('gatt','cata')  = D[4][4] =", D[4][4])
print("k('gatta','cata') = D[5][4] =", D[5][4])
print("brute force k('gatt','cata')  =", brute_common_subsequences("gatt", "cata"))
print("brute force k('gatta','cata') =", brute_common_subsequences("gatta", "cata"))
