"""Worked example: the gap-weighted subsequences kernel of length p=2 between
"cat" and "car", by the efficient dynamic program (Shawe-Taylor and
Cristianini 2004, Section 11.5.2, Computation 11.36).

Each length-2 subsequence u = s(i) is weighted by lambda^{l(i)}, where l(i) is
the span of the subsequence in the string (last index - first index + 1). The
suffix kernel and the auxiliary table DP_2 obey

    kS_1(i,j) = [s_i = t_j] lambda^2
    DP_p(k,l) = kS_{p-1}(k,l) + lam*DP_p(k-1,l) + lam*DP_p(k,l-1)
                - lam^2 * DP_p(k-1,l-1)
    kS_p(k,l) = [s_k = t_l] * lambda^2 * DP_p(k-1,l-1)
    k_p(s,t)  = sum_{k,l} kS_p(k,l)

Polynomials in lambda are carried exactly (coefficient dict power->coeff) and
also evaluated at lambda = 1/2. Reproduces Example 11.30: k("cat","car")=lam^4,
self-kernels 2 lam^4 + lam^6, normalized value (2 + lam^2)^{-1}.
"""

from collections import defaultdict


def padd(*polys):
    out = defaultdict(int)
    for p in polys:
        for k, v in p.items():
            out[k] += v
    return {k: v for k, v in out.items() if v != 0}


def pscale(p, coeff, shift):
    """Multiply polynomial p by coeff * lambda^shift."""
    return {k + shift: v * coeff for k, v in p.items()}


def pstr(p):
    if not p:
        return "0"
    terms = []
    for k in sorted(p):
        c = p[k]
        if k == 0:
            terms.append(f"{c}")
        elif c == 1:
            terms.append(f"lam^{k}")
        else:
            terms.append(f"{c}*lam^{k}")
    return " + ".join(terms)


def peval(p, lam):
    return sum(c * lam ** k for k, c in p.items())


def gap_kernel(s, t, p, lam=0.5):
    n, m = len(s), len(t)
    # kS_1 table (1-indexed via dicts)
    kS = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            kS[(i, j)] = {2: 1} if s[i - 1] == t[j - 1] else {}
    tables = {1: dict(kS)}
    for level in range(2, p + 1):
        DP = {}
        for k in range(0, n + 1):
            for l in range(0, m + 1):
                if k == 0 or l == 0:
                    DP[(k, l)] = {}
                else:
                    DP[(k, l)] = padd(
                        tables[level - 1].get((k, l), {}),
                        pscale(DP[(k - 1, l)], 1, 1),
                        pscale(DP[(k, l - 1)], 1, 1),
                        pscale(DP[(k - 1, l - 1)], -1, 2),
                    )
        kSp = {}
        for k in range(1, n + 1):
            for l in range(1, m + 1):
                if s[k - 1] == t[l - 1]:
                    kSp[(k, l)] = pscale(DP[(k - 1, l - 1)], 1, 2)
                else:
                    kSp[(k, l)] = {}
        tables[level] = kSp
        last_DP = DP
    kernel = {}
    for k in range(1, n + 1):
        for l in range(1, m + 1):
            kernel = padd(kernel, tables[p][(k, l)])
    return kernel, last_DP, tables


def gap_kernel_numeric_full(s, t, p, lam):
    """Full auxiliary tables in floating-point arithmetic."""
    n, m = len(s), len(t)
    suffix = [
        [lam ** 2 if s[i] == t[j] else 0.0 for j in range(m)]
        for i in range(n)
    ]
    if p == 1:
        return sum(map(sum, suffix))
    for _level in range(2, p + 1):
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        next_suffix = [[0.0] * m for _ in range(n)]
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                dp[i][j] = (
                    suffix[i - 1][j - 1]
                    + lam * dp[i - 1][j]
                    + lam * dp[i][j - 1]
                    - lam ** 2 * dp[i - 1][j - 1]
                )
                if s[i - 1] == t[j - 1]:
                    next_suffix[i - 1][j - 1] = lam ** 2 * dp[i - 1][j - 1]
        suffix = next_suffix
    return sum(map(sum, suffix))


def gap_kernel_numeric_rolling(s, t, p, lam):
    """Two-row auxiliary table while retaining the level suffix table."""
    n, m = len(s), len(t)
    suffix = [
        [lam ** 2 if s[i] == t[j] else 0.0 for j in range(m)]
        for i in range(n)
    ]
    if p == 1:
        return sum(map(sum, suffix))
    for _level in range(2, p + 1):
        previous = [0.0] * (m + 1)
        next_suffix = [[0.0] * m for _ in range(n)]
        for i in range(1, n + 1):
            current = [0.0] * (m + 1)
            for j in range(1, m + 1):
                current[j] = (
                    suffix[i - 1][j - 1]
                    + lam * previous[j]
                    + lam * current[j - 1]
                    - lam ** 2 * previous[j - 1]
                )
                if s[i - 1] == t[j - 1]:
                    next_suffix[i - 1][j - 1] = lam ** 2 * previous[j - 1]
            previous = current
        suffix = next_suffix
    return sum(map(sum, suffix))


lam = 0.5
p = 2
s, t = "cat", "car"
kst, DP2, tabs = gap_kernel(s, t, p, lam)

print("DP_2 table  (rows s =", s, ", cols t =", t, ")")
print("      " + "         ".join(list(t)))
for k in range(1, len(s) + 1):
    row = []
    for l in range(1, len(t) + 1):
        row.append(pstr(DP2[(k, l)]))
    print(f"{s[k-1]:>3}  " + "   |   ".join(row))

print()
print("DP_2 numeric at lambda = 0.5:")
for k in range(1, len(s) + 1):
    print(f"{s[k-1]:>3} ", [round(peval(DP2[(k, l)], lam), 6) for l in range(1, len(t) + 1)])

print()
print("k_2('cat','car')  =", pstr(kst), " = ", round(peval(kst, lam), 6))
kss, _, _ = gap_kernel("cat", "cat", 2, lam)
ktt, _, _ = gap_kernel("car", "car", 2, lam)
print("k_2('cat','cat')  =", pstr(kss), " = ", round(peval(kss, lam), 6))
print("k_2('car','car')  =", pstr(ktt), " = ", round(peval(ktt, lam), 6))
norm = peval(kst, lam) / (peval(kss, lam) * peval(ktt, lam)) ** 0.5
print("normalized k_hat  = k / sqrt(kss*ktt) =", round(norm, 6))
print("closed form (2+lam^2)^-1 =", round(1.0 / (2 + lam ** 2), 6))

# Verify that rolling the auxiliary table is cell-order equivalent to storing
# it in full across varied lengths, decay factors, and repeated symbols.
from itertools import product
test_strings = [
    "".join(chars)
    for length in range(1, 5)
    for chars in product("ab", repeat=length)
]
cases = 0
for left in test_strings:
    for right in test_strings:
        for level in range(1, min(len(left), len(right)) + 1):
            for decay in (0.2, 0.5, 0.9, 1.0):
                full = gap_kernel_numeric_full(left, right, level, decay)
                rolling = gap_kernel_numeric_rolling(left, right, level, decay)
                assert abs(full - rolling) <= 1e-12 * max(1.0, abs(full))
                cases += 1
print("full/rolling gap-DP agreement:", cases, "parameterized cases")
