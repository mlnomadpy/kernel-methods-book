"""Check for ch-vc Example: intervals on the line.

Class F = { indicator of a closed interval [a,b] on R : label +1 inside, -1 outside }.
We verify (i) the growth / shattering coefficient S_F(m) = m(m+1)/2 + 1 by brute force
over m ordered points, (ii) that F shatters 2 points but not 3 (so VC-dim = 2), and
(iii) plugs the growth function into the distribution-free VC risk bound.
"""
import numpy as np
from itertools import product


def realizable_labelings_intervals(m):
    """All 2-labelings of m ordered points realizable by 'inside an interval'.

    A point is labeled +1 iff it falls inside the interval. On m ordered points an
    interval selects a *contiguous* block (possibly empty). Enumerate every block.
    """
    labelings = set()
    # empty interval -> all -1
    labelings.add(tuple([-1] * m))
    for i in range(m):
        for j in range(i, m):  # contiguous block [i..j]
            lab = [-1] * m
            for k in range(i, j + 1):
                lab[k] = 1
            labelings.add(tuple(lab))
    return labelings


def shattering_coefficient(m):
    return len(realizable_labelings_intervals(m))


def formula(m):
    return m * (m + 1) // 2 + 1


print("m : S_F(m) brute  formula m(m+1)/2+1   2^m   shatters?")
for m in [1, 2, 3, 4, 5]:
    s = shattering_coefficient(m)
    f = formula(m)
    assert s == f, (m, s, f)
    print(f"{m} : {s:>12}  {f:>18}  {2**m:>5}   {s == 2**m}")

# All 2^2 = 4 labelings of 2 points are realizable -> shatters 2 points.
labs2 = realizable_labelings_intervals(2)
print("\n#labelings of 2 points =", len(labs2), "(= 2^2 =", 2**2, ") -> shatters")

# On 3 points the labeling (+1,-1,+1) is NOT realizable (non-contiguous) -> no shatter.
labs3 = realizable_labelings_intervals(3)
missing = [lab for lab in product([-1, 1], repeat=3) if lab not in labs3]
print("#labelings of 3 points =", len(labs3), "(< 2^3 =", 2**3, ")")
print("un-realizable 3-point labelings (the split ones):", missing)
print("VC-dim(intervals) = 2")

# ---- Plug the growth function into the VC bound ----
# With prob >= 1 - delta, for all f:  R[f] <= Remp[f] + sqrt( (8/m)(ln S(2m) + ln(4/delta)) ).
m_samp = 1000
delta = 0.05
S2m = formula(2 * m_samp)  # exact growth function at 2m points
conf = np.sqrt((8.0 / m_samp) * (np.log(S2m) + np.log(4.0 / delta)))
print("\nVC bound plug-in (intervals, VC-dim 2):")
print("  n =", m_samp, " delta =", delta)
print("  S_F(2n) =", S2m)
print("  ln S_F(2n) =", round(float(np.log(S2m)), 4))
print("  ln(4/delta) =", round(float(np.log(4.0 / delta)), 4))
print("  confidence term = sqrt((8/n)(ln S + ln(4/delta))) =", round(float(conf), 4))
