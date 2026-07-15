"""Worked example 1: the marginalized fixed-length HMM kernel on two short
sequences under a tiny 2-state hidden Markov model.

Alphabet Sigma = {A, B}, two hidden states {1, 2}.

    kappa(s, t) = sum_{h in {1,2}^n} P(s|h) P(t|h) P_M(h),
    P(s|h) = prod_i P(s_i | h_i),
    P_M(h) = P_M(h_1) prod_{i>=2} P_M(h_i | h_{i-1}).

We compute kappa("AB", "AA") two ways: (a) by enumerating the 4 hidden paths
and summing the joint products, and (b) by the forward-style recursion

    kappa_{1,a}   = P(s_1|a) P(t_1|a) P_M(a),
    kappa_{k+1,a} = P(s_{k+1}|a) P(t_{k+1}|a) sum_b P_M(a|b) kappa_{k,b},

with kappa(s,t) = sum_a kappa_{n,a}. The two must agree.
"""

from itertools import product

# emission P(symbol | state)
E = {1: {"A": 0.8, "B": 0.2},
     2: {"A": 0.3, "B": 0.7}}
# initial P_M(a)
pi = {1: 0.6, 2: 0.4}
# transition P_M(a | b): trans[b][a]
trans = {1: {1: 0.7, 2: 0.3},
         2: {1: 0.4, 2: 0.6}}
states = [1, 2]

s, t = "AB", "AA"
n = len(s)

# (a) brute-force enumeration of all hidden paths
print("path        P(s|h)   P(t|h)   P_M(h)     term")
total = 0.0
for h in product(states, repeat=n):
    ps = 1.0
    pt = 1.0
    pm = pi[h[0]]
    for i in range(n):
        ps *= E[h[i]][s[i]]
        pt *= E[h[i]][t[i]]
        if i >= 1:
            pm *= trans[h[i - 1]][h[i]]
    term = ps * pt * pm
    total += term
    print(f"{str(h):10}  {ps:6.3f}   {pt:6.3f}   {pm:6.4f}   {term:.6f}")
print(f"\nkappa(s,t) by enumeration = {total:.6f}")

# (b) forward-style recursion
kappa = {a: E[a][s[0]] * E[a][t[0]] * pi[a] for a in states}
print(f"\nkappa_1,1 = {kappa[1]:.6f}   kappa_1,2 = {kappa[2]:.6f}")
for k in range(1, n):
    new = {}
    for a in states:
        inner = sum(trans[b][a] * kappa[b] for b in states)
        new[a] = E[a][s[k]] * E[a][t[k]] * inner
    kappa = new
    print(f"kappa_{k+1},1 = {kappa[1]:.6f}   kappa_{k+1},2 = {kappa[2]:.6f}")
rec = sum(kappa[a] for a in states)
print(f"\nkappa(s,t) by recursion   = {rec:.6f}")
assert abs(total - rec) < 1e-12
print("agree:", abs(total - rec) < 1e-12)
