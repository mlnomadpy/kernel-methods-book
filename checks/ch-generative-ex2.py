"""Worked example 2: the Fisher score and Fisher kernel for a tiny iid
2-symbol generative model.

Model: symbols drawn iid from Sigma = {A, B} with unnormalized weights
theta = (theta_A, theta_B) and P(sigma | theta) = theta_sigma / (theta_A + theta_B).
For a sequence s of length n with symbol counts (n_A, n_B),

    log L(s) = sum_sigma n_sigma log theta_sigma - n log(theta_A + theta_B),
    score_sigma = d/dtheta_sigma log L = n_sigma / theta_sigma - n / Z.

At the uniform setting theta_A = theta_B = 1 (so Z = 2, P uniform = 1/2):

    g(s) = (n_A - n/2, n_B - n/2),

which lies on the line g_A + g_B = 0 (the centering the p-spectrum Fisher
kernel exhibits). The Fisher kernel with I replaced by the identity is
K(s, t) = g(s) . g(t). We also form the single-symbol Fisher information
matrix I_1 = E[g g'] under the uniform model and check it is singular.
"""

def counts(x):
    return x.count("A"), x.count("B")

def score(x):
    nA, nB = counts(x)
    n = len(x)
    return (nA - n / 2.0, nB - n / 2.0)

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]

seqs = {"s": "AAB", "t": "ABB", "u": "AAA"}
for name, x in seqs.items():
    nA, nB = counts(x)
    print(f"{name} = {x:4}  (n_A,n_B)=({nA},{nB})  g = {score(x)}")

g = {k: score(v) for k, v in seqs.items()}
print()
print(f"K(s,s) = {dot(g['s'], g['s']):.4f}")
print(f"K(t,t) = {dot(g['t'], g['t']):.4f}")
print(f"K(s,t) = {dot(g['s'], g['t']):.4f}")
print(f"K(s,u) = {dot(g['s'], g['u']):.4f}")

# single-symbol Fisher information under the uniform model:
# per-symbol score is (1[x=A]-1/2, 1[x=B]-1/2); average g g' over x uniform.
import itertools
I = [[0.0, 0.0], [0.0, 0.0]]
for x, p in [("A", 0.5), ("B", 0.5)]:
    gx = (1.0 * (x == "A") - 0.5, 1.0 * (x == "B") - 0.5)
    for a in range(2):
        for b in range(2):
            I[a][b] += p * gx[a] * gx[b]
det = I[0][0] * I[1][1] - I[0][1] * I[1][0]
print()
print(f"I_1 = {I}")
print(f"det(I_1) = {det:.4f}  (singular -> rank 1, why I is set to identity)")
