"""ch-text, Example 3: neural word embeddings, the embedding-space linear
kernel, and the Word Mover's Distance on a tiny toy embedding.

Setup. Five words live in a 2-D toy embedding space (think of a miniature
word2vec / GloVe table). The four pet words cluster together; the vehicle
word "car" sits far away:

    cat=(1,3)  kitten=(1,4)  dog=(2,3)  puppy=(2,4)  car=(8,0)

Three short documents:
    A = "cat kitten kitten"   (nBOW weights cat 1/3, kitten 2/3)
    B = "dog puppy"           (nBOW weights dog 1/2, puppy 1/2)
    C = "car"                 (nBOW weight  car 1)

A and B share NO word, and A and C share no word either, so the raw
bag-of-words kernel is 0 for both pairs and cannot tell that A is about the
same topic as B but a different one from C.

We compute three things and print every number the worked example / remark
displays:
  (1) the raw bag-of-words dot-product kernel (all the zeros),
  (2) the embedding-space document kernel: represent a document by its
      nBOW-weighted mean word vector (an empirical kernel mean embedding),
      then take the linear kernel <mu_i, mu_j> and its cosine,
  (3) the Word Mover's Distance WMD(d_i,d_j): the 1-Wasserstein / optimal
      transport distance between the two documents-as-distributions with
      Euclidean ground cost ||e_i - e_j||, solved as a small linear program.
Finally we show that the similarity matrix K = -WMD is indefinite (it has a
negative eigenvalue), so WMD is a distance, not a PD kernel.
"""
import numpy as np
from scipy.optimize import linprog

np.set_printoptions(precision=4, suppress=True)

# --- toy 2-D embedding table (a miniature word2vec/GloVe) ---
emb = {
    "cat":    np.array([1.0, 3.0]),
    "kitten": np.array([1.0, 4.0]),
    "dog":    np.array([2.0, 3.0]),
    "puppy":  np.array([2.0, 4.0]),
    "car":    np.array([8.0, 0.0]),
}
terms = list(emb)                       # dictionary order
idx = {t: j for j, t in enumerate(terms)}
E = np.stack([emb[t] for t in terms])   # T x 2 embedding matrix
print("embedding matrix E (rows = words) =\n", E)

docs = {
    "A": "cat kitten kitten",
    "B": "dog puppy",
    "C": "car",
}
names = list(docs)
T = len(terms)


def bow(doc):
    v = np.zeros(T)
    for w in doc.split():
        v[idx[w]] += 1.0
    return v


# --- (1) raw bag-of-words kernel ---
BOW = np.stack([bow(docs[n]) for n in names])   # 3 x T raw counts
print("\nraw bag-of-words count vectors (cols =", terms, ") =\n", BOW)
Kbow = BOW @ BOW.T
print("raw bag-of-words kernel BOW BOW^T =\n", Kbow)
print("raw BoW  <A,B> =", Kbow[0, 1], "   <A,C> =", Kbow[0, 2],
      "  (both zero: no shared words)")

# --- normalised bag-of-words (nBOW) distributions ---
P = BOW / BOW.sum(axis=1, keepdims=True)         # each row sums to 1
print("\nnBOW distributions (rows sum to 1) =\n", P)

# --- (2) embedding-space document kernel ---
# document mean embedding mu_i = sum_t P[i,t] * e_t  (a mean embedding)
MU = P @ E
print("mean word-embedding of each document (mu) =\n", MU)
Klin = MU @ MU.T
print("embedding linear kernel <mu_i,mu_j> =\n", Klin)
dn = np.sqrt(np.diag(Klin))
Kcos = Klin / np.outer(dn, dn)
print("embedding cosine kernel =\n", Kcos)
print("embedding cosine  A~B =", round(Kcos[0, 1], 4),
      "   A~C =", round(Kcos[0, 2], 4))


# --- (3) Word Mover's Distance via optimal transport (linear program) ---
def wmd(i, j, verbose=False):
    """1-Wasserstein distance between nBOW(i) and nBOW(j), Euclidean ground
    cost on word vectors, solved as an OT linear program."""
    a_idx = np.where(P[i] > 0)[0]
    b_idx = np.where(P[j] > 0)[0]
    a = P[i, a_idx]
    b = P[j, b_idx]
    # ground cost matrix C[p,q] = ||e_p - e_q||
    C = np.linalg.norm(E[a_idx][:, None, :] - E[b_idx][None, :, :], axis=2)
    na, nb = len(a_idx), len(b_idx)
    c = C.reshape(-1)
    # equality constraints: row sums = a, col sums = b
    A_eq = np.zeros((na + nb, na * nb))
    for p in range(na):
        A_eq[p, p * nb:(p + 1) * nb] = 1.0
    for q in range(nb):
        A_eq[na + q, q::nb] = 1.0
    b_eq = np.concatenate([a, b])
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (na * nb))
    Topt = res.x.reshape(na, nb)
    if verbose:
        print("  ground cost C =\n", C)
        print("  optimal transport plan T =\n", Topt)
    return res.fun, Topt, a_idx, b_idx


print("\n--- WMD(A,B): a genuine mass-splitting transport ---")
dAB, Tab, ai, bi = wmd(0, 1, verbose=True)
print("  rows(words) =", [terms[k] for k in ai],
      " cols(words) =", [terms[k] for k in bi])
print("WMD(A,B) =", round(dAB, 4))

dAC, _, _, _ = wmd(0, 2)
dBC, _, _, _ = wmd(1, 2)
print("WMD(A,C) =", round(dAC, 4))
print("WMD(B,C) =", round(dBC, 4))

# full 3x3 WMD distance matrix
Dwmd = np.zeros((3, 3))
for i in range(3):
    for j in range(i + 1, 3):
        d, _, _, _ = wmd(i, j)
        Dwmd[i, j] = Dwmd[j, i] = d
print("\nWMD distance matrix (A,B,C) =\n", Dwmd)

# --- WMD is a distance, not a PD kernel: -WMD is indefinite ---
Kneg = -Dwmd
w = np.linalg.eigvalsh(Kneg)
print("eigenvalues of the negative-distance kernel K = -WMD =", np.round(w, 4))
print("trace(K) =", round(float(np.trace(Kneg)), 4),
      " -> eigenvalues sum to 0; signs are mixed, so K is NOT PSD")
print("min eig =", round(float(w.min()), 4),
      "  max eig =", round(float(w.max()), 4))
