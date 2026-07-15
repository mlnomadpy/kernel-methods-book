"""ch-text, Example 2: latent semantic kernel (rank-2 LSI) on the tiny corpus.

Same 4 documents and 6 terms as Example 1. We take the raw document-term
matrix A (rows = documents), compute its SVD A = U S V^T via numpy.linalg.svd,
truncate to rank k = 2, project each document onto the top two term-concept
directions (columns of V), and form the latent semantic kernel Gram matrix
    K^LSI = (A V_2)(A V_2)^T = U_2 S_2^2 U_2^T.
The point: documents d1 and d3 share no terms (raw dot product 0) yet receive
a nonzero similarity in the rank-2 concept space, because the hub document d2
makes the four pet terms co-occur, so all four load on one latent concept.
Every number printed here is displayed in the worked example.
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)

terms = ["cat", "kitten", "dog", "puppy", "car", "engine"]
docs = {
    "d1": "cat kitten",
    "d2": "cat kitten kitten dog dog puppy",
    "d3": "dog puppy",
    "d4": "car car engine",
}
names = list(docs)
N, T = len(names), len(terms)
idx = {t: j for j, t in enumerate(terms)}

A = np.zeros((N, T))
for i, n in enumerate(names):
    for w in docs[n].split():
        A[i, idx[w]] += 1.0
print("document-term matrix A =\n", A)

# --- SVD ---
U, s, Vt = np.linalg.svd(A, full_matrices=False)
V = Vt.T
# sign convention: largest-magnitude entry of each concept direction positive
for j in range(V.shape[1]):
    if V[np.argmax(np.abs(V[:, j])), j] < 0:
        V[:, j] *= -1
        U[:, j] *= -1
print("singular values s =", s)
print("right singular vectors V (columns = term-concept directions) =\n", np.round(V, 4))

# --- truncate to rank 2 ---
k = 2
V2 = V[:, :k]
print("concept directions V_2 =\n", np.round(V2, 4))

# document coordinates in concept space
P = A @ V2
print("document coordinates A V_2 =\n", np.round(P, 4))

# latent semantic kernel Gram matrix
Klsi = P @ P.T
print("rank-2 latent kernel K^LSI =\n", np.round(Klsi, 4))

# cosine version
dd = np.sqrt(np.diag(Klsi))
Kn = Klsi / np.outer(dd, dd)
print("cosine latent similarity =\n", np.round(Kn, 4))

print("raw <d1,d3> =", (A @ A.T)[0, 2])
print("latent <d1,d3> =", round(Klsi[0, 2], 4),
      " cosine =", round(Kn[0, 2], 4))
print("latent cosine <d1,d4> =", round(Kn[0, 3], 4), " (pets vs vehicles)")

# --- GVSM: proximity P = A^T (co-occurrence via documents) ---
Gterm = A.T @ A                       # term-term co-occurrence matrix
Kg = A @ Gterm @ A.T                  # GVSM Gram = A A^T A A^T
ddg = np.sqrt(np.diag(Kg))
Kgn = Kg / np.outer(ddg, ddg)
print("term-term co-occurrence A^T A =\n", Gterm)
print("GVSM Gram A(A^T A)A^T =\n", Kg)
print("GVSM cosine <d1,d3> =", round(Kgn[0, 2], 4),
      " raw GVSM <d1,d3> =", Kg[0, 2])
