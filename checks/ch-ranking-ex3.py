"""ch-ranking, Example 3: AUC as a count of correctly ordered pairs.

Bipartite ranking: 3 positive items and 3 negative items receive real-valued
scores from a ranking function. The AUC equals the fraction of positive-negative
pairs the scorer orders correctly (positive scored above negative), which is the
Wilcoxon-Mann-Whitney statistic. It is exactly 1 minus the bipartite ranking
risk (fraction of misordered pos-neg pairs).

Every number printed here appears in the worked example.
"""
import numpy as np

# --- setup ---
pos = np.array([0.9, 0.6, 0.4])   # scores of positive (preferred) items
neg = np.array([0.7, 0.5, 0.2])   # scores of negative items
print("positive scores =", pos)
print("negative scores =", neg)

# --- count concordant / discordant pos-neg pairs ---
concordant = 0
discordant = 0
ties = 0
total = len(pos) * len(neg)
for p in pos:
    for n in neg:
        if p > n:
            concordant += 1
        elif p < n:
            discordant += 1
        else:
            ties += 1
print("total pos-neg pairs =", total)
print("concordant (pos > neg) =", concordant)
print("discordant (pos < neg) =", discordant)
print("ties =", ties)

U = concordant + 0.5 * ties            # Mann-Whitney U
auc = U / total
risk = discordant / total
print("Mann-Whitney U =", U)
print("AUC = U / total =", round(auc, 4))
print("bipartite ranking risk = discordant/total =", round(risk, 4))
print("AUC + risk =", round(auc + risk, 4))
