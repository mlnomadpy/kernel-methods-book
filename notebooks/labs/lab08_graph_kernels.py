# ---
# kernelspec: {display_name: Python 3, language: python, name: python3}
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
# ---
# %% [markdown]
# # Lab 8: Graph kernels
# **Objectives:** perform WL refinement; build explicit count features; verify graph-kernel PSD.  
# **Runtime:** under 1 minute. **Hardware:** CPU. **Data/license:** hand-built graphs, CC0.  
# **Seed:** 1729. **Expected output:** a PSD WL feature Gram matrix.  
# Book: [Graph kernels](../../graph-kernels.html)

# %%
import numpy as np
from collections import Counter
try: from notebooks.lab_utils import report
except ModuleNotFoundError: from lab_utils import report

graphs=[[[1],[0,2],[1]], [[1,2],[0,2],[0,1]], [[1],[0],[3],[2]]]
features=[]
for adj in graphs:
    labels=["0"]*len(adj); counts=Counter(labels)
    for _ in range(2):
        labels=[labels[i]+":"+",".join(sorted(labels[j] for j in adj[i])) for i in range(len(adj))]
        counts.update(labels)
    features.append(counts)
vocab=sorted({k for f in features for k in f}); phi=np.array([[f[k] for k in vocab] for f in features],float); k=phi@phi.T
assert np.linalg.eigvalsh(k).min()>-1e-10
report("graph-kernels", features=len(vocab), min_eigenvalue=float(np.linalg.eigvalsh(k).min()))
