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
# # Lab 5: KPCA, clustering, CCA, discriminants, and MDS
# **Objectives:** center kernels; solve spectral embeddings; connect KPCA and MDS.  
# **Runtime:** under 1 minute. **Hardware:** CPU. **Data/license:** synthetic circles, CC0.  
# **Seed:** 1729. **Expected output:** centered Gram row sums near zero and a real embedding.  
# Book: [KPCA](../../kernel-pca.html) · [CCA](../../kernel-cca-and-correlation.html) · [MDS](../../data-visualization-and-mds.html)

# %%
import numpy as np
try: from notebooks.lab_utils import centered, rbf_gram, report
except ModuleNotFoundError: from lab_utils import centered, rbf_gram, report

t=np.linspace(0,2*np.pi,50,endpoint=False); x=np.c_[np.cos(t),np.sin(t)]
kc=centered(rbf_gram(x,gamma=2)); vals,vecs=np.linalg.eigh(kc); emb=vecs[:,-2:]*np.sqrt(np.maximum(vals[-2:],0))
assert np.max(np.abs(kc.sum(axis=0)))<1e-10 and np.isrealobj(emb)
report("spectral-methods", leading_eigenvalue=float(vals[-1]), centered_residual=float(np.max(np.abs(kc.sum(0)))))
