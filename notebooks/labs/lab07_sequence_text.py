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
# # Lab 7: Sequence and text kernels
# **Objectives:** construct spectrum features; verify PSD; inspect normalization.  
# **Runtime:** under 1 minute. **Hardware:** CPU. **Data/license:** synthetic strings, CC0.  
# **Seed:** 1729. **Expected output:** PSD normalized trigram Gram matrix.  
# Book: [Sequence kernels](../../string-kernels.html) · [Text kernels](../../kernels-for-text.html)

# %%
import numpy as np
from collections import Counter
try: from notebooks.lab_utils import report
except ModuleNotFoundError: from lab_utils import report

docs=["kernel methods", "kernel machines", "graph methods", "string kernels"]
grams=sorted({s[i:i+3] for s in docs for i in range(len(s)-2)})
phi=np.array([[Counter(s[i:i+3] for i in range(len(s)-2))[g] for g in grams] for s in docs],float)
phi/=np.maximum(np.linalg.norm(phi,axis=1,keepdims=True),1); k=phi@phi.T
assert np.linalg.eigvalsh(k).min()>-1e-10 and np.allclose(np.diag(k),1)
report("sequence-text", vocabulary=len(grams), min_eigenvalue=float(np.linalg.eigvalsh(k).min()))
