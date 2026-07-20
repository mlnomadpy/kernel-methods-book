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
# # Lab 1: Gram matrices, PSD tests, and feature geometry
# **Objectives:** construct Gram matrices; distinguish symmetry from PSD; inspect feature geometry.  
# **Runtime:** under 1 minute fast, under 3 minutes full. **Hardware:** CPU, under 1 GB RAM.  
# **Data/license:** deterministic synthetic points, CC0. **Seed:** 1729.  
# **Expected output:** nonnegative RBF spectrum and a detected indefinite similarity.  
# Book: [RKHS](../../kernels-and-rkhs.html) · [Kernel trick](../../kernel-tricks.html)

# %%
import numpy as np
try: from notebooks.lab_utils import rbf_gram, rng, report
except ModuleNotFoundError: from lab_utils import rbf_gram, rng, report

x = rng(1).normal(size=(24, 3))
k = rbf_gram(x, gamma=0.4)
eig = np.linalg.eigvalsh(k)
bad = k.copy(); bad[0, 0] = -1
assert np.allclose(k, k.T) and eig.min() > -1e-10
assert np.linalg.eigvalsh(bad).min() < 0
report("gram-geometry", min_eigenvalue=float(eig.min()), rank=int(np.linalg.matrix_rank(k)))
