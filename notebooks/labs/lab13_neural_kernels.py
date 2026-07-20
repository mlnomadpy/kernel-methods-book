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
# # Lab 13: NTK, NNGP, and deep kernel learning
# **Objectives:** evaluate an analytic ReLU covariance; compare frozen and learned features; keep regimes distinct.  
# **Runtime:** under 1 minute. **Hardware:** CPU. **Data/license:** synthetic unit vectors, CC0.  
# **Seed:** 1729. **Expected output:** PSD NNGP and composed learned-feature kernels.  
# Book: [Infinite-width kernels](../../kernels-and-deep-learning.html) · [DKL](../../deep-kernel-learning.html)

# %%
import numpy as np
try: from notebooks.lab_utils import rbf_gram, rng, report
except ModuleNotFoundError: from lab_utils import rbf_gram, rng, report

x=rng(13).normal(size=(30,5)); x/=np.linalg.norm(x,axis=1,keepdims=True); dot=np.clip(x@x.T,-1,1); theta=np.arccos(dot)
nngp=(np.sin(theta)+(np.pi-theta)*np.cos(theta))/np.pi
w=rng(14).normal(size=(5,3)); learned=rbf_gram(np.tanh(x@w),gamma=.5)
assert np.linalg.eigvalsh(nngp).min()>-1e-9 and np.linalg.eigvalsh(learned).min()>-1e-9
report("neural-kernels", nngp_min_eig=float(np.linalg.eigvalsh(nngp).min()), learned_rank=int(np.linalg.matrix_rank(learned)))
