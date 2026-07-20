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
# # Lab 3: SVM, SVR, one-class learning, and ranking
# **Objectives:** compare four margin objectives; inspect support vectors; verify pairwise ranking inputs.  
# **Runtime:** under 1 minute. **Hardware:** CPU. **Data/license:** synthetic fixtures, CC0.  
# **Seed:** 1729. **Expected output:** successful fits and nonempty support sets.  
# Book: [SVM](../../support-vector-machines.html) · [SVR](../../support-vector-regression.html) · [Ranking](../../ranking-and-ordinal-regression.html)

# %%
import numpy as np
from sklearn.svm import SVC, SVR, OneClassSVM
try: from notebooks.lab_utils import rng, report
except ModuleNotFoundError: from lab_utils import rng, report

r = rng(3); x=np.r_[r.normal(-1,.3,(20,2)),r.normal(1,.3,(20,2))]; y=np.r_[-np.ones(20),np.ones(20)]
svc=SVC(kernel="rbf",gamma=.8,C=2).fit(x,y)
svr=SVR(kernel="rbf",gamma=.8,C=2).fit(x,np.sin(x[:,0]))
one=OneClassSVM(gamma=.8,nu=.1).fit(x[:20])
pairs=x[20:]-x[:20]; rank=SVC(kernel="linear",C=2).fit(np.r_[pairs,-pairs],np.r_[np.ones(20),-np.ones(20)])
assert min(len(svc.support_),len(svr.support_),len(one.support_),len(rank.support_))>0
report("margin-family", accuracy=float(svc.score(x,y)), support_vectors=int(len(svc.support_)))
