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
# # Lab 6: Kernel construction, alignment, Nyström, and random features
# **Objectives:** measure alignment; form a Nyström approximation; check approximation error.  
# **Runtime:** under 1 minute fast, under 5 minutes full. **Hardware:** CPU.  
# **Data/license:** synthetic regression data, CC0. **Seed:** 1729. **Expected output:** PSD approximation and relative error below one.  
# Book: [Kernel families](../../kernel-families.html) · [Large scale](../../large-scale-kernels.html)

# %%
import numpy as np
try: from notebooks.lab_utils import rbf_gram, rng, report
except ModuleNotFoundError: from lab_utils import rbf_gram, rng, report

x=rng(6).normal(size=(80,3)); k=rbf_gram(x,gamma=.3); idx=np.arange(0,80,4)
c=k[:,idx]; w=k[np.ix_(idx,idx)]; kn=c@np.linalg.pinv(w,rcond=1e-10)@c.T
y=np.sign(x[:,0]); align=float(y@k@y/(np.linalg.norm(k)*np.linalg.norm(np.outer(y,y))))
err=float(np.linalg.norm(k-kn)/np.linalg.norm(k))
assert np.linalg.eigvalsh((kn+kn.T)/2).min()>-1e-8 and err<1
report("design-scaling", alignment=align, nystrom_relative_error=err)
