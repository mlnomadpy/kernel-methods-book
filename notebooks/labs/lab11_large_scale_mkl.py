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
# # Lab 11: Large-scale kernels and MKL
# **Objectives:** combine PSD kernels; solve with conjugate gradients; record residuals and products.  
# **Runtime:** under 1 minute. **Hardware:** CPU. **Data/license:** synthetic vectors, CC0.  
# **Seed:** 1729. **Expected output:** converged matrix-free residual and a PSD mixture.  
# Book: [Scaling](../../large-scale-kernels.html) · [MKL](../../multiple-kernel-learning.html)

# %%
import numpy as np
try: from notebooks.lab_utils import rbf_gram, rng, report
except ModuleNotFoundError: from lab_utils import rbf_gram, rng, report

x=rng(11).normal(size=(100,4)); y=np.sin(x[:,0]); k=.3*rbf_gram(x,gamma=.1)+.7*rbf_gram(x,gamma=1.0); a=np.zeros(100); r=y.copy(); p=r.copy(); rs=r@r
for it in range(150):
    ap=k@p+.1*p; step=rs/(p@ap); a+=step*p; r-=step*ap; new=r@r
    if np.sqrt(new)<1e-8: break
    p=r+(new/rs)*p; rs=new
res=float(np.linalg.norm((k+.1*np.eye(100))@a-y))
assert res<1e-6 and np.linalg.eigvalsh(k).min()>-1e-9
report("large-scale-mkl", residual=res, iterations=it+1)
