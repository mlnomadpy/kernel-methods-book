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
# # Lab 10: Optimal transport, quadrature, CME, KSD, and SVGD
# **Objectives:** run Sinkhorn scaling; compute kernel quadrature weights; regularize an operator solve.  
# **Runtime:** under 1 minute. **Hardware:** CPU. **Data/license:** deterministic grids, CC0.  
# **Seed:** 1729. **Expected output:** transport marginals and a stable quadrature solve.  
# Book: [Optimal transport](../../optimal-transport-and-kernels.html) · [Quadrature](../../kernel-quadrature-and-herding.html) · [CME](../../conditional-mean-embeddings.html)

# %%
import numpy as np
try: from notebooks.lab_utils import rbf_gram, report
except ModuleNotFoundError: from lab_utils import rbf_gram, report

x=np.linspace(0,1,12)[:,None]; y=np.linspace(.1,.9,10)[:,None]; a=np.ones(12)/12; b=np.ones(10)/10
c=(x-y.T)**2; q=np.exp(-c/.03); u=np.ones(12)
for _ in range(200):
    v=b/(q.T@u); u=a/(q@v)
plan=u[:,None]*q*v[None,:]
k=rbf_gram(x,gamma=3); mean=np.array([(np.exp(-3*(xi-np.linspace(0,1,500))**2)).mean() for xi in x[:,0]])
w=np.linalg.solve(k+1e-6*np.eye(len(k)),mean)
assert np.max(abs(plan.sum(1)-a))<1e-8 and np.max(abs(plan.sum(0)-b))<1e-8
report("inference-toolkit", transport_mass=float(plan.sum()), quadrature_weight_sum=float(w.sum()))
