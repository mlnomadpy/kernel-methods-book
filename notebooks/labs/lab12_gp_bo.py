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
# # Lab 12: Gaussian processes and Bayesian optimization
# **Objectives:** compute posterior moments; verify nonnegative variance; select an acquisition maximizer.  
# **Runtime:** under 1 minute. **Hardware:** CPU. **Data/license:** synthetic objective, CC0.  
# **Seed:** 1729. **Expected output:** positive posterior variance and a finite next point.  
# Book: [Gaussian processes](../../gaussian-processes-and-rvm.html) · [Bayesian optimization](../../bayesian-optimization-and-bandits.html)

# %%
import numpy as np
try: from notebooks.lab_utils import rbf_gram, report
except ModuleNotFoundError: from lab_utils import rbf_gram, report

x=np.array([[-2.],[-.5],[1.5]]); y=np.sin(2*x[:,0]); grid=np.linspace(-3,3,200)[:,None]
k=rbf_gram(x,gamma=.6)+.03*np.eye(len(x)); ks=rbf_gram(x,grid,gamma=.6); alpha=np.linalg.solve(k,y)
mean=ks.T@alpha; var=1-np.sum(ks*np.linalg.solve(k,ks),axis=0); ucb=mean+1.96*np.sqrt(np.maximum(var,0)); nxt=float(grid[np.argmax(ucb),0])
assert var.min()>-1e-9 and -3<=nxt<=3
report("gp-bo", next_point=nxt, min_variance=float(var.min()), max_ucb=float(ucb.max()))
