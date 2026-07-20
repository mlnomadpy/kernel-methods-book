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
# # Lab 2: KRR and kernel logistic regression
# **Objectives:** fit KRR; optimize kernel logistic loss; compare calibrated probabilities.  
# **Runtime:** under 1 minute. **Hardware:** CPU. **Data/license:** synthetic binary data, CC0.  
# **Seed:** 1729. **Expected output:** finite ridge error and decreasing logistic objective.  
# Book: [KRR and smooth losses](../../kernel-ridge-and-friends.html)

# %%
import numpy as np
try: from notebooks.lab_utils import krr, rbf_gram, rng, report
except ModuleNotFoundError: from lab_utils import krr, rbf_gram, rng, report

x = np.linspace(-2, 2, 40)[:, None]
y = np.sin(2*x[:, 0]) + 0.05*rng(2).normal(size=len(x))
k = rbf_gram(x, gamma=1.2); alpha = krr(k, y, 1e-2)
rmse = float(np.sqrt(np.mean((k @ alpha-y)**2)))
labels = (y > 0).astype(float); beta = np.zeros(len(x)); losses=[]
for _ in range(60):
    score=k@beta; p=1/(1+np.exp(-np.clip(score,-30,30)))
    losses.append(float(np.mean(np.logaddexp(0, score)-labels*score)+5e-3*beta@k@beta))
    beta -= .3*(k@(p-labels)/len(x)+1e-2*k@beta)
assert rmse < .2 and losses[-1] < losses[0]
report("smooth-supervised", rmse=rmse, logistic_loss=losses[-1])
