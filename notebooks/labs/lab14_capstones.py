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
# # Lab 14: Protein, text, and molecular capstones
# **Objectives:** define leakage-safe splits; combine domain kernels; audit dataset and metric choices.  
# **Runtime:** under 1 minute fast; full mode is dataset-dependent. **Hardware:** CPU fast, Kaggle accelerator optional full.  
# **Fast data/license:** synthetic identifiers, CC0. **Full datasets:** AG News (CC BY-SA 4.0), Tox21 (CC0), and a protein benchmark whose redistribution status must be verified before release.  
# **Seed:** 1729. **Expected output:** normalized PSD multimodal kernel and split audit.  
# Book: [Applications](../../applications-and-practice.html) · [Graphs](../../graph-kernels.html) · [Text](../../kernels-for-text.html)

# %%
import numpy as np
try: from notebooks.lab_utils import rbf_gram, rng, report
except ModuleNotFoundError: from lab_utils import rbf_gram, rng, report

r=rng(14); protein=r.normal(size=(36,6)); text=r.normal(size=(36,8)); molecule=r.normal(size=(36,5))
k=sum(rbf_gram(v,gamma=1/v.shape[1]) for v in (protein,text,molecule))/3
k/=np.sqrt(np.outer(np.diag(k),np.diag(k))); groups=np.repeat(np.arange(12),3); train=groups<8; test=groups>=8
assert not set(groups[train])&set(groups[test]) and np.linalg.eigvalsh(k).min()>-1e-9
report("capstones", train=int(train.sum()), test=int(test.sum()), min_eigenvalue=float(np.linalg.eigvalsh(k).min()))
