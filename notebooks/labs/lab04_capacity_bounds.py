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
# # Lab 4: Generalization bounds and spectral capacity
# **Objectives:** compute effective dimension; compare eigenvalue decay; avoid treating a bound as an estimate.  
# **Runtime:** under 1 minute. **Hardware:** CPU. **Data/license:** generated spectra, CC0.  
# **Seed:** 1729. **Expected output:** faster decay gives smaller effective dimension.  
# Book: [Learning theory](../../learning-theory.html) · [Mercer rates](../../mercer-and-rates.html)

# %%
import numpy as np
try: from notebooks.lab_utils import effective_dimension, report
except ModuleNotFoundError: from lab_utils import effective_dimension, report

j=np.arange(1,201,dtype=float); fast=j**-3; slow=j**-1.2; ridge=.02
d_fast=effective_dimension(fast,ridge); d_slow=effective_dimension(slow,ridge)
bound=np.sqrt(d_fast/200)
assert d_fast<d_slow and 0<bound<1
report("capacity", effective_fast=d_fast, effective_slow=d_slow, illustrative_bound=float(bound))
