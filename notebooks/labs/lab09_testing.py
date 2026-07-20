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
# # Lab 9: MMD, HSIC, and permutation testing
# **Objectives:** compute unbiased MMD; construct a permutation null; report Monte Carlo resolution.  
# **Runtime:** under 1 minute fast, under 5 minutes full. **Hardware:** CPU.  
# **Data/license:** Gaussian fixtures, CC0. **Seed:** 1729. **Expected output:** finite MMD and valid corrected p-value.  
# Book: [MMD](../../kernel-mean-embeddings.html) · [Testing](../../kernel-hypothesis-testing.html)

# %%
import numpy as np
try: from notebooks.lab_utils import MODE, mmd2_unbiased, rbf_gram, rng, report
except ModuleNotFoundError: from lab_utils import MODE, mmd2_unbiased, rbf_gram, rng, report

r=rng(9); m=30; x=r.normal(size=(m,1)); y=r.normal(.6,size=(m,1)); z=np.r_[x,y]; full=rbf_gram(z,gamma=.5)
obs=mmd2_unbiased(full[:m,:m],full[m:,m:],full[:m,m:]); b=99 if MODE=="fast" else 999; null=[]
for _ in range(b):
    p=r.permutation(2*m); a=p[:m]; c=p[m:]; null.append(mmd2_unbiased(full[np.ix_(a,a)],full[np.ix_(c,c)],full[np.ix_(a,c)]))
pvalue=(1+sum(v>=obs for v in null))/(b+1)
assert 0<pvalue<=1
report("kernel-testing", mmd2=obs, p_value=float(pvalue), permutations=b)
