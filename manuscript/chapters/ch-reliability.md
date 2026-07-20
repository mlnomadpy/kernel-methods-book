---
id: ch-reliability
slug: distribution-shift-robustness-and-conformal-prediction
title: 'Distribution Shift, Robustness, and Conformal Prediction'
part: XV · Classical and Reliable Kernel Models
order: 51
tier: core
prerequisites:
  - kernel-ridge-and-friends
  - kernel-mean-embeddings
  - kernel-hypothesis-testing
objectives:
  - >-
    Distinguish covariate, label, concept, and support shift and state what each
    permits.
  - Derive kernel mean matching as an RKHS moment-balancing problem.
  - >-
    Use MMD witnesses and overlap diagnostics without confusing detection with
    correction.
  - >-
    Construct conformal prediction sets and state their finite-sample coverage
    assumptions.
  - >-
    Evaluate robust and shifted prediction with effective-sample-size and
    conditional-coverage diagnostics.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-reliability.yml
verification_date: null
bibliography:
  - feng2023covshift
  - burnaev2014conformal
  - gretton2012
  - sriperumbudur2012
---
# Distribution Shift, Robustness, and Conformal Prediction

<p class="lead">A model can be accurate on held-out data and fail immediately after deployment because the held-out data came from the same sampling system as training. Kernel embeddings make distribution change measurable, importance weighting makes a restricted form of change correctable, and conformal prediction turns residual ranks into finite-sample prediction sets. Each tool has precise assumptions, and none converts arbitrary shift into a solved problem.</p>

## A taxonomy of distribution change {#shift-taxonomy}

Let \(P\) denote the training distribution and \(Q\) the deployment distribution on \((X,Y)\).

::: {.definition #def-shift-types}
[Definition (common shift models)]{.box-title}

- **Covariate shift:** \(P_X\ne Q_X\) while \(P_{Y\mid X}=Q_{Y\mid X}\).
- **Label shift:** \(P_Y\ne Q_Y\) while \(P_{X\mid Y}=Q_{X\mid Y}\).
- **Concept shift:** the conditional law of \(Y\) given \(X\) changes.
- **Support shift:** deployment places mass where training has little or no support.

These models are restrictions on the joint distributions. Observed marginal change alone does not identify which model holds.
:::

Covariate shift is the most natural setting for reweighting. For a deployment risk

$$
R_Q(f)=\mathbb E_Q\ell\{Y,f(X)\},
$$

the conditional invariance assumption gives

$$
R_Q(f)=\mathbb E_P\left[w(X)\ell\{Y,f(X)\}\right],
\qquad
w(x)=\frac{dQ_X}{dP_X}(x),
$$

provided \(Q_X\) is absolutely continuous with respect to \(P_X\). No finite weight can repair a target region absent from training. This overlap condition is as important as the conditional invariance assumption.

## Detecting shift with kernel embeddings {#shift-detection}

The MMD from [[ch:kernel-mean-embeddings]] compares the input marginals through

$$
\operatorname{MMD}_k(P_X,Q_X)=\lVert\mu_{P_X}-\mu_{Q_X}\rVert_{\mathcal H_k}.
$$

A characteristic kernel makes zero population MMD equivalent to equality of distributions. The empirical test machinery in [[ch:kernel-hypothesis-testing]] determines whether an observed discrepancy is distinguishable from sampling variation [@gretton2012].

The witness function

$$
g(x)=\langle k(x,\cdot),\mu_{P_X}-\mu_{Q_X}\rangle
$$

localizes where the embedded distributions differ. It is useful for diagnosis, but its sign and scale depend on the kernel. A small MMD under one bandwidth is not proof that the deployment risk is unchanged.

::: {.example #example-shift-detection-correction}
[Example (detection does not identify a correction)]{.box-title}

Suppose an MMD test detects that deployment inputs differ from training inputs. The same observation is compatible with harmless covariate shift, a changed labeling mechanism, new unsupported regions, or a measurement-system change. Reweighting is justified only after defending conditional invariance and overlap. The test answers whether marginals differ, not why they differ.

**Verification artifact.** checks/example-ch-reliability-example-shift-detection-correction.json records the example source hash and verification scope.
:::

## Kernel mean matching {#kernel-mean-matching}

Kernel mean matching estimates weights without separately estimating two densities. Choose nonnegative weights \(\beta_i\) so the weighted training embedding approaches the deployment embedding:

$$
\min_{\beta}\left\lVert
\frac1n\sum_{i=1}^n\beta_i k(x_i,\cdot)
-\frac1m\sum_{j=1}^m k(z_j,\cdot)
\right\rVert_{\mathcal H_k}^2.
$$

Expanding the norm yields a quadratic program in Gram matrices. Constraints such as \(0\le\beta_i\le B\) and an approximately unit average prevent a few samples from absorbing all mass.

::: {.proposition #prop-kmm-moment-balance}
[Proposition (RKHS moment balance)]{.box-title}

If the empirical kernel-mean-matching objective is at most \(\varepsilon^2\), then every \(h\in\mathcal H_k\) satisfies

$$
\left|
\frac1n\sum_i\beta_i h(x_i)-\frac1m\sum_jh(z_j)
\right|
\le\varepsilon\lVert h\rVert_{\mathcal H_k}.
$$

**Assumptions.** Both empirical embeddings exist and \(h\) lies in the RKHS. **Proof status.** Proved by writing the difference as an RKHS inner product and applying Cauchy-Schwarz.
:::

The bound says exactly what is balanced: functions in the chosen RKHS. It does not guarantee balance for arbitrary losses or hidden confounders. Modern rate analyses additionally track the severity of the density ratio, source smoothness, and effective dimension [@feng2023covshift].

The weight effective sample size

$$
n_{\mathrm{eff}}=\frac{(\sum_i\beta_i)^2}{\sum_i\beta_i^2}
$$

is an indispensable diagnostic. A weighted risk based on a tiny effective sample can have large variance even when empirical moments match well.

## Robust losses and distributional neighborhoods {#robustness}

Robustness has several meanings. A robust loss reduces sensitivity to extreme residuals. A contamination model allows a fraction of observations to come from an arbitrary distribution. Distributionally robust optimization minimizes worst-case risk over a neighborhood of the empirical distribution. Adversarial robustness protects against input perturbations. These are different uncertainty sets and should not share a theorem by analogy.

An MMD ambiguity set takes

$$
\mathcal Q_\rho=\{Q:\operatorname{MMD}_k(Q,\widehat P)\le\rho\}.
$$

For losses lying in or controlled by the RKHS, the worst-case expectation can be bounded by empirical expectation plus a radius times an RKHS norm. This connects distributional robustness to regularization. If the loss is not in the RKHS or the chosen kernel ignores the relevant perturbation, the bound can be vacuous [@sriperumbudur2012].

Robust model design should report the perturbation model, radius selection, clean-data cost, shifted-data benefit, and failure outside the uncertainty set. Merely replacing squared loss by Huber loss does not address covariate shift.

## Conformal prediction from exchangeable scores {#conformal-prediction}

Conformal prediction wraps a fitted model with a rank calibration step. Split the data into a proper training set and a calibration set. Fit \(\widehat f\) on training data and compute nonconformity scores \(r_i=s(X_i,Y_i;\widehat f)\) on calibration data. For a new input \(x\), form

$$
\mathcal C_\alpha(x)=\{y:s(x,y;\widehat f)\le q_{1-\alpha}\},
$$

where \(q_{1-\alpha}\) is the finite-sample corrected empirical score quantile.

:::: {.theorem #thm-split-conformal-coverage}
[Theorem (split-conformal marginal coverage)]{.box-title}

If the calibration examples and the new example are exchangeable, and the fitted scoring rule is fixed with respect to calibration labels, then the split-conformal set satisfies

$$
\mathbb P\{Y_{\mathrm{new}}\in\mathcal C_\alpha(X_{\mathrm{new}})\}\ge 1-\alpha,
$$

with the usual finite-sample quantile convention.

**Assumptions.** Exchangeability; a measurable score; no reuse of calibration labels in fitting or tuning the score; valid randomized or conservative handling of ties. **Proof status.** Proved by exchangeability of the calibration and test score ranks.
::::

For regression, \(s(x,y)=|y-\widehat f(x)|\) gives a constant-width residual interval. Locally scaled or quantile-based scores adapt width to heteroscedasticity. For classification, scores based on class probabilities yield prediction sets. Conformalized ridge regression illustrates that validity can be added to a strong linear or kernel predictor with limited asymptotic efficiency loss under its working model [@burnaev2014conformal].

Marginal coverage is not conditional coverage at every \(x\). A method can attain the theorem while under-covering a small subgroup and over-covering elsewhere. Report coverage and width by scientifically relevant groups, regions, and difficulty strata.

## Conformal prediction under shift {#conformal-under-shift}

Ordinary conformal validity can fail when deployment examples are not exchangeable with calibration data. Under covariate shift with known or estimated density ratios, weighted conformal quantiles can target the deployment distribution. Their reliability depends on the same overlap and conditional invariance assumptions as importance weighting.

Under temporal dependence, block or sequential methods require assumptions adapted to the process. Under arbitrary concept shift, no calibration wrapper can guarantee future coverage without further information. Monitoring must therefore accompany prediction sets: detect changes, diagnose likely mechanisms, and trigger abstention or recalibration when assumptions become implausible.

:::: {.algorithm #algo-reliable-kernel-pipeline}
[Algorithm (shift-aware kernel prediction)]{.box-title}

**Input.** Training data, untouched calibration data, deployment covariates, candidate kernels, and a declared shift model.

**Output.** Predictions, prediction sets, and an assumption audit.

1. Define the deployment unit and split data by time, site, group, or source accordingly.
2. Test and visualize input shift with several justified kernel scales.
3. Diagnose support overlap and estimate weights only if covariate-shift assumptions are defensible.
4. Fit weighted and unweighted kernel baselines, recording effective sample size and conditioning.
5. Calibrate conformal scores without reusing calibration labels for model selection.
6. Report risk, coverage, set width, groupwise diagnostics, abstention rate, and sensitivity to weight clipping.

Kernel mean matching requires a constrained quadratic solve; large problems need low-rank features or stochastic moment matching. Stop when both the optimization residual and the embedding discrepancy are stable.
::::

## Common mistakes and practical implications {#reliability-practice}

- Detecting \(P_X\ne Q_X\) does not prove covariate shift.
- Density-ratio weights cannot create labels in unsupported target regions.
- Weight clipping trades bias for variance and must be reported.
- Choosing a shift kernel after inspecting test labels leaks deployment information.
- Conformal marginal coverage is not automatic subgroup or conditional coverage.
- Calibration data cease to be calibration data when repeatedly used for tuning.
- Robustness to one uncertainty set says nothing about a different perturbation mechanism.

The practical goal is not a universal certificate. It is a chain of explicit assumptions, diagnostics, corrections, prediction sets, and monitoring rules that can fail visibly.

## Summary and further reading {#reliability-summary}

Kernel embeddings detect and localize marginal distribution change. Kernel mean matching balances an RKHS function class and can correct covariate shift when conditional invariance and overlap hold. Robust losses and distributional neighborhoods address different perturbation models. Conformal prediction adds finite-sample marginal coverage under exchangeability, with weighted variants requiring stronger shift assumptions. See [@feng2023covshift] for kernel rates under shift and [@burnaev2014conformal] for the ridge-regression connection.

## Exercises {#exercises}

1. [warm-up]{.ex-tag} Give one data-generating example for each shift type and explain which observable marginals would change.
2. [computation]{.ex-tag} Expand the empirical kernel-mean-matching objective into its quadratic-program matrix and vector terms, including the normalization constraints.
3. [proof]{.ex-tag} Prove the RKHS moment-balance proposition and state why it does not control every bounded measurable loss.
4. [proof]{.ex-tag} Prove split-conformal marginal coverage from the exchangeability of score ranks, including the finite-sample quantile correction.
5. [synthesis]{.ex-tag} Design a deployment audit for a kernel classifier moving from one hospital or sensor network to another. Specify shift tests, overlap criteria, weighting, conformal scoring, subgroup diagnostics, and retraining triggers.
