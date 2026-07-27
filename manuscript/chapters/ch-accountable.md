---
id: ch-accountable
slug: accountable-kernels
title: 'Accountable Kernels: Uncertainty, Explanation, and Audit'
part: XII · Reliable Practice
order: 60
tier: advanced
prerequisites:
  - reproducing-kernel-banach-and-variation-spaces
objectives:
  - Separate model-based GP uncertainty from empirically calibrated coverage.
  - >-
    Compute example contributions and exact kernel-ridge leave-one-out
    influence.
  - >-
    Use MMD and HSIC as tests without confusing non-rejection with a
    certificate.
  - >-
    Assemble a reproducible model record that includes data, solver, tolerance,
    and approximation.
  - Define monitoring and recalibration triggers for a deployed kernel model.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-accountable.yml
verification_date: null
bibliography:
  - scholkopf2001
  - kimeldorf1971
  - cortes1995
  - rasmussen2006
  - vovk2005conformal
  - lei2018conformal
  - angelopoulos2021gentle
  - barber2021jackknife
  - kim2016critic
  - koh2017influence
  - basu2021fragile
  - ghorbani2019shapley
  - song2012bahsic
  - gretton2012
  - sriperumbudur2010
  - gretton2005hsic
  - gretton2008hsictest
  - perezsuay2017fair
  - mitchell2019cards
  - gebru2021datasheets
  - nist2023rmf
  - frazier2018
  - shahriari2016
  - kernelbook-code-ch-accountable-ex1
  - kernelbook-code-ch-accountable-ex2
  - kernelbook-code-ch-accountable-ex3
  - kernelbook-code-ch-accountable-ex4
  - kernelbook-code-ch-accountable-ex5
  - kernelbook-code-ch-accountable-audit
  - kennedy2001calibration
---
# Accountable Kernels: Uncertainty, Explanation, and Audit

<p class="lead">A prediction that will steer a spacecraft, propose a molecule for synthesis, or drive a satellite crop-yield estimate is not judged by its accuracy alone. Someone will ask how sure the model is, which training data support the decision, whether the inputs still resemble what the model was trained on, and whether the whole pipeline can be reproduced and documented for a review board. Kernel methods put unusually much of that evidence in one geometry. The representer theorem writes a prediction as a signed sum over named training points; convex objectives provide checkable optimality conditions; a Gaussian-process model supplies a predictive covariance; conformal calibration can add finite-sample marginal coverage under exchangeability; and kernel two-sample and independence statistics can monitor deployment. None of these is automatic trust. Contributions are not causal explanations, covariance is conditional on a model, numerical reproducibility needs a recorded implementation and tolerance, and statistical tests have finite power. This chapter turns those structural advantages into an audit chain whose assumptions and failure triggers can be stated before the model is deployed.</p>

## Accountability is a property of the model, not a wrapper {#a-property-not-a-wrapper}

Three facts, each proved earlier in the book, do all the work of this chapter, and it is worth stating them together before we develop any one of them, because their conjunction is the argument.

The first is the representer theorem of [[ch:kernel-tricks|the kernel-tricks chapter]], in the general form of Schölkopf, Herbrich, and Smola (2001): the minimizer of any regularized empirical risk over an RKHS lies in the span of the training data mapped into feature space, so the fitted predictor is

$$f(x)=\sum_{i=1}^n \alpha_i\,k(x_i,x).$$

Every value the model outputs is a weighted vote over the training examples, with contribution \(\alpha_i\,k(x_i,x)\) from example \(i\). This decomposition is exact for the fitted function, so it answers the narrow question “which represented cases contribute to this value?” It does not by itself answer a causal question, certify that the kernel features are meaningful to a person, or distinguish a genuinely influential case from a redundant cluster of near-duplicates.

The second is that many canonical fits are convex and have residuals or optimality conditions that can be checked. Kernel ridge regression and the Gaussian-process posterior mean solve a regularized linear system; the support vector machine of [[ch:support-vector-machines|its chapter]] solves a convex quadratic program (Cortes and Vapnik 1995). Strict regularization makes the primal predictor unique in the usual formulations, although dual coefficients need not be unique when the Gram matrix is singular. Fixed data, preprocessing, kernel, hyperparameters, arithmetic, solver, and stopping tolerance define a short reproducibility record. Different libraries or parallel reductions can still differ in their final bits, so numerical residuals and prediction tolerances are more meaningful than a claim of bitwise identity.

The third is that a Gaussian process, from [[ch:gaussian-processes-and-rvm|the Bayesian chapter]], returns not a point but a predictive distribution, whose variance

$$\sigma^2(x_\star)=k(x_\star,x_\star)-k(x_\star)^\top(K+\sigma^2 I)^{-1}k(x_\star)$$

depends only on where \(x_\star\) sits relative to the data, not on the observed labels. Under the stated covariance and likelihood, the model reports where that model has little information: far from the training geometry the variance often climbs toward the prior. The covariance is delivered by the same factorization as the mean, but it is a conditional model quantity, not a distribution-free certificate of error.

The comparison with other model families should therefore be made capability by capability, not as a blanket hierarchy. A deterministic neural predictor can also be calibrated, monitored, and documented, while a large approximate kernel model can lose the exact algebra used here. The kernel advantage is narrower and useful: for several important estimators, contributions, uncertainty under a GP model, and convex optimality are explicit in the fitted object. The rest of the chapter turns those facts into working tools and keeps their scopes visible.

## Uncertainty you can put in a report {#uncertainty-you-can-report}

A number without an error bar is not yet a scientific result, and a great deal of what makes kernel methods useful in the sciences of the next chapter is that they can return uncertainty as part of the answer. We take the native Gaussian-process interval first, identify the model assumptions behind it, and then layer on a conformal guarantee whose different assumption is exchangeability.

### The Gaussian-process interval, and its honest limits {#gp-error-bars}

The predictive variance above has a clean model-based reading. It starts at the prior variance \(k(x_\star,x_\star)\) and subtracts the quadratic form \(k(x_\star)^\top(K+\sigma_\epsilon^2 I)^{-1}k(x_\star)\). The exact boundary is worth stating formally [@rasmussen2006, Section 2.2, especially Equations 2.22--2.24].

:::: {.proposition #prop-accountable-gp-boundary}
[Proposition (what a Gaussian-process interval guarantees)]{.box-title}

Let \(f\sim\operatorname{GP}(m,k)\), where \(k\) is a finite-valued PSD covariance kernel. At fixed inputs \(X=(X_1,\ldots,X_n)\), observe \(Y_i=f(X_i)+\epsilon_i\), with independent \(\epsilon_i\sim N(0,\sigma_\epsilon^2)\) and \(\sigma_\epsilon^2\gt0\). Treat \(m\), \(k\), \(\sigma_\epsilon^2\), and \(X\) as fixed rather than estimated from the same response vector. Then, conditional on \(X,Y\),

$$
f(x_\star)\sim N\!\left(
m(x_\star)+k_\star^\top(K+\sigma_\epsilon^2I)^{-1}(Y-m_X),
\ k(x_\star,x_\star)-k_\star^\top(K+\sigma_\epsilon^2I)^{-1}k_\star
\right).
$$

Consequently the corresponding Gaussian interval has posterior probability \(1-\alpha\) for the latent \(f(x_\star)\). Adding \(\sigma_\epsilon^2\) to the variance gives the posterior-predictive interval for a new noisy observation. If the asserted joint Gaussian data-generating model is correct, averaging over repeated \(Y\) also gives nominal model-based coverage. The proposition supplies no distribution-free, conditional-on-\(x_\star\), or misspecification-robust coverage statement.

**Assumptions.** Fixed finite design, finite PSD covariance, positive Gaussian observation noise, fixed hyperparameters, and the asserted joint GP model.

**Proof status.** Complete proof below.

**Proof.** The displayed law is the conditional distribution of one block of a jointly Gaussian vector given the other; its covariance is the Schur complement. The Gaussian quantile has conditional posterior mass \(1-\alpha\). Every conclusion therefore inherits the asserted joint model. \(\square\)
::::

Hyperparameter fitting, covariance misspecification, non-Gaussian or heteroscedastic noise, and adaptive selection of \(x_\star\) move the deployed procedure outside this proposition unless separately justified. Frequentist coverage is therefore an empirical calibration target, not a consequence of a plotted posterior band.

The qualifier is the whole story, so we make both halves of it concrete on numbers.

::::: {.example #example-55-1}
[Example (an honest error bar, and a dishonest one)]{.box-title}

:::: wex
::: wex-setup
First the honest behavior. We fit a Gaussian process with an RBF kernel (length scale \(1\), noise \(0.05\)) to eight points of \(\sin x\) on \([0,6]\) with a deliberate gap in \((2.5,4.5)\), and read the posterior standard deviation at three places. Then the failure. We draw \(40\) points from a wiggly mean \(\sin(3x)\) on \([0,1]\) corrupted by heteroscedastic noise whose standard deviation grows from \(0.05\) at \(x=0\) to \(0.5\) at \(x=1\), and fit a GP that wrongly assumes a single homoscedastic noise equal to the average, \(\hat\sigma=0.292\). We then measure how often the nominal \(90\%\) interval actually covers fresh data. The values are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-accountable-ex1].
:::

1.  [Watch the band breathe.]{.wex-op} The posterior standard deviation is \(0.049\) at a training point (\(x=0.9\)), essentially the noise floor; it rises to \(0.472\) in the middle of the gap (\(x=3.5\)); and it is \(0.247\) just past the data (\(x=6.5\)). The model announces exactly where it has no evidence, and it does so from the inputs alone.
2.  [Now misspecify the noise.]{.wex-op} Averaged over the whole domain the nominal \(90\%\) interval covers \(89.1\%\) of fresh points, which looks fine. It is not fine: restricted to the high-noise half of the domain (\(x\gt0.5\)), coverage falls to \(79.2\%\). The single assumed noise is too small where the true noise is large, and the interval silently under-covers there.
3.  [Read the lesson.]{.wex-op} The GP error bar is honest about *where* the data are sparse (step 1), because that part depends only on the kernel geometry. It is only as honest about *how noisy* the data are as the noise model you gave it (step 2). Aggregate coverage hid a local failure that a report on the noisy regime would have relied on.

**Reading.** Two different promises live inside one interval. The width from the kernel geometry is trustworthy and is the source of the \"it knows when it is extrapolating\" behavior that makes GPs valued in the sciences. The width from the assumed noise model is trustworthy only under correct specification, and a misspecified kernel or noise level can make the nominal coverage a fiction exactly where it matters. This is the gap the next section closes.
::::
:::::

### Conformal prediction: a distribution-free coverage guarantee {#conformal}

The way to stop trusting the model's self-assessment and start certifying coverage is conformal prediction [@vovk2005conformal, Chapter 2; @lei2018conformal, Section 2.2 and Theorem 2.1; @angelopoulos2021gentle, Section 2]. It wraps any predictor, kernel or otherwise, and turns residuals into an interval. “Distribution-free” means no parametric family is assumed; it does not remove exchangeability, data-splitting, or score-definition requirements.

:::: {.algorithm #algo-55-1}
[Algorithm (Split-conformal prediction for regression)]{.box-title}

::: algo-io
[Input]{.algo-lab} a fitted predictor \(\hat f\) (trained on data disjoint from the calibration set); a calibration set \(\{(x_i,y_i)\}_{i=1}^n\); a miscoverage level \(\alpha\).

[Output]{.algo-lab} a prediction interval \(C(x)\) with \(\mathbb P\big(Y\in C(X)\big)\ge 1-\alpha\).
:::

1.  Compute the conformity scores \(s_i=\lvert y_i-\hat f(x_i)\rvert\) on the calibration set.
2.  Put \(r=\big\lceil (n+1)(1-\alpha)\big\rceil\). If \(r\le n\), set \(\hat q=s_{(r)}\); if \(r=n+1\), set \(\hat q=+\infty\).
3.  Return \(C(x)=[\,\hat f(x)-\hat q,\ \hat f(x)+\hat q\,]\).
::::

:::: {.theorem #thm-accountable-split-conformal}
[Theorem (finite-sample split-conformal coverage)]{.box-title}

Let the proper-training data be independent of the \(n\) calibration pairs and the test pair. Conditional on the proper-training data, suppose \((X_1,Y_1),\ldots,(X_n,Y_n),(X_{n+1},Y_{n+1})\) are exchangeable. Let \(\hat f\) be measurable with respect to proper-training data only, use the same measurable score \(S(x,y)=|y-\hat f(x)|\) for calibration and test, and construct \(C\) by Algorithm 55.1. Then

$$1-\alpha\ \le\ \mathbb P\big(Y_{n+1}\in C(X_{n+1})\big)\ \le\ 1-\alpha+\frac{1}{n+1},$$

where the upper bound holds when the \(n+1\) scores are almost surely distinct and \(r\le n\); when \(r=n+1\), coverage is one.

**Assumptions.** Split independence, conditional exchangeability, a fixed measurable score, and conservative handling of ties.

**Proof status.** Complete proof below.

**Proof.** Conditional on proper-training data, exchangeability makes the rank of the test score among the \(n+1\) scores uniform when ties are randomized. The test point is excluded only if its rank exceeds \(r\), whose probability is \((n+1-r)/(n+1)\le\alpha\). Conservative inclusion of ties can only increase coverage. If scores are distinct and \(r\le n\), inclusion has probability \(r/(n+1)\lt1-\alpha+1/(n+1)\). The \(r=n+1\) convention returns the whole line. Averaging over proper-training data proves the result. \(\square\)
::::

The theorem is marginal over a fresh exchangeable test pair. It is not conditional coverage at each \(x\), subgroup coverage, time-series validity, or validity after selecting the score or model on calibration responses. Predictor quality controls width, not validity. Jackknife+ supplies a different leave-one-out construction and guarantee [@barber2021jackknife, Theorem 1].

::::: {.example #example-55-2}
[Example (conformal repairs the coverage the GP lost)]{.box-title}

:::: wex
::: wex-setup
Same heteroscedastic truth as the failure above. We fit kernel ridge regression (RBF length scale \(0.15\), ridge \(10^{-2}\)) on a training split, calibrate on \(n=500\) held-out points at \(\alpha=0.10\), and test on \(8000\) fresh points. We compare the conformal band against a naive Gaussian band \(\hat f(x)\pm 1.645\,\hat\sigma\) using a single \(\hat\sigma\) estimated from the training residuals. The values are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-accountable-ex2].
:::

1.  [Read the quantile off the calibration residuals.]{.wex-op} The rank is \(\lceil 501\cdot 0.9\rceil=451\), so \(\hat q\) is the \(451\)st smallest of the \(500\) absolute residuals, \(\hat q=0.512\). The band is \(\hat f(x)\pm 0.512\), width \(1.025\).
2.  [Certify the coverage.]{.wex-op} On the fresh test set the conformal band covers \(89.9\%\), inside the guaranteed envelope \([0.900,\,0.902]\) up to test-sample noise. The naive Gaussian band, width \(0.909\), covers only \(87.1\%\): it is narrower and under-covers, because one \(\hat\sigma\) cannot describe noise that varies across the domain.
3.  [Note what conformal does and does not fix.]{.wex-op} Conformal restores *marginal* coverage to the target. It does not by itself equalize coverage across the domain: in the high-noise half the conformal band still covers only \(81.0\%\) (the naive band, \(76.0\%\)). The guarantee is an average over the input distribution, not a promise at every \(x\).

**Reading.** A defensible interval for a report is one whose coverage you can certify without believing the model's noise assumptions, and split conformal delivers exactly that from a held-out set and a single sorted list of residuals. Its honest limitation is that the certificate is marginal: to tighten it where the data are hard, one conditions the score on the input (normalized or locally-weighted conformal), which the exercises pursue. The pairing is the practical recipe of the chapter: let the Gaussian process propose the interval, and let conformal certify it.
::::
:::::

The reliability chapter returns to this construction under deployment shift and shows how retaining coverage can require wider sets; the present guarantee is the exchangeable baseline.

## Explanation for free: the representer theorem as attribution {#explanation-for-free}

When a kernel model makes a decision that someone wants to contest, the question is rarely \"which pixels\" and usually \"which prior cases, and how much.\" Kernel machines answer that question exactly, because the answer is written into the representer form, and for the convex ones the sensitivity of a prediction to each training point is closed form rather than estimated.

### Prediction as a weighted vote over training points {#representer-attribution}

Return to \(f(x)=\sum_i\alpha_i\,k(x_i,x)\). Each training point contributes \(\alpha_i k(x_i,x)\) to the prediction at \(x\). In an SVM, only support vectors have nonzero dual contribution, although the support set need not be small and nonunique dual coefficients can redistribute contributions when the Gram matrix is degenerate. In kernel ridge and Gaussian-process regression the coefficients are generally dense; for a local kernel, multiplication by \(k(x_i,x)\) often concentrates contributions near the query. This is an exact decomposition in the model's chosen representation. Calling it an explanation additionally requires that examples, labels, and the similarity itself are interpretable for the decision at hand.

The same RKHS machinery answers the dual question, \"which examples represent the data, and which are the edge cases,\" through the prototypes and criticisms of Kim, Khanna, and Koyejo (2016). Prototypes are the points whose empirical distribution best matches the data in the sense of maximum mean discrepancy from [[ch:kernel-mean-embeddings|the mean-embedding chapter]]; criticisms are the points in regions the prototypes represent worst, found by the witness function. Both are computed from the same Gram matrix the model already uses, and both are consumable by a human reviewer as \"these are the typical cases the model has learned, and these are the outliers it may not handle.\"

<figure class="viz" data-widget="influence-glow">

<figcaption>A live kernel-ridge fit on a handful of points. Drag the query \(x_\star\); each training point glows in proportion to its signed contribution \(\alpha_i\,k(x_i,x_\star)\) to the prediction there, and the readout names the three most influential points. Toggle to the exact leave-one-out influence, the change in \(f(x_\star)\) if that point were deleted, and the ranking shifts to what the decision truly rests on. Every number is read from the actual fitted expansion.</figcaption>
</figure>

### Exact influence and closed-form leave-one-out {#influence-loo}

Beyond \"which points support this prediction\" is the sharper counterfactual, \"how would this prediction change if a given training point had never been collected.\" General influence functions approximate infinitesimal up-weighting through an inverse Hessian [@koh2017influence, Section 2]. They require differentiability and a nonsingular Hessian, and they are not deletion-exact merely because an objective is convex; deep-network failures are documented by [@basu2021fragile, Sections 3--4]. Kernel ridge regression is special because it is a fixed linear smoother.

:::: {.proposition #prop-accountable-krr-loo}
[Proposition (exact kernel-ridge deletion sensitivity)]{.box-title}

Let \(K\in\mathbb R^{n\times n}\) be symmetric PSD, let \(\lambda\gt0\), and set \(A=K+\lambda I\), \(\alpha=A^{-1}y\), \(H=KA^{-1}\), and \(f_\star=k_\star^\top\alpha\). Fit the same kernel with the same \(\lambda\) after deleting observation \(i\). Then \(0\le H_{ii}\lt1\),

$$y_i-\hat f^{(-i)}(x_i)=\frac{y_i-\hat y_i}{1-H_{ii}},$$

and

$$
f_\star-f_\star^{(-i)}
=\frac{(A^{-1}k_\star)_i\,(y_i-\hat y_i)}{1-H_{ii}}.
$$

**Assumptions.** Fixed symmetric PSD Gram matrix, positive ridge, unchanged preprocessing, kernel, and regularization after deletion.

**Proof status.** Complete proof below.

**Proof.** Since \(A\succ0\), its inverse exists. Also \(I-H=\lambda A^{-1}\succ0\), hence \(H_{ii}\lt1\). Partitioning \(A\), \(y\), and \(k_\star\) around index \(i\), then applying the Schur-complement inverse formula, gives both identities after cancellation. \(\square\)
::::

This is sensitivity to deleting one record while holding the kernel, preprocessing, hyperparameters, and all other records fixed. It is not causal influence, uncertainty, or the effect of retuning. Numerically, an audit must fail rather than divide when \(1-H_{ii}\) is below a declared tolerance.

:::: {.algorithm #algo-55-2}
[Algorithm (Exact leave-one-out attribution for kernel ridge)]{.box-title}

::: algo-io
[Input]{.algo-lab} Gram matrix \(K\), labels \(y\), ridge \(\lambda\); a query \(x_\star\) with cross-kernel \(k_\star=[k(x_\star,x_i)]_i\).

[Output]{.algo-lab} the training points ranked by their influence on \(f(x_\star)\).
:::

1.  Solve once: \(G=(K+\lambda I)^{-1}\), \(\alpha=Gy\), \(H=KG\), \(\hat y=Hy\).
2.  Form the leave-one-out residuals \(r_i=(y_i-\hat y_i)/(1-H_{ii})\).
3.  For each \(i\), the exact change in the prediction on deleting point \(i\) is \(\Delta_i=(G\,k_\star)_i\,r_i\).
4.  Return the points sorted by \(\lvert\Delta_i\rvert\).
::::

::::: {.example #example-55-3}
[Example (which three points a decision rests on)]{.box-title}

:::: wex
::: wex-setup
Kernel ridge regression on \(40\) points of \(\sin(1.3x)\) with light noise on \([-3,3]\) (RBF length scale \(0.4\), ridge \(0.1\)). We ask which training points most determine the prediction at \(x_\star=0.7\), computing the deletion effect both by brute-force refitting and by the rank-one formula of the algorithm. The values are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-accountable-ex3].
:::

1.  [Confirm the closed form is exact.]{.wex-op} The full-model prediction is \(f(x_\star)=0.7249\). The rank-one deletion effect \(\Delta_i\) agrees with an actual refit-without-\(i\) to \(1.6\times 10^{-15}\), machine precision: the formula is not an approximation.
2.  [Rank the responsible points.]{.wex-op} The three largest \(\lvert\Delta_i\rvert\) come from \(x_i=0.781\) (weight \(\alpha_i=-1.344\), leverage \(H_{ii}=0.308\), \(k(x_i,x_\star)=0.980\), \(\Delta_i=-0.0555\)), then \(x_i=0.521\) (\(\Delta_i=+0.0325\)) and \(x_i=0.963\) (\(\Delta_i=+0.0270\)). They are the nearest high-weight neighbors of \(x_\star\), and deleting the first alone would move the prediction by \(0.056\).
3.  [Turn it into an audit statement.]{.wex-op} The decision at \(x_\star\) rests chiefly on three named training points; if any were later found mislabeled or unrepresentative, the exact effect on this prediction is known in advance, at the cost of one solve.

**Reading.** Data attribution that is a delicate approximation for deep models is, for a kernel ridge model, a closed-form consequence of the hat matrix. The same quantity \(1-H_{ii}\) that yields the leave-one-out error yields the influence of a point on any prediction, so cross-validation and attribution are the same linear algebra. Data-valuation schemes such as Data Shapley (Ghorbani and Zou 2019) build on exactly this notion of a training point's marginal worth.
::::
:::::

<figure class="viz" data-figure="influence-concentration-curve" data-alt="Cumulative absolute training-point contribution to one kernel-ridge prediction is plotted against the number of largest contributions retained for three RBF length scales."><figcaption>Example-based explanation has a concentration profile. Local kernels can place most of a prediction's absolute contribution on a small reviewable set, while broader kernels distribute responsibility. Reporting the mass captured by the top \(k\) cases is more informative than naming an arbitrary fixed number.</figcaption></figure>

## Auditing a deployed kernel model {#auditing-deployment}

A model that was accountable at training time can drift out of validity in deployment. An auditor then asks whether the input distribution still resembles training, whether outputs depend on attributes they should not use, and whether the whole pipeline can be reconstructed. Kernel tests address the first two questions; a recorded data and solver lineage addresses the third.

### Is the input still the training distribution? Kernel drift monitoring {#drift}

The natural monitor is the kernel two-sample test [@gretton2012, Sections 2 and 5]. Let \(k\) be measurable and bounded, or satisfy the integrability conditions needed for both mean embeddings. Writing \(\mu_P=\mathbb E\,k(x,\cdot)\), the squared MMD is

$$\mathrm{MMD}^2(P,Q)=\mathbb E\,k(x,x')+\mathbb E\,k(y,y')-2\,\mathbb E\,k(x,y),$$

with an unbiased U-statistic estimator built directly from Gram blocks.

:::: {.proposition #prop-accountable-mmd-monitor}
[Proposition (scope of an MMD deployment alarm)]{.box-title}

Let \(X_1,\ldots,X_m\overset{\mathrm{iid}}{\sim}P\) and \(Y_1,\ldots,Y_n\overset{\mathrm{iid}}{\sim}Q\) be mutually independent. If \(k\) is characteristic on the declared class of Borel probability measures and its mean embeddings exist, then \(\operatorname{MMD}_k(P,Q)=0\) if and only if \(P=Q\), which is precisely the characteristic-kernel definition [@sriperumbudur2010, Definition 7]. Under \(H_0:P=Q\), the pooled observations are exchangeable. A permutation test that fixes the entire test rule in advance, or repeats every label-dependent kernel-selection step within each permutation, therefore controls Type I error at its randomization level.

**Assumptions.** Mutually independent IID samples, existing mean embeddings, characteristic kernel for identification, and an exchangeability-preserving randomization design.

**Proof status.** Population identification is cited; permutation validity follows from invariance of the pooled null law under label permutations.
::::

Rejection is evidence against \(P=Q\); non-rejection is not evidence of equality and supplies no lower bound on power. Serial dependence, overlapping windows, adaptive repeated testing, or bandwidth selection after inspecting source-versus-production labels invalidates this elementary calibration unless the resampling and multiplicity correction reflect that design. An alarm says that input distributions differ in a direction visible to the chosen kernel. It does not establish covariate shift, identify the changed coordinate, or prove that predictions are wrong.

::::: {.example #example-55-4}
[Example (a drift alarm that does not cry wolf)]{.box-title}

:::: wex
::: wex-setup
A reference sample of \(200\) standard-normal inputs, an RBF kernel at the median-heuristic bandwidth, and \(2000\) permutations for the null. We test the reference against a fresh unshifted draw and against a window shifted in mean by \(0.5\). The values are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-accountable-ex4].
:::

1.  [Set the bandwidth from the data.]{.wex-op} The median pairwise distance is \(0.946\), giving \(\gamma=0.559\) in \(k(x,x')=e^{-\gamma(x-x')^2}\).
2.  [Pass the null case.]{.wex-op} Against the unshifted draw the statistic is \(\mathrm{MMD}^2=-0.004\) (the unbiased estimator can be negative) with p-value \(0.980\): no false alarm.
3.  [Catch the shift.]{.wex-op} Against the shifted window \(\mathrm{MMD}^2=+0.025\) with p-value \(0.001\): drift detected at any reasonable level.

**Reading.** This is a principled replacement for an ad-hoc threshold on some summary statistic: a hypothesis test whose null rejection means the production inputs no longer look like training. It is also the guard on the previous section's guarantee, because conformal coverage assumes the test point is exchangeable with calibration, and an MMD alarm is precisely a signal that the assumption has broken. Its honest cost is that a characteristic kernel and adequate sample size are needed for power, and that it reports that the distributions differ, not which feature moved.
::::
:::::

<figure class="viz" data-widget="drift-mmd">

<figcaption>A live drift monitor. Slide the covariate-shift amount applied to the production window; the unbiased \(\mathrm{MMD}^2\) between reference and window is recomputed, a fast permutation null is redrawn, and the statistic is shown against the null band with its p-value falling below the alarm level as the shift grows. Real statistic, real permutation null on every change.</figcaption>
</figure>

### Independence and fairness audits with HSIC {#independence-fairness}

A different audit asks whether outputs are statistically independent of an attribute, batch label, or sensor id. HSIC measures dependence as the squared Hilbert-Schmidt norm of the cross-covariance operator [@gretton2005hsic, Section 2], estimated here by

$$\widehat{\mathrm{HSIC}}=\frac{1}{(n-1)^2}\operatorname{tr}(KHLH),\qquad H=I-\tfrac1n\mathbf 1\mathbf 1^\top,$$

with \(K\) a kernel on predictions and \(L\) a kernel on the attribute.

:::: {.proposition #prop-accountable-hsic}
[Proposition (what an HSIC fairness diagnostic establishes)]{.box-title}

Let \((\widehat Y_i,A_i)_{i=1}^n\) be IID on compact domains. Suppose the two RKHSs have bounded universal kernels, with their unit balls uniformly bounded. Then population HSIC is zero if and only if \(\widehat Y\) and \(A\) are independent [@gretton2005hsic, Theorem 4]. Under the null, permuting the \(A_i\) gives an exact finite-sample randomization test when kernels and tuning choices are fixed independently of the pairing [@gretton2008hsictest, Sections 2--3].

**Assumptions.** IID held-out pairs on compact domains, bounded universal kernels as in the cited theorem, and fixed tuning under permutation.

**Proof status.** Population identification and the permutation construction are cited to the primary HSIC papers.
::::

Rejection establishes statistical dependence, not discrimination or causation. Non-rejection does not establish independence. Independence targets demographic parity; equalized odds instead requires \(\widehat Y\perp A\mid Y\), while calibration and individual fairness require different estimands. If predictions were produced by fitting on these same audit records, the pairs need not satisfy the IID requirement; use held-out audit data or repeat fitting inside a valid resampling design. An HSIC penalty can target empirical dependence [@perezsuay2017fair, Section 3], but a small training penalty is not a population fairness certificate.

::::: {.example #example-55-5}
[Example (auditing dependence on a protected attribute)]{.box-title}

:::: wex
::: wex-setup
Two hundred cases, a continuous protected attribute, and two predictors: a biased one whose score leans on the attribute, and a fair one from which that dependence has been removed. We test each with HSIC and a permutation null. The values are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-accountable-ex5].
:::

1.  [Test the biased model.]{.wex-op} \(\mathrm{HSIC}=0.0231\) with p-value \(0.000\): the hypothesis of independence is decisively rejected, so the predictions do depend on the attribute.
2.  [Test the fair model.]{.wex-op} \(\mathrm{HSIC}=0.0006\) with p-value \(0.775\): independence is not rejected. The same statistic that flagged the first model clears the second.

**Reading.** A group-fairness audit becomes a hypothesis test with a p-value, and because HSIC can also be added as a training penalty (Fair Kernel Learning, Pérez-Suay et al. 2017, which we meet again over satellite data in the next chapter), the property is both testable and enforceable with one kernel quantity. The honest caveat is that HSIC targets statistical independence, a demographic-parity-style notion, and other fairness definitions require other criteria.
::::
:::::

### Reproducibility and the audit trail {#reproducibility}

The last audit is procedural rather than statistical. For regularized kernel ridge regression, the predictor is the solution of a specified linear system; for an SVM, it is the primal predictor satisfying a specified convex program and KKT tolerance. Reproduction therefore requires the data version, preprocessing, kernel convention, hyperparameters, approximation, arithmetic precision, solver, stopping tolerance, and software environment. With those fixed, independent runs should agree within a declared numerical tolerance. This is stronger and more testable than saying “rerun the code,” but it is not a promise of bitwise-identical coefficients across hardware and libraries.

This is where the technical machinery meets the documentation frameworks that high-stakes deployment now expects. Model cards (Mitchell et al. 2019) ask for intended use, performance across conditions, and limitations; a kernel model can supply example contributions, feature-dependence tests, and intervals with stated assumptions. Datasheets for datasets (Gebru et al. 2021) ask for data provenance; exact kernel-ridge deletion influence can connect particular predictions to particular records. Risk-management frameworks such as the NIST AI Risk Management Framework (2023) call for measured uncertainty, robustness, and fairness, and conformal coverage, MMD drift tests, and HSIC independence tests are relevant measurements when their assumptions match the question. These artifacts support technical traceability; they are neither a legal determination nor a substitute for governance.

### A reproducible audit protocol {#reproducible-audit-protocol}

An audit becomes operational only when it can stop deployment. The protocol below separates numerical failure, assumption failure, statistical alarm, and insufficient evidence instead of compressing all four into a green or red badge.

:::: {.algorithm #algo-accountable-audit}
[Algorithm (Predeclared accountable-kernel release gate)]{.box-title}

::: algo-io
[Input]{.algo-lab} immutable identifiers for training, calibration, reference, and held-out audit data; preprocessing and kernel specifications; hyperparameters; solver and precision; random seed; familywise monitoring schedule; thresholds \(\tau_{\mathrm{solve}},\tau_{\mathrm{loo}},\alpha_{\mathrm{cov}},\alpha_{\mathrm{drift}},\alpha_{\mathrm{fair}}\); and minimum subgroup sizes.

[Output]{.algo-lab} a signed record with status `pass`, `fail`, or `insufficient evidence` for every gate.
:::

1.  Verify hashes, sample units, split disjointness, finiteness, Gram symmetry, and the smallest eigenvalue tolerance. Record the linear-system residual. Fail on mismatch, leakage, NaN/Inf, or residual above \(\tau_{\mathrm{solve}}\).
2.  Recompute predictions and intervals from a fresh process. Require prediction agreement within the declared tolerance; do not require bitwise equality.
3.  Compare every closed-form deletion effect against at least one direct deletion refit. Fail if the maximum discrepancy exceeds \(\tau_{\mathrm{loo}}\), or if any \(1-H_{ii}\) is below the leverage guard.
4.  On untouched labeled audit data, report marginal and predeclared subgroup coverage with binomial uncertainty. Fail the marginal gate when its one-sided confidence bound is below the target; mark a subgroup `insufficient evidence` rather than passing it when its sample size is below the preregistered minimum.
5.  Run the predeclared MMD and HSIC randomization tests on their declared independent sampling units. Correct for the number and cadence of tests. An adjusted drift rejection triggers investigation, abstention, or recollection; an adjusted HSIC rejection triggers a fairness investigation under the stated fairness estimand. Neither non-rejection is a certificate.
6.  Stress-test GP intervals by coverage stratum and standardized residuals. A model-based band may be reported as a GP posterior interval, but it may be labeled calibrated only if the held-out calibration gate passes.
7.  Store software versions, seed, thresholds, statistics, p-values, confidence bounds, warning states, and the exact action taken. Any changed model, kernel, threshold, or data version creates a new audit.
::::

::::: {.example #example-accountable-audit-fixture}
[Example (a deterministic release gate with deliberate failures)]{.box-title}

:::: wex
::: wex-setup
the chapter's computational reference [@kernelbook-code-ch-accountable-audit] uses a fixed seed and a one-dimensional fixture. It fits kernel ridge regression, verifies one deletion both algebraically and by refitting, constructs a split-conformal interval, and runs fixed-permutation MMD and HSIC diagnostics. It then injects a shifted production window and an attribute-dependent score.
:::

1.  [Pass the numerical gates.]{.wex-op} The linear-system relative residual must be below \(10^{-10}\), the Gram matrix must be symmetric within \(10^{-12}\), and the deletion formula must agree with a direct refit within \(10^{-10}\).
2.  [Separate a clean null from an alarm.]{.wex-op} The fixed randomization schedule must not reject the unshifted MMD fixture at \(0.05\), but must reject the shifted fixture. The same schedule must not reject the independent-score HSIC fixture and must reject the attribute-dependent fixture.
3.  [Fail visibly.]{.wex-op} The script exits nonzero if any numerical invariant, expected null behavior, expected alarm, or conformal-rank convention changes. These expected outcomes are regression fixtures, not claims about test power in every deployment.

**Reading.** A reproducible audit records both the calculation and the decision rule that consumes it. Determinism makes software regressions visible; it does not turn one synthetic fixture into external validation.
::::
:::::

## Uncertainty that drives the next experiment {#uncertainty-drives-experiment}

Model-based variance is not only something to report; it is something to act on. When each new label costs a wet-lab experiment or a supercomputer run, posterior variance can guide where the next label should be spent, provided the covariance is stress-tested against error. The basic rule is uncertainty sampling: query where \(\sigma^2(x)\) is largest. For a fixed GP covariance and noise model, this quantity depends only on input locations, so a query can be chosen before the expensive label is known. Its acquisition-function generalizations, which trade the mean against the variance to optimize rather than merely explore, are the subject of [[ch:bayesian-optimization-and-bandits|the Bayesian optimization chapter]] (Frazier 2018; Shahriari et al. 2016), and the calibration of a computer model against reality by Kennedy and O'Hagan (2001) is the emulation backdrop the next chapter builds on.

<figure class="viz" data-widget="active-variance">

<figcaption>Active learning on a one-dimensional target. Each step places the next sample at the point of maximum posterior variance (marked before it is evaluated), then refits the Gaussian process; a second trace samples at random for comparison. The uncertainty band collapses fastest exactly where the query was placed. Real GP conditioning per step, the mechanism the next chapter's on-the-fly force fields run at scale.</figcaption>
</figure>

The next chapter turns this from a demonstration into a working method. In an on-the-fly interatomic potential, a molecular-dynamics simulation runs on the cheap Gaussian-process energy until the predictive variance spikes at a configuration the model has never seen, at which point a single quantum-mechanical calculation is triggered, added to the training set, and the simulation continues. The variance is the whole control signal, and it is the same variance whose honest reading opened this chapter.

## Common mistakes and practical implications {#accountable-practice}

- A Gaussian-process variance is only as honest as its kernel and noise level; a badly chosen length scale reports confident nonsense, so the hyperparameters that set the error bar must be fit and disclosed, not assumed.
- Calibrated marginal coverage from conformal prediction is not conditional coverage: an interval correct on average can still be systematically too tight for an identifiable subgroup.
- Conformal guarantees rest on exchangeability; under drift the calibration set stops representing deployment, and coverage degrades silently unless a drift test guards it.
- A calibration or audit set consumed for tuning is no longer a valid calibration or audit set, and reusing it inflates every guarantee computed from it.
- Influence and leave-one-out identities are exact only for the convex, exactly solved kernel machine; imported wholesale onto an approximate or nonconvex fit they become the same fragile estimates they were meant to replace.
- An MMD or HSIC test that fails to reject is not a certificate of no drift or no dependence; it bounds only what the chosen kernel and sample size could have detected.

The practical goal is not a badge of trustworthiness but a documented chain: a disclosed kernel and noise model, an error bar with its assumptions, a coverage guarantee with its exchangeability caveat, and drift and dependence audits that can fail visibly and trigger a retrain.

## Summary and further reading {#summary}

Accountability is not a badge produced by one metric; it is a chain of evidence. The representer form supplies exact example contributions, while kernel-ridge deletion influence answers a sharper counterfactual with one hat matrix. A Gaussian process supplies covariance conditional on its kernel and likelihood; split conformal adds finite-sample marginal coverage under exchangeability. MMD and HSIC turn drift and dependence questions into tests, but population identifiability does not remove finite-sample power limits. Convex objectives make residuals and optimality conditions auditable, while reproducibility still requires a recorded preprocessing, approximation, solver, tolerance, and environment. The practical deliverable is therefore a model record with assumptions, diagnostics, and triggers: recalibrate when coverage fails, investigate when drift is detected, abstain when support is missing, and never promote a model-based variance into a real-world guarantee without validation.

For further reading, Rasmussen and Williams (2006) is the reference for the Gaussian-process variance and its hyperparameter learning; Vovk, Gammerman, and Shafer (2005) and Angelopoulos and Bates (2023) develop conformal prediction and its finite-sample coverage; Koh and Liang (2017) motivate influence functions for accountability and Cook and Weisberg (1982) give the classical leave-one-out and hat-matrix identities; Gretton et al. (2012) and Gretton et al. (2005) supply the MMD and HSIC audit statistics used here.

::: {.exercises}
## Exercises {#exercises}

1.  [warm-up]{.ex-tag} A kernel ridge model predicts a value with no error bar, and a stakeholder asks \"how sure are you.\" Name two different things that phrase could mean for this model, one answerable from the kernel geometry alone and one requiring an assumption about the noise, and say which of the two the Gaussian-process variance \(\sigma^2(x_\star)=k(x_\star,x_\star)-k(x_\star)^\top(K+\sigma^2I)^{-1}k(x_\star)\) reports honestly and why.
2.  [computation]{.ex-tag} A calibration set has \(n=19\) absolute residuals, sorted: \(0.10,0.12,\dots\). For \(\alpha=0.10\), which rank is the conformal quantile \(\hat q\), and what is the interval? Then show that as \(n\to\infty\) the rank \(\lceil(n+1)(1-\alpha)\rceil/n\) tends to the \((1-\alpha)\) empirical quantile, and explain why the finite-\(n\) correction always makes the interval at least as wide as the plug-in quantile would.
    Hint

    ::: hint-body
    \(\lceil 20\cdot 0.9\rceil=18\), so \(\hat q\) is the \(18\)th smallest residual and \(C(x)=\hat f(x)\pm\hat q\). The ceiling rounds up, so the finite-sample quantile sits at or above the plug-in one; this is the price of the exact \(1-\alpha\) lower bound.
    :::
3.  [proof]{.ex-tag} Kernel ridge regression is the linear smoother \(\hat y=Hy\), \(H=K(K+\lambda I)^{-1}\). Prove the closed-form leave-one-out identity \(y_i-\hat f^{(-i)}(x_i)=(y_i-\hat y_i)/(1-H_{ii})\).
    Hint

    ::: hint-body
    Let \(\hat f^{(-i)}\) be the fit with point \(i\) removed and consider the auxiliary problem in which \(y_i\) is replaced by \(\hat f^{(-i)}(x_i)\); its fit is unchanged at \(i\), so \(\hat f^{(-i)}(x_i)=\sum_j H_{ij}\tilde y_j\) with \(\tilde y_i=\hat f^{(-i)}(x_i)\). Solve the resulting scalar equation for \(\hat f^{(-i)}(x_i)\) and subtract from \(y_i\).
    :::
4.  [exploration]{.ex-tag} Take the influence example (RBF length scale \(0.4\), ridge \(0.1\), \(x_\star=0.7\)). Predict how the ranked list of the three most influential points changes as the ridge \(\lambda\) grows large, then check with a short script, and connect the change to the leverages \(H_{ii}\) and the effective dimension of [[ch:applications-and-practice|the practice chapter]].
    Hint

    ::: hint-body
    As \(\lambda\to\infty\), \(H\to0\), the leverages and all \(\Delta_i\) shrink, and the fit flattens toward a constant; the ranking is dominated by \((Gk_\star)_i\), which for a local kernel still favors the neighbors of \(x_\star\) but with far smaller magnitude.
    :::
5.  [challenge]{.ex-tag} Explain precisely why the conformal coverage guarantee requires exchangeability, and construct a deployment scenario (a slow covariate shift over time) in which split conformal silently loses coverage. Then argue why an MMD drift test between the calibration window and the current window is the right guard, and what its rejection tells you to do.
    Hint

    ::: hint-body
    Exchangeability makes the test residual's rank among the calibration residuals uniform; a time trend breaks it so the test residual is stochastically larger, and coverage falls below \(1-\alpha\). An MMD alarm signals the calibration set is no longer representative, so \(\hat q\) must be recomputed on a fresh window.
    :::
6.  [exploration]{.ex-tag} The MMD drift test used the median-heuristic bandwidth. Investigate its effect on power: rerun the shifted-versus-unshifted test as the bandwidth ranges over an order of magnitude around the median, and describe the trade-off. Then explain why a characteristic kernel is necessary for the guarantee \"\(\mathrm{MMD}=0\) iff \(P=Q\)\" and give a kernel for which it fails.
    Hint

    ::: hint-body
    Too small a bandwidth makes every point look dissimilar and kills power; too large makes every point look identical and also kills power; the median sits near the useful middle. A non-characteristic kernel, such as a polynomial of fixed degree, embeds distinct distributions to the same mean and can give \(\mathrm{MMD}=0\) for \(P\neq Q\).
    :::
:::
