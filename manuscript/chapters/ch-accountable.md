---
id: ch-accountable
slug: accountable-kernels
title: 'Accountable Kernels: Uncertainty, Explanation, and Audit'
part: XVIII · Kernels You Can Defend
order: 57
tier: advanced
prerequisites:
  - reproducing-kernel-banach-and-variation-spaces
objectives:
  - >-
    Explain the central definitions and claims in Accountable Kernels:
    Uncertainty, Explanation, and Audit.
  - Apply the chapter's principal methods and interpret their outputs.
  - >-
    State the assumptions behind formal results and connect them to earlier
    chapters.
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
  - kennedy2001calibration
---
# Accountable Kernels: Uncertainty, Explanation, and Audit

<p class="lead">A prediction that will steer a spacecraft, propose a molecule for synthesis, or drive a satellite crop-yield estimate is not judged by its accuracy alone. Someone will ask how sure the model is, which training data the decision rests on, whether the inputs still resemble what the model was trained on, and whether the whole thing can be reproduced and documented for a review board. For most modern models these are afterthoughts, bolted on with a separate uncertainty head, a post-hoc saliency map, or a monitoring dashboard that lives outside the model. For a kernel machine they are not afterthoughts at all: they fall out of the same geometry the book has been building. The representer theorem makes every prediction an explicit, signed sum over named training points, so attribution is exact and free. The fit is a deterministic convex solve, so it is reproducible bit for bit. The Gaussian-process posterior returns a calibrated variance as a first-class output, and where that calibration is not to be trusted, a distribution-free conformal wrapper restores a finite-sample guarantee. The audit tools, drift tests and independence tests, are themselves kernel statistics. Accountability, in short, is a structural property of a kernel model rather than a wrapper around it, and the next chapter shows the payoff in the sciences that demand it.</p>

## Accountability is a property of the model, not a wrapper {#a-property-not-a-wrapper}

Three facts, each proved earlier in the book, do all the work of this chapter, and it is worth stating them together before we develop any one of them, because their conjunction is the argument.

The first is the representer theorem of [[ch:kernel-tricks|the kernel-tricks chapter]], in the general form of Schölkopf, Herbrich, and Smola (2001): the minimizer of any regularized empirical risk over an RKHS lies in the span of the training data mapped into feature space, so the fitted predictor is

$$f(x)=\sum_{i=1}^n \alpha_i\,k(x_i,x).$$

Every value the model ever outputs is a weighted vote over the training examples, with the weight on example \(i\) equal to \(\alpha_i\,k(x_i,x)\). This is not an approximation to an explanation; it is the model. Ask why the prediction at \(x\) came out as it did and the answer is a list of past cases and their signed similarities to \(x\).

The second is that fitting is a deterministic convex program. Kernel ridge regression and the Gaussian-process posterior mean solve the linear system \((K+\lambda I)\alpha=y\); the support vector machine of [[ch:support-vector-machines|its chapter]] solves a convex quadratic program with a unique optimum (Cortes and Vapnik 1995). There is no random initialization, no mini-batch order, no early-stopping seed. Given the data, the kernel, and the hyperparameters, refitting reproduces the same solution to the last bit that the arithmetic allows. A regulator who asks for a reproducible training procedure is handed a short, closed specification rather than a stochastic recipe.

The third is that a Gaussian process, from [[ch:gaussian-processes-and-rvm|the Bayesian chapter]], returns not a point but a predictive distribution, whose variance

$$\sigma^2(x_\star)=k(x_\star,x_\star)-k(x_\star)^\top(K+\sigma^2 I)^{-1}k(x_\star)$$

depends only on where \(x_\star\) sits relative to the data, not on the observed labels. The model reports its own ignorance: far from the data the variance climbs back toward the prior. That error bar is delivered by the same solve as the mean, not by a separate procedure trained to imitate uncertainty.

The contrast with a deep network is not a matter of taste. A network's explanation must be reconstructed after the fact by a saliency method whose faithfulness is itself in question; its uncertainty must be approximated by an ensemble or by dropout; its training run is not reproducible without pinning a dozen sources of nondeterminism. The kernel model supplies all three natively because they are consequences of positive definiteness and convexity. The rest of this chapter turns each fact into a working tool, and is honest about where each one strains: calibration depends on the kernel being right, attribution is only as inspectable as the training set is documented, and every exact statement here is an exact statement about a model small enough to solve exactly.

## Uncertainty you can put in a report {#uncertainty-you-can-report}

A number without an error bar is not yet a scientific result, and a great deal of what makes kernel methods trusted in the sciences of the next chapter is that they hand back the error bar as part of the answer. We take the native Gaussian-process interval first, see exactly where it can be trusted and where it cannot, and then layer on the distribution-free guarantee that repairs the gap.

### The Gaussian-process interval, and its honest limits {#gp-error-bars}

The predictive variance above has a clean reading. It starts at the prior variance \(k(x_\star,x_\star)\) and subtracts the quadratic form \(k(x_\star)^\top(K+\sigma^2 I)^{-1}k(x_\star)\), which is exactly the information the training data supply about \(x_\star\). Near a cluster of data the subtraction is large and the posterior tightens; in a gap or beyond the data the subtraction vanishes and the interval balloons back toward the prior. Under a correctly specified model, the intervals \([\,\mu(x_\star)\pm z_{1-\alpha/2}\,\sigma(x_\star)\,]\) are genuine Bayesian credible intervals with the coverage they claim.

The qualifier is the whole story, so we make both halves of it concrete on numbers.

::::: {.example #example-55-1}
[Example (an honest error bar, and a dishonest one)]{.box-title}

:::: wex
::: wex-setup
First the honest behavior. We fit a Gaussian process with an RBF kernel (length scale \(1\), noise \(0.05\)) to eight points of \(\sin x\) on \([0,6]\) with a deliberate gap in \((2.5,4.5)\), and read the posterior standard deviation at three places. Then the failure. We draw \(40\) points from a wiggly mean \(\sin(3x)\) on \([0,1]\) corrupted by heteroscedastic noise whose standard deviation grows from \(0.05\) at \(x=0\) to \(0.5\) at \(x=1\), and fit a GP that wrongly assumes a single homoscedastic noise equal to the average, \(\hat\sigma=0.292\). We then measure how often the nominal \(90\%\) interval actually covers fresh data. All numbers from `checks/ch-accountable-ex1.py`.
:::

1.  [Watch the band breathe.]{.wex-op} The posterior standard deviation is \(0.049\) at a training point (\(x=0.9\)), essentially the noise floor; it rises to \(0.472\) in the middle of the gap (\(x=3.5\)); and it is \(0.247\) just past the data (\(x=6.5\)). The model announces exactly where it has no evidence, and it does so from the inputs alone.
2.  [Now misspecify the noise.]{.wex-op} Averaged over the whole domain the nominal \(90\%\) interval covers \(89.1\%\) of fresh points, which looks fine. It is not fine: restricted to the high-noise half of the domain (\(x\gt 0.5\)), coverage falls to \(79.2\%\). The single assumed noise is too small where the true noise is large, and the interval silently under-covers there.
3.  [Read the lesson.]{.wex-op} The GP error bar is honest about *where* the data are sparse (step 1), because that part depends only on the kernel geometry. It is only as honest about *how noisy* the data are as the noise model you gave it (step 2). Aggregate coverage hid a local failure that a report on the noisy regime would have relied on.

**Reading.** Two different promises live inside one interval. The width from the kernel geometry is trustworthy and is the source of the \"it knows when it is extrapolating\" behavior that makes GPs valued in the sciences. The width from the assumed noise model is trustworthy only under correct specification, and a misspecified kernel or noise level can make the nominal coverage a fiction exactly where it matters. This is the gap the next section closes.
::::

**Verification artifact.** checks/example-ch-accountable-example-55-1.json records the example source hash and verification scope.
:::::

### Conformal prediction: a distribution-free coverage guarantee {#conformal}

The way to stop trusting the model's self-assessment and start certifying coverage is conformal prediction (Vovk, Gammerman, and Shafer 2005; Lei et al. 2018; Angelopoulos and Bates 2021). It wraps any predictor, kernel or otherwise, and turns its residuals into an interval with a finite-sample coverage guarantee that assumes nothing about the data distribution beyond exchangeability. The split, or inductive, form is a few lines.

:::: {.algorithm #algo-55-1}
[Algorithm (Split-conformal prediction for regression)]{.box-title}

::: algo-io
[Input]{.algo-lab} a fitted predictor \(\hat f\) (trained on data disjoint from the calibration set); a calibration set \(\{(x_i,y_i)\}_{i=1}^n\); a miscoverage level \(\alpha\).

[Output]{.algo-lab} a prediction interval \(C(x)\) with \(\mathbb P\big(Y\in C(X)\big)\ge 1-\alpha\).
:::

1.  Compute the conformity scores \(s_i=\lvert y_i-\hat f(x_i)\rvert\) on the calibration set.
2.  Set \(\hat q\) to the \(\big\lceil (n+1)(1-\alpha)\big\rceil\)-th smallest score.
3.  Return \(C(x)=[\,\hat f(x)-\hat q,\ \hat f(x)+\hat q\,]\).
::::

The guarantee is the theorem of Vovk, Gammerman, and Shafer (2005), sharpened by Lei et al. (2018): if the calibration points and the test point are exchangeable, then

$$1-\alpha\ \le\ \mathbb P\big(Y_{n+1}\in C(X_{n+1})\big)\ \le\ 1-\alpha+\frac{1}{n+1},$$

the upper bound holding when the scores are almost surely distinct. Nothing about the shape of the data enters. The predictor \(\hat f\) can be a kernel ridge fit, a support vector regressor, or a Gaussian-process mean; if it is good the interval is tight, and if it is poor the interval is wide, but the coverage is certified either way. The jackknife+ of Barber et al. (2021) achieves a companion guarantee using leave-one-out residuals, which sits especially well with kernel models because their leave-one-out residuals are closed form, as the next section shows.

::::: {.example #example-55-2}
[Example (conformal repairs the coverage the GP lost)]{.box-title}

:::: wex
::: wex-setup
Same heteroscedastic truth as the failure above. We fit kernel ridge regression (RBF length scale \(0.15\), ridge \(10^{-2}\)) on a training split, calibrate on \(n=500\) held-out points at \(\alpha=0.10\), and test on \(8000\) fresh points. We compare the conformal band against a naive Gaussian band \(\hat f(x)\pm 1.645\,\hat\sigma\) using a single \(\hat\sigma\) estimated from the training residuals. All numbers from `checks/ch-accountable-ex2.py`.
:::

1.  [Read the quantile off the calibration residuals.]{.wex-op} The rank is \(\lceil 501\cdot 0.9\rceil=451\), so \(\hat q\) is the \(451\)st smallest of the \(500\) absolute residuals, \(\hat q=0.512\). The band is \(\hat f(x)\pm 0.512\), width \(1.025\).
2.  [Certify the coverage.]{.wex-op} On the fresh test set the conformal band covers \(89.9\%\), inside the guaranteed envelope \([0.900,\,0.902]\) up to test-sample noise. The naive Gaussian band, width \(0.909\), covers only \(87.1\%\): it is narrower and under-covers, because one \(\hat\sigma\) cannot describe noise that varies across the domain.
3.  [Note what conformal does and does not fix.]{.wex-op} Conformal restores *marginal* coverage to the target. It does not by itself equalize coverage across the domain: in the high-noise half the conformal band still covers only \(81.0\%\) (the naive band, \(76.0\%\)). The guarantee is an average over the input distribution, not a promise at every \(x\).

**Reading.** A defensible interval for a report is one whose coverage you can certify without believing the model's noise assumptions, and split conformal delivers exactly that from a held-out set and a single sorted list of residuals. Its honest limitation is that the certificate is marginal: to tighten it where the data are hard, one conditions the score on the input (normalized or locally-weighted conformal), which the exercises pursue. The pairing is the practical recipe of the chapter: let the Gaussian process propose the interval, and let conformal certify it.
::::

**Verification artifact.** checks/example-ch-accountable-example-55-2.json records the example source hash and verification scope.
:::::

<figure class="viz" data-widget="conformal-coverage">

<figcaption>Split conformal on a live kernel-ridge fit. The band half-width is the calibration-residual quantile \(\hat q\); as the target coverage \(1-\alpha\) slides, \(\hat q\) is recomputed from the sorted residuals and a running counter over a held-out stream reports the empirical coverage, which tracks the target inside the \(1/(n{+}1)\) envelope. Shrinking the calibration size loosens the guarantee visibly.</figcaption>
</figure>

## Explanation for free: the representer theorem as attribution {#explanation-for-free}

When a kernel model makes a decision that someone wants to contest, the question is rarely \"which pixels\" and usually \"which prior cases, and how much.\" Kernel machines answer that question exactly, because the answer is written into the representer form, and for the convex ones the sensitivity of a prediction to each training point is closed form rather than estimated.

### Prediction as a weighted vote over training points {#representer-attribution}

Return to \(f(x)=\sum_i\alpha_i\,k(x_i,x)\). Each training point casts a vote for the prediction at \(x\), of magnitude \(\lvert\alpha_i\rvert\,k(x_i,x)\) and sign \(\operatorname{sign}(\alpha_i)\). Two features make this a usable explanation rather than a formula. In the support vector machine the Karush-Kuhn-Tucker conditions force most \(\alpha_i\) to be exactly zero, so the prediction depends on a small, inspectable set of support vectors; the explanation is sparse by construction. In kernel ridge and Gaussian-process regression the weights \(\alpha=(K+\lambda I)^{-1}y\) are dense, so the credit is spread, but for a local kernel the factor \(k(x_i,x)\) concentrates the contribution on the training points near \(x\). Either way, \"why this prediction\" resolves to named examples with signed weights, and no surrogate model is needed to produce it.

The same RKHS machinery answers the dual question, \"which examples represent the data, and which are the edge cases,\" through the prototypes and criticisms of Kim, Khanna, and Koyejo (2016). Prototypes are the points whose empirical distribution best matches the data in the sense of maximum mean discrepancy from [[ch:kernel-mean-embeddings|the mean-embedding chapter]]; criticisms are the points in regions the prototypes represent worst, found by the witness function. Both are computed from the same Gram matrix the model already uses, and both are consumable by a human reviewer as \"these are the typical cases the model has learned, and these are the outliers it may not handle.\"

<figure class="viz" data-widget="influence-glow">

<figcaption>A live kernel-ridge fit on a handful of points. Drag the query \(x_\star\); each training point glows in proportion to its signed contribution \(\alpha_i\,k(x_i,x_\star)\) to the prediction there, and the readout names the three most influential points. Toggle to the exact leave-one-out influence, the change in \(f(x_\star)\) if that point were deleted, and the ranking shifts to what the decision truly rests on. Every number is read from the actual fitted expansion.</figcaption>
</figure>

### Exact influence and closed-form leave-one-out {#influence-loo}

Beyond \"which points support this prediction\" is the sharper counterfactual, \"how would this prediction change if a given training point had never been collected.\" For general models this is the province of influence functions (Koh and Liang 2017), which approximate the effect of up-weighting a training point through the inverse Hessian of the loss. Their derivation needs the empirical risk to be twice differentiable and strictly convex, so that the Hessian is positive definite and invertible. That condition is exactly what a convex kernel machine satisfies and a deep network does not, which is why influence estimates are trustworthy here and demonstrably fragile for deep nets (Basu, Pope, and Feizi 2021). Leaning into that contrast is the point: the tool that is a fragile approximation elsewhere is exact for a kernel ridge model.

Exact, and closed form. Kernel ridge regression is a linear smoother, \(\hat y = Hy\) with the hat matrix \(H=K(K+\lambda I)^{-1}\). The leave-one-out residual at a training point needs no refit,

$$y_i-\hat f^{(-i)}(x_i)=\frac{y_i-\hat y_i}{1-H_{ii}},$$

where \(H_{ii}\) is the leverage of point \(i\); and the same block-inverse identity gives the exact change in any test prediction \(f(x_\star)\) when point \(i\) is deleted, at the cost of one rank-one update rather than a refit. Ranking training points by that change identifies, for a specific decision, the data it rests on.

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
Kernel ridge regression on \(40\) points of \(\sin(1.3x)\) with light noise on \([-3,3]\) (RBF length scale \(0.4\), ridge \(0.1\)). We ask which training points most determine the prediction at \(x_\star=0.7\), computing the deletion effect both by brute-force refitting and by the rank-one formula of the algorithm. All numbers from `checks/ch-accountable-ex3.py`.
:::

1.  [Confirm the closed form is exact.]{.wex-op} The full-model prediction is \(f(x_\star)=0.7249\). The rank-one deletion effect \(\Delta_i\) agrees with an actual refit-without-\(i\) to \(1.6\times 10^{-15}\), machine precision: the formula is not an approximation.
2.  [Rank the responsible points.]{.wex-op} The three largest \(\lvert\Delta_i\rvert\) come from \(x_i=0.781\) (weight \(\alpha_i=-1.344\), leverage \(H_{ii}=0.308\), \(k(x_i,x_\star)=0.980\), \(\Delta_i=-0.0555\)), then \(x_i=0.521\) (\(\Delta_i=+0.0325\)) and \(x_i=0.963\) (\(\Delta_i=+0.0270\)). They are the nearest high-weight neighbors of \(x_\star\), and deleting the first alone would move the prediction by \(0.056\).
3.  [Turn it into an audit statement.]{.wex-op} The decision at \(x_\star\) rests chiefly on three named training points; if any were later found mislabeled or unrepresentative, the exact effect on this prediction is known in advance, at the cost of one solve.

**Reading.** Data attribution that is a delicate approximation for deep models is, for a kernel ridge model, a closed-form consequence of the hat matrix. The same quantity \(1-H_{ii}\) that yields the leave-one-out error yields the influence of a point on any prediction, so cross-validation and attribution are the same linear algebra. Data-valuation schemes such as Data Shapley (Ghorbani and Zou 2019) build on exactly this notion of a training point's marginal worth.
::::

**Verification artifact.** checks/example-ch-accountable-example-55-3.json records the example source hash and verification scope.
:::::

## Auditing a deployed kernel model {#auditing-deployment}

A model that was accountable at training time can drift out of validity in deployment, and the questions an auditor asks then are statistical: is the input distribution still the one the model was trained on, are the model's outputs independent of attributes they must not depend on, and can the whole pipeline be reproduced. Each of these is answered by a kernel statistic, and the first two come with p-values.

### Is the input still the training distribution? Kernel drift monitoring {#drift}

The natural monitor is the kernel two-sample test of Gretton et al. (2012), which measures the maximum mean discrepancy between a reference sample (the training inputs) and a recent production window. Writing the mean embedding of a distribution as \(\mu_P=\mathbb E\,k(x,\cdot)\), the squared MMD is the RKHS distance between embeddings,

$$\mathrm{MMD}^2(P,Q)=\mathbb E\,k(x,x')+\mathbb E\,k(y,y')-2\,\mathbb E\,k(x,y),$$

with an unbiased U-statistic estimator built directly from Gram blocks. For a characteristic kernel, from [[ch:kernel-mean-embeddings|the embedding chapter]] and Sriperumbudur et al. (2010), the embedding is injective, so \(\mathrm{MMD}=0\) if and only if \(P=Q\): the test can detect any distributional change, not only a shift in mean or variance. A permutation procedure supplies a finite-sample null and a p-value.

::::: {.example #example-55-4}
[Example (a drift alarm that does not cry wolf)]{.box-title}

:::: wex
::: wex-setup
A reference sample of \(200\) standard-normal inputs, an RBF kernel at the median-heuristic bandwidth, and \(2000\) permutations for the null. We test the reference against a fresh unshifted draw and against a window shifted in mean by \(0.5\). All numbers from `checks/ch-accountable-ex4.py`.
:::

1.  [Set the bandwidth from the data.]{.wex-op} The median pairwise distance is \(0.946\), giving \(\gamma=0.559\) in \(k(x,x')=e^{-\gamma(x-x')^2}\).
2.  [Pass the null case.]{.wex-op} Against the unshifted draw the statistic is \(\mathrm{MMD}^2=-0.004\) (the unbiased estimator can be negative) with p-value \(0.980\): no false alarm.
3.  [Catch the shift.]{.wex-op} Against the shifted window \(\mathrm{MMD}^2=+0.025\) with p-value \(0.001\): drift detected at any reasonable level.

**Reading.** This is a principled replacement for an ad-hoc threshold on some summary statistic: a hypothesis test whose null rejection means the production inputs no longer look like training. It is also the guard on the previous section's guarantee, because conformal coverage assumes the test point is exchangeable with calibration, and an MMD alarm is precisely a signal that the assumption has broken. Its honest cost is that a characteristic kernel and adequate sample size are needed for power, and that it reports that the distributions differ, not which feature moved.
::::

**Verification artifact.** checks/example-ch-accountable-example-55-4.json records the example source hash and verification scope.
:::::

<figure class="viz" data-widget="drift-mmd">

<figcaption>A live drift monitor. Slide the covariate-shift amount applied to the production window; the unbiased \(\mathrm{MMD}^2\) between reference and window is recomputed, a fast permutation null is redrawn, and the statistic is shown against the null band with its p-value falling below the alarm level as the shift grows. Real statistic, real permutation null on every change.</figcaption>
</figure>

### Independence and fairness audits with HSIC {#independence-fairness}

A different audit asks whether the model's outputs are statistically independent of an attribute they must not use, a protected characteristic, a batch label, a sensor id. The Hilbert-Schmidt Independence Criterion of Gretton et al. (2005) measures dependence as the squared Hilbert-Schmidt norm of the cross-covariance operator, estimated by

$$\widehat{\mathrm{HSIC}}=\frac{1}{(n-1)^2}\operatorname{tr}(KHLH),\qquad H=I-\tfrac1n\mathbf 1\mathbf 1^\top,$$

with \(K\) a kernel on the predictions and \(L\) a kernel on the attribute. For characteristic kernels \(\mathrm{HSIC}=0\) if and only if the two are independent, so it detects nonlinear dependence that a correlation would miss, and a permutation null turns it into a test (Gretton et al. 2008).

::::: {.example #example-55-5}
[Example (auditing dependence on a protected attribute)]{.box-title}

:::: wex
::: wex-setup
Two hundred cases, a continuous protected attribute, and two predictors: a biased one whose score leans on the attribute, and a fair one from which that dependence has been removed. We test each with HSIC and a permutation null. All numbers from `checks/ch-accountable-ex5.py`.
:::

1.  [Test the biased model.]{.wex-op} \(\mathrm{HSIC}=0.0231\) with p-value \(0.000\): the hypothesis of independence is decisively rejected, so the predictions do depend on the attribute.
2.  [Test the fair model.]{.wex-op} \(\mathrm{HSIC}=0.0006\) with p-value \(0.775\): independence is not rejected. The same statistic that flagged the first model clears the second.

**Reading.** A group-fairness audit becomes a hypothesis test with a p-value, and because HSIC can also be added as a training penalty (Fair Kernel Learning, Pérez-Suay et al. 2017, which we meet again over satellite data in the next chapter), the property is both testable and enforceable with one kernel quantity. The honest caveat is that HSIC targets statistical independence, a demographic-parity-style notion, and other fairness definitions require other criteria.
::::

**Verification artifact.** checks/example-ch-accountable-example-55-5.json records the example source hash and verification scope.
:::::

### Reproducibility and the audit trail {#reproducibility}

The last audit needs no experiment. Because a kernel machine is the unique optimum of a convex program, its solution is a deterministic function of the data, the kernel, and the hyperparameters: the linear system \((K+\lambda I)\alpha=y\) for ridge and Gaussian processes, a convex quadratic program for the support vector machine. Rerunning the fit reproduces the same parameters to the precision of the arithmetic, with no seed, no batch order, and no early-stopping choice to record. The documentable configuration is short and closed, which is precisely what a reproducible audit trail requires and what a stochastic-gradient training run struggles to provide.

This is where the technical machinery meets the documentation frameworks that high-stakes deployment now expects. Model cards (Mitchell et al. 2019) ask for intended use, performance across conditions, and limitations; a kernel model supplies example-based attributions, feature-relevance tests with p-values, and certified intervals to fill them. Datasheets for datasets (Gebru et al. 2021) ask for data provenance; exact influence and data valuation make each training point's contribution auditable. Risk-management frameworks such as the NIST AI Risk Management Framework (2023) call for measured uncertainty, robustness, and fairness, and conformal coverage, MMD drift tests, and HSIC independence tests are exactly such measurements. Where a regime like the European Union's AI Act designates a system high-risk and demands traceability and technical documentation, the deterministic solve and the explicit representer form make the \"how it works and how to reproduce it\" section a closed specification. None of this is a legal guarantee, and none of it discharges an obligation; the point is narrower and sound, that kernel methods produce, as a byproduct of their mathematics, the technical evidence these frameworks ask a model to supply.

## Uncertainty that drives the next experiment {#uncertainty-drives-experiment}

The calibrated variance is not only something to report; it is something to act on. When each new label costs a wet-lab experiment or a supercomputer run, the model's own uncertainty is the natural guide to where the next label should be spent, and this closes the loop from accountability to decision. The rule is uncertainty sampling: query where the posterior variance \(\sigma^2(x)\) is largest, since that is where a new observation most reduces the model's ignorance, and crucially the variance depends only on the input locations, so the most informative query can be chosen before the expensive label is known. Its acquisition-function generalizations, which trade the mean against the variance to optimize rather than merely explore, are the subject of [[ch:bayesian-optimization-and-bandits|the Bayesian optimization chapter]] (Frazier 2018; Shahriari et al. 2016), and the calibration of a computer model against reality by Kennedy and O'Hagan (2001) is the emulation backdrop the next chapter builds on.

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

Accountability is not a layer added to a kernel model; it is a reading of the model's own structure. The representer theorem makes every prediction an explicit signed sum over named training points, so example-based explanation is exact and free, and for convex kernel machines the influence of a training point and the leave-one-out error are closed-form consequences of one hat matrix, exact where the same tools are fragile approximations for deep networks. The Gaussian-process posterior returns a calibrated variance from the same solve as the mean, honest about where the data are sparse; where its noise assumptions cannot be trusted, split conformal prediction wraps any kernel predictor in an interval with a distribution-free, finite-sample coverage guarantee. Deployment is audited with kernel statistics: the MMD two-sample test flags distributional drift with a characteristic kernel that misses nothing, HSIC turns a fairness check into a hypothesis test, and the deterministic convex solve makes the whole fit reproducible for a documentation trail. Finally, the calibrated variance that made the model reportable also makes it active, choosing the next expensive experiment where it is least sure. Every claim here is exact for a model small enough to solve exactly, and the honest cost of exactness, the cubic solve and the need for a well-chosen kernel, is the same cost the next chapter's sciences pay gladly for an error bar they can defend.

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
