---
id: ch-reliability
slug: distribution-shift-robustness-and-conformal-prediction
title: 'Distribution Shift, Robustness, and Conformal Prediction'
part: XII · Reliable Practice
order: 58
tier: core
prerequisites:
  - kernel-ridge-and-friends
  - kernel-mean-embeddings
  - kernel-hypothesis-testing
objectives:
  - >-
    Distinguish covariate, label, concept, and support shift and state what each
    permits.
  - Derive MMD testing and kernel mean matching from RKHS geometry.
  - >-
    State covariate-shift rates with their overlap, curvature, and spectral
    assumptions.
  - >-
    Prove split-conformal marginal coverage and distinguish validity from
    efficiency.
  - >-
    Audit deployment with effective-sample-size, support, conditional-coverage,
    and shift diagnostics.
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

<p class="lead">The model was accurate yesterday and wrong today, and nothing in the training pipeline warned you. Held-out accuracy certifies performance under a sampling system; replace a sensor, move to a new hospital, or let time pass, and that certificate may become irrelevant while every validation dashboard still shows green. Kernel methods offer a useful chain of defenses. Mean embeddings turn distribution change into a geometric quantity. Importance weighting transfers risk under a restricted shift model. Robust optimization declares a neighborhood of distributions against which the predictor is protected. Conformal prediction turns score ranks into finite-sample prediction sets. This chapter develops those tools at research depth, including the assumptions behind the theorems, the derivations behind the algorithms, and the failure boundaries that determine whether a guarantee survives deployment.</p>

## A taxonomy of distribution change {#shift-taxonomy}

Not all change is alike. A shifted input distribution, a changed labeling mechanism, and a region absent from training require different responses. Let \(P\) be the source distribution and \(Q\) the target distribution on \(\mathcal X\times\mathcal Y\). Assume throughout that the relevant regular conditional distributions exist.

::: {.definition #def-shift-types}
[Definition (common shift models)]{.box-title}

- **Covariate shift:** \(P_X\ne Q_X\) while \(P_{Y\mid X}=Q_{Y\mid X}\), up to \(Q_X\)-null sets.
- **Label shift:** \(P_Y\ne Q_Y\) while \(P_{X\mid Y}=Q_{X\mid Y}\), up to \(Q_Y\)-null sets.
- **Concept shift:** \(P_{Y\mid X}\ne Q_{Y\mid X}\) on a set of positive target probability.
- **Support shift:** \(Q_X\not\ll P_X\), or the density ratio is so large that finite samples provide ineffective overlap.

These are restrictions on joint distributions, not labels inferred from a changed marginal. The observation \(P_X\ne Q_X\) does not identify covariate shift.
:::

For a measurable predictor \(f\) and integrable loss \(\ell_f(x,y)=\ell\{y,f(x)\}\), covariate shift gives the risk identity

$$
R_Q(f)=\mathbb E_Q\ell_f(X,Y)
=\mathbb E_P\!\left[w(X)\ell_f(X,Y)\right],
\qquad
w=\frac{dQ_X}{dP_X}.
$$

The identity requires two independent ingredients: conditional invariance and absolute continuity \(Q_X\ll P_X\). The first says that source labels remain relevant at a target input. The second says that source data contain that input region at all.

:::: {.proposition #prop-covariate-risk-identity}
[Proposition (risk transfer and its boundary)]{.box-title}

Suppose \(Q_X\ll P_X\), \(P_{Y\mid X}=Q_{Y\mid X}\) \(Q_X\)-almost surely, and \(\ell_f\in L^1(Q)\). Then

$$
R_Q(f)=\mathbb E_P\{w(X)\ell_f(X,Y)\}.
$$

If \(Q_X\not\ll P_X\), no finite source-only weighting function can reproduce every bounded target expectation.

**Assumptions.** The stated conditional invariance, absolute continuity, and integrability conditions.

**Proof status.** Complete proof below.

**Proof.** Let \(m_f(x)=\mathbb E\{\ell_f(X,Y)\mid X=x\}\), which is common to \(P\) and \(Q\) under conditional invariance. The Radon-Nikodym change of measure gives

$$
R_Q(f)=\int m_f\,dQ_X=\int w\,m_f\,dP_X
=\mathbb E_P\{w(X)\ell_f(X,Y)\}.
$$

For the second claim, choose a measurable set \(A\) with \(P_X(A)=0\) and \(Q_X(A)\gt0\). Every finite \(P_X\)-integrable weight satisfies
\(\mathbb E_P\{w(X)\mathbf 1_A(X)\}=0\), whereas
\(\mathbb E_Q\mathbf 1_A(X)=Q_X(A)\gt0\). Thus weighting cannot reproduce even the bounded test function \(\mathbf 1_A\). \(\square\)
::::

This proposition is the chapter's first hard boundary. A method may estimate weights, balance features, or minimize a robust objective, but none of these operations creates labels on a target-only set.

## Paper module: MMD as a two-sample test {#mmd-paper-module}

Gretton et al. ask a sharply defined question: given independent samples \(X_1,\ldots,X_m\sim P\) and \(Z_1,\ldots,Z_n\sim Q\), can we test \(H_0:P=Q\) without estimating either density? Their answer is an integral probability metric over the unit ball of an RKHS [@gretton2012].

**Exact setting.** Let \(k\) be measurable and positive semidefinite with RKHS \(\mathcal H\). Assume
\(\mathbb E_P\sqrt{k(X,X)}\lt\infty\) and
\(\mathbb E_Q\sqrt{k(Z,Z)}\lt\infty\), so the Bochner mean embeddings
\(\mu_P=\mathbb E_P k(X,\cdot)\) and
\(\mu_Q=\mathbb E_Q k(Z,\cdot)\) exist. Define

$$
\operatorname{MMD}_{k}(P,Q)
=\sup_{\lVert h\rVert_{\mathcal H}\le 1}
\{\mathbb E_P h(X)-\mathbb E_Q h(Z)\}.
$$

**Contribution and geometric result.** The population statistic is not merely analogous to a distance between means. It is exactly the RKHS distance

$$
\operatorname{MMD}_{k}(P,Q)=\lVert\mu_P-\mu_Q\rVert_{\mathcal H}.
$$

When \(P\ne Q\), the unit-norm maximizer is the normalized witness

$$
h^\star=\frac{\mu_P-\mu_Q}
{\lVert\mu_P-\mu_Q\rVert_{\mathcal H}},
\qquad
h^\star(t)\propto \mathbb E_P k(X,t)-\mathbb E_Q k(Z,t).
$$

The witness localizes differences visible to the kernel. It does not identify their causal mechanism.

:::: {.proposition #prop-mmd-ustat}
[Proposition (MMD identity and unbiased estimator)]{.box-title}

Under the embedding assumptions,

$$
\operatorname{MMD}_{k}^{2}(P,Q)
=\mathbb E k(X,X')+\mathbb E k(Z,Z')-2\mathbb E k(X,Z),
$$

where primed variables are independent copies. An unbiased estimator is

$$
\widehat{\operatorname{MMD}}_{u}^{2}
=\frac{1}{m(m-1)}\sum_{i\ne j}k(X_i,X_j)
+\frac{1}{n(n-1)}\sum_{i\ne j}k(Z_i,Z_j)
-\frac{2}{mn}\sum_{i,j}k(X_i,Z_j).
$$

**Assumptions.** Measurable PSD kernel, existing Bochner mean embeddings, and independent samples within and between the two groups.

**Proof status.** Complete proof below.

**Proof.** By reproducing and Fubini,

$$
\begin{aligned}
\lVert\mu_P-\mu_Q\rVert_{\mathcal H}^{2}
&=\langle\mu_P,\mu_P\rangle
+\langle\mu_Q,\mu_Q\rangle
-2\langle\mu_P,\mu_Q\rangle\\
&=\mathbb E k(X,X')+\mathbb E k(Z,Z')
-2\mathbb E k(X,Z).
\end{aligned}
$$

Each off-diagonal average is a U-statistic for its corresponding expectation, while the cross average is an ordinary unbiased sample average. Linearity of expectation proves unbiasedness. The estimator can be negative because unbiasedness is obtained by removing diagonal terms; population squared MMD remains nonnegative. \(\square\)
::::

**Finite-sample guarantee.** Under the stronger assumption \(0\le k(x,z)\le K\), Theorem 7 of [@gretton2012] bounds the deviation of the biased empirical MMD:

$$
\Pr\!\left[
\left|\widehat{\operatorname{MMD}}_b-\operatorname{MMD}_k(P,Q)\right|
\gt2\!\left(\sqrt{\frac K m}+\sqrt{\frac K n}\right)+\varepsilon
\right]
\le
2\exp\!\left\{-\frac{\varepsilon^2mn}{2K(m+n)}\right\}.
$$

The proof combines a bounded-difference inequality with a bound on the expected empirical process. The result supplies a conservative distribution-free threshold. The paper's sharper asymptotic null analysis concerns a degenerate U-statistic: when \(P=Q\), the scaled statistic converges to an infinite weighted sum of centered chi-squared variables determined by the centered-kernel spectrum. A Gaussian approximation to the ordinary central limit theorem is therefore wrong under the null.

**Algorithmic object.** Computing the quadratic statistic costs \(O((m+n)^2)\) kernel evaluations and \(O(m+n)\) additional storage if accumulated in blocks. Calibration by permutation is valid when the pooled observations are exchangeable under \(H_0\) and every data-dependent kernel choice is repeated inside each permutation or fixed before seeing the labels identifying source and target.

**Comparison.** MMD detection asks whether two distributions differ. Kernel mean matching, developed next, chooses weights to reduce a particular empirical MMD. The second operation does not inherit the first operation's test interpretation, and a rejected MMD test does not prove that weighting is valid.

**Failure boundary.** A non-characteristic kernel may miss real changes. A linear kernel sees only a mean shift. A degree-two polynomial kernel sees only moments represented by that feature space. Dependence invalidates the IID permutation argument unless blocks or another valid resampling design are used. Kernel or bandwidth selection after inspecting target labels inflates Type I error unless selection is included in calibration.

<figure class="viz" data-figure="shift-detection-delay" data-alt="Sequential squared MMD traces begin near zero and rise after a distribution mean shift. A companion curve shows detection delay decreasing as the shift magnitude grows."><figcaption>A detector has a latency curve, not merely power at one alternative. With the window and threshold fixed, large shifts cross quickly while subtle shifts accumulate evidence slowly. Deployment requirements should therefore specify the smallest consequential shift and the latest acceptable alarm.</figcaption></figure>

## Kernel mean matching as an inverse problem {#kernel-mean-matching}

The density ratio \(w=dQ_X/dP_X\) involves two unknown distributions. Kernel mean matching (KMM) avoids separate density estimation by matching target and weighted-source embeddings. Given source inputs \(x_1,\ldots,x_n\) and target inputs \(z_1,\ldots,z_m\), solve

$$
\min_{\beta\in\mathbb R^n}
\left\lVert
\frac1n\sum_{i=1}^n\beta_i k(x_i,\cdot)
-\frac1m\sum_{j=1}^m k(z_j,\cdot)
\right\rVert_{\mathcal H}^{2}
$$

subject, for example, to \(0\le\beta_i\le B\) and
\(\left|n^{-1}\sum_i\beta_i-1\right|\le\varepsilon\).

Let \(K_{SS}\in\mathbb R^{n\times n}\) and
\(K_{ST}\in\mathbb R^{n\times m}\) be source-source and source-target Gram matrices. Expanding the norm gives

$$
\frac{1}{n^2}\beta^\top K_{SS}\beta
-\frac{2}{nm}\beta^\top K_{ST}\mathbf 1_m
+\frac{1}{m^2}\mathbf 1_m^\top K_{TT}\mathbf 1_m.
$$

The last term is constant in \(\beta\). Thus KMM is a convex quadratic program because \(K_{SS}\succeq0\). If \(K_{SS}\) is singular, the embedding may identify many weight vectors equally well; box and normalization constraints do not guarantee uniqueness.

:::: {.proposition #prop-kmm-moment-balance}
[Proposition (what KMM balances)]{.box-title}

If the empirical embedding discrepancy is at most \(\varepsilon\), then every \(h\in\mathcal H\) satisfies

$$
\left|
\frac1n\sum_i\beta_i h(x_i)-\frac1m\sum_jh(z_j)
\right|
\le\varepsilon\lVert h\rVert_{\mathcal H}.
$$

**Assumptions.** The empirical embeddings exist, the discrepancy is at most \(\varepsilon\), and \(h\in\mathcal H\).

**Proof status.** Complete proof below.

**Proof.** Write the empirical embedding difference as \(\Delta\). By the reproducing property, the expression inside the absolute value is \(\langle h,\Delta\rangle_{\mathcal H}\). Cauchy-Schwarz gives
\(|\langle h,\Delta\rangle|\le\lVert h\rVert\lVert\Delta\rVert\le
\varepsilon\lVert h\rVert\). \(\square\)
::::

This is an exact function-class statement. It does not control a loss outside \(\mathcal H\), the conditional law \(Y\mid X\), or a hidden variable not represented in \(X\). Characteristicness identifies a population distribution from all RKHS moments, but a small finite-sample discrepancy is not equality of distributions.

Weight variability determines the statistical price of transfer. For normalized nonnegative weights, define

$$
n_{\mathrm{eff}}
=\frac{(\sum_i\beta_i)^2}{\sum_i\beta_i^2}.
$$

Uniform weights give \(n_{\mathrm{eff}}=n\). Concentration on one observation gives \(n_{\mathrm{eff}}\) near one. Weight clipping increases effective sample size but changes the estimand, creating a bias-variance tradeoff that must be reported rather than hidden.

## Paper module: rates under covariate shift {#covariate-rates}

Feng et al. study regularized empirical risk minimization in an RKHS for a broad family of losses under covariate shift [@feng2023covshift]. Their analysis explains why the phrase "importance weighting corrects shift" is incomplete: the answer depends on density-ratio tails, loss curvature, and the kernel spectrum.

**Exact setting.** Source observations \((X_i,Y_i)_{i=1}^n\) are IID from \(P^S\), while performance is measured under \(P^T\). The conditional law \(P_{Y\mid X}\) is common to source and target, \(P_X^T\ll P_X^S\), and
\(\phi=dP_X^T/dP_X^S\). The target minimizer

$$
f^\star\in\arg\min_{f\in\mathcal H_K}
\mathbb E_T L\{Y,f(X)\}
$$

is normalized in the paper to \(\lVert f^\star\rVert_K=1\). The kernel is symmetric, continuous, positive semidefinite, bounded on the diagonal, and has a Mercer expansion in \(L^2(P_X^T)\) with uniformly bounded eigenfunctions. The expected loss has local quadratic curvature around \(f^\star\) in both source and target \(L^2\) norms:

$$
\mathbb E_I L(Y,f(X))-\mathbb E_I L(Y,f^\star(X))
\ge c_0\lVert f-f^\star\rVert_I^2,
\quad I\in\{S,T\},
$$

whenever \(\lVert f-f^\star\rVert_I\) is sufficiently small. Rates for excess risk also use a matching local upper-curvature condition.

Let \((\mu_j)\) be the target-measure Mercer eigenvalues and define

$$
\mathcal R_n(\delta)
=\left\{\frac1n\sum_{j\ge1}\min(\delta^2,\mu_j)\right\}^{1/2}.
$$

This localized complexity is the bridge from kernel spectrum to sample rate.

**Two ratio regimes.**

1. Uniform overlap: \(\sup_x\phi(x)\le\alpha\).
2. Second-moment overlap: \(\mathbb E_S\phi(X)^2\le\beta^2\).

The first implies \(\lVert g\rVert_T^2\le\alpha\lVert g\rVert_S^2\). The second does not, because Cauchy-Schwarz introduces a fourth moment of \(g\). That single missing inequality explains why an unweighted estimator can be rate-optimal in the first regime and suboptimal in the second.

**Estimators.** The unweighted estimator is

$$
\widehat f
\in\arg\min_{f\in\mathcal H_K}
\left\{\frac1n\sum_{i=1}^n L(Y_i,f(X_i))
+\lambda\lVert f\rVert_K^2\right\}.
$$

For the second-moment regime, the paper uses truncated importance weights
\(\phi_n(x)=\min\{\phi(x),\gamma_n\}\):

$$
\widehat f_\phi
\in\arg\min_{f\in\mathcal H_K}
\left\{\frac1n\sum_{i=1}^n
\phi_n(X_i)L(Y_i,f(X_i))
+\lambda\lVert f\rVert_K^2\right\}.
$$

**Primary result.** Under the bounded-kernel, bounded-eigenfunction, and local-curvature assumptions, the uniformly bounded-ratio theorem controls target \(L^2\) error by a constant multiple of
\(\alpha(\delta_n^2+\lambda)\), where \(\delta_n\) solves a localized fixed-point inequality involving
\(\sqrt{\log n}\mathcal R_n(\sqrt{\alpha}\delta)\). Under only
\(\mathbb E_S\phi^2\le\beta^2\), the paper shows that the unweighted rate degrades. Truncating at
\(\gamma_n=\sqrt{n\beta^2}\) restores the sharp form

$$
\lVert\widehat f_\phi-f^\star\rVert_T^2
\lesssim \delta_n^2+\lambda
$$

with high probability, where now
\(\sqrt{\beta^2\log n}\mathcal R_n(\delta)\) enters the fixed point. For polynomial eigenvalue decay
\(\mu_j\lesssim j^{-2r}\), \(r\gt1/2\), the bounded-ratio result yields, up to constants and logarithms,

$$
\lVert\widehat f-f^\star\rVert_T^2
\lesssim
\left(\frac{\alpha^2\log n}{n}\right)^{\frac{2r}{2r+1}}.
$$

**Why truncation works.** The decomposition contains a stochastic term and a clipping bias. Bounding \(\phi_n\) permits concentration, while

$$
\mathbb E_S\{(\phi-\phi_n)|g|\}
\le
\frac{\mathbb E_S(\phi^2)}{\gamma_n}\,\lVert g\rVert_\infty
\le\frac{\beta^2}{\gamma_n}\lVert g\rVert_\infty
$$

uses \((\phi-\gamma_n)_+\le\phi^2/\gamma_n\). With
\(\gamma_n\asymp\sqrt{n}\beta\), concentration and clipping bias are balanced. This derivation explains the threshold's order, though the paper's theorem requires a longer localized empirical-process argument for the stated rate.

**Failure boundary and comparison.** These are oracle-weight results: replacing \(\phi\) by an estimated ratio adds another error term. Local curvature can fail for poorly identified targets or badly scaled losses. Uniformly bounded eigenfunctions are a real restriction, not decoration. KMM controls an empirical RKHS moment discrepancy; it does not automatically satisfy the oracle ratio assumptions of the rate theorem. Conversely, exact ratios can have disastrous variance when overlap is weak. The correct comparison reports target error, ratio error, clipping bias, and effective sample size.

## Robust losses and distributional neighborhoods {#robustness}

Robustness must name its adversary. A Huber loss limits the influence of extreme residuals. A contamination model replaces a fraction of observations by arbitrary points. Adversarial robustness perturbs individual inputs. Distributionally robust optimization protects against a set of probability measures. None implies the others.

For a function class \(\mathcal F\), the integral probability metric

$$
d_{\mathcal F}(Q,\widehat P)
=\sup_{h\in\mathcal F}
\left|\mathbb E_Qh-\mathbb E_{\widehat P}h\right|
$$

defines an ambiguity set \(\mathcal Q_\rho=\{Q:d_{\mathcal F}(Q,\widehat P)\le\rho\}\). If
\(\mathcal F\) is the RKHS unit ball, \(d_{\mathcal F}\) is MMD. For any loss function
\(\ell_f\in\mathcal H\),

$$
\sup_{Q\in\mathcal Q_\rho}\mathbb E_Q\ell_f
\le
\mathbb E_{\widehat P}\ell_f+\rho\lVert\ell_f\rVert_{\mathcal H}.
$$

The inequality is immediate from the definition after scaling
\(\ell_f/\lVert\ell_f\rVert_{\mathcal H}\). It is useful only when the loss belongs to the RKHS and has a manageable norm. Empirical estimation of an IPM has its own sampling error and function-class dependence [@sriperumbudur2012].

An MMD ball also encodes a kernel-dependent notion of nearness. A perturbation invisible to the feature map can lie at zero or small MMD while changing operational risk. Radius selection therefore needs a scientific perturbation model, not only cross-validation on source data.

## Conformal prediction from exchangeable ranks {#conformal-prediction}

Importance weighting attempts to transfer a risk. Conformal prediction asks a different question: can a prediction set attain a declared marginal coverage without correctly specifying the conditional distribution?

Split the observations into proper training data and calibration data
\((X_i,Y_i)_{i=1}^m\). Fit a predictor using only the proper training data. Let
\(S_i=s(X_i,Y_i)\) be measurable nonconformity scores, where the fitted scoring rule is fixed conditional on the proper training data. For a new input \(x\), define

$$
\mathcal C_\alpha(x)
=\{y:s(x,y)\le S_{(k)}\},
\qquad
k=\left\lceil(m+1)(1-\alpha)\right\rceil,
$$

where \(S_{(k)}\) is the \(k\)-th calibration order statistic and is interpreted as \(+\infty\) if \(k=m+1\).

:::: {.theorem #thm-split-conformal-coverage}
[Theorem (finite-sample marginal coverage)]{.box-title}

Suppose that, conditional on the proper training data, the calibration examples and the test example are exchangeable. Suppose also that the score is measurable and calibration labels were not used to fit or tune it. Then

$$
\Pr\{Y_{m+1}\in\mathcal C_\alpha(X_{m+1})\}
\ge 1-\alpha.
$$

With randomized tie handling, the rank argument can give exact rather than conservative coverage.

**Assumptions.** Conditional exchangeability of calibration and test examples, score measurability, and no calibration-label reuse in score fitting or tuning.

**Proof status.** Complete proof below.

**Proof.** Conditional on the proper training data, the scores
\(S_1,\ldots,S_m,S_{m+1}\) are exchangeable. If ties are broken by independent continuous random variables, the rank \(R\) of \(S_{m+1}\) among the \(m+1\) scores is uniform on
\(\{1,\ldots,m+1\}\). The test response is excluded only if
\(S_{m+1}\gt S_{(k)}\), which implies \(R\gt k\). Therefore

$$
\Pr\{Y_{m+1}\notin\mathcal C_\alpha(X_{m+1})\mid\text{training}\}
\le\frac{m+1-k}{m+1}\le\alpha.
$$

Conservative treatment of ties can only reduce the exclusion probability. Averaging over the proper training data proves the marginal statement. \(\square\)
::::

The guarantee is finite-sample and distribution-free within exchangeability. It is also marginal over the random test input. It does not state
\(\Pr\{Y\in\mathcal C_\alpha(X)\mid X=x\}\ge1-\alpha\) for every \(x\), nor does it promise narrow sets. Validity and efficiency are different properties.

For regression, \(s(x,y)=|y-\widehat f(x)|\) yields
\([\widehat f(x)-S_{(k)},\widehat f(x)+S_{(k)}]\). Locally scaled residuals can adapt width, but the scale model must be trained without calibration-label leakage. For classification, a score based on fitted class probabilities yields a set of labels.

<figure class="viz" data-widget="conformal-coverage">

<figcaption>Split conformal turns a sorted calibration-score list into a marginal coverage statement. The finite-sample rank changes discretely with the calibration size and target level. Local or subgroup behavior remains an empirical diagnostic unless a stronger theorem is invoked.</figcaption>
</figure>

## Paper module: conformalized ridge regression {#crr-paper-module}

Burnaev and Vovk investigate whether conformal validity destroys the efficiency of a well-specified ridge predictor [@burnaev2014conformal]. This is a deeper question than coverage. A method can cover at the requested rate by returning an uninformative set.

**Exact setting.** Their primal analysis uses objects \(x_i\in\mathbb R^p\) and labels

$$
y_i=w^\top x_i+\xi_i.
$$

For unconditional conformal validity, the observations need only be IID. For the paper's asymptotic efficiency theorem, the assumptions are stronger:

1. \(x_1,x_2,\ldots\) are IID.
2. \(\Sigma=\mathbb E(x_1x_1^\top)\) exists and is nonsingular.
3. \(w\) is independent of the object sequence.
4. \(\xi_i\) are independent \(N(0,\sigma^2)\) variables, independent of the objects and \(w\).

Ridge residuals are computed after appending a candidate test label \(y\). If \(H_y\) is the ridge hat matrix for the augmented design, then

$$
r(y)=(I-H_y)
\begin{bmatrix}
Y\\y
\end{bmatrix}
=a+by
$$

for vectors \(a,b\) independent of the candidate value. Every comparison between the candidate residual and a calibration residual changes only when two affine functions cross. Sorting these breakpoints produces the conformal set in \(O(n\log n)\) after the linear-algebra quantities are available.

**Primary results.** Under IID observations, the conformalized ridge set has coverage at least \(1-\alpha\), whether or not the linear Gaussian model is true. Under the four stronger assumptions above, the paper proves that the conformal set is eventually an interval almost surely and that its two endpoints differ from the Bayesian ridge interval endpoints by \(O_p(n^{-1/2})\), with explicit asymptotic Gaussian limits.

**Proof architecture.** The validity result is a rank argument over the augmented residual scores. The efficiency result is different. It reduces the conformal endpoints to residual order statistics, controls ridge leverages, applies a Bahadur representation to the relevant residual quantile, and then compares that quantile with the Gaussian Bayesian endpoint. The distinction matters: exchangeability proves coverage, while Gaussian linear structure proves the endpoint comparison.

**Comparison with split conformal.** Split conformal pays for a held-out calibration set and is computationally simple. Full conformalized ridge regression refits conceptually over candidate labels but exploits affine ridge algebra to avoid brute-force refitting. Its validity uses more data symmetrically. The asymptotic efficiency theorem is not a general theorem about kernel ridge regression, nonlinear kernels, heteroscedastic noise, or distribution shift.

**Failure boundary.** Under exchangeability but model misspecification, coverage survives while the endpoint-efficiency theorem may fail. Under covariate or concept shift, even marginal coverage may fail. A singular or nearly singular design is outside the nonsingular second-moment assumption and can make the comparison unstable. At significance \(\alpha\), a conformal method cannot give an informative finite-sample set until enough ranks exist; very small calibration samples therefore impose coarse or infinite thresholds.

## Conformal prediction under shift {#conformal-under-shift}

Ordinary conformal validity can fail when calibration and deployment examples are not exchangeable. Under covariate shift, suppose the calibration pairs follow \(P\), the test pair follows \(Q\), the conditional law is invariant, and the density ratio \(w=dQ_X/dP_X\) is known. A weighted rank construction assigns calibration mass proportional to \(w(X_i)\) and test mass proportional to \(w(x)\). The target point is no longer uniformly ranked, but it is ranked according to these likelihood weights.

This correction has the same boundaries as importance weighting:

- \(Q_X\ll P_X\) is required.
- Conditional invariance is required because the ratio uses inputs only.
- Estimated weights add error unless the procedure accounts for estimation.
- Large weights make weighted quantiles unstable.
- Arbitrary concept drift admits no distribution-free future-coverage theorem from old labels alone.

Temporal data require another design. Blocking can make calibration units approximately exchangeable only under a dependence model and a block length justified by mixing or domain knowledge. Repeated online recalibration creates adaptive dependence that must be addressed explicitly.

<figure class="viz" data-figure="conformal-coverage-width-shift" data-alt="Coverage and interval width are plotted against the ratio of deployment noise to calibration noise. Frozen conformal intervals lose coverage as noise grows, while shift-aware recalibration maintains ninety percent coverage by widening the interval."><figcaption>Exchangeability failure appears as a coverage-width tradeoff. Keeping the old residual quantile keeps the old width but loses the certificate; restoring \(90\%\) coverage under a scale change requires wider sets. A reliability report should show both quantities.</figcaption></figure>

## A worked deployment calculation {#reliability-worked-example}

Consider four source observations with binary outcomes

$$
y=(0,0,1,1)
$$

and KMM weights

$$
\beta=(0.2,0.6,1.2,2.0).
$$

The weights are nonnegative and sum to four, so their average is one. The unweighted outcome mean is \(0.5\), while the weighted mean is

$$
\frac14\sum_i\beta_i y_i
=\frac{1.2+2.0}{4}=0.8.
$$

The correction is large because the target-like source observations are concentrated among the positive outcomes. Its effective sample size is

$$
n_{\mathrm{eff}}
=\frac{4^2}{0.2^2+0.6^2+1.2^2+2.0^2}
=\frac{16}{5.84}\approx2.74.
$$

Thus four weighted observations carry variance comparable to fewer than three uniform observations. If the last weight is clipped from \(2.0\) to \(1.5\) and the weights are renormalized, effective sample size rises, but the weighted mean moves away from \(0.8\). That movement is clipping bias, not numerical noise.

Now suppose a separate calibration set has nine absolute residuals

$$
0.1,\ 0.2,\ 0.4,\ 0.7,\ 1.1,\ 1.6,\ 2.3,\ 4.0,\ 7.0.
$$

For target coverage \(1-\alpha=0.75\),

$$
k=\lceil(9+1)0.75\rceil=8,
$$

so the split-conformal radius is \(4.0\). Using the seventh order statistic \(2.3\) because it is the ordinary empirical \(75\%\) quantile omits the finite-sample correction and loses the stated guarantee. The wide interval is not a defect in the proof. It is evidence that nine calibration residuals support only coarse tail resolution and that this fitted predictor sometimes errs substantially.

Finally, suppose the deployment system adds a new input region \(A\) with
\(P_X(A)=0\) and \(Q_X(A)=0.05\). Neither the weights above nor the residual quantile provides a certificate on \(A\). The appropriate output is an overlap alarm and, depending on the application, abstention or targeted label collection.

## An auditable reliability pipeline {#reliable-pipeline}

:::: {.algorithm #algo-reliable-kernel-pipeline}
[Algorithm (shift-aware kernel prediction)]{.box-title}

**Input.** Source training data, untouched calibration data, target covariates, candidate kernels, a declared shift model, and operational loss.

**Output.** Predictions, prediction sets, and an assumption ledger.

1. Define the deployment unit and split by time, site, subject, or device so that validation matches deployment dependence.
2. State which conditional distribution is assumed invariant and which support relation is required.
3. Test and visualize input shift using preregistered kernel scales; calibrate any adaptive scale selection.
4. Diagnose overlap with ratio tails, nearest-neighbor or leverage diagnostics, and target mass outside the source support.
5. Estimate or balance weights only when covariate shift is defensible. Report clipping, embedding residual, ratio diagnostics, and \(n_{\mathrm{eff}}\).
6. Fit weighted and unweighted kernel baselines using stable solves and untouched target labels for final evaluation.
7. Calibrate conformal scores without reusing calibration labels for model or score selection.
8. Report target risk, marginal and groupwise coverage, set width, abstention, conditioning, and sensitivity to kernel and clipping choices.
9. Trigger recollection, recalibration, fallback, or retraining when declared support, shift, or coverage thresholds fail.

For KMM, stop only when both the quadratic-program residual and embedding discrepancy are stable. For conformal prediction, record the exact calibration sample and rank convention used for every deployed model version.
::::

## Common mistakes and practical implications {#reliability-practice}

- Detecting \(P_X\ne Q_X\) does not establish \(P_{Y\mid X}=Q_{Y\mid X}\).
- A negative unbiased MMD estimate is possible and is not a negative population distance.
- Choosing a kernel after observing source-target labels requires selection-aware test calibration.
- Small empirical MMD does not prove support overlap.
- KMM balances an RKHS, not every bounded loss or hidden confounder.
- Exact importance weighting can have intolerable variance.
- Weight clipping changes the target functional.
- Conformal marginal coverage is not pointwise or subgroup coverage.
- Coverage does not imply useful set width.
- Calibration labels stop being calibration labels when repeatedly used for tuning.
- GP or kernel uncertainty is model-based until calibrated under a relevant sampling design.
- Robustness against one ambiguity set says nothing about a different perturbation mechanism.

The practical objective is not a universal certificate. It is a chain of claims whose assumptions can be checked separately and whose failures lead to a declared action.

## Summary and further reading {#reliability-summary}

Distribution shift first requires identification of the quantity that changed. MMD turns a difference between distributions into an RKHS norm and supports a calibrated two-sample test under an explicit sampling design [@gretton2012]. KMM minimizes an empirical version of that norm, but guarantees balance only over the chosen RKHS. Covariate-shift rates depend jointly on overlap, loss curvature, and kernel spectral complexity; truncated ratios trade clipping bias for concentration [@feng2023covshift]. MMD ambiguity sets protect only losses controlled by the corresponding function class [@sriperumbudur2012]. Split conformal prediction proves finite-sample marginal coverage from exchangeable ranks, while conformalized ridge regression shows that validity can coexist with asymptotic efficiency under a much stronger Gaussian linear model [@burnaev2014conformal]. None of these results repairs target-only support or arbitrary concept drift.

## Exercises {#exercises}

1. [warm-up]{.ex-tag} Give one data-generating example for covariate, label, concept, and support shift. For each example, state which observable marginal changes and which conditional invariance would have to be defended.
2. [proof]{.ex-tag} Prove the covariate-shift risk identity using conditional expectation and the Radon-Nikodym derivative. Then construct a bounded loss showing why the identity cannot hold for every target risk when \(Q_X\not\ll P_X\).
3. [computation]{.ex-tag} Expand the empirical KMM objective into its quadratic-program matrix and vector terms. For weights \((0.2,0.6,1.2,2.0)\), compute the effective sample size and explain what it says about variance.
4. [proof]{.ex-tag} Derive the population squared-MMD kernel formula and prove that the off-diagonal empirical estimator is unbiased. Explain why the unbiased estimate may be negative.
5. [proof]{.ex-tag} Prove split-conformal marginal coverage, including the corrected rank \(k=\lceil(m+1)(1-\alpha)\rceil\), conservative ties, and the case \(k=m+1\).
6. [computation]{.ex-tag} With calibration scores \(0.1,0.2,0.4,0.7,1.1,1.6,2.3,4.0,7.0\), compute split-conformal radii for target coverages \(0.50\), \(0.75\), \(0.80\), and \(0.90\). Identify every level for which the set is forced to be infinite.
7. [synthesis]{.ex-tag} Compare the assumptions and conclusions of the MMD test, KMM moment balance, the Feng et al. covariate-shift rate, split conformal, and the Burnaev-Vovk efficiency theorem. Give one implication that is invalid in each direction between neighboring methods.
8. [challenge]{.ex-tag} Design a deployment audit for a kernel classifier moving between hospitals. Specify the sampling unit, shift tests, kernel-selection protocol, overlap criteria, weight estimation and clipping, conformal score, subgroup diagnostics, and actions triggered by support or coverage failure.
