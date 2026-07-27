---
id: ch-lower
slug: limits-and-lower-bounds-for-kernel-learning
title: Limits and Lower Bounds for Kernel Learning
part: 'IV · Generalization, Approximation, and Limits'
order: 20
tier: core
prerequisites:
  - learning-theory
  - mercer-and-rates
  - random-features-sketches-and-randomized-kernel-linear-algebra
objectives:
  - >-
    Explain why an upper bound is informative only when compared with a lower
    bound in the same statistical or computational currency.
  - >-
    Derive two-point testing lower bounds with total variation, Hellinger
    distance, and Kullback-Leibler divergence.
  - >-
    Use Fano and Assouad reductions to turn packings into minimax risk lower
    bounds.
  - >-
    Match kernel-ridge upper rates to source and capacity classes while stating
    the sampling, noise, and operator assumptions exactly.
  - >-
    Separate lower bounds for random features, Nyström methods, sketches, linear
    solves, and population prediction risk.
  - >-
    Diagnose adaptation barriers and no-free-lunch claims by constructing
    explicit failure witnesses.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-lower.yml
verification_date: null
bibliography:
  - tsybakov2009
  - caponnetto2007
  - dicker2015
  - bach2017quadrature
  - avron2017rff
  - yang2017sketch
  - nesterov2004
---
# Limits and Lower Bounds for Kernel Learning

<p class="lead">An upper bound says that one route works. It does not say that the route is optimal, that its assumptions are necessary, or that another algorithm cannot do better. Kernel learning makes this ambiguity especially dangerous because four different bottlenecks can wear the same exponent: the data may contain too little information, the chosen feature map may have too little rank, the sketch may erase a statistically important direction, or the solver may have taken too few oracle calls. This chapter develops the tools that separate those obstructions. We begin by reducing estimation to hypothesis testing, grow two alternatives into Fano and Assouad packings, and then recover minimax lower bounds for source-and-capacity classes. We next examine random features, Nyström approximations, sketches, and first-order kernel solves without exchanging one error currency for another. The final sections explain adaptation limits and build small failure witnesses that can be checked by hand.</p>

## What an upper bound leaves unanswered {#lower-upper-incomplete}

Let \(\mathcal P\) be a class of data-generating distributions, let \(a(P)\) be the object to estimate, and let \(L(\widehat a,a(P))\geq0\) be a loss. The minimax risk from \(n\) observations is

$$
\mathfrak R_n(\mathcal P,L)
=
\inf_{\widehat a}\sup_{P\in\mathcal P}
\mathbb E_{P^n}L\!\left(\widehat a,a(P)\right),
$$

where the infimum ranges over all measurable estimators, including randomized ones. An upper bound analyzes one estimator. A lower bound controls this infimum and therefore every estimator in the declared information model.

::: {.definition #def-lower-currencies}
[Definition (four lower-bound currencies)]{.box-title}

In this chapter a lower bound is always labeled by its target:

1. **statistical:** prediction, estimation, or excess risk under a distribution class;
2. **representation:** rank, number of features, landmarks, or sketch rows needed to preserve a declared matrix or function-space quantity;
3. **optimization:** objective gap, energy-norm error, or residual after a number of oracle calls;
4. **systems:** arithmetic, communication, or memory in a specified machine model.

A theorem in one currency does not automatically imply a theorem in another.
:::

The quantifiers matter. A minimax result has the form “for every estimator there exists an admissible distribution.” It does not say that every distribution is hard. A representation lower bound for i.i.d. Fourier features does not rule out adaptive features. An oracle lower bound for matrix-vector products does not rule out a direct factorization that uses richer access.

## Two-point testing and Le Cam's method {#lower-le-cam}

The smallest hard family has two members. Suppose \(P_0\) and \(P_1\) produce parameters separated in loss but \(P_0^n\) and \(P_1^n\) are difficult to distinguish. Any accurate estimator would induce an accurate test, so testing hardness becomes estimation hardness.

::: {.theorem #thm-lower-le-cam}
[Theorem (two-point lower bound; proved here)]{.box-title}

Let \(P_0,P_1\) be probability measures on the observation space and let \(a_0,a_1\) be their parameters. Assume the loss \(L\) is symmetric and satisfies

$$
L(u,a_0)+L(u,a_1)\geq \Delta
$$

for every admissible estimate \(u\), with \(\Delta\gt0\). Then every estimator \(\widehat a\) based on \(n\) independent observations satisfies

$$
\max_{j\in\{0,1\}}
\mathbb E_{P_j^n}L(\widehat a,a_j)
\geq
\frac{\Delta}{4}
\left(1-\operatorname{TV}(P_0^n,P_1^n)\right).
$$

No domination assumption is needed for this total-variation form. If \(P_0\ll P_1\), Pinsker's inequality further gives

$$
\operatorname{TV}(P_0^n,P_1^n)
\leq
\sqrt{\frac{n}{2}\operatorname{KL}(P_0\Vert P_1)}.
$$

**Assumptions.** The observations have product laws \(P_0^n\) and \(P_1^n\);
the loss is measurable, symmetric in the declared parameter arguments, and
satisfies the displayed separation inequality for every admissible estimate.
The Pinsker corollary additionally requires \(P_0\ll P_1\) and finite
\(\operatorname{KL}(P_0\Vert P_1)\).

**Proof status.** The testing reduction is proved immediately below. The
testing-error identity and Pinsker inequality are standard and may also be
found in [@tsybakov2009, Sections 2.2--2.3].
:::

::: {.proof}
Define the test \(\phi=1\) when \(L(\widehat a,a_1)\leq L(\widehat a,a_0)\), and \(\phi=0\) otherwise. On \(\{\phi=1\}\), the separation assumption and the ordering of the two losses imply \(L(\widehat a,a_0)\geq\Delta/2\). The analogous statement holds under \(P_1\) on \(\{\phi=0\}\). Therefore

$$
\mathbb E_{P_0^n}L(\widehat a,a_0)
+\mathbb E_{P_1^n}L(\widehat a,a_1)
\geq
\frac{\Delta}{2}
\left(P_0^n\{\phi=1\}+P_1^n\{\phi=0\}\right).
$$

The minimum sum of the two testing errors is
\(1-\operatorname{TV}(P_0^n,P_1^n)\). At least one of the two risks is half their sum, proving the first display. Tensorization of KL and Pinsker's inequality prove the second. \(\square\)
:::

The theorem is deliberately modular. Geometry supplies \(\Delta\); probability supplies an upper bound on divergence. If either piece is weak, the lower bound is weak.

### A Gaussian regression reduction {#lower-gaussian-reduction}

Fix design points \(x_1,\ldots,x_n\), observe

$$
y_i=f(x_i)+\varepsilon_i,
\qquad
\varepsilon_i\stackrel{\mathrm{iid}}{\sim}N(0,\sigma^2),
$$

and define \(\lVert g\rVert_n^2=n^{-1}\sum_i g(x_i)^2\). For two regression functions,

$$
\operatorname{KL}(P_f\Vert P_g)
=\frac{n}{2\sigma^2}\lVert f-g\rVert_n^2.
$$

Choosing functions far enough apart to incur prediction loss but close enough that the KL divergence remains constant gives the familiar noise floor. This identity is exact for fixed-design Gaussian regression; it is not an assertion about random design, heavy-tailed noise, or classification.

## From two points to many: Fano and Assouad {#lower-fano-assouad}

Two points detect one difficult direction. Kernel classes can contain many nearly orthogonal difficult directions, and their number is controlled by the spectrum. Packing methods aggregate those directions.

::: {.theorem #thm-lower-fano}
[Theorem (metric Fano reduction; stated with source)]{.box-title}

Let \(P_1,\ldots,P_M\) be distributions with parameters \(a_1,\ldots,a_M\) satisfying
\(d(a_j,a_k)\geq2s\) for \(j\ne k\), where \(d\) is a metric. Assume \(M\geq2\) and

$$
\frac1M\sum_{j=1}^M\operatorname{KL}(P_j\Vert Q)
\leq \alpha\log M
$$

for some probability measure \(Q\) and \(0\lt\alpha\lt1\). Then every estimator obeys

$$
\sup_j P_j\!\left(d(\widehat a,a_j)\geq s\right)
\geq
1-\alpha-\frac{\log2}{\log M}.
$$

Consequently, for squared metric loss,

$$
\inf_{\widehat a}\sup_j\mathbb E_j d(\widehat a,a_j)^2
\geq
s^2\left(1-\alpha-\frac{\log2}{\log M}\right).
$$

**Assumptions.** The parameter set contains an \(M\)-point measurable packing
in the metric \(d\); \(M\geq2\); the average KL divergence to the declared
reference measure \(Q\) is finite and satisfies the displayed information
budget; the estimator is measurable.

**Proof status.** The decoding reduction is proved below; the information inequality is cited to [@tsybakov2009, Section 2.5].
:::

::: {.proof}
Decode \(\widehat J\) as a nearest packing point to \(\widehat a\). If
\(d(\widehat a,a_J)\lt s\), the \(2s\)-separation and triangle inequality make \(J\) the unique nearest point, so \(\widehat J=J\). Hence the estimation failure probability dominates the decoding error. Fano's information inequality gives the displayed lower bound on that error. Multiplying the event probability by \(s^2\) gives the squared-loss statement. \(\square\)
:::

Assouad's method is better when the hard family is a hypercube. Let
\(\omega\in\{-1,+1\}^m\), and make coordinate \(j\) control one eigen-direction. If neighboring vertices are hard to distinguish, every estimator makes errors across many coordinates.

::: {.theorem #thm-lower-assouad}
[Theorem (Assouad reduction; stated with proof sketch)]{.box-title}

Let \(\{P_\omega:\omega\in\{-1,+1\}^m\}\) be an experiment. Suppose there is a measurable decoder
\(\widehat\omega(u)\) from the estimate space to the hypercube such that

$$
L(u,a_\omega)
\geq
\delta^2 d_H(\widehat\omega(u),\omega)
$$

for every estimate \(u\) and vertex \(\omega\), where \(d_H\) is Hamming distance. For each \(j\), let
\(\overline P_{j,+}\) and \(\overline P_{j,-}\) be the uniform mixtures over vertices with
\(\omega_j=+1\) and \(-1\). Then

$$
\inf_{\widehat a}\sup_\omega
\mathbb E_\omega L(\widehat a,a_\omega)
\geq
\frac{m\delta^2}{2}
\min_j\left(1-\operatorname{TV}(\overline P_{j,+},\overline P_{j,-})\right).
$$

**Assumptions.** The hypercube experiments and their coordinate mixtures are
well-defined probability measures, the decoder is measurable, and the loss
dominates Hamming error with the displayed scale \(\delta^2\).

**Proof status.** The coordinatewise testing reduction is given here; the standard measurable nearest-vertex step is sourced to [@tsybakov2009, Section 2.7].
:::

::: {.proof}
Apply the declared decoder. The loss domination converts expected loss into \(\delta^2\) times expected Hamming error. Average uniformly over vertices and expand Hamming loss coordinate by coordinate. For coordinate \(j\), the induced sign decision tests the two mixtures \(\overline P_{j,+}\) and \(\overline P_{j,-}\), whose minimum sum of errors is \(1-\operatorname{TV}\). Summing the \(m\) coordinate bounds and using maximum at least average yields the result. \(\square\)
:::

Fano uses the logarithm of packing size. Assouad uses the number of independently flippable coordinates. Neither theorem manufactures a packing: the difficult work is choosing coefficients that satisfy the RKHS source constraint while keeping neighboring experiments statistically close.

## Minimax lower bounds for source and capacity classes {#lower-krr-minimax}

Let \(P_X\) be a probability measure, let \(k\) be measurable and bounded with
\(\sup_x k(x,x)\leq\kappa^2\), and let

$$
(Tf)(x)=\int k(x,x')f(x')\,dP_X(x')
$$

be the positive compact integral operator on \(L^2(P_X)\). Write its positive eigenvalues as
\(\mu_1\geq\mu_2\geq\cdots\). A source ball is

$$
\mathcal F(r,R)=
\{f_\rho=T^r g:\lVert g\rVert_{L^2(P_X)}\leq R\}.
$$

A capacity condition can be expressed by
\(\mathcal N(\lambda)=\operatorname{tr}(T(T+\lambda I)^{-1})\lesssim\lambda^{-p}\).
For a lower bound, an upper capacity condition is insufficient: the class could be finite rank. We need a matching supply of eigen-directions, for example

$$
c_-j^{-1/p}\leq\mu_j\leq c_+j^{-1/p},
\qquad 0\lt p\lt1.
$$

::: {.theorem #thm-lower-source-capacity}
[Theorem (source-capacity minimax rate; sourced, proof architecture reconstructed)]{.box-title}

Assume:

1. \(X_i\stackrel{\mathrm{iid}}{\sim}P_X\);
2. \(Y_i=f_\rho(X_i)+\varepsilon_i\), with conditionally centered noise satisfying a nondegenerate Gaussian or a two-sided Bernstein-type condition with variance scale \(\sigma^2\gt0\);
3. \(T\) is positive, compact, and has two-sided polynomial eigenvalue decay
   \(\mu_j\asymp j^{-1/p}\), \(0\lt p\lt1\);
4. \(f_\rho\in\mathcal F(r,R)\), \(r\gt0\);
5. the eigenfunctions satisfy the regularity conditions needed to realize the packing as admissible regression distributions.

Then, up to constants depending on the declared class parameters,

$$
\inf_{\widehat f}
\sup_{\rho}
\mathbb E_\rho
\lVert\widehat f-f_\rho\rVert_{L^2(P_X)}^2
\gtrsim
n^{-\,2r/(2r+p)}.
$$

Matching upper bounds hold for a regularizer whose qualification covers \(r\); ordinary KRR saturates when its qualification is exceeded. The exact endpoint and noise hypotheses depend on the source convention. This statement follows the integral-operator minimax framework of [@caponnetto2007] and the qualification comparison of [@dicker2015].

**Assumptions.** Items 1--5 above define the sampling model, noise law,
two-sided spectral supply, source ball, and eigenfunction regularity. The
claimed exponent is for expected squared \(L^2(P_X)\) error; it is not an
RKHS-norm, high-probability, or fixed-design result.

**Proof status.** The exponent derivation and packing geometry are reconstructed below. The full random-design measure-theoretic realization and constants are not reproved here.
:::

::: {.proof}
Choose a block of \(m\) eigenfunctions with eigenvalues of order
\(\mu_m\asymp m^{-1/p}\), and define hypercube alternatives

$$
f_\omega
=a\sum_{j=m+1}^{2m}\omega_j\psi_j.
$$

The source constraint is satisfied when
\(m a^2\mu_m^{-2r}\lesssim R^2\), or
\(a^2\lesssim R^2\mu_m^{2r}/m\). Neighboring vertices differ in one coefficient, so their squared \(L^2(P_X)\) distance is \(4a^2\). Under Gaussian noise, the one-sample KL is proportional to \(a^2/\sigma^2\), and the \(n\)-sample KL is proportional to \(na^2/\sigma^2\). Keep neighboring tests hard by taking \(na^2\lesssim\sigma^2\). Assouad then yields risk of order \(ma^2\).

Balance the source and testing restrictions:

$$
\frac{R^2\mu_m^{2r}}{m}
\asymp
\frac{\sigma^2}{n}.
$$

Since \(\mu_m\asymp m^{-1/p}\), this gives
\(m\asymp n^{p/(2r+p)}\). Therefore

$$
ma^2\asymp\frac{m}{n}
\asymp n^{-2r/(2r+p)}.
$$

The omitted technical step verifies that the perturbed regression laws remain in the declared distribution class and controls random-design divergences. That step is part of the sourced theorem, not proved by the exponent calculation alone. \(\square\)
:::

### What exactly is matched {#lower-krr-match}

The lower bound matches an upper bound only if the source exponent, capacity exponent, error norm, noise class, sampling model, and probability mode agree. Replacing two-sided eigenvalue decay by
\(\mathcal N(\lambda)\leq C\lambda^{-p}\) changes a minimax class into a one-sided upper-bound class. Replacing expected \(L^2(P_X)\) risk by high-probability RKHS-norm error changes the currency. Calling KRR minimax for every \(r\) ignores saturation.

## A finite-rank Gaussian example {#lower-finite-rank-example}

The testing reduction is visible without asymptotics.

::: {.example #example-lower-gaussian}
[Example (one active kernel eigen-direction)]{.box-title}

Let \(X\) be uniform on \(\{-1,+1\}\), let
\(k(x,x')=xx'\), and observe

$$
Y=\theta X+\varepsilon,
\qquad
\varepsilon\sim N(0,1).
$$

The kernel has rank one and feature \(\phi(x)=x\). Compare
\(\theta_0=-\delta\) and \(\theta_1=\delta\) from \(n=8\) observations, with
\(\delta=1/4\). Their regression functions have squared \(L^2(P_X)\) distance
\(4\delta^2=1/4\). The \(n\)-sample KL divergence is

$$
\operatorname{KL}(P_{-\delta}^n\Vert P_\delta^n)
=\frac n2(2\delta)^2
=2n\delta^2
=1.
$$

Pinsker gives \(\operatorname{TV}\leq1/\sqrt2\). For squared loss,
\(\lVert u-f_{-\delta}\rVert^2+\lVert u-f_\delta\rVert^2\geq2\delta^2=1/8\).
The two-point theorem therefore gives

$$
\inf_{\widehat f}\max_{\theta\in\{-1/4,1/4\}}
\mathbb E_\theta\lVert\widehat f-f_\theta\rVert_{L^2(P_X)}^2
\geq
\frac1{32}\left(1-\frac1{\sqrt2}\right)
\approx0.009153.
$$

The bound is conservative because Pinsker discards the exact Gaussian testing error. Its value is conceptual: even a rank-one RKHS has a nonzero finite-sample noise floor.
:::

## Random features, Nyström methods, and sketches {#lower-randomized}

Lower bounds become easier to misread when several bottlenecks are active at once. The plate below is a diagnostic map, not a universal theorem: its stylized error terms show how changing sample size and approximation rank can move the dominant obstruction among noise, representation, computation, and an irreducible floor. Any concrete lower bound must replace these model terms with the rates proved for its own source class, spectrum, oracle, and loss.

<figure class="viz" data-figure="kernel-minimax-phase-diagram" data-alt="A logarithmic phase diagram over sample size and approximation rank marks regions dominated by noise, insufficient rank, computational cost, and an irreducible floor."><figcaption>A lower bound has a currency and a regime. Increasing rank helps only in the rank-limited region; increasing data helps only while sampling noise dominates. The boundaries are generated from explicit stylized terms, so they organize the questions a theorem must answer without pretending that one phase diagram applies to every kernel problem.</figcaption></figure>

Approximation lower bounds answer different questions depending on what the algorithm sees and what it must preserve.

::: {.proposition #prop-lower-rank}
[Proposition (rank barrier for regularized spectral approximation; proved here)]{.box-title}

Let \(K\succeq0\) have eigenvalues
\(\lambda_1\geq\cdots\geq\lambda_n\), let \(\gamma\gt0\), and let
\(\widetilde K\succeq0\) have rank at most \(m\lt n\). If

$$
\left\lVert
(K+\gamma I)^{-1/2}(K-\widetilde K)(K+\gamma I)^{-1/2}
\right\rVert_2
\leq\varepsilon,
$$

then necessarily

$$
\frac{\lambda_{m+1}}{\lambda_{m+1}+\gamma}\leq\varepsilon.
$$

No sampling assumptions are required. The conclusion concerns regularized spectral error, not prediction risk.

**Assumptions.** \(K\) and \(\widetilde K\) are finite real symmetric PSD
matrices on the same \(n\)-dimensional space, \(\gamma\gt0\), and
\(\operatorname{rank}(\widetilde K)\leq m\lt n\). The norm is the spectral norm.

**Proof status.** Proved immediately below by a subspace-intersection argument
and the Rayleigh quotient.
:::

::: {.proof}
The null space of \(\widetilde K\) has dimension at least \(n-m\). The span of the top
\(m+1\) eigenvectors of \(K\) has dimension \(m+1\), so it intersects that null space in a nonzero vector \(v\). Normalize \(v\) and note
\(v^\top Kv\geq\lambda_{m+1}\). On this vector,

$$
\frac{v^\top(K-\widetilde K)v}{v^\top(K+\gamma I)v}
=
\frac{v^\top Kv}{v^\top Kv+\gamma}
\geq
\frac{\lambda_{m+1}}{\lambda_{m+1}+\gamma}.
$$

The left side is bounded by the declared operator norm. \(\square\)
:::

The proposition applies to any rank-\(m\) approximation, including feature matrices and Nyström matrices, but only for this spectral target. Sharper restrictions depend on the access model:

- ordinary i.i.d. Fourier sampling can require more features than a rank argument predicts for Gaussian kernels [@avron2017rff, Theorem 8];
- any selected feature or quadrature family encounters eigenvalue-controlled approximation barriers in the corresponding integral-operator model [@bach2017quadrature, Propositions 2--3];
- an oblivious sketch below the critical statistical dimension can fail to retain the fixed-design minimax KRR rate [@yang2017sketch, Theorem 2].

| Lower bound | Budget | Access model | Protected quantity | Does not imply |
|---|---:|---|---|---|
| rank barrier above | rank \(m\) | arbitrary rank-\(m\) PSD matrix | regularized spectral error | population excess risk |
| ordinary RFF barrier | features | i.i.d. spectral samples in stated Gaussian-kernel regime | regularized spectral approximation | impossibility for adaptive features |
| eigenvalue feature barrier | atoms/features | selected atoms in integral-operator model | approximation or quadrature objective | Nyström risk lower bound without reduction |
| statistical sketch barrier | sketch rows | stated oblivious sketch family | fixed-design KRR prediction rate | matrix approximation for every input |

### Failure witness: Frobenius accuracy can miss the used direction {#lower-frobenius-witness}

Let \(K=\operatorname{diag}(100,1)\), \(\widetilde K=\operatorname{diag}(100,0)\), and
\(\gamma=0.01\). The relative Frobenius error is about \(0.01\), yet the regularized relative error on the second coordinate is

$$
\frac{1}{1+0.01}\approx0.9901.
$$

With response \(y=e_2\), exact KRR predicts along that direction while the approximate kernel deletes it. A small global matrix percentage is not a task-aware guarantee.

## Optimization-oracle limits for kernel solves {#lower-optimization}

Kernel ridge regression solves \(A\alpha=y\) with \(A=K+\gamma I\succ0\), equivalently minimizing

$$
q(\alpha)=\frac12\alpha^\top A\alpha-y^\top\alpha.
$$

The optimization model must be declared. Here a first-order method receives gradients
\(\nabla q(\alpha)=A\alpha-y\), which are equivalent to matrix-vector products with \(A\).

::: {.theorem #thm-lower-first-order}
[Theorem (first-order quadratic lower bound; stated with source)]{.box-title}

Fix \(t\geq0\) and \(\kappa\gt1\). In dimension at least \(2t+1\), for every deterministic first-order method whose \(t\)-th iterate lies in the span generated by previous gradients, there exists a positive-definite quadratic with condition number at most \(\kappa\) such that

$$
\frac{\lVert\alpha_t-\alpha^\star\rVert_A}
{\lVert\alpha_0-\alpha^\star\rVert_A}
\geq
c
\left(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^t
$$

for a universal constant \(c\gt0\). Thus \(\Omega(\sqrt\kappa\log(1/\varepsilon))\) first-order calls are necessary in the worst case for relative energy error \(\varepsilon\).

**Assumptions.** Exact arithmetic, a black-box first-order oracle, the linear-span restriction, and sufficiently large dimension. The theorem does not cover preconditioners built from richer access, direct factorizations, finite termination after discovering the full spectrum, or randomized algorithms without an additional minimax argument.

**Proof status.** Stated from the resisting-oracle/Chebyshev theory in [@nesterov2004, Sections 2.1--2.2]; not reproved in full.
:::

For kernel systems,
\(\kappa(A)=(\lambda_{\max}(K)+\gamma)/(\lambda_{\min}(K)+\gamma)\), and
\(\lambda_{\min}(K)\) may be zero. Ridge caps the condition number by
\(1+\lambda_{\max}(K)/\gamma\), but small regularization can still make the solve hard. Conjugate gradients has a matching Chebyshev-type upper bound in this oracle model. Preconditioning changes the spectrum and therefore changes the problem to which the lower bound applies; it does not violate the theorem.

### Residual, solution error, and statistical risk {#lower-opt-currencies}

The residual \(r_t=y-A\alpha_t\) satisfies

$$
\alpha_t-\alpha^\star=-A^{-1}r_t.
$$

A small Euclidean residual implies a solution bound containing
\(\lVert A^{-1}\rVert\). It says nothing by itself about population risk. Conversely, early stopping may have useful statistical regularization even when the linear system is not solved accurately. An optimization lower bound becomes a learning lower bound only after a reduction that preserves the sampling model and loss.

## Adaptation and no-free-lunch limits {#lower-adaptation}

An estimator is adaptive if it attains, without knowing the class index, the minimax rate over several source, capacity, noise, or margin classes. Adaptation is possible in some regimes through validation, aggregation, Lepski-type comparison, or hierarchical penalties. It is not free.

::: {.proposition #prop-lower-nested}
[Proposition (nested-class warning; proved here)]{.box-title}

Let \(\mathcal P_1\subset\mathcal P_2\). If two distributions
\(P_0\in\mathcal P_1\) and \(P_1\in\mathcal P_2\) satisfy the assumptions of the two-point theorem with separation \(\Delta\), then no estimator can have risk below

$$
\frac{\Delta}{4}
\left(1-\operatorname{TV}(P_0^n,P_1^n)\right)
$$

simultaneously at both points.

**Assumptions.** The two distributions belong to the stated nested classes,
their product experiments satisfy the two-point theorem, and the loss uses the
same parameter map and separation constant \(\Delta\) at both points.

**Proof status.** Immediate from Theorem [two-point lower bound](#thm-lower-le-cam).
:::

This elementary statement is not a complete adaptation theorem. It identifies the mechanism: if membership in the smoother class cannot be tested at the desired resolution, an estimator cannot safely switch to the smoother-class tuning at that resolution. Full adaptation results must state the family of classes, whether logarithmic penalties are unavoidable, and whether the guarantee is pointwise, minimax, or honest and uniform.

No-free-lunch claims require equal care. Without restrictions on the target or distribution, density of an RKHS gives no uniform rate. Without a lower spectral condition, an effective-dimension upper bound cannot certify hardness. Without restrictions on feature sampling, an i.i.d. RFF lower bound says nothing about optimized features. “No method can do better” is valid only after the estimator class and information interface have been quantified.

## Failure witnesses and diagnostic protocol {#lower-failures}

Every lower-bound argument should be tested against a witness that breaks one of its hypotheses.

1. **Identical alternatives.** If \(a(P_0)=a(P_1)\), testing may be hard but estimation separation is zero.
2. **Disjoint alternatives.** If total variation is one, the alternatives are perfectly testable and Le Cam gives nothing.
3. **One-sided capacity.** A finite-rank operator satisfies many upper capacity bounds but does not contain infinitely many hard directions.
4. **Vanishing noise.** A noisy-regression lower bound need not survive noiseless interpolation.
5. **Wrong currency.** A rank lower bound for spectral approximation does not force excess risk when labels avoid the omitted subspace.
6. **Richer access.** A matrix-vector oracle lower bound does not constrain an algorithm handed an exact eigendecomposition.
7. **Changed quantifiers.** A worst-case lower bound does not predict the median difficulty of a benchmark dataset.

A useful lower-bound audit records:

| Item | Required question |
|---|---|
| parameter class | What distributions, functions, spectra, and noise laws are allowed? |
| estimator class | All measurable estimators, fixed features, oblivious sketches, or first-order spans? |
| observation model | Random design, fixed design, oracle access, matrix entries, or samples? |
| loss | \(L^2(P_X)\), empirical prediction, RKHS norm, matrix norm, energy error, or runtime? |
| quantifiers | Which object is chosen first, and where does randomness live? |
| proof bridge | How does testing, packing, rank, or polynomial approximation imply the target loss? |
| failure boundary | Which stronger access or narrower class escapes the result? |

## Practice: constructing a lower bound responsibly {#lower-practice}

Start from the desired impossibility statement, not from a favorite inequality.

1. Name the currency and normalize the loss.
2. Specify the estimator's information interface.
3. Choose a finite hard family inside the parameter class.
4. Compute its separation in the target loss.
5. Compute or bound KL, Hellinger, total variation, or oracle indistinguishability.
6. Apply the weakest testing or approximation lemma that closes the argument.
7. Check that every alternative satisfies positivity, boundedness, source, capacity, and noise assumptions.
8. State whether constants and endpoints are proved, sourced, or only the exponent is reconstructed.
9. Exhibit an escape route showing the theorem's boundary.

The most common mistake is to perform steps four and five in incompatible geometries. An RKHS-norm packing may have the wrong \(L^2(P_X)\) separation. A Frobenius-small matrix perturbation may be regularized-spectrally large. A hard quadratic may not be a Gram matrix in the kernel family under discussion.

## Common mistakes and practical implications {#lower-common-mistakes}

Three mistakes recur because they allow a true lower bound to answer the wrong
question. First, changing the loss changes the theorem: a Frobenius-norm rank
barrier does not establish population prediction hardness unless a reduction
connects the omitted matrix directions to admissible labels. Second, changing
the information interface changes the adversary: a matrix-vector oracle result
does not constrain an algorithm that receives landmarks, entries, or a
factorization. Third, changing one side of a rate comparison breaks the match:
a minimax expectation bound under random design cannot certify optimality of a
high-probability fixed-design algorithm.

In practice, place upper and lower results in the same ledger row before
calling a rate optimal. The row must agree on parameter class, observation
model, loss, probability mode, computational access, and asymptotic regime. If
one cell differs, report the results as complementary rather than matching.
This comparison prevents an impossibility theorem from becoming a rhetorical
flourish and tells an implementer which escape route (richer access,
adaptivity, a narrower target class, or a different tolerance) remains open.

## Summary and further reading {#lower-summary}

Lower bounds complete an upper-bound story by identifying what no admissible procedure can uniformly improve. Le Cam reduces estimation to one binary test. Fano turns metric entropy into a many-way test. Assouad accumulates difficulty across independently flippable coordinates. In kernel regression, source smoothness sets the amplitude of hard eigen-directions, capacity determines how many such directions are available, and noise limits how well their signs can be recovered. Balancing these ingredients gives the exponent \(2r/(2r+p)\).

Randomized computation adds representation restrictions. Rank, feature, landmark, and sketch lower bounds must name what is preserved and what access is allowed. Optimization lower bounds add an oracle model: the familiar square-root dependence on condition number is a statement about first-order access to a worst-case quadratic, not about every implementation of every kernel solve. Adaptation limits are testing limits between nested classes, and no-free-lunch statements are meaningful only with explicit quantifiers.

For statistical reductions and packing methods, see [@tsybakov2009]. For source-and-capacity rates and qualification, see [@caponnetto2007; @dicker2015]. For feature and sketch barriers, see [@bach2017quadrature; @avron2017rff; @yang2017sketch]. For first-order oracle complexity, see [@nesterov2004].

## Exercises {#exercises}

::: {.exercises}
1. [warm-up]{.ex-tag} Classify each claim by currency: (a) every rank-\(m\) approximation has regularized spectral error at least \(0.2\); (b) every estimator has expected prediction error at least \(cn^{-2/3}\); (c) every linear-span method needs \(c\sqrt\kappa\log(1/\varepsilon)\) matrix-vector products; (d) every distributed implementation communicates at least \(nd\) words. For each claim, name one conclusion in a different currency that does not follow automatically.
2. [computation]{.ex-tag} Reproduce the finite-rank Gaussian example with \(n=8\), \(\delta=1/4\), and noise variance one. Compute parameter separation, KL divergence, the Pinsker total-variation bound, and the final Le Cam risk lower bound to six decimals.
3. [proof]{.ex-tag} Prove the total-variation form of the two-point theorem when the loss is squared metric loss. Show that \(d(u,a_0)^2+d(u,a_1)^2\geq d(a_0,a_1)^2/2\), and identify \(\Delta\).
4. [proof]{.ex-tag} Fill in the source-capacity packing algebra. With \(\mu_m\asymp m^{-1/p}\), show that balancing \(a^2\lesssim R^2\mu_m^{2r}/m\) and \(a^2\lesssim\sigma^2/n\) yields \(m\asymp n^{p/(2r+p)}\) and risk \(ma^2\asymp n^{-2r/(2r+p)}\). Explain why a capacity upper bound alone does not justify this packing.
5. [computation]{.ex-tag} Let \(K=\operatorname{diag}(9,4,1,0.25)\), \(\gamma=1\), and require regularized spectral error at most \(0.4\). Use the rank barrier to find the smallest rank not ruled out. Compare with the smallest rank not ruled out when \(\gamma=0.1\).
6. [proof]{.ex-tag} Prove Proposition [rank barrier](#prop-lower-rank) using the dimension formula for intersecting subspaces. Then construct a rank-\(m\) truncated eigendecomposition that attains error \(\lambda_{m+1}/(\lambda_{m+1}+\gamma)\).
7. [synthesis]{.ex-tag} A paper proves that \(m\) ordinary random Fourier features cannot achieve a stated regularized spectral approximation for a Gaussian kernel. List the additional reductions and assumptions needed before concluding that no \(m\)-feature method can attain the minimax population KRR rate.
8. [exploration]{.ex-tag} For a kernel system \(A=K+\gamma I\), compare three interfaces: matrix-vector products only, entry queries plus adaptive Nyström sampling, and an explicit dense matrix. State which first-order lower bound applies to each, what preprocessing is allowed, and which cost must be included for a fair comparison.
9. [challenge]{.ex-tag} Construct a two-class adaptation witness. Choose \(P_0\) in a smaller source ball and \(P_1\) only in a larger ball so that their parameters are separated by \(2s\) but \(n\operatorname{KL}(P_0\Vert P_1)\leq1/8\). Use the two-point theorem to show that an estimator cannot have squared error \(o(s^2)\) at both points. State why this does not prove that adaptation over the full pair of classes is impossible.
:::
