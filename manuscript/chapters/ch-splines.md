---
id: ch-splines
slug: smoothing-splines-and-additive-rkhs
title: Smoothing Splines and Additive RKHS Models
part: XV · Classical and Reliable Kernel Models
order: 51
tier: core
prerequisites:
  - kernels-and-rkhs
  - kernel-ridge-and-friends
  - mercer-and-rates
objectives:
  - Derive a smoothing spline from a roughness-penalized variational problem.
  - >-
    Separate the penalized RKHS component from the unpenalized polynomial null
    space.
  - >-
    Interpret the smoother matrix through effective degrees of freedom and
    generalized cross-validation.
  - >-
    Construct additive and tensor-product RKHS models with identifiable
    interactions.
  - >-
    Relate splines, kernel ridge regression, Gaussian processes, and Green
    functions.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-splines.yml
verification_date: null
bibliography:
  - kimeldorf1971
  - cravenwahba1979
  - wahba1990
  - green1984
---
# Smoothing Splines and Additive RKHS Models

<p class="lead">Long before the phrase kernel machine became standard, statisticians faced a dilemma this book keeps meeting: a curve that passes through every observation oscillates wildly between the points, while a curve rigid enough to be stable ignores the data. Their answer was to penalize derivatives, letting the data pull against a roughness charge. The minimizer of that variational problem was a spline, and its apparently special finite form was an instance of the representer theorem. This chapter develops that classical route, explains the unpenalized null space that ordinary kernel ridge regression often hides and that silently decides how the fit extrapolates, and extends the construction to additive models with interpretable interactions.</p>

## From roughness to a finite estimator {#roughness-to-estimator}

Suppose observations satisfy \(y_i=f^\star(x_i)+\varepsilon_i\) on an interval. Interpolation alone is unstable because a function can pass through every observation while oscillating violently between them. A smoothing spline balances fit against a differential measure of roughness:

$$
\min_{f\in\mathcal V}\;\frac{1}{n}\sum_{i=1}^n\{y_i-f(x_i)\}^2
+\lambda J(f),
\qquad
J(f)=\int \{D^m f(t)\}^2\,dt.
$$

The derivative order \(m\) specifies which behavior is considered rough. The tuning parameter \(\lambda\) controls the tradeoff: small \(\lambda\) follows the observations, while large \(\lambda\) drives the solution toward functions annihilated by \(D^m\).

::: {.definition #def-spline-null-space}
[Definition (penalty null space)]{.box-title}

The null space of a seminorm penalty \(J\) is

$$
\mathcal N_J=\{f\in\mathcal V:J(f)=0\}.
$$

For the integrated squared \(m\)-th derivative, \(\mathcal N_J\) is the space of polynomials of degree strictly smaller than \(m\), subject to the chosen domain and boundary convention.
:::

This null space is not a nuisance. It identifies the trends that should remain unpenalized. Penalizing an intercept or a linear trend merely because they appear in a coefficient vector changes the statistical model.

## The spline representer theorem {#spline-representer}

The objective searches an infinite-dimensional space, yet its minimizer is pinned down by finitely many coefficients. The mechanism is the orthogonality argument behind every representer theorem, once the free trends are given their own coordinates. Decompose the function space as \(\mathcal V=\mathcal N_J\oplus\mathcal H_J\), where \(\mathcal H_J\) is an RKHS on which \(J(f)=\lVert f\rVert_{\mathcal H_J}^2\). Let \(p_1,\ldots,p_q\) span the null space and let \(R\) be the reproducing kernel of \(\mathcal H_J\).

:::: {.theorem #thm-spline-representer}
[Theorem (smoothing-spline representer form)]{.box-title}

Every minimizer of the squared-error and roughness objective has a representative of the form

$$
\widehat f(x)=\sum_{a=1}^q d_a p_a(x)+\sum_{i=1}^n c_i R(x_i,x).
$$

When the loss is strictly convex on the observed values and the usual side constraints identify the null-space component, the fitted values are unique.

**Assumptions.** Evaluation is continuous on \(\mathcal H_J\); \(\mathcal N_J\) is finite dimensional; a minimizer exists; the design identifies the null-space basis. **Proof status.** Proved by decomposing the penalized component into the span of \(R(x_i,\cdot)\) and its orthogonal complement. The complement changes no observation and only increases \(J\) [@kimeldorf1971; @wahba1990].
::::

With \(K_{ij}=R(x_i,x_j)\) and \(P_{ia}=p_a(x_i)\), the coefficients solve a saddle system such as

$$
\begin{pmatrix}
K+n\lambda I & P\\
P^\top & 0
\end{pmatrix}
\begin{pmatrix}c\\d\end{pmatrix}
=
\begin{pmatrix}y\\0\end{pmatrix}.
$$

The constraint \(P^\top c=0\) prevents the penalized kernel expansion from duplicating the unpenalized trend. A stable implementation should factor the null space first, solve in its orthogonal complement, and report the condition number after scaling the inputs.

## Green functions, splines, and kernel ridge regression {#green-spline-krr}

The kernel \(R\) did not come from a catalogue; the penalty itself manufactures it. The penalty operator and its boundary conditions determine a Green function. That Green function becomes the reproducing kernel of the penalized space. The construction in [[ch:mercer-and-rates|Mercer's theorem, spectra, and rates]] therefore has a differential counterpart: the eigenfunctions of the boundary-value problem define the smoothing directions, and the inverse eigenvalues determine how strongly each direction is penalized [@green1984].

If the null space is absent or has already been projected out, the spline estimator is kernel ridge regression:

$$
\widehat f(x)=k_x^\top(K+n\lambda I)^{-1}y.
$$

The distinction is conceptual rather than cosmetic. A generic positive-definite kernel penalizes every RKHS direction. A spline seminorm leaves selected low-order trends free. This is why the extrapolation of a natural spline can differ sharply from that of an RBF kernel even when both interpolate the observed region well.

::: {.example #example-spline-nullspace}
[Example (the null space controls extrapolation)]{.box-title}

Consider two estimators with similar in-sample residuals. One uses a derivative seminorm whose null space contains affine functions; the other uses a strictly positive-definite stationary kernel with a zero mean. Beyond the observations, the first estimator tends toward its fitted affine component, while the second tends toward its prior mean as correlations decay. The difference is a consequence of the function-space decomposition, not merely a different bandwidth.

**Verification artifact.** checks/example-ch-splines-example-spline-nullspace.json records the example source hash and verification scope.
:::

## The smoother matrix and effective complexity {#smoother-matrix}

How flexible is a spline fit? Counting basis functions gives the wrong answer, because shrinkage leaves most of them barely used; the honest measure watches how the fitted values respond to the observations. For fixed design and tuning parameters, a squared-loss spline is a linear smoother:

$$
\widehat y=S_\lambda y.
$$

The eigenvalues of \(S_\lambda\) lie between zero and one under the standard construction. Directions with eigenvalues near one are retained; directions near zero are suppressed. The quantity

$$
\operatorname{df}(\lambda)=\operatorname{tr}(S_\lambda)
$$

is the effective degrees of freedom. It is generally not the number of nonzero coefficients. It measures how much the fitted vector responds, in total, to perturbations of the observations.

::: {.proposition #prop-spline-loocv}
[Proposition (leave-one-out residual identity)]{.box-title}

For a linear smoother whose matrix does not depend on \(y\), the leave-one-out residual at observation \(i\) is

$$
\frac{y_i-\widehat y_i}{1-S_{\lambda,ii}},
$$

whenever the denominator is nonzero.

**Assumptions.** The fitted values are linear in \(y\), and deleting one observation leaves the same smoothing rule applied to the reduced problem. **Proof status.** Proved by the rank-one deletion identity; the result is the spline analogue of the KRR identity in [[ch:kernel-ridge-and-friends]] [@cravenwahba1979].
:::

Generalized cross-validation replaces the individual diagonal values by their average:

$$
\operatorname{GCV}(\lambda)=
\frac{n^{-1}\lVert(I-S_\lambda)y\rVert^2}
{\{1-n^{-1}\operatorname{tr}(S_\lambda)\}^2}.
$$

GCV is rotation invariant in observation space and inexpensive once the trace is available. It is not immune to dependence, heteroscedasticity, or model selection bias. Those conditions require blocked validation, weighted criteria, or a likelihood-based alternative.

## Smoothing-spline ANOVA {#spline-anova}

A single input has carried the chapter this far. With many inputs, a fully general multivariate fit is expensive to compute and impossible to plot; the classical compromise splits the function into components a person can read. For multivariate inputs \(x=(x_1,\ldots,x_d)\), an additive decomposition separates main effects and interactions:

$$
f(x)=\mu+\sum_j f_j(x_j)+\sum_{j\lt k}f_{jk}(x_j,x_k)+\cdots.
$$

Each component belongs to an RKHS, and centering constraints make the decomposition identifiable. Tensor products construct interaction spaces. If \(\mathcal H_j\) has kernel \(k_j\), then a two-factor interaction has kernel \(k_j k_k\). Expanding

$$
\prod_j(1+k_j)
$$

reveals the constant, main-effect, and interaction subspaces. This is the functional counterpart of the ANOVA kernel algebra in [[ch:kernel-families]].

The model can use separate penalties \(\lambda_r J_r(f_r)\), allowing different smoothness for different components. Selection then concerns both smoothing and structure. A hierarchy rule can require an interaction to enter only when its main effects are present, while sparsity penalties can suppress whole components.

:::: {.algorithm #algo-spline-anova-fit}
[Algorithm (auditable additive RKHS fit)]{.box-title}

**Input.** Training observations, component spaces, null-space bases, candidate smoothing parameters, and a validation design.

**Output.** An identified additive predictor with componentwise uncertainty and complexity diagnostics.

1. Center covariates and construct each null-space and penalized basis using training data only.
2. Impose centering constraints on main effects and marginal constraints on interactions.
3. Solve the penalized system by a stable direct or matrix-free method.
4. Select smoothing parameters with nested validation, GCV under justified assumptions, or marginal likelihood.
5. Report residual diagnostics, componentwise effective degrees of freedom, condition estimates, and sensitivity to the interaction hierarchy.
6. Stop iterative solves when both the linear residual and held-out predictions stabilize.

Dense direct cost is cubic in the total basis size. Tensor structure, low-rank bases, and backfitting can lower cost, but their convergence tolerance becomes part of the estimator.
::::

## Bayesian interpretation and uncertainty {#spline-bayesian-view}

A fitted curve, however carefully tuned, does not report how far to trust it; the Bayesian reading of the same optimization supplies the missing distribution. The Kimeldorf-Wahba correspondence reads the penalized component as a Gaussian process and the null-space component as an unpenalized or diffuse trend [@kimeldorf1971]. Under Gaussian observation noise, the posterior mean equals the smoothing-spline estimator after matching the variance ratio to \(\lambda\). This connection explains why GCV, empirical Bayes, and marginal likelihood can produce related but nonidentical smoothing choices.

Posterior intervals inherit the GP model assumptions. Frequentist confidence intervals inherit approximations about bias and smoothing-parameter selection. Neither interpretation justifies ignoring selection uncertainty. When coverage matters, the conformal methods in [[ch:distribution-shift-robustness-and-conformal-prediction]] provide a complementary layer with different assumptions.

## Common mistakes and practical implications {#spline-practice}

- Treating a seminorm as a norm can accidentally penalize the intercept or leave the system unidentified.
- Using raw coordinates without scaling makes derivative penalties and tensor products difficult to interpret.
- Selecting \(\lambda\) and reporting error on the same validation data is optimistic.
- A large basis is not automatically a complex fit; the effective degrees of freedom depend on shrinkage.
- High-order interactions can dominate computation and destroy interpretability without hierarchy constraints.
- Pointwise intervals do not automatically provide simultaneous or post-selection coverage.

Use a spline when the roughness operator and its null space encode scientifically meaningful behavior. Use a generic kernel when similarity is easier to specify than derivatives. In either case, report the smoother spectrum rather than only the nominal number of coefficients.

## Summary and further reading {#spline-summary}

Smoothing splines are kernel machines derived from differential roughness penalties. Their representer form contains both an unpenalized null space and a penalized RKHS expansion. The smoother matrix exposes effective complexity and fast cross-validation, while tensor products lead to additive and interaction models. The foundational correspondence is developed by [@kimeldorf1971] and [@wahba1990], with generalized cross-validation in [@cravenwahba1979].

## Exercises {#exercises}

1. [warm-up]{.ex-tag} For the integrated squared second-derivative penalty, identify the null space and explain why its basis must appear outside the penalized kernel expansion.
2. [computation]{.ex-tag} Given a symmetric smoother matrix and a residual vector, compute the leave-one-out residuals, effective degrees of freedom, and generalized cross-validation score.
3. [proof]{.ex-tag} Prove the smoothing-spline representer theorem by decomposing the penalized component into observed kernel sections and their orthogonal complement.
4. [challenge]{.ex-tag} Derive the identifiable kernel decomposition for a two-variable smoothing-spline ANOVA model containing both main effects and one interaction.
5. [synthesis]{.ex-tag} Design a comparison among a natural spline, an RBF KRR model, and a Matérn GP for data requiring extrapolation. Specify the null-space assumptions, selection protocol, uncertainty metrics, and conditioning diagnostics.
