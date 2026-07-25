---
id: ch-approx
slug: kernel-interpolation-and-approximation
title: Kernel Interpolation and Approximation Theory
part: 'IV · Generalization, Approximation, and Limits'
order: 18
tier: advanced
prerequisites:
  - mercer-and-rates
objectives:
  - >-
    Formulate kernel interpolation as a linear system and prove that its
    solution is the minimum-RKHS-norm interpolant.
  - >-
    Read the native space of a translation-invariant kernel off the Fourier
    transform of the kernel and place Matern and Gaussian kernels on the
    smoothness scale.
  - >-
    Derive the power function and use it to bound pointwise interpolation error
    for any target in the native space.
  - >-
    State fill-distance convergence rates and the accuracy-stability uncertainty
    relation, and draw their consequences for shape-parameter choice.
  - >-
    Explain Schoenberg's characterization of radial positive definiteness and
    the role of conditionally positive definite kernels.
  - >-
    Relate the worst-case error over a native-space ball to the Gaussian-process
    posterior standard deviation.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-approx.yml
verification_date: null
bibliography:
  - wendland2005
  - schaback1995
  - narcowich2005
  - fasshauer2007
  - schoenberg1938
  - micchelli1986
  - aronszajn1950
  - rasmussen2006
  - wahba1990
  - caponnetto2007
  - steinwart2008
---
# Kernel Interpolation and Approximation Theory

<p class="lead">A network of weather stations reports temperature at a few hundred scattered sites, and a forecaster must fill in the map between them; a wind-tunnel campaign yields drag at a dozen operating points, and an engineer must predict it everywhere else. Both want a surface that passes exactly through the measurements, together with an honest statement of how wrong it can be in between. Machine learning instinct says exact fitting is overfitting, yet approximation theory reaches the opposite verdict: when the data are noiseless, the kernel interpolant is the optimal recovery of every function the kernel's norm can measure, and it comes with a computable pointwise error bar. This chapter builds that theory. We solve the interpolation system and prove the solution has minimum RKHS norm, read the native space as a smoothness scale through the Fourier transform, derive the power function and recognize it as the Gaussian-process posterior standard deviation, convert point-set geometry into convergence rates, and prove the uncertainty relation that couples spectral accuracy to exponential ill-conditioning.</p>

## Kernel interpolation as minimum-norm recovery {#approx-minimum-norm}

Suppose we observe a function exactly, with no noise, at finitely many sites. What should we do with a kernel then? The regularized machinery of [[ch:kernel-ridge-and-friends]] was built to trade data fit against smoothness, but with exact data there is nothing to trade: we should fit the observations exactly and let the RKHS norm arbitrate among the infinitely many functions that do.

::: {.definition #def-approx-interpolant}
[Definition (kernel interpolation problem)]{.box-title}

Let \(k\) be a strictly positive definite kernel on \(\mathcal X\) with RKHS \(\mathcal H_k\), let \(X = \{x_1, \dots, x_n\} \subset \mathcal X\) be distinct sites, and let \(y_1, \dots, y_n\) be target values. The kernel interpolant is

$$
s(x) = \sum_{j=1}^n c_j\, k(x, x_j),
\qquad
K c = y,
$$

where \(K = [k(x_i, x_j)]_{i,j=1}^n\) is the Gram matrix. Strict positive definiteness and distinctness of the sites make \(K\) invertible, so \(c = K^{-1} y\) exists and is unique, and \(s(x_i) = y_i\) for every \(i\).
:::

The ansatz \(s \in \operatorname{span}\{k(\cdot, x_j)\}\) is not a convenience; it is forced by optimality. Among all functions in \(\mathcal H_k\) that pass through the data, the interpolant is the one the norm likes best.

::: {.proposition #prop-approx-min-norm}
[Proposition (minimum-norm interpolation)]{.box-title}

Let \(s = \sum_j c_j k(\cdot, x_j)\) with \(Kc = y\) be the kernel interpolant of the data. Then for every \(g \in \mathcal H_k\) with \(g(x_i) = y_i\) for all \(i\),

$$
\lVert g \rVert_{\mathcal H_k}^2 = \lVert s \rVert_{\mathcal H_k}^2 + \lVert g - s \rVert_{\mathcal H_k}^2,
$$

so \(s\) is the unique minimum-norm interpolant of the data.

**Assumptions.** \(k\) strictly positive definite, sites distinct, at least one interpolant exists in \(\mathcal H_k\). **Proof status.** complete.
:::

::: {.proof}
[Proof]{.box-title}

Let \(g\) interpolate the data and set \(u = g - s\). Then \(u(x_i) = 0\) for every \(i\). By the reproducing property,

$$
\langle u, s \rangle_{\mathcal H_k}
= \Big\langle u, \sum_j c_j k(\cdot, x_j) \Big\rangle_{\mathcal H_k}
= \sum_j c_j\, u(x_j) = 0,
$$

so \(u \perp s\) and the Pythagorean identity gives \(\lVert g \rVert^2 = \lVert s \rVert^2 + \lVert u \rVert^2\). The norm of any competitor strictly exceeds \(\lVert s \rVert\) unless \(u = 0\), which proves uniqueness. [\(\square\)]{.qed}
:::

Three familiar objects are special cases. First, kernel ridge regression: the ridge coefficients \((K + \lambda I)^{-1} y\) converge to \(K^{-1} y\) as \(\lambda \to 0^+\), so interpolation is noiseless ridge, the endpoint of the regularization path studied in [[ch:kernel-ridge-and-friends]]. Second, kriging: geostatisticians write the same linear system with a covariance function in place of \(k\) and call \(s\) the simple kriging predictor, the reading developed in [[ch:spatial-and-spatiotemporal-kernels]]. Third, splines: with the kernels of [[ch:smoothing-splines-and-additive-rkhs]], minimum-norm interpolation is exactly natural spline interpolation, the historical root of the whole subject [@wahba1990]. Approximation theorists arrived at this triangle from scattered-data interpolation with radial basis functions, and their monographs [@wendland2005; @fasshauer2007] are the source for most of what follows.

## The native space {#approx-native-space}

Whose functions does the error theory cover? Every bound in this chapter is stated for targets in the RKHS of \(k\), which the approximation literature calls the native space of the kernel and constructs, following [@aronszajn1950], as the completion of finite kernel combinations. The name change signals a change of viewpoint: rather than a feature-space device, the native space is read as a concrete smoothness class, and the cleanest reading is through the Fourier transform.

::: {.theorem #thm-approx-fourier}
[Theorem (Fourier form of the native space)]{.box-title}

Let \(k(x, y) = \varphi(x - y)\) on \(\mathbb R^d\) with \(\varphi\) continuous, integrable, and with Fourier transform \(\hat\varphi(\omega) \gt 0\) for all \(\omega\). Then

$$
\mathcal H_k = \Big\{ f \in L^2(\mathbb R^d) \cap C(\mathbb R^d) \,:\,
\lVert f \rVert_{\mathcal H_k}^2 = (2\pi)^{-d/2} \int_{\mathbb R^d}
\frac{\lvert \hat f(\omega) \rvert^2}{\hat\varphi(\omega)}\, d\omega \lt \infty \Big\},
$$

with the corresponding inner product.

**Assumptions.** \(\varphi \in C(\mathbb R^d) \cap L^1(\mathbb R^d)\), \(\hat\varphi\) positive everywhere. **Proof status.** cited from [@wendland2005].
:::

The formula says that membership in \(\mathcal H_k\) is a decay condition on \(\hat f\): the spectrum of \(f\) must vanish at high frequencies at least as fast as the square root of the kernel's spectrum. The kernel's Fourier decay is therefore a smoothness dial, the stationary sibling of the eigenvalue-decay dial of [[ch:mercer-and-rates]].

Two families calibrate the dial. The Matern kernel with smoothness parameter \(\nu\) has \(\hat\varphi(\omega)\) proportional to \((1 + \lVert\omega\rVert^2)^{-(\nu + d/2)}\), so its native space is norm-equivalent to the Sobolev space \(H^{\nu + d/2}(\mathbb R^d)\): functions with \(\nu + d/2\) square-integrable derivatives, no more and no less [@wendland2005]. This is the reason Matern kernels are the workhorse of spatial statistics: their native spaces are exactly the classical smoothness classes.

The Gaussian kernel sits at the opposite extreme. Its Fourier transform is again a Gaussian, so membership requires \(\lvert\hat f\rvert^2\) to decay faster than \(e^{-\sigma^2\lVert\omega\rVert^2/2}\). The native space is tiny: every member extends to an analytic function, and even modest functions such as \(x \mapsto \lvert x \rvert\), or any function with a single kink, are excluded [@steinwart2008]. The cost is concrete. The error bounds of this chapter are multiplied by \(\lVert f \rVert_{\mathcal H_k}\), and for a target outside the native space that factor is infinite, so the bound says nothing. Gaussian interpolation of merely Sobolev-smooth targets still works, but its analysis needs the escape results of a later section, and its fast-convergence promises apply only to the very smooth.

## The power function {#approx-power-function}

An interpolant without an error estimate is a guess. The question this section answers: at a point \(x\) where we did not measure, how wrong can \(s(x)\) be, over all targets the native space admits? The answer is the norm of the pointwise error functional.

Write the interpolant of data \(f(x_1), \dots, f(x_n)\) in cardinal form: \(s_f(x) = k_X(x)^\top K^{-1} y = \sum_j w_j(x) f(x_j)\), where \(k_X(x) = (k(x, x_1), \dots, k(x, x_n))^\top\) and \(w(x) = K^{-1} k_X(x)\) are weights independent of \(f\). The error at \(x\) is then a linear functional of \(f\),

$$
\varepsilon_x(f) = f(x) - s_f(x)
= \Big\langle f,\; k(\cdot, x) - \sum_{j=1}^n w_j(x)\, k(\cdot, x_j) \Big\rangle_{\mathcal H_k},
$$

by the reproducing property, so it is bounded, and its Riesz representer is explicit.

::: {.definition #def-approx-power}
[Definition (power function)]{.box-title}

The power function of the site set \(X\) is the norm of the pointwise error functional,

$$
P_X(x) = \sup_{\lVert f \rVert_{\mathcal H_k} \le 1} \lvert f(x) - s_f(x) \rvert
= \Big\lVert k(\cdot, x) - \sum_j w_j(x) k(\cdot, x_j) \Big\rVert_{\mathcal H_k}.
$$

Expanding the squared norm of the representer and using \(w(x) = K^{-1} k_X(x)\) gives the closed form

$$
P_X(x)^2 = k(x, x) - k_X(x)^\top K^{-1} k_X(x).
$$
:::

::: {.theorem #thm-approx-power-bound}
[Theorem (pointwise error bound)]{.box-title}

For every \(f \in \mathcal H_k\) and every \(x \in \mathcal X\),

$$
\lvert f(x) - s_f(x) \rvert \le P_X(x)\, \lVert f \rVert_{\mathcal H_k},
$$

and the bound is sharp: for each \(x\) there is a unit-norm \(f\) attaining it.

**Assumptions.** \(k\) strictly positive definite, \(X\) distinct sites, \(f\) in the native space. **Proof status.** complete.
:::

::: {.proof}
[Proof]{.box-title}

The error is the inner product of \(f\) with the representer \(r_x = k(\cdot, x) - \sum_j w_j(x) k(\cdot, x_j)\), so Cauchy-Schwarz gives \(\lvert \varepsilon_x(f) \rvert \le \lVert r_x \rVert\, \lVert f \rVert = P_X(x) \lVert f \rVert\). Equality holds for \(f = r_x / \lVert r_x \rVert\) whenever \(P_X(x) \gt 0\); at sites, both sides vanish. [\(\square\)]{.qed}
:::

The closed form deserves a second look, because we have met it before. It is precisely the posterior variance of a noise-free Gaussian process with covariance \(k\) conditioned on observations at \(X\), as derived in [[ch:gaussian-processes-and-rvm]]. The identical formula carries two meanings: for the Bayesian it is an average-case error bar under a prior, and for the approximation theorist it is a worst-case error bar over the unit ball of the native space [@rasmussen2006]. The same object, integrated over the domain, became the worst-case quadrature error of [[ch:kernel-quadrature-and-herding]]; here we meet its pointwise ancestor. We return to this two-way reading at the end of the chapter.

The pointwise shape matters as much as the formula. Uncertainty vanishes at every interpolation site, grows in the gaps between sites, and becomes largest where the design leaves the domain least covered. For the minimum kernel this geometry can be computed exactly, so the interpolant and its sharp native-space envelope can be placed on the same axes.

<figure class="viz" data-figure="power-function" data-alt="A piecewise-linear minimum-kernel interpolant passes through three observations of a quadratic target. A shaded error envelope has zero width at each interpolation site and widens between sites in proportion to the power function."><figcaption>The power function is a geometric error bar: it vanishes where data pin down the interpolant and widens in poorly covered gaps, simultaneously representing worst-case native-space error and noise-free Gaussian-process uncertainty.</figcaption></figure>

## Fill distance, separation radius, and convergence rates {#approx-fill-distance}

Point count alone does not measure coverage. Eleven quasi-uniform sites and eleven clustered sites have the same nominal budget, yet the latter leave a large unresolved gap. Computing the Gaussian power function with a Cholesky solve makes the geometric consequence explicit before any asymptotic rate is quoted.

<figure class="viz" data-figure="fill-distance-power-rate" data-alt="Power functions for equally many quasi-uniform and clustered interpolation sites. The quasi-uniform design has a low envelope, while the clustered design has a large peak in its uncovered gap."><figcaption>Fill distance, not sample count, controls the worst uncovered region. The power function vanishes at the sites, remains uniformly small for the quasi-uniform design, and rises sharply across the hole left by clustered sampling.</figcaption></figure>

The power function converts geometry into error, so the next step is to summarize the geometry of \(X\) in two numbers.

::: {.definition #def-approx-fill-sep}
[Definition (fill distance and separation radius)]{.box-title}

For a site set \(X \subset \Omega \subset \mathbb R^d\), the fill distance and separation radius are

$$
h_{X,\Omega} = \sup_{x \in \Omega} \min_{1 \le j \le n} \lVert x - x_j \rVert,
\qquad
q_X = \tfrac{1}{2} \min_{i \ne j} \lVert x_i - x_j \rVert.
$$

\(X\) is quasi-uniform with mesh ratio \(\rho\) when \(h_{X,\Omega} \le \rho\, q_X\).
:::

The fill distance is the radius of the largest data-free ball: it measures how well \(X\) covers \(\Omega\) and governs accuracy. The separation radius measures how close two sites come, governs stability, and appears in the next section. Convergence theorems bound the power function by powers of \(h\).

::: {.theorem #thm-approx-rates}
[Theorem (fill-distance error rates)]{.box-title}

Let \(\Omega \subset \mathbb R^d\) be a bounded Lipschitz domain and let \(k\) have native space norm-equivalent to \(H^\tau(\Omega)\) with \(\tau \gt d/2\), as for the Matern family. There are constants \(C, h_0\) such that for all \(X\) with \(h = h_{X,\Omega} \le h_0\) and all \(f \in \mathcal H_k\),

$$
\lVert f - s_f \rVert_{L^\infty(\Omega)} \le C\, h^{\tau - d/2}\, \lVert f \rVert_{\mathcal H_k},
\qquad
\lVert f - s_f \rVert_{L^2(\Omega)} \le C\, h^{\tau}\, \lVert f \rVert_{\mathcal H_k}.
$$

For the Gaussian kernel the rate is spectral: \(\lVert f - s_f \rVert_{L^\infty(\Omega)} \le C e^{-c\,\lvert \log h \rvert / h}\, \lVert f \rVert_{\mathcal H_k}\) for some \(c \gt 0\).

**Assumptions.** Bounded Lipschitz domain, \(h\) below a domain-dependent threshold, target in the native space. **Proof status.** cited from [@wendland2005].
:::

The reading is direct: each extra order of kernel smoothness buys one extra power of \(h\), and infinitely smooth kernels buy rates faster than any polynomial. The rates depend on \(X\) only through \(h\), so they reward covering, not abundance: a thousand points crowded in one corner have the same fill distance as a handful spread well. Quasi-uniformity enters when constants must not degrade as points are added, and in the sharper results of the escape section below; in practice near-uniform designs, or sequential designs that greedily shrink the largest power-function value, keep \(\rho\) bounded. The statistical, noisy analogue of this bias analysis, where rates are driven by eigenvalue decay rather than fill distance, is the subject of [[ch:mercer-and-rates]] and of [@caponnetto2007]; the two theories describe the same estimator under different data models.

## Stability and the uncertainty principle {#approx-stability}

Approximation and numerical stability can move in opposite directions. For analytic kernels on increasingly dense sites, an error bound may fall exponentially while the Gram condition number grows exponentially. The schematic experiment below uses explicit monotone laws to expose that conflict; it is a regime map, not a claim that the displayed constants apply to every kernel or point set.

<figure class="viz" data-figure="approximation-conditioning-frontier" data-alt="As fill distance decreases, an approximation-error curve falls exponentially while a Gram-matrix condition-number curve rises exponentially on a second logarithmic axis."><figcaption>Dense sampling can buy approximation accuracy with numerical instability. Stable factorizations, scaling, regularization, or a better basis are therefore part of approximation theory rather than implementation afterthoughts.</figcaption></figure>

Everything so far argues for the smoothest kernel available: more smoothness, faster rates. The linear algebra objects. The system \(Kc = y\) must be solved, and its condition number is governed by the separation radius: for a kernel of Sobolev smoothness \(\tau\), the smallest Gram eigenvalue obeys a lower bound of order \(q_X^{2\tau - d}\), which degrades rapidly as smoothness grows or points cluster [@wendland2005]. For the Gaussian kernel, no polynomial bound holds and \(\lambda_{\min}(K)\) decays exponentially. This is not an artifact of a bad basis; there is a theorem in the way, made quantitative by Schaback [@schaback1995].

::: {.theorem #thm-approx-uncertainty}
[Theorem (uncertainty relation)]{.box-title}

Let \(k\) be strictly positive definite, \(X\) a set of distinct sites, and \(x \notin X\). Write \(K'\) for the Gram matrix of \(X \cup \{x\}\). Then

$$
P_X(x)^2 = \frac{1}{\big[(K')^{-1}\big]_{xx}} \ge \lambda_{\min}(K').
$$

Accuracy and stability are therefore coupled: wherever the power function is small, the augmented Gram matrix has a small eigenvalue.

**Assumptions.** \(k\) strictly positive definite; the sites of \(X \cup \{x\}\) distinct. **Proof status.** complete; the trade-off principle is due to [@schaback1995].
:::

::: {.proof}
[Proof]{.box-title}

Order \(K'\) with \(x\) last, so \(K'\) has blocks \(K\), \(k_X(x)\), and \(k(x,x)\). The Schur complement of the block \(K\) is \(k(x,x) - k_X(x)^\top K^{-1} k_X(x) = P_X(x)^2\), and the block-inverse formula identifies its reciprocal as the \((x,x)\) diagonal entry of \((K')^{-1}\). Since \(K'\) is symmetric positive definite, every diagonal entry of \((K')^{-1}\) is at most \(\lambda_{\max}\big((K')^{-1}\big) = 1/\lambda_{\min}(K')\). Taking reciprocals gives \(P_X(x)^2 \ge \lambda_{\min}(K')\). [\(\square\)]{.qed}
:::

Read the inequality as a conservation law. A kernel whose power function decays spectrally fast, such as the Gaussian, forces \(\lambda_{\min}\) of its Gram matrices to decay at least as fast, so spectral accuracy and well-conditioned linear algebra cannot coexist in the standard basis. You may have either fast convergence in exact arithmetic or a benign linear system in floating point, not both.

The practical consequences center on the shape parameter \(\epsilon\), the inverse length-scale in \(k(x,y) = \varphi(\epsilon \lVert x - y \rVert)\). Small \(\epsilon\) flattens the kernel, improves accuracy in exact arithmetic, and destroys conditioning; large \(\epsilon\) localizes the kernel, tames \(K\), and wastes the data. Flat kernels are dangerous precisely because the naive solve fails long before the mathematical limit turns bad; in fact the \(\epsilon \to 0\) limit of the interpolant is typically a polynomial interpolant, and stable algorithms such as RBF-QR reach it by changing basis rather than by inverting \(K\) [@fasshauer2007]. When \(n\) is also large, conditioning interacts with the iterative solvers and preconditioners of [[ch:large-scale-kernels]]; a Gram matrix that is nearly singular by design is the worst client an iterative method can have.

## Escaping the native space {#approx-escape}

The error theorem has a gap: it prices every target by its native-space norm, and for a target outside the native space it goes silent. This matters most for smooth kernels, whose native spaces are smallest. What happens when we interpolate a function rougher than the kernel expects? Sampling inequalities give the answer: convergence survives, at the rate the target's own smoothness supports.

::: {.theorem #thm-approx-escape}
[Theorem (rates outside the native space)]{.box-title}

Let \(\Omega\) be a bounded Lipschitz domain and let \(k\) have native space norm-equivalent to \(H^\tau(\Omega)\). Let \(f \in H^\beta(\Omega)\) with \(d/2 \lt \beta \le \tau\), and let \(X\) have fill distance \(h\) and mesh ratio \(\rho = h/q_X\). Then

$$
\lVert f - s_f \rVert_{L^\infty(\Omega)} \le C\, h^{\beta - d/2}\, \rho^{\tau - \beta}\, \lVert f \rVert_{H^\beta(\Omega)}.
$$

**Assumptions.** Bounded Lipschitz domain, \(\beta \gt d/2\) so that point evaluations of \(f\) are defined, \(h\) below a threshold. **Proof status.** cited from [@narcowich2005].
:::

The rate \(h^{\beta - d/2}\) is the one a kernel of exactly the target's smoothness would deliver, so the interpolant adapts downward: using a kernel that is too smooth costs only the mesh-ratio factor \(\rho^{\tau - \beta}\), which quasi-uniform designs keep bounded, plus the conditioning penalty of the previous section. The condition \(\beta \gt d/2\) is not decoration; below it, point values of \(f\) are not even well defined and interpolation is the wrong question. Together with the uncertainty relation, this theorem gives balanced advice: oversmooth kernels are statistically forgiving but numerically brittle, and the Matern family with a moderate \(\nu\) matched to the plausible roughness of the target is usually the sound default.

## Schoenberg theory and conditional positive definiteness {#approx-schoenberg}

Radial kernels raise a structural question that we have so far dodged: which functions of distance are legitimate kernels in the first place, on every \(\mathbb R^d\) at once? The classical answer predates the RKHS literature and belongs to Schoenberg.

::: {.definition #def-approx-completely-monotone}
[Definition (completely monotone function)]{.box-title}

A function \(g \in C[0, \infty) \cap C^\infty(0, \infty)\) is completely monotone when \((-1)^{\ell} g^{(\ell)}(r) \ge 0\) for all \(r \gt 0\) and all integers \(\ell \ge 0\): \(g\) is nonnegative, nonincreasing, convex, and so on through alternating signs of all derivatives.
:::

::: {.theorem #thm-approx-schoenberg}
[Theorem (Schoenberg)]{.box-title}

Let \(g: [0, \infty) \to \mathbb R\) be continuous. The kernel \(k(x, y) = g(\lVert x - y \rVert^2)\) is positive definite on \(\mathbb R^d\) for every \(d \ge 1\) if and only if \(g\) is completely monotone; it is strictly positive definite on every \(\mathbb R^d\) if and only if \(g\) is completely monotone and not constant.

**Assumptions.** \(g\) continuous on \([0, \infty)\). **Proof status.** cited from [@schoenberg1938].
:::

The theorem certifies the standard inventory at a glance: \(g(r) = e^{-\epsilon^2 r}\) gives the Gaussian, \(g(r) = (1 + r)^{-1/2}\) the inverse multiquadric, and any Laplace transform of a nonnegative measure qualifies. But it also excludes useful functions: the multiquadric \(\sqrt{1 + r}\) grows, and the thin-plate term \(r \log \sqrt r\) changes sign, yet both interpolate well in practice. The fix is to relax positive definiteness on a finite-dimensional subspace.

::: {.definition #def-approx-cpd}
[Definition (conditionally positive definite kernel)]{.box-title}

A symmetric kernel \(k\) on \(\mathbb R^d\) is conditionally positive definite of order \(m\) when for all distinct sites \(x_1, \dots, x_n\) and all coefficients \(c \ne 0\) satisfying \(\sum_i c_i\, p(x_i) = 0\) for every polynomial \(p\) of degree at most \(m - 1\), the quadratic form \(\sum_{i,j} c_i c_j k(x_i, x_j)\) is strictly positive.
:::

Interpolation with a conditionally positive definite kernel augments the expansion with the polynomials it exempts and imposes the moment conditions on the coefficients:

$$
s(x) = \sum_{j=1}^n c_j k(x, x_j) + \sum_{\ell} b_\ell\, p_\ell(x),
\qquad
\begin{pmatrix} K & P \\ P^\top & 0 \end{pmatrix}
\begin{pmatrix} c \\ b \end{pmatrix}
=
\begin{pmatrix} y \\ 0 \end{pmatrix},
$$

where the columns of \(P\) evaluate a basis of the exempt polynomials at the sites. Micchelli's theorem supplies the workhorse criterion: if \(g'\) is completely monotone and non-constant, then \(g(\lVert x - y \rVert^2)\) is conditionally positive definite of order one and the interpolation matrix is nonsingular for distinct sites, which covers the multiquadric [@micchelli1986]. Thin-plate splines \(\lVert x - y \rVert^2 \log \lVert x - y \rVert\) and the polyharmonic family are conditionally positive definite of higher order, and their augmented systems reproduce polynomials exactly: if the data come from a polynomial of exempt degree, the interpolant returns it. This is the same semiparametric structure as the representer theorem with unpenalized null space in [[ch:smoothing-splines-and-additive-rkhs]], and the same device that universal kriging uses to model a mean drift in [[ch:spatial-and-spatiotemporal-kernels]]. The moment conditions \(P^\top c = 0\) are exactly what make the quadratic form, and hence the native-space geometry, well defined modulo polynomials.

## Worst case meets average case {#approx-worst-average}

We close the theory by making precise a claim implicit throughout: kernel interpolation is not merely a method with good bounds, it is the best possible use of the data.

::: {.proposition #prop-approx-optimal-recovery}
[Proposition (minimax optimal recovery)]{.box-title}

Fix a site set \(X\), a point \(x\), and the data map \(f \mapsto (f(x_1), \dots, f(x_n))\). For any algorithm \(A\), linear or not, that predicts \(f(x)\) from the data,

$$
\sup_{\lVert f \rVert_{\mathcal H_k} \le 1} \big\lvert f(x) - A\big(f(x_1), \dots, f(x_n)\big) \big\rvert \;\ge\; P_X(x),
$$

and the kernel interpolant attains this bound. Interpolation is therefore minimax optimal among all algorithms using the same data.

**Assumptions.** \(k\) strictly positive definite, distinct sites. **Proof status.** sketched.
:::

The argument is a two-point trick. Let \(r_x\) be the error representer and set \(f_\pm = \pm r_x / \lVert r_x \rVert\). Both functions vanish at every site, so any algorithm receives identical, all-zero data and returns the same prediction for both; yet \(f_+(x) - f_-(x) = 2 P_X(x)\), so on at least one of them the algorithm errs by \(P_X(x)\) or more. The interpolant of zero data is zero, whose error on \(f_\pm\) is exactly \(P_X(x)\).

Now place this beside the probabilistic reading. Under a Gaussian process prior with covariance \(k\), the posterior mean given noise-free data is the kernel interpolant, and the posterior standard deviation at \(x\) is \(P_X(x)\) [@rasmussen2006]. The identical formulas thus answer two different questions: the worst case over a native-space ball, and the average case under a Gaussian prior, a correspondence running through the spline literature since [@wahba1990]. Nothing is mystical here; both derivations optimize the same quadratic functional. But the interpretations diverge exactly where practice needs care: the worst-case bound requires \(f\) in the native space with known norm, while the Bayesian error bar requires the prior to be honest, and GP sample paths themselves lie outside the native space almost surely. The acquisition rules of [[ch:bayesian-optimization-and-bandits]] consume the same mean and error bar, and inherit whichever reading one can defend.

One refinement is worth flagging: superconvergence. For targets smoother than the generic native-space element, in the sense of lying in the range of the kernel's integral operator, the \(L^2\) error rate roughly doubles, from \(h^\tau\) toward \(h^{2\tau}\)-type behavior, because the error functional then acts through the operator a second time [@wendland2005]. The generic bound of the power function is sharp over the whole ball, but the ball's smoother core converges faster; the source conditions of [[ch:mercer-and-rates]] tell the same story in spectral language.

## A three-point interpolant by hand {#approx-example}

Every formula in this chapter fits in a hand computation with the min kernel, whose RKHS we built explicitly in [[ch:mercer-and-rates]]: \(k(x, y) = \min(x, y)\) on \([0, 1]\), with native space the functions vanishing at zero whose derivative is square-integrable, and norm \(\lVert f \rVert^2 = \int_0^1 f'(t)^2\, dt\).

::: {.example #example-approx-power}
[Example (min-kernel interpolation of \(f(x) = x^2\))]{.box-title}

Interpolate \(f(x) = x^2\) at the sites \(X = \{0.25,\, 0.5,\, 1\}\), so \(y = (0.0625,\, 0.25,\, 1)^\top\). The Gram matrix is

$$
K = \begin{pmatrix} 0.25 & 0.25 & 0.25 \\ 0.25 & 0.5 & 0.5 \\ 0.25 & 0.5 & 1 \end{pmatrix}.
$$

**Coefficients.** Solve \(Kc = y\) by forward elimination: subtracting row one from rows two and three, then row two from row three, gives \(0.25(c_1 + c_2 + c_3) = 0.0625\), \(0.25(c_2 + c_3) = 0.1875\), and \(0.5\, c_3 = 0.75\). Back-substitution yields exactly

$$
c = (-0.5,\; -0.75,\; 1.5)^\top,
$$

and the checks confirm it: row two gives \(0.25(-0.5) + 0.5(-0.75) + 0.5(1.5) = -0.125 - 0.375 + 0.75 = 0.25\), and row three gives \(-0.125 - 0.375 + 1.5 = 1\). Since each \(\min(x, x_j)\) is piecewise linear, \(s\) is the piecewise-linear interpolant through \((0,0)\) and the data, with slopes \(0.25,\, 0.75,\, 1.5\) on the three subintervals; the slope drops \(c_1 + c_2 + c_3 = 0.25\) at the left, consistent with \(s'(0^+) = c_1 + c_2 + c_3\).

**Power function at \(x_* = 0.75\).** Here \(k(x_*, x_*) = 0.75\) and \(k_X(x_*) = (0.25,\, 0.5,\, 0.75)^\top\). Solving \(K w = k_X(x_*)\) the same way gives \(w = (0,\, 0.5,\, 0.5)^\top\), so

$$
P_X(0.75)^2 = 0.75 - \big(0.25 \cdot 0 + 0.5 \cdot 0.5 + 0.75 \cdot 0.5\big) = 0.75 - 0.625 = 0.125,
$$

and \(P_X(0.75) = \sqrt{0.125} = 0.3536\).

**The bound versus the truth.** The native-space norm of the target is \(\lVert f \rVert^2 = \int_0^1 (2t)^2 dt = 4/3\), so \(\lVert f \rVert = 1.1547\) and the theorem promises \(\lvert f(0.75) - s(0.75) \rvert \le 0.3536 \times 1.1547 = 0.4082\). The actual values are \(s(0.75) = 0.25 + 1.5 \times 0.25 = 0.625\) and \(f(0.75) = 0.5625\), an error of \(0.0625\), comfortably inside the worst-case bar. The Bayesian reading of the same numbers: a Brownian-motion prior conditioned on these three observations has posterior mean \(0.625\) and posterior standard deviation \(0.3536\) at \(x_* = 0.75\).

**Verification artifact.** checks/example-ch-approx-example-approx-power.json records the example source hash and verification scope.
:::

## Common mistakes and practical implications {#approx-practice}

- Interpolation is optimal for noiseless data only. With observation noise the minimum-norm interpolant chases the noise, and the correct estimator is the regularized one; the power function then understates the error because it prices bias but not variance.
- The error bound \(P_X(x) \lVert f \rVert\) is vacuous when \(f\) is outside the native space, and native spaces of smooth kernels are far smaller than intuition suggests; a Gaussian kernel's native space contains no function with a kink [@steinwart2008]. Check target smoothness before quoting spectral rates.
- Do not report accuracy without conditioning. By the uncertainty relation, a power function near machine precision forces \(\lambda_{\min}(K)\) near machine precision, and the printed solution of \(Kc = y\) may carry no correct digits. Report \(h_{X,\Omega}\), \(q_X\), and an estimate of \(\lambda_{\min}\) or the condition number together.
- Shrinking the shape parameter until the error on a test grid stops improving is a trap: the apparent optimum often sits where rounding error and truncated convergence cancel. Stable evaluations of the flat limit change basis instead of inverting \(K\) [@fasshauer2007].
- Rates are driven by the fill distance, not the sample size. Clustered designs waste points; near-uniform or greedy power-function designs achieve the same \(h\) with far fewer sites, at bounded mesh ratio.
- Conditionally positive definite kernels require their polynomial block. Dropping the moment conditions \(P^\top c = 0\) for thin-plate or multiquadric interpolation produces a system that may be singular and an interpolant without polynomial reproduction.
- The GP posterior standard deviation and the power function are the same formula, but their guarantees differ. Quoting a worst-case bound requires a norm estimate for the target; quoting a credible interval requires defending the prior. Choose one and say which.

## Summary and further reading {#approx-summary}

Noiseless kernel interpolation solves \(Kc = y\) and returns the unique minimum-norm interpolant in the native space, the limit of kernel ridge regression as regularization vanishes and the common core of kriging and spline interpolation. The native space of a translation-invariant kernel is a Fourier-weighted smoothness class: Sobolev for Matern kernels, a small analytic class for the Gaussian. The power function \(P_X(x)^2 = k(x,x) - k_X(x)^\top K^{-1} k_X(x)\) is the norm of the pointwise error functional, yields the sharp bound \(\lvert f(x) - s_f(x) \rvert \le P_X(x) \lVert f \rVert\), and coincides with the noise-free GP posterior standard deviation, so worst-case and average-case analyses share their formulas while differing in guarantees. Fill distance drives convergence, with rate \(h^{\tau - d/2}\) in the sup norm for smoothness \(\tau\) and spectral rates for the Gaussian; separation radius drives stability, and the Schur-complement uncertainty relation \(P_X(x)^2 \ge \lambda_{\min}(K')\) shows accuracy and conditioning cannot be had together. Sampling inequalities extend the theory to targets rougher than the native space at their own natural rate, and Schoenberg's completely monotone characterization, with its conditionally positive definite extension, settles which radial functions interpolate legitimately.

For further reading, Wendland's monograph [@wendland2005] is the standard rigorous account of native spaces, error estimates, and stability; Fasshauer [@fasshauer2007] gives a computational treatment with extensive shape-parameter practice. Schaback's trade-off principle appears in his 1995 paper [@schaback1995], and the escape from the native space is due to Narcowich, Ward, and Wendland [@narcowich2005]. The classical roots are Schoenberg's 1938 characterization [@schoenberg1938], Micchelli's 1986 interpolation theorem [@micchelli1986], and Aronszajn's 1950 theory of reproducing kernels [@aronszajn1950]. Wahba [@wahba1990] connects splines, kriging, and Bayesian readings; Rasmussen and Williams [@rasmussen2006] give the GP side, and Steinwart and Christmann [@steinwart2008] the fine structure of Gaussian RKHSs; Caponnetto and De Vito [@caponnetto2007] develop the noisy statistical counterpart of these rates.

::: {.exercises}
## Exercises {#exercises}

1.  [warm-up]{.ex-tag} Show directly from the closed form \(P_X(x)^2 = k(x,x) - k_X(x)^\top K^{-1} k_X(x)\) that \(P_X(x_i) = 0\) at every site \(x_i \in X\) and that \(0 \le P_X(x)^2 \le k(x,x)\) everywhere. Interpret both facts in the Gaussian-process reading of the power function.
2.  [computation]{.ex-tag} In the setting of the worked example (min kernel, sites \(\{0.25, 0.5, 1\}\), target \(f(x) = x^2\)), take the test point \(x_* = 0.375\). Compute \(k_X(x_*)\), solve \(Kw = k_X(x_*)\), and evaluate \(P_X(0.375)^2\) and \(P_X(0.375)\). Then compute \(s(0.375)\), the true value \(f(0.375)\), and verify that the error respects the bound \(P_X(0.375)\, \lVert f \rVert\) with \(\lVert f \rVert = 1.1547\).
3.  [proof]{.ex-tag} Derive the power-function formula. Starting from the cardinal form \(s_f(x) = \sum_j w_j(x) f(x_j)\) with \(w(x) = K^{-1} k_X(x)\), show that the error functional \(f \mapsto f(x) - s_f(x)\) is represented by \(r_x = k(\cdot, x) - \sum_j w_j(x) k(\cdot, x_j)\), expand \(\lVert r_x \rVert_{\mathcal H_k}^2\), and simplify to \(k(x,x) - k_X(x)^\top K^{-1} k_X(x)\). Conclude the sharp bound \(\lvert f(x) - s_f(x) \rvert \le P_X(x) \lVert f \rVert_{\mathcal H_k}\).
    Hint

    ::: hint-body
    The expansion gives \(k(x,x) - 2 w^\top k_X(x) + w^\top K w\); substitute \(w = K^{-1} k_X(x)\) so the last two terms combine into \(-k_X(x)^\top K^{-1} k_X(x)\). Sharpness follows by taking \(f = r_x / \lVert r_x \rVert\) in Cauchy-Schwarz.
    :::
4.  [proof]{.ex-tag} Let \(c_\lambda = (K + \lambda I)^{-1} y\) be the kernel ridge coefficients and \(c_0 = K^{-1} y\) the interpolation coefficients. Using the eigendecomposition of \(K\), show that \(\lVert c_\lambda - c_0 \rVert \le \dfrac{\lambda}{\lambda_{\min}(K)} \lVert c_0 \rVert\), so the ridge solution converges to the interpolant as \(\lambda \to 0^+\), and explain why the convergence is slow exactly when the uncertainty relation says the site set is accurate.
5.  [synthesis]{.ex-tag} Write down the posterior mean and posterior variance at a point \(x\) of a Gaussian process with covariance \(k\) conditioned on noise-free observations \(y_i = f(x_i)\). Verify that the mean is the kernel interpolant and the variance is \(P_X(x)^2\). Then explain carefully what each framework must assume for its error statement to be valid, and give one scenario in which the worst-case statement holds while the Bayesian one is misleading, or conversely.
    Hint

    ::: hint-body
    The noise-free posterior at \(x\) is \(\mathcal N\big(k_X(x)^\top K^{-1} y,\; k(x,x) - k_X(x)^\top K^{-1} k_X(x)\big)\). For the divergence of guarantees, consider a target in the native space whose norm is known (worst case valid) but that is atypical under the prior, or a prior that is well calibrated while no useful norm bound on \(f\) is available.
    :::
6.  [exploration]{.ex-tag} Take the min kernel and the two-point clustered site set \(\{0.5, 0.5 + \delta\}\) with \(\delta = 0.01\), so \(K = \begin{pmatrix} 0.5 & 0.5 \\ 0.5 & 0.51 \end{pmatrix}\). Compute the trace, the determinant, both eigenvalues to four decimals, and the condition number. Repeat symbolically for general small \(\delta\) and describe how \(\lambda_{\min}\) and the condition number scale with the separation radius. Relate the finding to the uncertainty relation: what does the near-singularity of \(K\) say about the power function of the first point relative to the second?
7.  [challenge]{.ex-tag} Prove that for the min kernel on \([0,1]\), between two adjacent sites \(a \lt x \lt b\) the power function satisfies \(P_X(x)^2 = \dfrac{(x - a)(b - x)}{b - a}\), depending only on the two neighbors. Locate the point of maximal uncertainty on each subinterval, verify that the formula reproduces \(P_X(0.75)^2 = 0.125\) and \(P_X(0.375)^2 = 0.0625\) from the example and exercise 2, and explain the Markov-property reason the other sites drop out.
    Hint

    ::: hint-body
    The representer \(r_x\) vanishes at all sites, equals a piecewise-linear tent between \(a\) and \(b\) with value \(P_X(x)^2\) scaling at \(x\), and its squared norm is the Dirichlet energy \(\int (r_x')^2\). Alternatively condition Brownian motion on its values at \(a\) and \(b\): the Brownian bridge variance at \(x\) is exactly \((x-a)(b-x)/(b-a)\).
    :::
8.  [exploration]{.ex-tag} With the Gaussian kernel \(k(x, y) = e^{-\epsilon^2 (x - y)^2}\) on the two sites \(\{0, 1\}\), the Gram matrix is \(\begin{pmatrix} 1 & e^{-\epsilon^2} \\ e^{-\epsilon^2} & 1 \end{pmatrix}\) with eigenvalues \(1 \pm e^{-\epsilon^2}\). Compute the condition number for \(\epsilon = 3\) and for \(\epsilon = 0.1\) to four decimals. Which regime is better conditioned, which regime uses the data more efficiently for a smooth target, and how does the pair illustrate the shape-parameter trade-off of the uncertainty principle?
:::
