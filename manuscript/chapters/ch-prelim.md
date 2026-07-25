---
id: ch-prelim
slug: preliminaries
title: Mathematical Preliminaries
part: Reference · Mathematical Preliminaries
order: 0
tier: core
prerequisites: []
objectives:
  - >-
    Diagnose positive semidefiniteness, rank, conditioning, and spectral
    structure.
  - >-
    Use Hilbert-space projections, bounded operators, and compact spectral
    theory.
  - >-
    Work with measure-theoretic and Hilbert-valued expectations used by
    embeddings.
  - >-
    Apply matrix concentration to empirical covariance and Gram approximations
    with explicit normalization.
  - >-
    Apply subgradients, Fenchel duality, KKT conditions, and convergence
    assumptions.
  - >-
    Recognize the graph, manifold, group, tensor, path, and Krein structures
    used later.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-prelim.yml
verification_date: null
bibliography:
  - scholkopf2002
  - shawe2004
---
# Mathematical Preliminaries

<p class="lead">Kernel methods sit at a crossroads of linear algebra, real and functional analysis, probability, and convex optimization, and the book's later parts reach further, into spectral graph theory, groups and their actions, couplings of measures, the tensor calculus of paths, and inner products without positivity. This chapter gathers, in one place, the results the rest of the book uses by name. It is a reference, not a course: each section states the definitions and facts a later chapter leans on, with just enough intuition to make them usable. Read it straight through for a refresher, or jump to a result when a chapter cites it.</p>

## Linear algebra and matrices {#linear-algebra}

The Gram matrix is the object every kernel algorithm actually computes with, so the linear algebra of symmetric matrices is the bedrock of the whole subject. We work in \(\mathbb{R}^d\) with the standard inner product \(\langle x, y\rangle = x^\top y = \sum_{i=1}^d x_i y_i\) and Euclidean norm \(\lVert x\rVert = \sqrt{x^\top x}\).

::: {.definition #def-p-1}
[Symmetric and positive semidefinite matrices]{.box-title}

A real matrix \(A \in \mathbb{R}^{n\times n}\) is **symmetric** if \(A = A^\top\). A symmetric \(A\) is **positive semidefinite** (PSD), written \(A \succeq 0\), if \(v^\top A v \geq 0\) for every \(v \in \mathbb{R}^n\); it is **strictly positive definite** (or just positive definite in the matrix sense), written \(A \succ 0\), if the inequality is strict, \(v^\top A v \gt 0\) for every \(v \neq 0\). The two are genuinely different properties: a PSD matrix may be singular, while a strictly positive definite one is invertible.

A word of caution on terminology, because the two adjacent uses of \"positive definite\" are a standing source of confusion. In the kernel-methods literature we follow Aronszajn's convention: a **positive definite kernel** is one whose every Gram matrix \(K_{ij} = k(x_i,x_j)\) is positive *semi*definite (\(K \succeq 0\)). So the phrase \"positive definite kernel,\" used throughout this book, produces PSD Gram matrices, not strictly positive definite ones. When we need the stronger matrix property, that a particular Gram matrix is invertible, we say **strictly positive definite** explicitly. Positive semidefiniteness is thus the matrix shadow of a valid kernel: a kernel is positive definite (Aronszajn) exactly when every Gram matrix it produces is PSD.
:::

:::: {.theorem #thm-p-2}
[Spectral theorem (symmetric case)]{.box-title}

Every symmetric \(A \in \mathbb{R}^{n\times n}\) has an orthonormal basis of eigenvectors: there is an orthogonal matrix \(U\) (so \(U^\top U = I\)) and a real diagonal matrix \(\Lambda = \operatorname{diag}(\lambda_1,\dots,\lambda_n)\) with

$$ A = U\Lambda U^\top = \sum_{k=1}^n \lambda_k\, u_k u_k^\top . $$

The eigenvalues \(\lambda_k\) are real. \(A \succeq 0\) iff all \(\lambda_k \geq 0\), and \(A \succ 0\) iff all \(\lambda_k \gt 0\). This decomposition is the finite-dimensional ancestor of Mercer's theorem: the eigenvectors are the discrete eigenfunctions, and \(\sqrt{\lambda_k}\, u_k\) gives an explicit feature map.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
::::

Several consequences recur. A PSD matrix has a unique PSD **square root** \(A^{1/2} = U\Lambda^{1/2}U^\top\), and it factors as \(A = B^\top B\) (take \(B = \Lambda^{1/2}U^\top\)); conversely any \(B^\top B\) is PSD, which is why a Gram matrix \(K = \Phi\Phi^\top\) is always PSD. The **trace** \(\operatorname{tr}(A) = \sum_i A_{ii} = \sum_k \lambda_k\) is the sum of eigenvalues and is cyclic, \(\operatorname{tr}(AB) = \operatorname{tr}(BA)\); the **determinant** \(\det(A) = \prod_k \lambda_k\). The **rank** is the number of nonzero eigenvalues. For the eigenvalue extremes, the **Rayleigh quotient** gives

$$ \lambda_{\max}(A) = \max_{v \neq 0} \frac{v^\top A v}{v^\top v}, \qquad \lambda_{\min}(A) = \min_{v \neq 0} \frac{v^\top A v}{v^\top v}, $$

and the higher eigenvalues follow the Courant-Fischer min-max principle. Maximizing a Rayleigh quotient is exactly what kernel PCA and spectral clustering do.

::: {.definition #def-p-3}
[Singular value decomposition and pseudo-inverse]{.box-title}

Any \(M \in \mathbb{R}^{m\times n}\) factors as \(M = U\Sigma V^\top\) with \(U, V\) orthogonal and \(\Sigma\) diagonal with nonnegative **singular values** \(\sigma_1 \geq \dots \geq 0\). When \(M\) is not invertible, the **Moore-Penrose pseudo-inverse** \(M^{+} = V\Sigma^{+}U^\top\) (inverting the nonzero singular values) gives the minimum-norm least-squares solution of \(Mx = b\); it is what \"solving\" a rank-deficient normal equation means.
:::

Numerical solvability is not the same as algebraic invertibility. For a strictly positive definite symmetric matrix, the spectral **condition number** is \(\kappa_2(A)=\lambda_{\max}(A)/\lambda_{\min}(A)\). Relative perturbations can be amplified by roughly this factor. Kernel matrices with repeated or nearly repeated observations can be positive definite yet numerically unusable. Prefer a Cholesky or QR solve to forming \(A^{-1}\), add a documented ridge or jitter only when the model permits it, and report the residual in the original system.

The spectrum makes the conditioning problem visible. In the next plate, one direction is a million times weaker than the leading direction; adding \(\lambda I\) shifts every eigenvalue by the same amount, barely moving the strong directions while lifting the weak one. This improves the solve, but it also changes the estimator, so the numerical benefit and the statistical bias must be reported together.

<figure class="viz" data-figure="conditioning-clinic" data-alt="Two logarithmic bar charts compare four Gram-matrix eigenvalues before and after adding a ridge of 0.01. The smallest eigenvalue rises from one hundred-thousandth to roughly one hundredth, reducing the condition number from one million to one thousand."><figcaption>Ridge regularization stabilizes a Gram solve by lifting weak eigendirections: here \(\kappa_2\) falls from \(10^6\) to about \(10^3\), while the leading eigenvalues barely change. The same lift that repairs conditioning also suppresses information in those weak directions.</figcaption></figure>

For the displayed spectrum \((10,1,10^{-2},10^{-5})\), the raw condition number is \(10/10^{-5}=10^6\). Adding \(10^{-2}I\) changes it to \(10.01/0.01001=1000\). The calculation explains both the gain and the failure mode: a perturbation aligned with the weakest eigenvector is far less amplified, but a true signal component in that direction is also shrunk. A stable residual therefore does not prove that the regularized answer solves the original statistical problem.

Two identities save real computation in the book. The **matrix inversion lemma** (Sherman-Morrison-Woodbury),

$$ (A + UCV)^{-1} = A^{-1} - A^{-1}U\bigl(C^{-1} + VA^{-1}U\bigr)^{-1}VA^{-1}, $$

lets kernel ridge regression switch between an \(n\times n\) solve in dual variables and a \(d\times d\) solve in primal variables, whichever is smaller. The **Schur (Hadamard) product** \(A \circ B\), the entrywise product, preserves positive semidefiniteness: if \(A, B \succeq 0\) then \(A \circ B \succeq 0\). That single fact is why the product of two kernels is a kernel.

## Real analysis, metrics, and Hilbert spaces {#analysis}

Feature spaces are infinite-dimensional, so the finite intuition above has to survive a limit. The right setting is a Hilbert space, and the facts below are what make an infinite-dimensional \"space of features\" behave like Euclidean space.

::: {.definition #def-p-4}
[Metric and normed spaces; completeness]{.box-title}

A **metric** \(d\) on a set \(X\) is a nonnegative, symmetric function with \(d(x,y)=0 \iff x=y\) that satisfies the triangle inequality \(d(x,z) \leq d(x,y) + d(y,z)\). A **norm** \(\lVert\cdot\rVert\) on a vector space induces the metric \(d(x,y) = \lVert x-y\rVert\). A sequence is **Cauchy** if its terms get arbitrarily close to each other; a space is **complete** if every Cauchy sequence converges to a point of the space. Completeness is the property that rules out \"holes,\" and it is exactly the condition an inner-product space must add to become a Hilbert space.
:::

::: {.definition #def-p-5}
[Inner product and Hilbert space]{.box-title}

An **inner product** \(\langle\cdot,\cdot\rangle\) on a real vector space \(H\) is symmetric, linear in each argument, and positive: \(\langle f, f\rangle \geq 0\) with equality only at \(f = 0\). It induces the norm \(\lVert f\rVert = \sqrt{\langle f,f\rangle}\). A **Hilbert space** is an inner-product space that is complete in this norm. \(\mathbb{R}^d\) and the sequence space \(\ell^2\) of square-summable sequences are the model examples; a reproducing kernel Hilbert space is a Hilbert space of functions.
:::

Three inequalities are used constantly. **Cauchy-Schwarz**, \(\lvert\langle f, g\rangle\rvert \leq \lVert f\rVert\,\lVert g\rVert\), with equality iff \(f\) and \(g\) are parallel, is the workhorse behind nearly every bound in the book (it turns the reproducing property into the pointwise bound \(\lvert f(x)\rvert \leq \lVert f\rVert\sqrt{k(x,x)}\)). The **triangle inequality** \(\lVert f+g\rVert \leq \lVert f\rVert + \lVert g\rVert\) follows from it. And the **Pythagorean identity** \(\lVert f+g\rVert^2 = \lVert f\rVert^2 + \lVert g\rVert^2\) when \(\langle f,g\rangle = 0\) drives the orthogonal-decomposition proof of the representer theorem. All three lean on positivity of the inner product; the section on [indefinite inner products](#indefinite-inner-products) at the end of this chapter records what breaks when positivity is dropped.

::: {.theorem #thm-p-6}
[Orthogonal projection and the Riesz representation theorem]{.box-title}

In a Hilbert space, any closed subspace \(V\) admits a unique **orthogonal projection**: every \(f\) splits as \(f = f_V + f_\perp\) with \(f_V \in V\), \(f_\perp \perp V\), and \(f_V\) the closest point of \(V\) to \(f\). The **Riesz representation theorem** says every continuous linear functional \(L: H \to \mathbb{R}\) is an inner product against a unique vector: \(L(f) = \langle f, g_L\rangle\) for some \(g_L \in H\). Applied to the evaluation functional \(f \mapsto f(x)\), Riesz produces the reproducing kernel; that is the definitional route to an RKHS.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
:::

A few pointwise-analysis notions round this out. A function is **continuous** if small input changes give small output changes, and **Lipschitz** with constant \(L\) if \(\lvert f(x) - f(y)\rvert \leq L\,d(x,y)\). For an RKHS function the reproducing property and Cauchy-Schwarz give the exact bound \(\lvert f(x) - f(x')\rvert \leq \lVert f\rVert_{\mathcal H}\, d_k(x,x')\), where \(d_k(x,x')^2 = k(x,x) - 2k(x,x') + k(x',x')\) is the metric induced by the kernel's own feature map. This is automatic and needs no smoothness of \(k\). Turning it into Lipschitz-ness in the ordinary metric \(d(x,x')\), so that \(\lVert f\rVert_{\mathcal H}\) bounds a genuine Lipschitz constant, needs a further hypothesis on the kernel: that \(k\) is smooth enough (for instance differentiable with bounded gradient) that \(d_k(x,x') \leq C\, d(x,x')\). Under that hypothesis the RKHS norm bounds a Lipschitz constant, which is the precise sense in which \"small norm means smooth.\" A set is **compact** if every sequence has a convergent subsequence (on \(\mathbb{R}^d\), compact means closed and bounded); compactness of the domain is the hypothesis under which Mercer's theorem holds. A basis \((e_k)\) of a Hilbert space is **orthonormal** if \(\langle e_i, e_j\rangle = \delta_{ij}\), and then \(f = \sum_k \langle f, e_k\rangle e_k\) with \(\lVert f\rVert^2 = \sum_k \langle f, e_k\rangle^2\) (Parseval); the eigenfunction expansion of Mercer's theorem is one such basis.

## Measure, integration, and Hilbert-valued expectation {#measure-integration}

Once a kernel mean or a distribution embedding enters the picture, expectations are taken over abstract sample spaces and land in a Hilbert space, so the informal calculus of averages needs a measure-theoretic footing. A probability space is a triple \((\Omega,\mathcal{F},P)\): \(\mathcal{F}\) is a sigma-algebra of events and \(P\) is a countably additive measure with total mass one. A random variable is a measurable map. Statements that hold **almost everywhere** may fail on a set of measure zero. The spaces \(L^p(P)\) identify functions that agree almost everywhere and satisfy \(\int|f|^p\,dP\lt\infty\); conditional expectations live in \(L^2\), while kernel mean existence begins with an \(L^1\) condition.

Three exchange rules recur. Tonelli permits swapping integrals of a nonnegative measurable function. Fubini permits the same for an absolutely integrable function. Dominated convergence permits moving a limit through an integral when the sequence converges almost everywhere and is bounded by an integrable envelope. Each rule has a hypothesis; expectation notation alone does not make an interchange legal.

::: {.definition #def-bochner-integral}
[Bochner integrability]{.box-title}

Let \(Z:\Omega\to\mathcal{H}\) take values in a separable Hilbert space. If \(Z\) is strongly measurable and \(\mathbb{E}\lVert Z\rVert_{\mathcal H}\lt\infty\), its **Bochner expectation** \(\mathbb{E}Z\in\mathcal{H}\) is the norm limit of expectations of simple functions. It satisfies

$$
\left\lVert\mathbb{E}Z\right\rVert_{\mathcal H}\le\mathbb{E}\lVert Z\rVert_{\mathcal H},
\qquad A(\mathbb{E}Z)=\mathbb{E}[AZ]
$$

for every bounded linear operator \(A\). For \(Z=k(X,\cdot)\), a sufficient condition is \(\mathbb{E}\sqrt{k(X,X)}\lt\infty\), and \(\mathbb{E}Z\) is the kernel mean embedding.
:::

## Operators, spectra, and Fourier analysis {#functional-analysis}

Mercer's theorem and the translation-invariant kernels developed across Parts IV and VI need a little operator theory and harmonic analysis. The finite spectral theorem generalizes to certain operators on function spaces, and Fourier analysis is what characterizes which functions are valid kernels.

::: {.definition #def-p-7}
[Bounded, compact, self-adjoint, positive operators]{.box-title}

A linear operator \(T: H \to H\) is **bounded** if \(\lVert Tf\rVert \leq C\lVert f\rVert\) for some \(C\); **self-adjoint** if \(\langle Tf, g\rangle = \langle f, Tg\rangle\) (the operator analogue of a symmetric matrix); **positive** if \(\langle Tf, f\rangle \geq 0\); and **compact** if it maps bounded sets to sets whose closure is compact. Compactness is the infinite-dimensional substitute for finite rank: it is what forces a discrete spectrum.
:::

::: {.theorem #thm-p-8}
[Spectral theorem for compact self-adjoint operators]{.box-title}

A compact self-adjoint operator \(T\) on a Hilbert space has a countable orthonormal system of eigenfunctions \((\psi_k)\) with real eigenvalues \(\lambda_k \to 0\), and \(Tf = \sum_k \lambda_k \langle f, \psi_k\rangle \psi_k\). The **integral operator** \((T_k f)(x) = \int k(x, y) f(y)\, d\nu(y)\) of a continuous PSD kernel on a compact domain is compact, self-adjoint, and positive, so this theorem applies; its eigenvalues and eigenfunctions are precisely the pieces of Mercer's decomposition \(k(x,y) = \sum_k \lambda_k \psi_k(x)\psi_k(y)\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
:::

For kernels that depend only on the difference \(x - y\), the relevant spectrum is the Fourier one. The **Fourier transform** of an integrable function \(f\) on \(\mathbb{R}^d\) is \(\hat f(\omega) = \int f(x)\, e^{-i\omega^\top x}\, dx\); it diagonalizes convolution and translation. A finite nonnegative measure has a Fourier transform that is a positive definite function, and **Bochner's theorem** is the converse: a continuous translation-invariant function \(k(x,y) = \kappa(x-y)\) is positive definite exactly when \(\kappa\) is the Fourier transform of a finite nonnegative measure. That characterization is the source of the Gaussian and Laplace kernels and the starting point for random Fourier features.

## Probability and statistics {#probability}

Learning is estimation from samples, so probability supplies both the objects (distributions, expectations) and the guarantees (concentration). We treat data as independent draws from an unknown distribution and ask what can be inferred.

::: {.definition #def-p-9}
[Distributions, expectation, and independence]{.box-title}

A random variable \(X\) has a distribution \(P\); its **expectation** is \(\mathbb{E}[X] = \int x\, dP(x)\), a linear operator, and its **variance** is \(\operatorname{Var}(X) = \mathbb{E}[(X - \mathbb{E}X)^2]\). Variables are **independent** if their joint distribution factors; a sample is **i.i.d.** when its points are independent and identically distributed. The kernel mean embedding \(\mu_P = \mathbb{E}_{X\sim P}[k(\cdot, X)]\) of Part VII is just an expectation taken in an RKHS, and the maximum mean discrepancy compares two such embeddings.
:::

Two families of results are used to turn samples into guarantees. The **laws of large numbers** say the empirical average \(\frac1n\sum_i X_i\) converges to \(\mathbb{E}[X]\) as \(n\to\infty\), and the **central limit theorem** says its fluctuations are Gaussian of size \(1/\sqrt n\); this \(1/\sqrt n\) rate is the one that appears in every generalization bound. Sharper, non-asymptotic control comes from **concentration inequalities**. **Hoeffding's inequality** bounds the deviation of a bounded average, and its generalization, **McDiarmid's bounded-differences inequality**, bounds any function of independent variables that changes little when one variable is altered:

$$ \Pr\bigl(\,f(X_1,\dots,X_n) - \mathbb{E}f \geq t\,\bigr) \leq \exp\!\Bigl(\frac{-2t^2}{\sum_i c_i^2}\Bigr), $$

where \(c_i\) bounds the change from moving coordinate \(i\). McDiarmid applied to the worst-case empirical error is the engine behind the Rademacher bounds of [[ch:learning-theory]]. A last modeling link: minimizing a loss is often maximum-likelihood estimation in disguise, since a loss \(\ell(y, f)\) equal to \(-\log p(y \mid f)\) makes empirical risk minimization the same as fitting a probabilistic noise model, the reading that connects the squared loss to Gaussian noise and the logistic loss to a Bernoulli model.

This section covers what empirical risk minimization needs. The inference chapters ask for more, and the deeper material, Gaussian conditioning, Bayes' rule, score functions, U-statistics, and a precise statement of Hoeffding's inequality, is collected in the section on [conditioning, scores, and U-statistics](#probability-for-inference) below.

### Matrix concentration for kernel approximations {#matrix-concentration}

Scalar concentration controls one test direction at a time. A kernel approximation usually has to preserve every direction at once: an empirical covariance \(\widehat C=n^{-1}\sum_i z_i z_i^\top\) should be close to \(C=\mathbb E[z z^\top]\), or a random-feature Gram matrix should approximate its expectation in spectral norm. A union bound over entries obscures the geometry and can lose factors of dimension. Matrix concentration works directly with eigenvalues and operator norms.

::: {.definition #def-p-matrix-variance}
[Matrix variance and intrinsic dimension]{.box-title}

For independent, mean-zero, self-adjoint random matrices \(Y_1,\ldots,Y_n\in\mathbb R^{d\times d}\), define

$$
V=\sum_{i=1}^n\mathbb E[Y_i^2],
\qquad
v=\lVert V\rVert_2.
$$

The scalar \(v\) is the **matrix variance proxy**. When \(v\gt0\), the ratio

$$
r(V)=\frac{\operatorname{tr}(V)}{\lVert V\rVert_2}
$$

is its **intrinsic dimension** or stable rank. It lies in \([1,d]\) and counts how many eigendirections carry substantial variance. This is a variance-side notion. The regularized effective dimension \(\operatorname{tr}(K(K+\lambda I)^{-1})\) used later in [[ch:mercer-and-rates]] and [[ch:random-features-sketches-and-randomized-kernel-linear-algebra]] is a different, regularization-dependent quantity, although both replace ambient dimension by occupied spectral dimension.
:::

::: {.theorem #thm-p-matrix-bernstein}
[Self-adjoint matrix Bernstein inequality]{.box-title}

Let \(Y_1,\ldots,Y_n\in\mathbb R^{d\times d}\) be independent, self-adjoint random matrices satisfying

$$
\mathbb E Y_i=0
\qquad\text{and}\qquad
\lVert Y_i\rVert_2\leq L
\quad\text{almost surely}
$$

for every \(i\). With \(v=\lVert\sum_i\mathbb E[Y_i^2]\rVert_2\), for every \(t\geq0\),

$$
\Pr\!\left(
\left\lVert\sum_{i=1}^nY_i\right\rVert_2\geq t
\right)
\leq
2d\exp\!\left(
-\frac{t^2}{2(v+Lt/3)}
\right).
$$

Equivalently, with probability at least \(1-\delta\),

$$
\left\lVert\sum_{i=1}^nY_i\right\rVert_2
\leq
\sqrt{2v\log\!\frac{2d}{\delta}}
+\frac{2L}{3}\log\!\frac{2d}{\delta}.
$$

**Assumptions.** The matrices have fixed finite dimension, are independent and self-adjoint, are centered individually, and obey the same almost-sure spectral-norm bound \(L\). The variance uses the unnormalized sum. If the target is an average, put the factor \(1/n\) inside each \(Y_i\); then typically \(L=O(n^{-1})\) and \(v=O(n^{-1})\).

**Proof status.** Standard matrix Laplace-transform result; no proof is reproduced here. The displayed two-sided norm bound follows by controlling the maximum eigenvalues of both \(\sum_iY_i\) and \(-\sum_iY_i\).
:::

The leading term is the familiar square-root variance scale, while the linear term controls rare large summands. Refined Bernstein inequalities can replace the ambient factor \(d\) by a constant multiple of \(r(V)\), subject to the precise threshold and one-sided/two-sided convention of the chosen version. This matters for kernels whose covariance lives near a low-dimensional subspace. It does not license replacing \(d\) by any convenient effective dimension: the dimension factor must be derived from the same variance or regularized operator used in the theorem.

For positive random matrices there is a sharper multiplicative question: does sampling preserve the spectrum relative to its expectation?

::: {.theorem #thm-p-matrix-chernoff}
[Matrix Chernoff inequality]{.box-title}

Let \(X_1,\ldots,X_n\in\mathbb R^{d\times d}\) be independent, self-adjoint random matrices with

$$
0\preceq X_i\preceq L I
\quad\text{almost surely}.
$$

Write \(\mu_{\min}=\lambda_{\min}(\sum_i\mathbb E X_i)\) and \(\mu_{\max}=\lambda_{\max}(\sum_i\mathbb E X_i)\). For \(0\leq\varepsilon\lt1\),

$$
\Pr\!\left\{
\lambda_{\min}\!\left(\sum_iX_i\right)
\leq(1-\varepsilon)\mu_{\min}
\right\}
\leq
d\exp\!\left(-\frac{\varepsilon^2\mu_{\min}}{2L}\right),
$$

and, for \(\varepsilon\geq0\),

$$
\Pr\!\left\{
\lambda_{\max}\!\left(\sum_iX_i\right)
\geq(1+\varepsilon)\mu_{\max}
\right\}
\leq
d\left(\frac{e^\varepsilon}{(1+\varepsilon)^{1+\varepsilon}}\right)^{\mu_{\max}/L}.
$$

**Assumptions.** Independence, finite-dimensional self-adjointness, and the Loewner bound \(0\preceq X_i\preceq LI\) are essential. The theorem controls the unnormalized sum. For an empirical average, either divide the conclusion by \(n\) or define \(X_i/n\) as the summands.

**Proof status.** Standard matrix Laplace-transform result; no proof is reproduced here.
:::

::: {.example #example-prelim-matrix-concentration}
[A two-direction covariance fixture]{.box-title}

Let \(z\) equal \(\sqrt2e_1\) or \(\sqrt2e_2\), each with probability \(1/2\). Then \(C=\mathbb E[zz^\top]=I_2\). For \(n=4\), the balanced deterministic sample \((e_1,e_1,e_2,e_2)\) after the same \(\sqrt2\) scaling gives \(\widehat C=I_2\). A \(3{:}1\) split gives

$$
\widehat C
=
\begin{pmatrix}
3/2&0\\
0&1/2
\end{pmatrix},
\qquad
\lVert\widehat C-C\rVert_2=\frac12.
$$

To use Bernstein, set \(Y_i=(z_i z_i^\top-I_2)/n\). Then \(\mathbb EY_i=0\), \(\lVert Y_i\rVert_2=1/n\), \(Y_i^2=I_2/n^2\), and therefore \(L=1/n\), \(V=I_2/n\), \(v=1/n\), and \(r(V)=2\). The calculation checks every normalization by hand. At \(n=4\) and \(t=1/2\), the probability upper bound exceeds one and is therefore vacuous; concentration becomes informative only as \(n\) grows. A correct theorem need not be a sharp small-sample certificate.

**Verification artifact.** checks/example-ch-prelim-example-prelim-matrix-concentration.json records the example source hash and verification scope.
:::

This module is the probabilistic bridge to regularized spectral approximation in [[ch:random-features-sketches-and-randomized-kernel-linear-algebra]] and to Nyström and iterative scaling methods in [[ch:large-scale-kernels]]. In each use, first identify the random self-adjoint summand, then its expectation, spectral bound, variance proxy, normalization, and the exact matrix quantity the downstream algorithm needs.

## Convex optimization and duality {#optimization}

Fitting a kernel machine is solving an optimization problem, and almost always a convex one, which is why the solutions are unique and computable. The vocabulary of convexity and Lagrangian duality is used throughout Parts II and VII.

::: {.definition #def-p-10}
[Convex sets, convex functions, and gradients]{.box-title}

A set is **convex** if it contains the segment between any two of its points. A function \(f\) is **convex** if its graph lies below its chords, \(f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)\); equivalently, for differentiable \(f\), the graph lies above every tangent, \(f(y) \geq f(x) + \nabla f(x)^\top(y-x)\). A convex function has no local minima other than global ones, so gradient methods find the optimum. The **gradient** \(\nabla f\) is the vector of partial derivatives; where \(f\) is not differentiable, a **subgradient** is any \(g\) with \(f(y) \geq f(x) + g^\top(y-x)\), which is how the hinge loss is handled at its kink.
:::

:::::: {.definition #def-p-11}
[Lagrangian duality and the KKT conditions]{.box-title}

For a constrained problem \(\min_x f(x)\) subject to inequality constraints \(g_i(x) \leq 0\) and equality constraints \(h_j(x) = 0\), the **Lagrangian** is \(L(x, \alpha, \nu) = f(x) + \sum_i \alpha_i g_i(x) + \sum_j \nu_j h_j(x)\), where the inequality multipliers satisfy \(\alpha_i \geq 0\) while the equality multipliers \(\nu_j\) are unrestricted in sign. The **dual function** \(q(\alpha, \nu) = \min_x L(x,\alpha,\nu)\) lower-bounds the optimum (weak duality), and for convex problems meeting a regularity condition the bound is tight (strong duality). At an optimum the **Karush-Kuhn-Tucker (KKT) conditions** hold:

$$ \nabla f(x) + \sum_i \alpha_i \nabla g_i(x) + \sum_j \nu_j \nabla h_j(x) = 0 \quad\text{(stationarity)}, $$

$$ g_i(x) \leq 0, \quad h_j(x) = 0 \quad\text{(primal feasibility)}, $$

$$ \alpha_i \geq 0 \quad\text{(dual feasibility)}, \qquad \alpha_i\, g_i(x) = 0 \quad\text{(complementary slackness)}. $$

Complementary slackness is the operative one for kernel machines: the SVM's support vectors are exactly the points whose inequality multiplier \(\alpha_i\) is nonzero, read straight off this condition.
::::::

The algorithms that solve these problems at scale are covered in [[ch:large-scale-kernels]], but their basics belong here. **Gradient descent** steps against the gradient, \(x_{t+1} = x_t - \eta\nabla f(x_t)\), and converges on convex, smooth objectives; **stochastic gradient descent** replaces the full gradient with a cheap unbiased estimate from one example or a minibatch, trading exactness for a per-step cost independent of \(n\). Smoothness (a Lipschitz gradient) and strong convexity (a curvature lower bound) are the two properties that set the convergence rate. Finally, a **quadratic program**, minimizing a convex quadratic subject to linear constraints, is the exact form of the SVM dual, and interior-point and decomposition methods are how it is solved.

The **Fenchel conjugate** of a proper convex function \(f\) is \(f^*(u)=\sup_x\{\langle u,x\rangle-f(x)\}\). Fenchel-Young gives \(f(x)+f^*(u)\ge\langle u,x\rangle\), with equality exactly when \(u\in\partial f(x)\). This is the compact route from a primal loss to its dual constraint. Strong Fenchel duality still needs a qualification condition; convexity alone does not erase a duality gap.

## More probability: conditioning, scores, and U-statistics {#probability-for-inference}

The distributional chapters of the book do inference, not only estimation: they condition one Gaussian block on another, differentiate log-densities they cannot normalize, and average symmetric functions over pairs of samples. The five facts below are their toolkit.

::::: {.theorem #thm-p-12}
[Conditioning a joint Gaussian]{.box-title}

Partition a Gaussian vector as \(x = (x_1, x_2) \sim \mathcal{N}(\mu, \Sigma)\) with

$$ \mu = \begin{pmatrix} \mu_1 \\ \mu_2 \end{pmatrix}, \qquad \Sigma = \begin{pmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{pmatrix}, \qquad \Sigma_{22} \succ 0 . $$

Then conditionally on \(x_2\), the block \(x_1\) is again Gaussian, with

$$ \mathbb{E}[x_1 \mid x_2] = \mu_1 + \Sigma_{12}\, \Sigma_{22}^{-1} (x_2 - \mu_2), \qquad \operatorname{Cov}(x_1 \mid x_2) = \Sigma_{11} - \Sigma_{12}\, \Sigma_{22}^{-1} \Sigma_{21} . $$

The conditional mean is affine in the observed block, and the conditional covariance is the **Schur complement** of \(\Sigma_{22}\) in \(\Sigma\); it does not depend on the observed value at all.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
:::::

Read with \(\Sigma\) a kernel Gram matrix over training and test points, this theorem *is* Gaussian process regression: the posterior mean is a linear smoother in the observations and the posterior variance is a Schur complement, which is how [[ch:gaussian-processes-and-rvm]] derives its predictive equations and how the acquisition rules of [[ch:bayesian-optimization-and-bandits]] know where they are uncertain. The Schur complement is also the block that appears when the matrix inversion lemma of the linear-algebra section inverts a partitioned matrix.

:::: {.definition #def-p-13}
[Bayes' rule and conditional expectation]{.box-title}

For random variables \(X, Y\) with joint density \(p(x, y)\), the **conditional density** of \(y\) given \(x\) is \(p(y \mid x) = p(x, y) / p(x)\) wherever \(p(x) \gt 0\), and **Bayes' rule** reverses the conditioning,

$$ p(y \mid x) = \frac{p(x \mid y)\, p(y)}{\int p(x \mid y')\, p(y')\, dy'} . $$

The **conditional expectation** \(\mathbb{E}[f(Y) \mid X = x] = \int f(y)\, p(y \mid x)\, dy\) is, as a function of \(x\), the best mean-square predictor of \(f(Y)\): it minimizes \(\mathbb{E}[(f(Y) - g(X))^2]\) over all measurable \(g\), and it obeys the tower property \(\mathbb{E}\bigl[\mathbb{E}[f(Y) \mid X]\bigr] = \mathbb{E}[f(Y)]\).
::::

The conditional mean embeddings of [[ch:conditional-mean-embeddings]] represent the map \(x \mapsto \mathbb{E}[\varphi(Y) \mid X = x]\) as an RKHS-valued regression and turn Bayes' rule into linear algebra on Gram matrices; the adjustment formulas of [[ch:causal-inference-with-kernels]] are conditional expectations averaged over confounders.

::::: {.proposition #prop-p-14}
[The score function and integration by parts]{.box-title}

Let \(p\) be a continuously differentiable, everywhere positive density on \(\mathbb{R}^d\). Its **score function** is

$$ s_p(x) = \nabla_x \log p(x) = \frac{\nabla p(x)}{p(x)}, $$

which is unchanged if \(p\) is multiplied by a constant: the score is computable even when the normalizer is not. If \(f\) is differentiable, \(\mathbb{E}_p\lVert s_p(X) f(X)\rVert\) and \(\mathbb{E}_p\lVert \nabla f(X)\rVert\) are finite, and \(p(x) f(x) \to 0\) as \(\lVert x \rVert \to \infty\), then

$$ \mathbb{E}_p\bigl[\, s_p(X)\, f(X) + \nabla f(X) \,\bigr] = 0 . $$

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::::

::: {.proof}
[Proof]{.box-title}

In each coordinate \(i\), \(\mathbb{E}_p[\partial_i \log p \cdot f + \partial_i f] = \int \bigl( (\partial_i p)\, f + p\, \partial_i f \bigr)\, dx = \int \partial_i (p f)\, dx = 0\), the last step by integration by parts using the decay of \(p f\) at infinity. [\(\square\)]{.qed}
:::

The identity says the operator \(f \mapsto s_p f + \nabla f\) has mean zero under \(p\) for a whole class of test functions, a fingerprint of \(p\) that never touches its normalizing constant; running the test functions over an RKHS ball turns the largest violated mean into the computable discrepancy of [[ch:kernel-stein-discrepancy]].

:::: {.definition #def-p-15}
[U-statistics and V-statistics]{.box-title}

Let \(h(x, x')\) be a symmetric function, called the **kernel** of the statistic (an older usage, distinct from a positive definite kernel), and let \(X_1, \dots, X_n\) be i.i.d. from \(P\). The **U-statistic** and **V-statistic** with kernel \(h\) are

$$ U_n = \frac{1}{n(n-1)} \sum_{i \neq j} h(X_i, X_j), \qquad V_n = \frac{1}{n^2} \sum_{i, j=1}^{n} h(X_i, X_j) . $$

Every term of \(U_n\) has expectation \(\theta = \mathbb{E}[h(X, X')]\) for independent \(X, X' \sim P\), so the U-statistic is an **unbiased** estimator of \(\theta\) for every \(n\). The V-statistic keeps the diagonal terms \(h(X_i, X_i)\) and is biased by \(O(1/n)\), in exchange for being a plug-in: \(V_n\) is \(\theta\) computed under the empirical distribution. Kernels of order \(m\) are averaged over \(m\)-tuples of distinct indices in the same way.
::::

The squared maximum mean discrepancy of [[ch:kernel-mean-embeddings]] is estimated exactly this way, as a two-sample U- or V-statistic whose kernel is built from \(k\); the null distributions that calibrate the tests of [[ch:kernel-hypothesis-testing]] come from the degenerate case of U-statistic limit theory, where the usual Gaussian limit collapses and a weighted sum of \(\chi^2\) variables takes its place.

:::: {.theorem #thm-p-16}
[Hoeffding's inequality]{.box-title}

Let \(X_1, \dots, X_n\) be independent with \(X_i \in [a_i, b_i]\) almost surely, and let \(\bar X = \frac{1}{n} \sum_{i=1}^n X_i\). For every \(t \gt 0\),

$$ \Pr\bigl(\, \bar X - \mathbb{E}[\bar X] \geq t \,\bigr) \;\leq\; \exp\!\Bigl( \frac{-2 n^2 t^2}{\sum_{i=1}^{n} (b_i - a_i)^2} \Bigr), $$

and the same bound holds for the deviation below the mean. When all ranges equal \([a, b]\) the exponent is \(-2 n t^2 / (b - a)^2\): a bounded average deviates by more than \(t\) with probability exponentially small in \(n\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
::::

McDiarmid's bounded-differences inequality, displayed in the probability section above, generalizes this from averages to any stable function of independent variables (take \(f = \bar X\), so \(c_i = (b_i - a_i)/n\)); it is proved in [[ch:vc-theory-and-generalization]]. Hoeffding is the bound behind the single-function deviation estimates of [[ch:learning-theory]] and the sampling-error control of the randomized approximations in [[ch:large-scale-kernels]].

## Graphs, manifolds, and groups {#graphs-manifolds-groups}

Several chapters replace \(\mathbb{R}^d\) by a structured domain: a similarity graph built on the sample, a curved surface, or a space acted on by a symmetry group. One matrix and one averaging construction cover most of what they need.

:::: {.definition #def-p-17}
[The graph Laplacian]{.box-title}

Let \(W = (w_{ij})\) be symmetric nonnegative edge weights on vertices \(\{1, \dots, n\}\), with degrees \(d_i = \sum_j w_{ij}\) and degree matrix \(D = \operatorname{diag}(d_1, \dots, d_n)\). The **graph Laplacian** is \(L = D - W\). Its quadratic form is

$$ f^\top L f \;=\; \frac{1}{2} \sum_{i,j=1}^{n} w_{ij}\, (f_i - f_j)^2 \;\geq\; 0, $$

so \(L\) is symmetric PSD, and \(f^\top L f\) measures how much \(f\) varies across heavy edges. The constant vector satisfies \(L \mathbf{1} = 0\), and the multiplicity of the eigenvalue \(0\) is the number of connected components. The **normalized Laplacians** \(L_{\mathrm{sym}} = I - D^{-1/2} W D^{-1/2}\) and \(L_{\mathrm{rw}} = I - D^{-1} W\) (defined when all \(d_i \gt 0\)) rescale by degree and inherit the same structure.
::::

Eigenvectors of \(L\) with small eigenvalue are the smoothest functions on the graph, which is why spectral clustering in [[ch:kernel-clustering]] cuts along them; matrix functions of \(L\) such as the heat kernel \(e^{-tL}\) and the regularized inverse are the kernels on graphs of [[ch:graph-kernels]]; and \(L\) is the discrete stand-in for the manifold operator of the next paragraph in [[ch:geometric-and-equivariant-kernels]].

On a smooth compact Riemannian manifold the role of \(L\) is played by the **Laplace-Beltrami operator**, the divergence of the gradient computed in the manifold's own metric. It is self-adjoint and positive, its spectrum is discrete, and its eigenfunctions generalize the Fourier modes: on the circle they are the sines and cosines, on the sphere the spherical harmonics. Heat and Matérn kernels on manifolds are built from these eigenpairs in [[ch:geometric-and-equivariant-kernels]], and graph Laplacians estimated from samples approximate the operator, the reading behind the spectral embeddings of [[ch:data-visualization-and-mds]].

:::: {.definition #def-p-18}
[Group actions, invariance, and averaging]{.box-title}

A group \(G\) **acts** on a set \(X\) when each \(g \in G\) is assigned a transformation \(x \mapsto g \cdot x\) with \(e \cdot x = x\) and \(g \cdot (h \cdot x) = (gh) \cdot x\). A function \(f\) on \(X\) is **invariant** if \(f(g \cdot x) = f(x)\) for every \(g\) and \(x\). For a finite group, the **averaging operator**

$$ (\bar f)(x) \;=\; \frac{1}{\lvert G \rvert} \sum_{g \in G} f(g \cdot x) $$

always produces an invariant function, fixes every function that is already invariant, and, when the action preserves the underlying measure, is idempotent and self-adjoint: it is the orthogonal projection onto the subspace of invariant functions. For a compact group the sum becomes an integral against the **Haar measure**, the unique probability measure on \(G\) invariant under the group's own translations, and the projection statement is unchanged.
::::

Averaging a kernel over the action in each argument yields an invariant kernel whose RKHS consists of invariant functions, the construction at the center of [[ch:geometric-and-equivariant-kernels]]; the virtual-sample and jittered kernels of [[ch:invariances-and-pre-images]] are this projection applied approximately, over a subset of the group.

## Couplings and linear programs {#couplings-and-transport}

Comparing two distributions by physically moving the mass of one onto the other requires a name for a joint plan of movement.

::: {.definition #def-p-19}
[Couplings of two probability measures]{.box-title}

A **coupling** of probability measures \(P\) on \(X\) and \(Q\) on \(Y\) is a probability measure \(\pi\) on the product \(X \times Y\) whose marginals are \(P\) and \(Q\): \(\pi(A \times Y) = P(A)\) and \(\pi(X \times B) = Q(B)\) for all measurable \(A, B\). The set \(\Pi(P, Q)\) of couplings is convex, since a mixture of couplings has the same marginals, and it is never empty: the **independent coupling** \(P \otimes Q\) always belongs to it.
:::

Optimal transport selects the coupling minimizing an expected cost \(\mathbb{E}_{\pi}[c(X, Y)]\); for empirical measures this is a linear program over transport matrices with fixed row and column sums, the problem [[ch:optimal-transport-and-kernels]] solves, regularizes, and compares against the MMD. A linear program is the Lagrangian framework of the optimization section with every function affine: its dual is again a linear program, strong duality holds with no extra regularity, and dualizing the marginal constraints of the transport program is precisely what produces the Kantorovich dual used there.

## Paths, tensors, and iterated integrals {#paths-tensors-signatures}

A time series is a path, and the natural polynomial features of a path are integrals against its own increments, which live in tensor powers of the state space. The **tensor product** \(V \otimes W\) of two vector spaces is the space of formal bilinear combinations of symbols \(v \otimes w\); for \(V = W = \mathbb{R}^d\) with basis \((e_i)\), the \(m\)-fold power \((\mathbb{R}^d)^{\otimes m}\) has basis \(e_{i_1} \otimes \cdots \otimes e_{i_m}\), so its elements are \(m\)-way arrays of \(d^m\) coordinates.

:::: {.definition #def-p-20}
[Iterated integrals of a path]{.box-title}

Let \(x : [0, T] \to \mathbb{R}^d\) be a continuous path of bounded variation, so that integration against its increments \(dx_t\) is well defined. For each multi-index \((i_1, \dots, i_m) \in \{1, \dots, d\}^m\), the **iterated integral** of \(x\) is the number

$$ S^{(i_1, \dots, i_m)}(x) \;=\; \int_{0 \lt t_1 \lt t_2 \lt \cdots \lt t_m \lt T} dx^{i_1}_{t_1}\, dx^{i_2}_{t_2} \cdots dx^{i_m}_{t_m}, $$

the integral over the ordered simplex of times. For \(m = 1\) these are the total increments \(x^i_T - x^i_0\); for \(m = 2\) they record the ordered co-movement of pairs of coordinates. Collecting the degree-\(m\) integrals into a tensor \(S^{(m)}(x) \in (\mathbb{R}^d)^{\otimes m}\) for every \(m\) gives the raw material of the path's **signature**.
::::

The signature \((1, S^{(1)}(x), S^{(2)}(x), \dots)\) is a feature expansion of the path that is invariant to time reparameterization and determines the path up to a natural equivalence; inner products of signatures define the sequence kernels of [[ch:signature-and-time-series-kernels]], computed there by recursion on the data rather than by materializing any tensor.

## Indefinite inner products {#indefinite-inner-products}

Some natural similarity functions fail positive semidefiniteness, and the geometry that accommodates them keeps the bilinear algebra of an inner product while giving up positivity.

:::: {.definition #def-p-21}
[Indefinite and Krein inner products]{.box-title}

An **indefinite inner product** on a vector space \(\mathcal{K}\) is a symmetric bilinear form \(\langle \cdot, \cdot \rangle_{\mathcal K}\) with no positivity requirement, so \(\langle f, f \rangle_{\mathcal K}\) may be negative, or zero for \(f \neq 0\). The workable case is a **Krein space**: \(\mathcal{K}\) decomposes as an orthogonal direct sum of two Hilbert spaces \(\mathcal{H}_+\) and \(\mathcal{H}_-\) with

$$ \langle f, g \rangle_{\mathcal K} \;=\; \langle f_+, g_+ \rangle_{\mathcal{H}_+} - \langle f_-, g_- \rangle_{\mathcal{H}_-}, $$

a difference of two genuine inner products.
::::

Cauchy-Schwarz fails outright: \(\langle f, f \rangle_{\mathcal K}\) can vanish on nonzero (neutral) vectors, so it no longer induces a norm and nothing bounds \(\lvert \langle f, g \rangle_{\mathcal K} \rvert\); topology and projections must instead be borrowed from the associated Hilbert space \(\mathcal{H}_+ \oplus \mathcal{H}_-\). Kernels that decompose as a difference \(k_+ - k_-\) of two positive definite kernels reproduce in exactly such spaces, the subject of [[ch:indefinite-and-krein-kernels]].

## Numerical linear algebra clinic {#numerical-linear-algebra-clinic}

Exact identities do not specify stable algorithms. Normal equations square the condition number, explicit inverses waste work and magnify rounding error, and a small ridge changes the statistical problem even when introduced as "jitter." For every Gram solve, distinguish four quantities: backward error, forward coefficient error, prediction error, and optimization residual. They need not move together.

A dependable workflow scales features, inspects the Gram diagonal and a spectral estimate, uses a factorization or matrix-free solver appropriate to definiteness, and checks the residual in the original system. Cholesky is preferred for a well-conditioned positive-definite matrix; pivoted Cholesky or eigendecomposition exposes low rank; QR is safer for rectangular least squares; iterative methods require a stopping rule and preconditioner. The scaling chapter [[ch:large-scale-kernels]] turns these choices into algorithms.

## Measure and probability clinic {#measure-probability-clinic}

Three distinctions prevent many later errors. Almost-sure equality is not pointwise equality; convergence in probability is weaker than almost-sure convergence and does not by itself permit exchanging a limit and expectation; and an expectation under \(Q\) can be rewritten under \(P\) only when \(Q\) is absolutely continuous with respect to \(P\). In that last case the Radon-Nikodym derivative \(dQ/dP\) is the abstract density ratio behind importance weighting.

Conditional expectation is defined relative to a sigma-algebra, even when no density exists. Regular conditional distributions need assumptions on the measurable spaces. Hilbert-valued expectations need measurability and norm integrability, not only formal linearity. When a proof moves an operator, limit, derivative, or supremum through an expectation, name the theorem that permits it.

## Functional and convex analysis clinic {#functional-convex-clinic}

Finite-dimensional intuition fails in two recurring ways. Bounded closed sets need not be compact in an infinite-dimensional norm topology, and a bounded sequence can converge weakly without converging in norm. Compact operators convert some weak convergence into strong convergence and make inverse problems spectrally tractable, but their inverses are typically unbounded on their ranges.

For convex programs, KKT conditions are necessary and sufficient only under the relevant convexity and qualification conditions. Slater's condition is a common sufficient qualification for convex inequality constraints. Existence also needs coercivity, compactness, or lower-semicontinuity plus an appropriate compactness argument. Algorithmic convergence adds smoothness, step-size, stochastic-noise, and stopping assumptions beyond the existence of an optimum.

## Structured-domain clinic {#structured-domain-clinic}

Graphs, manifolds, groups, tensors, and paths each have more than one natural normalization. A graph Laplacian depends on the affinity scale and degree convention; a manifold kernel depends on metric, boundary, and volume measure; group averaging depends on the action and invariant measure; tensor features grow exponentially with degree; path signatures depend on augmentation and truncation. State these choices before invoking a theorem.

The test for a structured kernel remains the ordinary Gram test: for every finite input set, the matrix must be PSD. Structure supplies principled constructions, not an exemption from positivity, measurability, conditioning, or validation.

## Summary and further reading {#prelim-summary}

Kernel methods reduce nonlinear learning to linear algebra in Hilbert spaces, but the reduction remains sound only when its analytic hypotheses travel with it. Spectra govern geometry and conditioning; bounded evaluation and projection produce RKHS representations; measure theory and Bochner integration make distribution embeddings well defined; convex duality turns losses into coefficient problems; and structured domains contribute their own operators and symmetries. Readers needing proof-level measure theory, functional analysis, or convex analysis should consult a dedicated graduate text before relying on an interchange, compactness claim, or duality theorem.

## Notation and conventions {#notation-conventions}

A few conventions hold throughout the book. Vectors are columns; \(x^\top y\) is an inner product and \(xy^\top\) an outer product. \(\mathbf{K}\) or \(K\) is the Gram matrix with entries \(K_{ij} = k(x_i, x_j)\); \(\mathcal{H}\) is a reproducing kernel Hilbert space with inner product \(\langle\cdot,\cdot\rangle_{\mathcal H}\) and norm \(\lVert\cdot\rVert_{\mathcal H}\); \(\Phi\) is a feature map with \(k(x,x') = \langle \Phi(x), \Phi(x')\rangle\). Training data is \((x_i, y_i)_{i=1}^n\), and \(n\) is the sample size, \(d\) the input dimension. The symbols \(\lambda\) (a regularization weight or an eigenvalue), \(\alpha\) (dual coefficients), and \(\sigma\) (a kernel bandwidth) recur; where an overload could confuse, the surrounding text disambiguates. The full symbol table and a glossary of terms are collected in the [Notation and Glossary](glossary.html).

::: {.remark}
[How to use this chapter]{.box-title}

Nothing here is proved in full (the one-line integration by parts above is the exception); each result is stated so a later chapter can cite it and keep moving. When a chapter writes \"by the spectral theorem\" or \"Bochner's theorem gives,\" this is where the statement lives. The first five sections are the core, used from the opening chapters onward. The sections from [conditioning, scores, and U-statistics](#probability-for-inference) through [indefinite inner products](#indefinite-inner-products) serve the later parts of the book, on probabilistic inference, structured domains, transport, paths, and indefinite geometry; read them when the consuming chapter sends you here. Standard references for the material are, for linear algebra and analysis, any graduate text; for the probability and optimization used in learning, the treatments in the three sources this book synthesizes, and in particular the statistical-learning and optimization chapters of Schölkopf and Smola (2002) and the appendices of Shawe-Taylor and Cristianini (2004).
:::

## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

Do not treat a formal inverse as a numerical algorithm: inspect the spectrum, solve by factorization, and distinguish regularization from harmless roundoff protection. In analysis, name the hypothesis that permits each interchange of limit, derivative, supremum, and expectation. In optimization, convexity alone does not guarantee existence, strong duality, or convergence of a chosen algorithm; carry the qualification, coercivity, and step-size assumptions with the result. Across structured domains, state the measure, normalization, metric, and boundary convention before importing a Euclidean theorem.

## Exercises {#exercises}

1. [computation]{.ex-tag} A Gram matrix has eigenvalues \(10,1,10^{-8}\). Compute its condition number, then compute the condition number after adding ridge \(10^{-3}I\). Explain why the regularized solve is more stable.
2. [proof]{.ex-tag} Starting from the projection theorem, prove the Pythagorean identity for \(f=f_V+f_\perp\) and identify the exact step used in a representer-theorem proof.
3. [proof]{.ex-tag} Let \(Z=k(X,\cdot)\). Under \(\mathbb{E}\sqrt{k(X,X)}\lt\infty\), use reproduction and the Bochner operator rule to show \(\langle f,\mathbb{E}Z\rangle=\mathbb{E}f(X)\).
4. [computation]{.ex-tag} Compute the Fenchel conjugate of \(f(x)=x^2/2\) and verify Fenchel-Young equality at \(u=f'(x)\).
5. [synthesis]{.ex-tag} For a graph-kernel pipeline, list one assumption or diagnostic drawn from each of linear algebra, probability, convex optimization, and graph geometry.
6. [challenge]{.ex-tag} Give an example in which pointwise convergence does not justify exchanging limit and expectation, then state a domination condition that repairs the argument.
7. [computation]{.ex-tag} In the two-direction covariance fixture, let \(N_1\) of the \(n\) samples fall on \(\sqrt2e_1\). Derive \(\widehat C\) and \(\lVert\widehat C-I_2\rVert_2\) as functions of \(N_1\), then verify \(L\), \(v\), and \(r(V)\) for the Bernstein normalization \(Y_i=(z_i z_i^\top-I_2)/n\).
