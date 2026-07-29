---
id: ch-cca
example_code_policy: visible-for-executable
slug: kernel-cca-and-correlation
title: Kernel CCA and Correlation Analysis
part: V · Spectral Geometry and Unlabeled Structure
order: 24
tier: advanced
prerequisites:
  - kernel-clustering
objectives:
  - >-
    Derive classical CCA as a generalized eigenproblem and interpret canonical
    scores as paired, variance-normalized projections.
  - >-
    Kernelize both views through representer expansions and prove why the
    unregularized empirical problem saturates at correlation one.
  - >-
    Form the regularized kernel CCA eigenproblem and explain how its shrinkage
    parameter trades correlation against covariance.
  - >-
    Compute canonical scores for new paired observations and distinguish sample
    correlation from population dependence.
  - >-
    Compare linear CCA, kernel CCA, and deep CCA by representation, capacity,
    regularization, and computational cost.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-cca.yml
verification_date: null
bibliography:
  - andrew2013dcca
  - bach2002
  - fukumizu2007cca
  - gong2014
  - hardoon2004
  - lai2000kcca
  - scholkopf2002
  - kernelbook-code-ch-cca-ex2
  - shawe2004
narrative_link_policy: exact
---
# Kernel CCA and Correlation Analysis

<p class="lead">Often every object comes with two representations at once: an image and its caption, a gene's sequence and its expression profile, an audio clip and its transcript. Canonical correlation analysis looks for a direction in each view so that the two projected signals track each other as closely as possible, extracting the structure the two views share. This chapter kernelizes that search, meets an instructive failure along the way (without regularization the kernelized problem reports perfect correlation on any data), and shows how penalizing rough directions repairs it and ties correlation, covariance, and variance together as one eigen-decomposition family.</p>

## Kernel canonical correlation analysis {#kernel-cca}

The [[ch:kernel-pca|kernel-PCA chapter]] diagonalized variation in one feature
space; CCA asks which directions co-vary across two feature spaces.

The last method in this chapter relates two views of the same objects rather than analyzing one. Suppose every object comes with two representations: an image and its caption, a gene's sequence and its expression profile, an audio clip and its transcript. Canonical correlation analysis (CCA) looks for a direction in each view such that the two projected signals are maximally correlated, extracting the shared structure between the views.

<span id="cca-classical"></span>

**Classical CCA.**

What does it mean, concretely, for two views to share structure? CCA answers with a single number to maximize: the correlation between one projection of each view. The kernel version will inherit the linear one's eigenstructure wholesale, so we set up the linear problem first. Given two views collected as data matrices \(X \in \mathbb{R}^{n \times p}\), whose rows are \(\mathbf{x}_1^\top, \ldots, \mathbf{x}_n^\top\), and \(Y \in \mathbb{R}^{n \times d}\), whose rows are \(\mathbf{y}_1^\top, \ldots, \mathbf{y}_n^\top\), of the same \(n\) objects (so \(\mathbf{x}_i \in \mathbb{R}^p\) and \(\mathbf{y}_i \in \mathbb{R}^d\) are the two descriptions of object \(i\)), and assuming both views centered, CCA seeks directions \(\mathbf{w}_a \in \mathbb{R}^p\) and \(\mathbf{w}_b \in \mathbb{R}^d\) maximizing the empirical correlation of the projections \(\mathbf{w}_a^\top \mathbf{x}_i\) and \(\mathbf{w}_b^\top \mathbf{y}_i\):

$$
\max_{\mathbf{w}_a \in \mathbb{R}^p,\ \mathbf{w}_b \in \mathbb{R}^d} \ \frac{\frac{1}{n} \sum_{i=1}^n \mathbf{w}_a^\top \mathbf{x}_i \, \mathbf{y}_i^\top \mathbf{w}_b}{\big(\frac{1}{n} \sum_{i=1}^n \mathbf{w}_a^\top \mathbf{x}_i \mathbf{x}_i^\top \mathbf{w}_a\big)^{1/2} \big(\frac{1}{n} \sum_{i=1}^n \mathbf{w}_b^\top \mathbf{y}_i \mathbf{y}_i^\top \mathbf{w}_b\big)^{1/2}}.
$$

Interpreting the sums as sample covariance and variances, this is the population objective

$$
\max_{\mathbf{w}_a, \mathbf{w}_b} \ \frac{\operatorname{cov}(\mathbf{w}_a^\top X, \mathbf{w}_b^\top Y)}{\sqrt{\operatorname{var}(\mathbf{w}_a^\top X)} \, \sqrt{\operatorname{var}(\mathbf{w}_b^\top Y)}},
$$

the correlation coefficient between the two one-dimensional projections. In matrix notation the objective is

$$
\max_{\mathbf{w}_a, \mathbf{w}_b} \ \frac{\mathbf{w}_a^\top X^\top Y \mathbf{w}_b}{\big(\mathbf{w}_a^\top X^\top X \mathbf{w}_a\big)^{1/2} \big(\mathbf{w}_b^\top Y^\top Y \mathbf{w}_b\big)^{1/2}}.
$$

This first optimum gives the first pair of canonical directions; subsequent pairs are found by solving the same problem under the constraint of orthogonality to the directions already found.

<span id="cca-eigenproblem"></span>

**CCA is a generalized eigenvalue problem.**

The ratio is invariant to rescaling \(\mathbf{w}_a\) and \(\mathbf{w}_b\), so we may fix the scale by demanding unit projected variance and maximize only the numerator:

$$
\max_{\mathbf{w}_a, \mathbf{w}_b} \ \mathbf{w}_a^\top X^\top Y \mathbf{w}_b \quad \text{s.t.} \quad \mathbf{w}_a^\top X^\top X \mathbf{w}_a = 1 \ \text{ and } \ \mathbf{w}_b^\top Y^\top Y \mathbf{w}_b = 1.
$$

Introducing multipliers \(\lambda_a, \lambda_b\) for the two constraints, the stationarity conditions of the Lagrangian

$$
\min_{\mathbf{w}_a, \mathbf{w}_b} \ -\mathbf{w}_a^\top X^\top Y \mathbf{w}_b + \frac{\lambda_a}{2}\big(\mathbf{w}_a^\top X^\top X \mathbf{w}_a - 1\big) + \frac{\lambda_b}{2}\big(\mathbf{w}_b^\top Y^\top Y \mathbf{w}_b - 1\big)
$$

are obtained by setting the gradients to zero:

$$
-X^\top Y \mathbf{w}_b + \lambda_a X^\top X \mathbf{w}_a = 0, \qquad -Y^\top X \mathbf{w}_a + \lambda_b Y^\top Y \mathbf{w}_b = 0.
$$

Left-multiplying the first by \(\mathbf{w}_a^\top\) and the second by \(\mathbf{w}_b^\top\) and subtracting, the numerator terms cancel and the constraints give \(\lambda_a \mathbf{w}_a^\top X^\top X \mathbf{w}_a = \lambda_b \mathbf{w}_b^\top Y^\top Y \mathbf{w}_b\), hence \(\lambda_a = \lambda_b =: \lambda\). The two stationarity equations then assemble into a single generalized eigenvalue problem:

$$
\begin{pmatrix} 0 & X^\top Y \\ Y^\top X & 0 \end{pmatrix} \begin{pmatrix} \mathbf{w}_a \\ \mathbf{w}_b \end{pmatrix} = \lambda \begin{pmatrix} X^\top X & 0 \\ 0 & Y^\top Y \end{pmatrix} \begin{pmatrix} \mathbf{w}_a \\ \mathbf{w}_b \end{pmatrix}.
$$

Writing \(\Sigma_A = \begin{pmatrix} 0 & X^\top Y \\ Y^\top X & 0 \end{pmatrix}\), \(\Sigma_B = \begin{pmatrix} X^\top X & 0 \\ 0 & Y^\top Y \end{pmatrix}\), and \(\mathbf{w} = \begin{pmatrix} \mathbf{w}_a \\ \mathbf{w}_b \end{pmatrix}\), this is \(\Sigma_A \mathbf{w} = \lambda \Sigma_B \mathbf{w}\). If the block covariances are invertible, multiplying through and substituting \(\mathbf{v} = \Sigma_B^{1/2} \mathbf{w}\) turns it into the ordinary symmetric eigenproblem

$$
\Sigma_B^{-1/2} \Sigma_A \Sigma_B^{-1/2} \, \mathbf{v} = \lambda \, \mathbf{v},
$$

whose eigenvalues are the canonical correlations and whose eigenvectors, transformed back, give the canonical directions. The derivation follows Section 6.5 of Shawe-Taylor and Cristianini (2004).

<span id="kernel-cca-formulation"></span>

**Kernelizing CCA.**

Exactly as with PCA, we can replace each view by an RKHS embedding. Take two p.d. kernels \(K_a, K_b : \mathcal{X} \times \mathcal{X} \to \mathbb{R}\), with embeddings \(\varphi_a : \mathcal{X} \to \mathcal{H}_a\) and \(\varphi_b : \mathcal{X} \to \mathcal{H}_b\), giving two views \((\varphi_a(\mathbf{x}_i))_i\) and \((\varphi_b(\mathbf{x}_i))_i\) of the dataset. We look for directions \(f_a \in \mathcal{H}_a\) and \(f_b \in \mathcal{H}_b\) maximizing

$$
\max_{f_a \in \mathcal{H}_a,\ f_b \in \mathcal{H}_b} \ \frac{\frac{1}{n} \sum_{i=1}^n \langle f_a, \varphi_a(\mathbf{x}_i) \rangle_{\mathcal{H}_a} \langle \varphi_b(\mathbf{x}_i), f_b \rangle_{\mathcal{H}_b}}{\big(\frac{1}{n} \sum_{i=1}^n \langle f_a, \varphi_a(\mathbf{x}_i) \rangle_{\mathcal{H}_a}^2\big)^{1/2} \big(\frac{1}{n} \sum_{i=1}^n \langle f_b, \varphi_b(\mathbf{x}_i) \rangle_{\mathcal{H}_b}^2\big)^{1/2}},
$$

and again the reproducing property replaces every inner product by a function evaluation, \(\langle f_a, \varphi_a(\mathbf{x}_i)\rangle_{\mathcal{H}_a} = f_a(\mathbf{x}_i)\), giving

$$
\max_{f_a \in \mathcal{H}_a,\ f_b \in \mathcal{H}_b} \ \frac{\frac{1}{n} \sum_{i=1}^n f_a(\mathbf{x}_i) f_b(\mathbf{x}_i)}{\big(\frac{1}{n} \sum_{i=1}^n f_a(\mathbf{x}_i)^2\big)^{1/2} \big(\frac{1}{n} \sum_{i=1}^n f_b(\mathbf{x}_i)^2\big)^{1/2}}.
$$

Up to a few technical details (again left as an exercise), the representer theorem applies, and we seek solutions of the form \(f_a(\cdot) = \sum_{i=1}^n \alpha_i K_a(\mathbf{x}_i, \cdot)\) and \(f_b(\cdot) = \sum_{i=1}^n \beta_i K_b(\mathbf{x}_i, \cdot)\). Then \(f_a(\mathbf{x}_i) = [K_a \boldsymbol{\alpha}]_i\) and \(f_b(\mathbf{x}_i) = [K_b \boldsymbol{\beta}]_i\), and the objective becomes finite-dimensional:

$$
\max_{\boldsymbol{\alpha}, \boldsymbol{\beta} \in \mathbb{R}^n} \ \frac{\boldsymbol{\alpha}^\top K_a K_b \boldsymbol{\beta}}{\big(\boldsymbol{\alpha}^\top K_a^2 \boldsymbol{\alpha}\big)^{1/2} \big(\boldsymbol{\beta}^\top K_b^2 \boldsymbol{\beta}\big)^{1/2}}.
$$

Removing the scaling ambiguity as before, this is

$$
\max_{\boldsymbol{\alpha}, \boldsymbol{\beta} \in \mathbb{R}^n} \ \boldsymbol{\alpha}^\top K_a K_b \boldsymbol{\beta} \quad \text{s.t.} \quad \boldsymbol{\alpha}^\top K_a^2 \boldsymbol{\alpha} = 1 \ \text{ and } \ \boldsymbol{\beta}^\top K_b^2 \boldsymbol{\beta} = 1,
$$

which, by the same Lagrangian argument as in the linear case, leads to a generalized eigenvalue problem. Subsequent canonical directions come from the same problem with added orthogonality constraints.

### The population operator and its domains {#cca-population-operator}

The finite Gram problem hides the analytic difficulty: population CCA whitens two covariance operators, and covariance operators in an infinite-dimensional RKHS normally have eigenvalues tending to zero. Let \((X,Y)\sim P_{XY}\) on measurable spaces \(\mathcal X\times\mathcal Y\). Let \(\mathcal H_X,\mathcal H_Y\) be separable RKHSs with measurable kernels satisfying

$$
\mathbb E k_X(X,X) \lt \infty,\qquad \mathbb E k_Y(Y,Y) \lt \infty.
$$

The mean elements exist, and the centered covariance and cross-covariance operators are defined by

$$
\langle f,C_{XX}h\rangle_{\mathcal H_X}
  =\operatorname{Cov}(f(X),h(X)),\qquad
\langle g,C_{YX}f\rangle_{\mathcal H_Y}
  =\operatorname{Cov}(f(X),g(Y)).
$$

These operators are bounded; \(C_{XX}\) and \(C_{YY}\) are positive and self-adjoint, while \(C_{XY}=C_{YX}^\ast\). Moreover,

$$
\mathcal N(C_{XX})=\{f\in\mathcal H_X:\operatorname{Var}f(X)=0\}.
$$

Thus injectivity means that no nonzero RKHS function is almost surely constant after centering. Injectivity is stronger than finite-sample Gram invertibility and does not make \(C_{XX}^{-1/2}\) bounded. If \(C_{XX}\) is compact with eigenpairs \((\mu_j,e_j)\) and \(\mu_j\downarrow0\), its inverse square root is defined only on

$$
\mathcal D(C_{XX}^{-1/2})
=\left\{u:\sum_{j:\mu_j\gt0}\frac{|\langle u,e_j\rangle|^2}{\mu_j}\lt\infty\right\}.
$$

The safe population object comes from the factorization

$$
C_{YX}=C_{YY}^{1/2}V_{YX}C_{XX}^{1/2},
$$

where \(V_{YX}\) is the unique contraction satisfying \(V_{YX}=Q_YV_{YX}Q_X\), with \(Q_X,Q_Y\) the projections onto the closures of the covariance ranges. The notation \(C_{YY}^{-1/2}C_{YX}C_{XX}^{-1/2}\) is shorthand for this bounded extension, not permission to multiply three everywhere-defined bounded operators. Section 2.2, Equation (5), of [@fukumizu2007cca] makes this warning explicit.

::: {.theorem #thm-cca-population}
[Theorem (population canonical correlation and attainment)]{.box-title}

Under the moment and separability assumptions above, the supremal RKHS correlation equals \(\lVert V_{YX}\rVert\). If \(V_{YX}\) is compact, its largest singular value is attained by singular vectors \((\phi,\psi)\). Canonical functions \(f=C_{XX}^{-1/2}\phi\) and \(g=C_{YY}^{-1/2}\psi\) exist in the RKHS only when \(\phi\in\mathcal R(C_{XX}^{1/2})\) and \(\psi\in\mathcal R(C_{YY}^{1/2})\).

**Assumptions.** The kernels are measurable, the RKHSs separable, diagonal kernel moments finite, and \(V_{YX}\) compact for attainment. Null-variance functions are quotiented out. The stated range conditions are required before applying inverse square roots.
**Proof status.** The factorization and population formulation are given in [@fukumizu2007cca, Section 2.2, Equation (5), and Section 2.3, Equations (6) and (8)]. The attainment step is the compact-operator singular-value theorem.
:::

If compactness fails, the supremum need not identify a stable finite-dimensional direction. The extreme witness is \(Y=X\) in an infinite-dimensional RKHS: then \(V_{YX}=I\), every unit direction has correlation one, and no leading direction is distinguished. This is a population ambiguity, not finite-sample overfitting.

### What goes wrong, and why we must regularize {#kernel-cca-overfitting}

There is a serious flaw hiding in the kernelized problem. Suppose \(K_a\) and \(K_b\) are invertible, and change variables to \(\boldsymbol{\alpha}' = K_a \boldsymbol{\alpha}\) and \(\boldsymbol{\beta}' = K_b \boldsymbol{\beta}\). The objective and constraints turn into

$$
\max_{\boldsymbol{\alpha}', \boldsymbol{\beta}' \in \mathbb{R}^n} \ \boldsymbol{\alpha}'^\top \boldsymbol{\beta}' \quad \text{s.t.} \quad \boldsymbol{\alpha}'^\top \boldsymbol{\alpha}' = 1 \ \text{ and } \ \boldsymbol{\beta}'^\top \boldsymbol{\beta}' = 1.
$$

This is maximized, at value \(1\), by any \(\boldsymbol{\alpha}' = \boldsymbol{\beta}'\) on the unit sphere. In other words the maximal correlation is always attainable and the solution is completely undetermined: the method reports perfect correlation between the two views no matter what data it is fed. This is not a numerical accident but the RKHS version of a familiar disease. In high or infinite dimension it is trivially easy to find directions that make any two finite samples look perfectly correlated, precisely the phenomenon of *spurious correlations* that populate collections like Tyler Vigen's gallery of nonsense coincidences. Unregularized kernel CCA overfits by construction.

It is worth saying plainly where the free lunch comes from. When \(K_a\) is invertible the map \(\boldsymbol{\alpha} \mapsto K_a \boldsymbol{\alpha}\) is a bijection of \(\mathbb{R}^n\), so the vector of projected values \((f_a(\mathbf{x}_1), \ldots, f_a(\mathbf{x}_n))\) can be steered to any pattern we like by choosing \(\boldsymbol{\alpha}\); the same holds for the second view. Two arbitrary patterns can then be matched exactly, which is why the correlation saturates at its maximum of one. The kernel has handed each view as many adjustable directions as there are data points, and with that much freedom the two samples can always be fitted together, whether or not any real dependence links them. Regularization works by making some of those directions expensive: the added \(\lVert f_a \rVert_{\mathcal{H}_a}^2\) term charges for wiggly, high-norm functions, so only shared structure smooth enough to be worth the price survives.

The failure has two faces. It is an overfitting problem, since the fit is perfect for reasons having nothing to do with genuine shared structure, and it is a numerical instability problem, since the derivation needed to invert the kernel matrices, which are typically ill-conditioned. Both are cured by the same remedy: regularize by preferring smooth directions, penalizing the RKHS norms \(\lVert f_a \rVert_{\mathcal{H}_a}\) and \(\lVert f_b \rVert_{\mathcal{H}_b}\). Concretely, recall \(\lVert f_a \rVert_{\mathcal{H}_a}^2 = \boldsymbol{\alpha}^\top K_a \boldsymbol{\alpha}\), and replace the constraint \(\boldsymbol{\alpha}^\top K_a^2 \boldsymbol{\alpha} = 1\) by the convex combination

$$
(1 - \tau)\, \boldsymbol{\alpha}^\top K_a^2 \boldsymbol{\alpha} + \tau \, \underbrace{\boldsymbol{\alpha}^\top K_a \boldsymbol{\alpha}}_{\lVert f_a \rVert_{\mathcal{H}_a}^2} = 1,
$$

and symmetrically for \(\boldsymbol{\beta}\) with \(K_b\). The parameter \(\tau \in [0, 1]\) interpolates between the raw, degenerate problem at \(\tau = 0\) and a pure norm penalty at \(\tau = 1\). The added term charges for wild directions, but \(K^2+\tau K\) remains singular whenever \(K\) has a null space. A strict ridge such as \((K+\kappa I)^2\), or the population constraint \(C_{XX}+\varepsilon I\), is the safe formulation when injectivity is unavailable. Reading the penalty as a preference for directions of small RKHS norm ties the fix back to the supervised chapters: there we penalized \(\lVert f \rVert_{\mathcal{H}}\) to keep a fitted function from chasing noise, and here we do the same to keep a canonical direction from manufacturing correlation. The regularized problem is again a generalized eigenvalue problem; this form of kernel CCA is developed in [@bach2002, Section 3] and [@fukumizu2007cca, Section 2.1, Equation (3)].

Kernel CCA in this regularized form is a practical tool for learning shared representations across modalities. A representative application is finding a joint latent representation of images and their text tags, aligning the visual and textual views of a collection so that the two can be compared or retrieved against each other (Gong and Lazebnik, 2014).

The output is easier to read in score space than in coefficient space. Each paired observation produces two numbers, one from each view; useful canonical directions make those numbers line up without collapsing either view to a constant. The figure contrasts the raw coordinates, where the shared factor is obscured by view-specific variation, with the regularized canonical scores, where paired observations track one another.

<figure class="viz" data-figure="cca-paired-projections" data-alt="Two paired two-dimensional views share a latent variable hidden by nuisance variation. After regularized CCA, their one-dimensional canonical scores lie close to the diagonal, so matching observations receive similar scores."><figcaption>CCA is successful when paired observations agree after projection: regularization suppresses view-specific directions and exposes the shared coordinate rather than manufacturing perfect in-sample correlation.</figcaption></figure>

<span id="cca-covariance-family"></span>

**Variance, covariance, and correlation as one family.**

It is worth stepping back to see that the unsupervised methods of this chapter are the same construction tuned by a single choice: what we ask of a pair of directions. Shawe-Taylor and Cristianini (2004) organize their whole treatment this way. Given one view, asking for the direction of maximal variance gives PCA, the eigenvectors of the covariance matrix. Given two paired views \(X\) and \(Y\), we can instead ask for a pair of directions \((\mathbf{w}_a, \mathbf{w}_b)\) maximizing the raw covariance of the projections,

$$
\max_{\lVert \mathbf{w}_a \rVert = \lVert \mathbf{w}_b \rVert = 1} \ \mathbf{w}_a^\top C_{XY} \mathbf{w}_b, \qquad C_{XY} = \tfrac{1}{n} X^\top Y,
$$

whose solution is the leading pair of singular vectors of the cross-covariance matrix \(C_{XY}\), the covariance itself being the top singular value, with subsequent pairs found by deflation. Or we can ask for the directions of maximal correlation, dividing that same covariance by the projected standard deviations, and we are back to CCA and its generalized eigenproblem. Variance is a self-covariance diagonalized, covariance is a cross-term factorized by singular value decomposition, and correlation is covariance normalized by variance; the eigen-decomposition machinery is identical, and only the matrix in the quadratic form changes.

This taxonomy also clarifies what regularization does to kernel CCA. The penalized constraint \((1-\tau)\,\boldsymbol{\alpha}^\top K_a^2 \boldsymbol{\alpha} + \tau\,\boldsymbol{\alpha}^\top K_a \boldsymbol{\alpha} = 1\) does more than restore invertibility: as \(\tau\) runs from \(0\) to \(1\) it slides the objective smoothly from pure correlation toward pure covariance (Shawe-Taylor and Cristianini, 2004). At \(\tau = 0\) we have the degenerate correlation problem that reports a perfect match on any data; as \(\tau\) grows, the projected-variance term is progressively replaced by a plain norm, and in the limit the method rewards directions of large covariance rather than large correlation, which is exactly the well-posed maximal-covariance problem above. The regularizer is thus not an ad hoc patch but a dial between two members of the same family, and a modest \(\tau\) buys the numerical stability and the resistance to spurious correlation that pure correlation lacks.

## Why regularization is mandatory {#regularization-mandatory}

The last two sections diagnosed the degeneracy of unregularized kernel CCA and offered a dial, the parameter \(\tau\), for it. It is worth pinning down why that degeneracy is unavoidable rather than a corner case one might dodge with a lucky choice of kernel. The overfitting argument needed only one fact: that \(K_a\) and \(K_b\) are invertible, so that \(\boldsymbol{\alpha} \mapsto K_a \boldsymbol{\alpha}\) is a bijection. Asking when that fact holds shows the trouble is forced by exactly the property that makes a kernel worth using.

### Characteristic kernels force the saturation {#characteristic-kernels-force-saturation}

The expressive kernels of the earlier chapters are the universal and characteristic ones: the Gaussian, the Laplace, and their relatives, whose RKHS is dense in the continuous functions and whose mean embedding separates probability measures. That defining richness casts a concrete finite-sample shadow. On any set of \(n\) distinct points a strictly positive definite kernel produces a Gram matrix of full rank \(n\), hence invertible. So for precisely the kernels one reaches for, both \(K_a\) and \(K_b\) are invertible on any sample in general position, and the change of variables of the overfitting section applies verbatim.

::: {.proposition #prop-15-1}
[Proposition (unregularized kernel CCA saturates)]{.box-title}

Let \(K_a\) and \(K_b\) be strictly positive definite kernels, for instance Gaussian kernels on Euclidean space, evaluated at \(n\) distinct points in their respective views. Then the Gram matrices are invertible, and unregularized kernel CCA attains its maximum canonical correlation \(1\) on every such sample, whatever the pairing between the two views.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::

::: {.proof}
[Proof]{.box-title}

Strict positive definiteness makes \(K_a\) and \(K_b\) invertible, so \(\boldsymbol{\alpha}' = K_a \boldsymbol{\alpha}\) and \(\boldsymbol{\beta}' = K_b \boldsymbol{\beta}\) range over all of \(\mathbb{R}^n\). Under this substitution the objective \(\boldsymbol{\alpha}^\top K_a K_b \boldsymbol{\beta}\) with constraints \(\boldsymbol{\alpha}^\top K_a^2 \boldsymbol{\alpha} = \boldsymbol{\beta}^\top K_b^2 \boldsymbol{\beta} = 1\) becomes \(\max \, \boldsymbol{\alpha}'^\top \boldsymbol{\beta}'\) subject to \(\lVert \boldsymbol{\alpha}' \rVert = \lVert \boldsymbol{\beta}' \rVert = 1\), whose value is \(1\) by Cauchy-Schwarz, attained at any \(\boldsymbol{\alpha}' = \boldsymbol{\beta}'\) on the unit sphere. The pairing of the two views never entered the argument, so the reported correlation is \(1\) whether or not any real dependence links them. [\(\square\)]{.qed}
:::

This is the same disease that makes the regularized Fisher discriminant of [[ch:kernel-discriminants-and-projections]] replace its singular scatter matrix by \(N + \mu I\): an expressive feature map hands each view as many free directions as there are points, and left unconstrained they will fit anything. Fukumizu, Bach, and Gretton (2007) make the statement asymptotic and precise. Working with the population cross-covariance operator between the two RKHSs, they show that the unregularized canonical correlation is degenerate, and that empirical kernel CCA is a consistent estimator of the true population dependence only when a regularizer \(\kappa_n\) is present and is sent to zero slowly as the sample grows. Regularization is therefore not a numerical convenience bolted on for conditioning; it is the condition under which kernel CCA estimates anything at all. Kernel CCA in this form was proposed by Lai and Fyfe (2000) and, as a kernel measure of dependence, by Bach and Jordan (2002), and given its standard overview by Hardoon, Szedmak, and Shawe-Taylor (2004).

### The shrinkage form of the regularized eigenproblem {#shrinkage-eigenproblem}

We restored well-posedness earlier with the convex combination \((1 - \tau)\, \boldsymbol{\alpha}^\top K_a^2 \boldsymbol{\alpha} + \tau\, \boldsymbol{\alpha}^\top K_a \boldsymbol{\alpha}\). The parameterization used in most implementations, following Hardoon, Szedmak, and Shawe-Taylor (2004), writes the same idea as a shrinkage of the kernel matrix itself: add \(\kappa\) to its diagonal before squaring, replacing \(K_a^2\) by \((K_a + \kappa I)^2\). The two are the same penalty to first order, since

$$
(K_a + \kappa I)^2 = K_a^2 + 2\kappa\, K_a + \kappa^2 I,
$$

where the middle term \(2\kappa\, K_a\) is exactly the RKHS-norm penalty \(\boldsymbol{\alpha}^\top K_a \boldsymbol{\alpha} = \lVert f_a \rVert_{\mathcal{H}_a}^2\) that charges for wiggly directions, and the trailing \(\kappa^2 I\) is a strict ridge that makes the constraint matrix positive definite even when \(K_a\) is only positive semidefinite. That last point matters for the non-universal kernels: a linear or low-degree polynomial kernel has \(\operatorname{rank} K_a \lt n\), so \(K_a^2\) is singular and even the penalized matrix \(K_a^2 + \tau K_a\) can stay singular, whereas \((K_a + \kappa I)^2\) is invertible for every \(\kappa \gt 0\).

:::: {.definition #def-15-2}
[Definition (regularized kernel CCA)]{.box-title}

For a shrinkage parameter \(\kappa \gt 0\), regularized kernel CCA solves the generalized eigenvalue problem

$$
\begin{pmatrix} 0 & K_a K_b \\ K_b K_a & 0 \end{pmatrix} \begin{pmatrix} \boldsymbol{\alpha} \\ \boldsymbol{\beta} \end{pmatrix} = \rho \begin{pmatrix} (K_a + \kappa I)^2 & 0 \\ 0 & (K_b + \kappa I)^2 \end{pmatrix} \begin{pmatrix} \boldsymbol{\alpha} \\ \boldsymbol{\beta} \end{pmatrix},
$$

whose largest eigenvalue \(\rho \in [0, 1]\) is the leading canonical correlation and whose eigenvector supplies the dual weights \((\boldsymbol{\alpha}, \boldsymbol{\beta})\) of the first canonical pair.
::::

Because the right-hand matrix is block diagonal and positive definite, the substitution \(\mathbf{u} = (K_a + \kappa I) \boldsymbol{\alpha}\), \(\mathbf{v} = (K_b + \kappa I) \boldsymbol{\beta}\) reduces this to an ordinary symmetric eigenproblem exactly as in the linear derivation above: the canonical correlations are the eigenvalues of \((K_a + \kappa I)^{-1} K_a K_b (K_b + \kappa I)^{-1}\) and its transpose, so the shrinkage simply whitens each view through the well-conditioned \((K + \kappa I)^{-1}\) in place of the singular or near-singular \(K^{-1}\). The consistency theory of Fukumizu, Bach, and Gretton (2007) is the statement that letting \(\kappa = \kappa_n \to 0\) at a controlled rate recovers the population canonical correlations as \(n \to \infty\); holding \(\kappa\) fixed trades a little bias for the stability that the next example makes numerical.

### A worked example: saturation and its cure {#kcca-worked-example}

A three-object sample makes both halves visible at once: the unregularized correlation is \(1\) by the proposition above, and the shrinkage dials it down to an honest, data-dependent value.

:::::: {.example #example-15-1}
[Example (kernel CCA on three paired points)]{.box-title}

::::: wex
:::: wex-setup
Three objects with scalar views \(x^a = (-1,\, 0,\, 2)\) and \(x^b = (0,\, 1,\, 1.5)\), each embedded with the Gaussian kernel \(k(u, v) = e^{-(u - v)^2 / 2}\) (bandwidth \(\sigma = 1\), universal). The Gram matrices are

$$
K_a = \begin{pmatrix} 1 & 0.607 & 0.011 \\ 0.607 & 1 & 0.135 \\ 0.011 & 0.135 & 1 \end{pmatrix}, \qquad K_b = \begin{pmatrix} 1 & 0.607 & 0.325 \\ 0.607 & 1 & 0.882 \\ 0.325 & 0.882 & 1 \end{pmatrix}.
$$
::::

1.  [Confirm both embeddings are full rank.]{.wex-op} The determinants \(\det K_a = 0.6155\) and \(\det K_b = 0.0955\) are nonzero, so \(K_a\) and \(K_b\) are invertible and the hypothesis of the saturation proposition holds.
2.  [Solve the unregularized eigenproblem.]{.wex-op} With the constraint matrix \(\operatorname{diag}(K_a^2, K_b^2)\) the top generalized eigenvalue is \(\rho = 1.000\): the two views report a perfect canonical correlation, and the same \(1\) would appear for any other pairing of the six numbers.
3.  [Apply the \((K + \kappa I)\) shrinkage.]{.wex-op} Replacing each \(K^2\) by \((K + \kappa I)^2\) and solving the regularized eigenproblem gives a leading correlation that falls monotonically as the shrinkage grows: \(\rho = 0.896\) at \(\kappa = 0.1\), then \(\rho = 0.611\) at \(\kappa = 0.5\), and \(\rho = 0.413\) at \(\kappa = 1\).
4.  [Read off the canonical directions.]{.wex-op} At \(\kappa = 1\) the leading dual weights, normalized to unit length, are \(\boldsymbol{\alpha} = (0.519,\, 0.635,\, 0.572)\) and \(\boldsymbol{\beta} = (0.632,\, 0.627,\, 0.456)\), giving one concrete pair of canonical functions \(f_a = \sum_i \alpha_i K_a(x_i^a, \cdot)\) and \(f_b = \sum_i \beta_i K_b(x_i^b, \cdot)\).

**Reading.** The unregularized \(1.000\) is an artifact of an invertible, expressive kernel and carries no information about the data. The shrinkage turns \(\kappa\) into a dial that trades that illusory perfect fit for a genuine, data-dependent correlation, which is exactly the \((K + \kappa I)\) ridge whose vanishing rate the consistency theory prescribes.

```python
import numpy as np
xa, xb = np.array([-1.,0.,2.]), np.array([0.,1.,1.5])
rbf_gram = lambda z: np.exp(-(z[:,None]-z[None,:])**2/2)
Ka, Kb = rbf_gram(xa), rbf_gram(xb)
def correlation(ridge):
    A = np.linalg.solve(Ka + ridge*np.eye(3), Ka)
    B = np.linalg.solve(Kb + ridge*np.eye(3), Kb)
    return np.linalg.eigvals(np.matmul(A, B)).real.max()
values = np.array([correlation(r) for r in [.1,.5,1.]])
assert np.allclose(values, [.896,.611,.413], atol=8e-4)
assert np.all(np.diff(values) < 0)
print(values)
```
:::::
::::::

### What consistency actually requires {#cca-consistency}

Regularization has two incompatible jobs. At each finite \(n\), \(\varepsilon_n\gt0\) stabilizes inverse square roots; for consistency toward the unregularized population target, it must eventually vanish. It cannot vanish too quickly because empirical covariance error is amplified by inverse powers of \(\varepsilon_n\).

::: {.theorem #thm-cca-consistency}
[Theorem (a sufficient kernel-CCA consistency schedule)]{.box-title}

Let \((X_i,Y_i)_{i=1}^n\) be i.i.d. copies of \((X,Y)\). Assume the measurable-kernel moment conditions of the population section, compactness of \(V_{YX}\), a one-dimensional leading left and right singular subspace, and the range conditions \(\phi\in\mathcal R(C_{XX})\), \(\psi\in\mathcal R(C_{YY})\). If

$$
\varepsilon_n\downarrow0,\qquad n^{1/3}\varepsilon_n\longrightarrow\infty,
$$

then the regularized empirical canonical functions converge, up to independent signs, to the population canonical functions in \(L^2(P_X)\) and \(L^2(P_Y)\), in probability.

**Assumptions.** I.i.d. paired sampling, separable RKHSs, finite diagonal kernel moments, compact normalized cross-covariance, simple leading canonical correlation, the stated range conditions, and the displayed deterministic schedule.
**Proof status.** This is [@fukumizu2007cca, Theorem 2], using its Equation (9). Its Lemma 6 gives the operator-estimation term \(O_p(\varepsilon_n^{-3/2}n^{-1/2})\).
:::

Fixed \(\varepsilon\) converges to a regularized population estimand rather than the raw one. The schedule \(\varepsilon_n=n^{-1/2}\) is too fast for this theorem, whereas \(\varepsilon_n=n^{-1/4}\) satisfies it. Cross-validation may select a better finite-sample value, but that random choice is not automatically covered by this deterministic-schedule theorem.

### Repeated canonical correlations {#cca-repeated-correlations}

A canonical correlation can be identified while its directions are not. If \(\rho_1=\cdots=\rho_m\gt\rho_{m+1}\), any simultaneous orthogonal rotation of the \(m\) left and right singular vectors produces another valid canonical basis.

::: {.proposition #prop-cca-repeated}
[Proposition (identifiability under multiplicity)]{.box-title}

Suppose \(V_{YX}\) is compact and its leading singular value has multiplicity \(m\). The leading value and corresponding left and right \(m\)-dimensional singular subspaces are intrinsic, but individual canonical directions inside them are not.

**Assumptions.** Compactness and a positive gap \(\rho_m-\rho_{m+1}\).
**Proof status.** Proved immediately below.
:::

::: {.proof}
[Proof]{.box-title}

Let \(V_{YX}\phi_j=\rho\psi_j\) and \(V_{YX}^\ast\psi_j=\rho\phi_j\) for \(j\le m\). For any orthogonal \(R\in\mathbb R^{m\times m}\), set \(\widetilde\phi_j=\sum_\ell R_{\ell j}\phi_\ell\) and \(\widetilde\psi_j=\sum_\ell R_{\ell j}\psi_\ell\). Linearity gives the same singular equations, while orthogonality preserves normalization. Every rotated basis is therefore equally valid, but the spectral projectors onto the two subspaces are unchanged. [\(\square\)]{.qed}
:::

Compare projection matrices or principal angles between estimated canonical subspaces. Raw coefficient vectors can disagree even when two fits recover exactly the same shared subspace.

### A deterministic multiview regularization study {#cca-regularization-study}

The following controlled experiment makes training saturation and held-out utility disagree. Each view has \(30\) coordinates and only \(36\) training pairs. The first coordinate in each view is a noisy measurement of the same latent Gaussian variable; the other \(29\) coordinates are independent nuisance noise. A separate set of \(400\) paired observations measures generalization.

::: {.example #example-cca-regularization-study}
[Example (regularization reveals the shared coordinate)]{.box-title}

With deterministic seed \(1502\), nearly unregularized linear CCA reports training correlation \(0.999999\) but only \(0.381473\) held-out correlation. Ridge values \(0.03,0.3,3,30\) produce held-out correlations \(0.387012,0.414535,0.424547,0.395060\). Moderate shrinkage improves the held-out quantity even though it lowers training correlation.

As a negative control, permuting the training pairing destroys the shared relation. Across the same ridge sweep, every absolute held-out correlation is below \(0.027\). A large training canonical correlation without held-out paired alignment is a failure witness, not evidence of dependence.

**Verification.** The deterministic generator, train/test centering, permutation control, and assertions are checked by the chapter's computational reference [@kernelbook-code-ch-cca-ex2].

```python
import numpy as np
def rcca(X, Y, ridge):
    n = len(X)
    Cxx, Cyy, Cxy = np.matmul(X.T, X)/n, np.matmul(Y.T, Y)/n, np.matmul(X.T, Y)/n
    lx, Ux = np.linalg.eigh(Cxx+ridge*np.eye(X.shape[1]))
    ly, Uy = np.linalg.eigh(Cyy+ridge*np.eye(Y.shape[1]))
    Wx, Wy = np.matmul(Ux*(1/np.sqrt(lx)), Ux.T), np.matmul(Uy*(1/np.sqrt(ly)), Uy.T)
    U, s, Vt = np.linalg.svd(np.linalg.multi_dot([Wx, Cxy, Wy]))
    return np.matmul(Wx, U[:, 0]), np.matmul(Wy, Vt.T[:, 0]), s[0]
rng = np.random.default_rng(1502)
def sample(n, d=30):
    z = rng.normal(size=(n, 1))
    return (np.c_[z+.2*rng.normal(size=(n, 1)), rng.normal(size=(n, d-1))],
            np.c_[z+.2*rng.normal(size=(n, 1)), rng.normal(size=(n, d-1))])
Xtr, Ytr = sample(36); Xte, Yte = sample(400)
Xte -= Xtr.mean(0); Yte -= Ytr.mean(0); Xtr -= Xtr.mean(0); Ytr -= Ytr.mean(0)
for ridge in [1e-6, .03, .3, 3., 30.]:
    a, b, train = rcca(Xtr, Ytr, ridge)
    print(ridge, train, np.corrcoef(np.matmul(Xte, a), np.matmul(Yte, b))[0, 1])
```
:::

This is not a universal tuning recommendation. It demonstrates the protocol: preserve pairs when splitting, fit centering on training data only, select shrinkage by held-out paired performance, and include a pairing permutation as a negative control.

## Multi-view learning and deep CCA {#multi-view-deep-cca}

Kernel CCA is one point in a wider space of multi-view methods, all of which read shared structure from paired data and differ only in how each view is represented before the correlations are taken. Linear CCA maps each view by a matrix, kernel CCA maps it by a fixed RKHS feature map, and one can instead learn the maps outright.

  Method       View map                                                 What is learned                                            Scaling in \(n\)
  ------------ -------------------------------------------------------- ---------------------------------------------------------- ---------------------------------------------------
  Linear CCA   linear \(\mathbf{x} \mapsto W \mathbf{x}\)              the two projection matrices                                cheap, dimension bound
  Kernel CCA   fixed feature map \(\varphi(\mathbf{x})\)   the dual weights \(\boldsymbol{\alpha}, \boldsymbol{\beta}\)      \(O(n^3)\) eigenproblem
  Deep CCA     neural network \(f_\theta(\mathbf{x})\)      the network weights \(\theta, \psi\)   minibatch, \(O(n)\)

Deep CCA (Andrew, Arora, Bilmes, and Livescu, 2013) keeps the correlation objective but replaces the two fixed feature maps by trainable neural networks \(f_\theta\) and \(g_\psi\), one per view. The networks are trained to maximize the total canonical correlation of their outputs, the sum of the top singular values of the normalized cross-covariance \(\hat{\Sigma}_{ff}^{-1/2} \hat{\Sigma}_{fg} \hat{\Sigma}_{gg}^{-1/2}\), and the gradient of that spectral objective is backpropagated through both networks. This buys the nonlinearity of kernel CCA while escaping the \(O(n^3)\) Gram-matrix eigenproblem, since the networks are optimized on minibatches rather than on the full sample. Crucially, the whitening matrices \(\hat{\Sigma}_{ff}\) and \(\hat{\Sigma}_{gg}\) are inverted with the very same \((\hat{\Sigma} + r I)\) shrinkage of the previous section: the expressive-map degeneracy does not vanish when the map is learned rather than fixed, so deep CCA regularizes its covariances for exactly the reason kernel CCA regularizes its Gram matrices.

The setting that has driven these methods most is a paired corpus. In the cross-lingual case each object is a document present in two languages, the two views are its representations under the per-language kernels of [[ch:kernels-for-text]], and the leading canonical directions span a shared, language-independent semantic space in which a query in one language can be matched against candidates in the other; the same construction induces bilingual word and document embeddings from parallel text. The image-and-tags alignment mentioned earlier (Gong and Lazebnik, 2014) is the vision analogue, with one view a visual kernel and the other a text kernel over the tags. In every case the family relationship to the single-view spectral methods of [[ch:kernel-pca]] is the organizing thread: kernel PCA diagonalizes one view's covariance, kernel CCA factorizes the cross-covariance of two, and deep CCA does the latter with learned features, but all three read geometry out of a spectral decomposition.

::: {.remark}
[Further reading]{.box-title}

For a book-length treatment of the methods in this chapter, Schölkopf and Smola (2002), *Learning with Kernels*, develops kernel PCA in its chapter 14 and takes up the related kernel feature-extraction methods in the neighbouring chapters, with the pre-image problem and kernel-PCA denoising worked out in detail. Shawe-Taylor and Cristianini (2004), *Kernel Methods for Pattern Analysis*, covers the same territory from the pattern-analysis viewpoint: its chapters 5 and 6 treat the spectral methods, kernel PCA, and canonical correlation analysis, and Section 6.5 is the source of the generalized-eigenvalue derivation of CCA used above. Both books also situate these unsupervised methods within the wider representer-theorem program that organizes these chapters.
:::

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

- Never report the unregularized training correlation as evidence of dependence: with invertible Gram matrices it equals one by construction.
- Center both Gram matrices using the training sample, and reuse those training means for test points. Re-centering a query changes the fitted problem.
- Tune shrinkage on held-out paired data. A tiny ridge can make the eigenproblem numerically solvable while still leaving a statistically unstable direction.
- Inspect canonical scores as well as the leading correlation. A high score driven by one or two pairs, or one that disappears under resampling, is not a robust shared representation.
- Compare against linear CCA. A nonlinear map earns its complexity only when its held-out alignment improves.

## Summary and further reading {#summary-and-further-reading}

The next chapter, [[ch:kernel-discriminants-and-projections|kernel
discriminants and projections]], changes the controlled numerator from
cross-view covariance to label relevance.

CCA searches for two variance-normalized projections whose scores covary as strongly as possible. Kernelization makes those projections nonlinear but also makes perfect empirical correlation trivial whenever both Gram matrices are invertible. RKHS-norm shrinkage is therefore part of the estimand, not merely a numerical patch: it restricts which score patterns each view may realize and turns the problem back into a meaningful generalized eigenproblem. Covariance analysis, CCA, kernel CCA, and deep CCA differ mainly in the representation and normalization they choose; all must control expressive directions and be judged on held-out paired observations. For primary treatments, see [@bach2002], [@hardoon2004], [@fukumizu2007cca], and [@andrew2013dcca].

## Exercises {#exercises}

1.  [proof]{.ex-tag} Show why unregularized kernel CCA saturates at correlation \(1\). Assume \(K_a\) and \(K_b\) are invertible and substitute \(\boldsymbol{\alpha}' = K_a \boldsymbol{\alpha}\), \(\boldsymbol{\beta}' = K_b \boldsymbol{\beta}\) into the finite-dimensional objective \(\boldsymbol{\alpha}^\top K_a K_b \boldsymbol{\beta}\) with constraints \(\boldsymbol{\alpha}^\top K_a^2 \boldsymbol{\alpha} = 1\) and \(\boldsymbol{\beta}^\top K_b^2 \boldsymbol{\beta} = 1\). Show the problem becomes \(\max\, \boldsymbol{\alpha}'^\top \boldsymbol{\beta}'\) subject to \(\lVert\boldsymbol{\alpha}'\rVert = \lVert\boldsymbol{\beta}'\rVert = 1\), whose maximum is \(1\), attained by any \(\boldsymbol{\alpha}' = \boldsymbol{\beta}'\) on the unit sphere. Interpret the result: the map \(\boldsymbol{\alpha} \mapsto K_a\boldsymbol{\alpha}\) being a bijection means the projected values \((f_a(\mathbf{x}_1), \ldots, f_a(\mathbf{x}_n))\) can be steered to any pattern, so any two samples can be matched exactly. Then explain how replacing the constraint by \((1-\tau)\,\boldsymbol{\alpha}^\top K_a^2 \boldsymbol{\alpha} + \tau\,\boldsymbol{\alpha}^\top K_a \boldsymbol{\alpha} = 1\) breaks the degeneracy.
    Hint

    ::: hint-body
    The Cauchy-Schwarz inequality bounds \(\boldsymbol{\alpha}'^\top\boldsymbol{\beta}' \le \lVert\boldsymbol{\alpha}'\rVert\,\lVert\boldsymbol{\beta}'\rVert = 1\), with equality exactly when the two unit vectors coincide. For the regularization step, note that \(\boldsymbol{\alpha}^\top K_a \boldsymbol{\alpha} = \lVert f_a\rVert_{\mathcal{H}_a}^2\) is the RKHS norm of the direction: adding \(\tau \gt 0\) of it makes the constraint matrix strictly positive definite and charges for high-norm, wiggly directions, so the change of variable no longer decouples the problem.
    :::
2.  [challenge]{.ex-tag} The chapter argues that variance, covariance, and correlation are one eigen-decomposition family tuned by a single choice. Fill in the family for a pair of centered views \(X, Y\): (a) which matrix's eigenvectors solve the single-view maximal-variance problem (PCA); (b) which matrix's singular vectors solve the maximal-covariance problem \(\max_{\lVert\mathbf{w}_a\rVert = \lVert\mathbf{w}_b\rVert = 1} \mathbf{w}_a^\top C_{XY}\mathbf{w}_b\); and (c) which generalized eigenproblem solves maximal correlation (CCA). Then explain, using the regularized constraint \((1-\tau)\,\boldsymbol{\alpha}^\top K_a^2 \boldsymbol{\alpha} + \tau\,\boldsymbol{\alpha}^\top K_a \boldsymbol{\alpha} = 1\), why sending \(\tau\) from \(0\) to \(1\) slides kernel CCA from pure correlation toward pure covariance, and why a modest \(\tau\) is what buys resistance to spurious correlation.
3.  [computation]{.ex-tag} Expand the shrinkage constraint matrix \((K + \kappa I)^2\) and name its three terms: the raw projected-variance matrix \(K^2\), the RKHS-norm penalty, and the strict ridge. Then take the linear kernel \(k(u, v) = uv\) on the one-dimensional points \(u \in \{1, 2, 3\}\), whose Gram matrix \(K\) has rank \(1\). Check that \(K^2\) and the penalized matrix \(K^2 + \tau K\) both remain singular, while \((K + \kappa I)^2\) is invertible for every \(\kappa \gt 0\). Conclude why the shrinkage form is the safe default once the kernel is not universal.
    Hint

    ::: hint-body
    The rank of \(K^2\) equals the rank of \(K\), and adding \(\tau K\) cannot raise it because \(\tau K\) shares the null space of \(K\); only the \(\kappa^2 I\) term fills that null space, lifting every eigenvalue to at least \(\kappa^2\).
    :::
4.  [challenge]{.ex-tag} Reproduce the worked example's shrinkage sweep and confirm the leading canonical correlation \(\rho(\kappa)\) falls monotonically from \(1\) as \(\kappa\) grows (the check script prints \(1.000,\, 0.896,\, 0.611,\, 0.413\) at \(\kappa = 0,\, 0.1,\, 0.5,\, 1\)). Then prove monotonicity in general. Writing the top correlation as the maximum of \(\boldsymbol{\alpha}^\top K_a K_b \boldsymbol{\beta}\) over \(\sqrt{\boldsymbol{\alpha}^\top (K_a + \kappa I)^2 \boldsymbol{\alpha}} \, \sqrt{\boldsymbol{\beta}^\top (K_b + \kappa I)^2 \boldsymbol{\beta}} = 1\), argue that enlarging the constraint matrices from \(K^2\) to \((K + \kappa I)^2\) can only shrink the attainable maximum, so \(\rho(\kappa)\) is nonincreasing.
    Hint

    ::: hint-body
    For a positive semidefinite \(K\) and \(\kappa \gt 0\), \((K + \kappa I)^2 - K^2 = 2\kappa K + \kappa^2 I \succeq 0\), so each denominator quadratic form only grows with \(\kappa\) while the numerator is untouched: every feasible ratio is bounded by its value at smaller \(\kappa\).
    :::
5.  [warm-up]{.ex-tag} Place kernel CCA in the multi-view family. (a) State precisely what deep CCA (Andrew et al., 2013) replaces relative to kernel CCA, and explain why it still needs the covariance shrinkage \((\hat{\Sigma} + r I)\) of this chapter. (b) Describe the cross-lingual paired-corpus application: what are the two views, what does a leading canonical direction represent, and how is it used for retrieval across languages? (c) Relate the objective to kernel PCA of [[ch:kernel-pca]]: which covariance or cross-covariance does each of PCA, CCA, and deep CCA decompose?
6.  [proof]{.ex-tag} Let \(C\) be a positive compact operator with eigenpairs \((\mu_j,e_j)\), \(\mu_j\downarrow0\). Show that injectivity of \(C\) does not make \(C^{-1/2}\) bounded. State its domain and construct a sequence of unit vectors on which the inverse-square-root norm diverges.
7.  [synthesis]{.ex-tag} Explain why the schedule \(\varepsilon_n=n^{-1/4}\) satisfies the consistency theorem but \(\varepsilon_n=n^{-1/2}\) does not. Separate the claims “the finite problem is invertible,” “the regularized estimator converges,” and “cross-validation selected a useful finite-sample value.”
8.  [computation]{.ex-tag} Run the deterministic multiview study. Report the nearly unregularized training and held-out correlations, the ridge with the best held-out correlation, and the permutation-control maximum. Explain why selecting by training correlation chooses the failure mode.
:::
