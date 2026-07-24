---
id: ch-rkbs
slug: reproducing-kernel-banach-and-variation-spaces
title: Reproducing-Kernel Banach and Variation Spaces
part: XVII · Beyond Hilbert Geometry
order: 56
tier: research
prerequisites:
  - kernels-and-rkhs
  - vector-and-operator-valued-kernels
  - inverse-learning-and-spectral-regularization
objectives:
  - Explain which RKHS arguments rely essentially on an inner product.
  - >-
    Define reproducing-kernel Banach spaces using bounded evaluation and
    duality.
  - >-
    State Banach representer theorems without assuming a linear coefficient
    expansion.
  - >-
    Connect nonquadratic norms to sparsity, atomic measures, and variation
    spaces.
  - >-
    Relate finite-width neural ridge splines to continuous-domain variational
    problems.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-rkbs.yml
verification_date: null
bibliography:
  - zhang2009rkbs
  - wang2024sparserkbs
  - parhi2021banach
---
# Reproducing-Kernel Banach and Variation Spaces

<p class="lead">Ask kernel ridge regression for a sparse solution and it will refuse. The refusal is geometric: a squared Hilbert norm spreads energy across every correlated direction, so the minimizer touches every kernel section, and no amount of tuning makes most coefficients exactly zero. The same geometry blocks a second wish. A shallow neural network is a sum of finitely many ridge functions, yet an optimization over all such sums, with a quadratic penalty, has no reason to return a finite sum. Both wishes call for a space with a different unit ball. Reproducing-kernel Banach spaces keep the one property learning cannot do without, bounded point evaluation, while replacing the inner product by duality and the squared norm by nonsmooth atomic norms whose extreme points are single features. By the end of the chapter the representer theorem lives in the dual, sparsity falls out of convex geometry rather than heuristics, and a finite-width network appears as the exact solution of a convex variational problem over a variation space.</p>

## What the Hilbert structure was buying {#rkbs-hilbert-structure}

Before trading away the inner product we should know exactly what it was paying for, because each convenience will need either a replacement or an obituary. Three familiar RKHS moves depend on it:

1. Every bounded functional has a unique representer in the same space.
2. Orthogonal projection separates observed kernel sections from an irrelevant complement.
3. The squared norm has the linear gradient \(f\mapsto f\) after identifying the space with its dual.

In a Banach space there may be no orthogonality, no symmetric inner product, and no canonical identification with the dual. Bounded evaluation can still hold, but the reproducing object and the representer theorem must be expressed through duality.

::: {.definition #def-rkbs}
[Definition (reproducing-kernel Banach space)]{.box-title}

A Banach space \(\mathcal B\) of functions on \(\mathcal X\) is a reproducing-kernel Banach space when every point evaluation \(\delta_x:f\mapsto f(x)\) is a bounded linear functional and the resulting evaluation functionals admit a kernel representation through an appropriate dual pairing.

The exact kernel may be scalar, two-sided, operator valued, or induced by a semi-inner product, depending on the geometry of \(\mathcal B\).
:::

Unlike the Hilbert case, a positive-definite symmetric scalar kernel is not by itself a complete description of an arbitrary RKBS. Smoothness, strict convexity, reflexivity, and a chosen duality map determine how primal and dual representations relate [@zhang2009rkbs].

## Semi-inner products and duality mappings {#rkbs-duality}

An RKHS silently identifies each function with the functional it represents, and every representer argument leans on that identification. A Banach space grants no such identification for free, but a smooth one offers a substitute. In a smooth Banach space, the normalized duality map \(J:\mathcal B\to\mathcal B^\ast\) satisfies

$$
\langle f,J(f)\rangle=\lVert f\rVert_{\mathcal B}^2,
\qquad
\lVert J(f)\rVert_{\mathcal B^\ast}=\lVert f\rVert_{\mathcal B}.
$$

The induced semi-inner product can be written \([g,f]=\langle g,J(f)\rangle\). It is linear in its first argument but need not be symmetric. Reproduction then takes a form such as

$$
f(x)=[f,K_x]
$$

for a suitable section \(K_x\). Because the second slot is nonlinear through \(J\), expansions that are linear in the dual variable need not be linear in the primal function.

For \(\ell_p\)-type geometry with \(p\ne2\), the duality map raises coordinate magnitudes to a power and preserves their signs. This changes the shrinkage path and the meaning of similarity. It also explains why carrying an RKHS coefficient formula into a Banach norm by substitution can be wrong.

## Banach-space representer theorems {#rkbs-representer}

The question that decides whether any of this is usable is whether regularized learning over an infinite-dimensional Banach space still collapses to finitely many degrees of freedom. Consider

$$
\min_{f\in\mathcal B}
\Phi\{f(x_1),\ldots,f(x_n)\}+\lambda\Omega(\lVert f\rVert_{\mathcal B}).
$$

In smooth strictly convex settings, optimality often yields a finite representation in the dual:

$$
J(\widehat f)\in\operatorname{span}\{\delta_{x_1},\ldots,\delta_{x_n}\}.
$$

Recovering \(\widehat f\) requires applying the inverse duality map, which can be nonlinear.

:::: {.theorem #thm-rkbs-dual-representer}
[Theorem (dual representer principle)]{.box-title}

Suppose \(\mathcal B\) is reflexive, strictly convex, and smooth; evaluations are bounded; \(\Omega\) is strictly increasing; and a minimizer exists. Then a minimum-norm solution can be chosen so that its norming functional lies in the span of the observed evaluation functionals.

**Assumptions.** Reflexivity, strict convexity, smoothness, bounded evaluations, existence, and the stated monotonicity. **Proof status.** Cited at the level of a general RKBS representer principle; different Banach constructions require different technical forms [@zhang2009rkbs].
::::

The theorem is finite dimensional in the dual, not necessarily a linear primal kernel expansion. Algorithms may solve for dual coefficients and repeatedly evaluate \(J^{-1}\). Conditioning now involves both a kernel matrix and the nonlinear geometry of the norm.

## Nonsmooth norms and sparse atoms {#rkbs-sparse-atoms}

Sparsity lives where the norm has corners, so the smooth theory above cannot deliver it. Nonsmooth spaces such as \(\ell_1\), spaces of finite measures, and total-variation spaces do not have a single smooth duality map. Optimality uses subgradients and extreme points. The representer can become sparse because an optimizer of a convex problem over a measure ball may be supported on finitely many atoms.

::: {.proposition #prop-rkbs-sparse-extreme}
[Proposition (finite atomic extreme solutions)]{.box-title}

For a convex data-fit problem depending on finitely many bounded linear measurements and regularized by a measure or atomic norm, an extreme minimizer can be chosen with finite atomic support, under the relevant compactness and existence conditions.

**Assumptions.** Convex lower-semicontinuous loss, bounded measurements, existence of a minimizer, and an atomic norm whose unit ball has the required compactness. **Proof status.** This is an abstract convex-geometric principle; precise support bounds and topologies depend on the space [@wang2024sparserkbs].
:::

This sparsity differs from support-vector sparsity. In an SVM, selected atoms are kernel sections centered at observed inputs. In a variation space, atom parameters may be learned continuously and need not coincide with training locations.

::: {.example #example-rkbs-dense-sparse}
[Example (quadratic and atomic regularization select different geometry)]{.box-title}

An RKHS squared norm spreads a solution across correlated directions because energy adds quadratically. An atomic norm pays for the total magnitude of selected atoms and can prefer a small set of extreme directions. Both estimators can interpolate the same observations, yet one expresses smooth distributed energy while the other expresses sparse learned features.

**Verification artifact.** checks/example-ch-rkbs-example-rkbs-dense-sparse.json records the example source hash and verification scope.
:::

<figure class="viz" data-figure="hilbert-vs-variation" data-alt="Two fitted curves nearly overlap a target made from two localized bumps. A coefficient plot shows the quadratic solution spreading weight across most correlated atoms, while the atomic-norm solution uses only a few nonzero coefficients."><figcaption>Similar prediction error can hide very different representations. A Hilbert penalty spreads energy smoothly across correlated atoms; a variation penalty pays total atomic mass and therefore lands on a sparse face of the feasible geometry.</figcaption></figure>

## Variation spaces and ridge splines {#variation-ridge-splines}

The most consequential client of this atomic machinery is the shallow neural network. A shallow ridge-function model has

$$
f(x)=\int a(w,b)\,\sigma(w^\top x-b)\,d\nu(w,b).
$$

Treating the coefficient as a signed measure and penalizing its total variation produces a Banach variation space. Under suitable activations and side conditions, extreme solutions are finite sums of ridge atoms, hence finite-width neural networks. The width bound comes from the number of measurements and convex geometry rather than being fixed before optimization [@parhi2021banach].

For truncated-power activations, these functions are multivariate ridge splines. The connection extends the spline story in [[ch:smoothing-splines-and-additive-rkhs]]: classical splines place knots in the input domain, while ridge splines learn oriented hyperplane knots.

This is a bridge to feature learning. A fixed RKHS optimizes coefficients in a fixed feature geometry. A variation norm can optimize the locations and directions of atoms while retaining a convex infinite-dimensional formulation. The finite neural network is a representer of that variational problem.

## Algorithms: from duality to column generation {#rkbs-algorithms}

A finite representer theorem guarantees that a sparse solution exists; it says nothing about how to find its atoms. Finite dictionaries reduce atomic regularization to lasso-like optimization. Continuous dictionaries require finding an atom most correlated with a residual or dual certificate. Conditional-gradient and exchange methods alternate between a linear minimization oracle and coefficient refitting.

:::: {.algorithm #algo-rkbs-column-generation}
[Algorithm (atomic column generation)]{.box-title}

**Input.** Measurements, a loss, an atomic family, a regularization level, and an oracle tolerance.

**Output.** A finite atomic predictor and a dual-gap certificate.

1. Initialize with an empty or small atom set.
2. Solve the restricted convex coefficient problem on active atoms.
3. Form the dual residual and find an atom maximizing its absolute correlation.
4. Add the atom, optionally remove inactive atoms, and refit coefficients.
5. Stop when the oracle value satisfies the dual optimality tolerance and the held-out objective is stable.
6. Report active atoms, primal and dual objectives, oracle accuracy, and sensitivity to initialization.

The outer convergence guarantee assumes an exact or controlled linear oracle. For neural ridge atoms that oracle can itself be nonconvex, so its approximation error must be distinguished from the convex master problem.
::::

## Statistical and computational tradeoffs {#rkbs-tradeoffs}

Changing the norm changes the capacity measure. RKHS analysis naturally uses eigenvalues, effective dimension, and Rademacher complexity of Hilbert balls. Atomic and variation spaces use dual norms, covering numbers, sparsity, or path-like complexity. A sparse representation can reduce evaluation cost without improving statistical error if the atom search overfits.

Banach methods also lose some computational conveniences. Gram matrices may no longer capture the entire problem, duality maps can be nonlinear, and nonreflexive spaces require weak-star arguments. The gain is a geometry aligned with robustness, sparsity, or learned features.

## Common mistakes and practical implications {#rkbs-practice}

- A Banach norm cannot generally be inserted into an RKHS derivation while keeping the same linear representer.
- Bounded evaluation must be verified for the chosen space and topology.
- Dual sparsity and primal sparsity are not identical.
- A finite atomic representer theorem guarantees existence, not that atom search is easy.
- Approximate linear oracles can invalidate claimed dual certificates.
- Learned atoms need independent regularization and validation even when the master problem is convex.
- A neural-network representation does not imply that ordinary nonconvex network training solves the variation-space problem.

Use RKBS or variation geometry when the desired inductive bias is genuinely nonquadratic. The extra functional analysis is justified only if it changes sparsity, robustness, or feature learning in a measurable way.

## Summary and further reading {#rkbs-summary}

RKBSs preserve bounded evaluation while replacing Hilbert inner products by duality. Smooth Banach spaces yield representer principles in the dual; nonsmooth measure and atomic norms yield finite sparse extreme solutions. Variation spaces connect continuous-domain regularization to finite-width neural ridge splines. Foundations appear in [@zhang2009rkbs], sparse representers in [@wang2024sparserkbs], and the neural ridge-spline bridge in [@parhi2021banach].

## Exercises {#exercises}

1. [warm-up]{.ex-tag} Identify the exact steps of the standard RKHS representer proof that fail without an inner product.
2. [computation]{.ex-tag} Compute the normalized duality map and its inverse for a finite-dimensional \(\ell_p\) space at a supplied vector.
3. [proof]{.ex-tag} Derive the first-order optimality condition showing that the norming functional of a smooth RKBS solution lies in the span of evaluation functionals.
4. [challenge]{.ex-tag} Prove a finite-support bound for an extreme solution of an atomic-norm interpolation problem with finitely many scalar measurements.
5. [synthesis]{.ex-tag} Compare an RKHS estimator, a finite-dictionary lasso, and a continuous variation-space model on the same regression task. Specify capacity control, optimization certificates, sparsity, compute, and extrapolation diagnostics.
