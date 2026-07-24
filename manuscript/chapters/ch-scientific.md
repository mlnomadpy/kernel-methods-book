---
id: ch-scientific
slug: scientific-computing-and-operator-learning
title: Kernels for Scientific Computing and Operator Learning
part: XVI · Dynamics and Scientific Learning
order: 55
tier: advanced
prerequisites:
  - mercer-and-rates
  - vector-and-operator-valued-kernels
  - inverse-learning-and-spectral-regularization
  - gaussian-processes-and-rvm
objectives:
  - >-
    Construct kernel collocation methods for differential equations and boundary
    constraints.
  - >-
    Derive covariance formulas for derivative and linear-functional
    observations.
  - >-
    Separate discretization, approximation, regularization, and observation
    errors.
  - >-
    Interpret probabilistic numerical solvers without overstating posterior
    calibration.
  - >-
    Compare kernel operator regression with neural operators across resolution
    and geometry.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-scientific.yml
verification_date: null
bibliography:
  - green1984
  - wendland2005
  - raissi2017gpde
  - cockayne2019probnum
  - li2020fno
---
# Kernels for Scientific Computing and Operator Learning

<p class="lead">A curve that passes through every measurement can still be physically impossible: it can violate the equation that governs the field, contradict the boundary conditions, or fail to conserve what the physics conserves. Scientific learning asks more than prediction at another row of a data table; it may require a function satisfying a differential equation, an inverse coefficient compatible with boundary measurements, or an operator mapping one field to another across discretizations. Ordinary kernel regression, built on point evaluations of a scalar response, does not obviously accommodate derivatives, fluxes, integrals, or equation residuals. It does, and cheaply: every bounded linear functional of an RKHS function has a representer obtained by applying the functional to a kernel argument. From that one fact we build symmetric collocation solvers for boundary-value problems, Gaussian processes conditioned on derivative and residual observations, probabilistic numerical methods with honest interpretations, and kernel operator regression that stands comparison with neural operators, provided numerical error is decomposed rather than hidden.</p>

## Linear information about an unknown function {#scientific-linear-information}

What do a thermocouple reading, a measured flux, and a PDE residual have in common? Each extracts one number from the unknown function in a linear way, and that shared structure is all the theory needs. Let \(u\in\mathcal H_k\) be an unknown function and suppose the available information consists of bounded linear functionals

$$
y_i=L_i u+\varepsilon_i.
$$

Point evaluation, derivatives, integrals, boundary traces, and differential-equation residuals all fit this form when they are bounded on the chosen RKHS. By the Riesz representation theorem, each functional has a representer \(r_i\in\mathcal H_k\) satisfying \(L_i u=\langle u,r_i\rangle_{\mathcal H_k}\).

:::: {.theorem #thm-functional-representer}
[Theorem (representer theorem for linear scientific information)]{.box-title}

The minimum-norm regularized estimator

$$
\min_{u\in\mathcal H_k}\sum_i\ell_i(y_i,L_i u)+\lambda\lVert u\rVert_{\mathcal H_k}^2
$$

has a solution in the span of the functional representers:

$$
\widehat u=\sum_i c_i r_i.
$$

For a sufficiently smooth scalar kernel, \(r_i(x)=L_i^{(z)}k(x,z)\), with the functional applied to the indicated kernel argument.

**Assumptions.** Each \(L_i\) is bounded on \(\mathcal H_k\); the regularized problem has a minimizer; the penalty is strictly increasing in the norm. **Proof status.** Proved by orthogonally decomposing \(u\) into the representer span and its complement.
::::

The Gram matrix becomes

$$
G_{ij}=L_i^{(x)}L_j^{(z)}k(x,z).
$$

Positive semidefiniteness follows because \(G\) is the Gram matrix of the Riesz representers. Differentiating a kernel formula is legitimate only when the required derivatives exist and evaluation remains bounded.

## Symmetric kernel collocation {#kernel-collocation}

The most direct use of functional representers is to solve a differential equation by regression: make the equation and its boundary conditions the observations. Consider a linear boundary-value problem

$$
\mathcal A u=f\quad\text{in }\Omega,
\qquad
\mathcal B u=g\quad\text{on }\partial\Omega.
$$

Choose interior points and boundary points, then create functionals from \(\mathcal A\) and \(\mathcal B\) evaluated at those locations. Solving the functional Gram system produces a function that matches the equation and boundary information at the collocation sites.

::: {.proposition #prop-collocation-psd}
[Proposition (validity of the symmetric collocation matrix)]{.box-title}

If all collocation functionals are bounded on \(\mathcal H_k\), the matrix obtained by applying each pair of functionals to the two kernel arguments is positive semidefinite.

**Assumptions.** The kernel is differentiable to the order required by the differential and boundary operators; the functionals are bounded. **Proof status.** Proved by identifying the matrix with the Gram matrix of functional representers.
:::

Unsymmetric collocation applies the operator to only one argument and can be cheaper, but it loses this direct symmetric positive-definite structure. Least-squares collocation uses more residual points than basis centers and can improve robustness to noise and inconsistent information.

Native-space error estimates relate fill distance, kernel smoothness, and target regularity. A very smooth kernel is not automatically better: it can be badly conditioned and can impose regularity the true solution does not possess [@wendland2005]. Shape-parameter selection must balance approximation and numerical stability.

::: {.example #example-scientific-boundary}
[Example (boundary constraints change the admissible space)]{.box-title}

Two kernel solvers may use the same interior differential residuals but different boundary functionals. Their solutions can agree at every interior collocation site and still represent different boundary-value problems. Boundary conditions are part of the operator and function space, not an afterthought to be patched onto an unconstrained regression fit.

**Verification artifact.** checks/example-ch-scientific-example-scientific-boundary.json records the example source hash and verification scope.
:::

## Green kernels and physics-informed covariance {#green-physics-kernels}

When a Green function for \(\mathcal A\) and the boundary conditions is available, it can define a kernel whose sections already respect the operator geometry. This mirrors the spline construction in [[ch:smoothing-splines-and-additive-rkhs]]. Physics can also be encoded by applying projection operators to a base kernel, yielding divergence-free, curl-free, or conservation-compatible vector fields.

A hard constraint restricts the hypothesis space. A soft residual penalty allows model discrepancy:

$$
\sum_i\{y_i-u(x_i)\}^2
+\rho\sum_j\{\mathcal A u(z_j)-f(z_j)\}^2
+\lambda\lVert u\rVert_{\mathcal H_k}^2.
$$

The parameter \(\rho\) expresses trust in the equation relative to observations. Sending it to infinity is justified only when the numerical operator and physical model are exact at the required scale.

## Gaussian processes with differential observations {#gp-differential-observations}

Collocation returns a single function; it does not say how much to trust it between the collocation sites. A probabilistic reading of the same construction does. If \(u\sim\mathcal{GP}(m,k)\), bounded linear transformations remain jointly Gaussian. For linear functionals \(L_i\) and \(M_j\),

$$
\operatorname{Cov}(L_i u,M_j u)
=L_i^{(x)}M_j^{(z)}k(x,z).
$$

This permits conditioning on values, derivatives, fluxes, integrals, and linear PDE residuals in one Gaussian system. Unknown forcing or coefficients make the problem nonlinear and may require approximation, hierarchical priors, or alternating inference [@raissi2017gpde].

The posterior covariance measures uncertainty under the GP prior, observation model, and exact functional evaluations. Numerical differentiation, operator discretization, boundary approximation, and kernel selection are additional sources of uncertainty. A narrow posterior obtained with a misspecified operator is not a calibrated numerical guarantee.

## Probabilistic numerical methods {#probabilistic-numerics}

The GP treatment of a differential equation is one instance of a broader stance: treat numerical error itself as something to be modeled, not merely bounded. A probabilistic numerical method places a probability model on an unknown mathematical object and conditions on information produced by a numerical procedure. Bayesian quadrature in [[ch:kernel-quadrature-and-herding]] is one example; differential-equation solvers are another [@cockayne2019probnum].

Three questions keep the interpretation honest:

1. What is random: epistemic uncertainty about a fixed function, a genuinely random physical field, or a device for numerical error representation?
2. Which information operator was observed, and was it evaluated exactly?
3. Is posterior contraction calibrated to actual numerical error under a stated class of problems?

Posterior variance can guide adaptive mesh refinement by selecting information where uncertainty or an error indicator is large. The selection policy then becomes part of the method and must be tested against deterministic residual-based refinement.

## Inverse scientific problems {#scientific-inverse-problems}

So far the unknown was the solution field itself. Often the real target sits one level deeper: the conductivity, source, or material coefficient that produced the field we can measure. In an inverse problem, observations depend on an unknown coefficient through a forward solution operator:

$$
y=\mathcal G(a)+\varepsilon.
$$

Kernel regularization may act on \(a\), on the state \(u\), or on both. Linear inverse problems connect directly to [[ch:inverse-learning-and-spectral-regularization]]. Nonlinear forward maps require local linearization, adjoints, sampling, or surrogate models.

Identifiability precedes optimization. If distinct coefficients produce indistinguishable observations, a smooth kernel selects one representative but does not recover information absent from the experiment. Report the observation operator, sensitivity spectrum, prior or RKHS penalty, and uncertainty conditional on nonidentifiable directions.

## Learning operators between function spaces {#operator-learning}

Solving one boundary-value problem per parameter setting is wasteful when the same equation must be solved thousands of times. The alternative is to learn the solution map itself. Many scientific tasks observe pairs of functions \((a_i,u_i)\) and seek an operator \(\mathcal G:a\mapsto u\). A scalar kernel on input functions combined with an operator-valued output kernel yields

$$
\widehat{\mathcal G}(a)=\sum_i K(a,a_i)c_i,
$$

where each coefficient may itself be a function. Basis discretization reduces the problem to the vector-valued framework of [[ch:vector-and-operator-valued-kernels]].

Resolution transfer requires more than evaluating a matrix on a finer grid. The input and output spaces, sampling operators, and norms must be defined independently of discretization. Training and testing on different meshes is a necessary diagnostic.

Neural operators learn nonlinear maps between function spaces. Fourier neural operators parameterize an integral kernel in spectral coordinates and have become an important comparator for parametric PDE families [@li2020fno]. Kernel operator regression offers convex fitting for fixed representations and transparent regularization; neural operators offer learned representations and often greater expressivity. Comparisons should match training trajectories, meshes, parameter counts, error norms, and rollout horizons.

:::: {.algorithm #algo-scientific-kernel-workflow}
[Algorithm (auditable scientific kernel solver)]{.box-title}

**Input.** A domain, operators and boundary conditions, observations or simulation pairs, a kernel family, and target error norms.

**Output.** A continuous estimator or learned operator with decomposed error diagnostics.

1. State the continuous problem, units, boundary conditions, and observation functionals before discretization.
2. Choose a kernel whose native space supports every required functional and plausible solution regularity.
3. Separate training sites, validation functions or parameter settings, and test meshes.
4. Solve the regularized functional Gram system with scaling, preconditioning, and residual monitoring.
5. Measure data error, equation residual, boundary residual, discretization sensitivity, and conditioning separately.
6. Compare against an established numerical method and, for operator learning, a resolution-independent neural baseline.

Stop adaptive refinement when the target norm and independent residual indicator stabilize, not merely when posterior variance becomes small.
::::

## Common mistakes and practical implications {#scientific-practice}

- Differentiating a nonsmooth kernel beyond its supported order invalidates the functional Gram matrix.
- Small collocation residual at training sites does not guarantee small solution error between sites.
- A physics penalty does not compensate for wrong boundary conditions.
- Treating discretized vectors as functions without specifying sampling and norms makes resolution claims meaningless.
- Posterior numerical uncertainty can omit model and discretization error.
- Comparing a kernel solver with a neural operator on different meshes or training budgets is inconclusive.
- Ill-conditioning can dominate approximation error for flat or excessively smooth kernels.

Scientific kernel models are most compelling when the continuous information operators are explicit and when numerical error is decomposed rather than hidden inside a single test metric.

## Summary and further reading {#scientific-summary}

Bounded linear scientific information has RKHS representers obtained by applying functionals to kernel arguments. This yields symmetric collocation, derivative-observation GPs, and constrained vector fields. Probabilistic numerical methods attach uncertainty to numerical information under an explicit model. Operator regression extends the same ideas to maps between functions and provides a principled comparator to neural operators. See [@wendland2005], [@cockayne2019probnum], [@raissi2017gpde], and [@li2020fno].

## Exercises {#exercises}

1. [warm-up]{.ex-tag} State the smoothness needed to use point, first-derivative, and second-derivative observations with an RKHS kernel.
2. [computation]{.ex-tag} Form the functional Gram matrix for value and derivative observations of a one-dimensional RBF kernel and verify its symmetry.
3. [proof]{.ex-tag} Prove positive semidefiniteness of the symmetric collocation matrix using Riesz representers.
4. [challenge]{.ex-tag} Derive a kernel formulation that enforces a linear boundary condition exactly while penalizing an interior differential residual softly.
5. [synthesis]{.ex-tag} Design a resolution-transfer benchmark comparing kernel operator regression and a Fourier neural operator on a parametric PDE family. Specify meshes, norms, boundary handling, compute, uncertainty, and failure diagnostics.
