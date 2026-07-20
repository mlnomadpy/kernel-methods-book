---
id: ch-spatial
slug: spatial-and-spatiotemporal-kernels
title: Spatial and Spatiotemporal Kernel Models
part: XV · Classical and Reliable Kernel Models
order: 50
tier: advanced
prerequisites:
  - kernel-families
  - gaussian-processes-and-rvm
  - geometric-and-equivariant-kernels
objectives:
  - >-
    Distinguish covariance, correlation, and variogram descriptions of spatial
    dependence.
  - >-
    Construct anisotropic, nonstationary, and space-time kernels without losing
    positive definiteness.
  - >-
    Derive ordinary and universal kriging as Gaussian-process prediction with an
    explicit trend space.
  - >-
    Explain the Matérn SPDE connection and the computational value of sparse
    precision matrices.
  - Design validation that respects spatial and temporal dependence.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-spatial.yml
verification_date: null
bibliography:
  - matern1960
  - stein1999
  - lindgren2011spde
  - gneiting2002space
---
# Spatial and Spatiotemporal Kernel Models

<p class="lead">Spatial data violate the independent-sample picture: nearby observations share weather, geology, ecology, or measurement conditions, and prediction is often required at an unobserved location rather than for another exchangeable case. Covariance kernels provide the geometry, kriging supplies the predictor, and sparse stochastic differential constructions make large fields computationally accessible.</p>

## Random fields, covariance, and variograms {#spatial-random-fields}

Let \(Z(s)\) be a second-order random field indexed by a spatial location \(s\in\mathcal D\). Write \(m(s)=\mathbb E Z(s)\) and

$$
C(s,t)=\operatorname{Cov}\{Z(s),Z(t)\}.
$$

The covariance function must be positive definite, exactly as in [[ch:kernels-and-rkhs]]. A stationary model has \(C(s,t)=c(s-t)\); an isotropic model further reduces this to a function of distance.

::: {.definition #def-semivariogram}
[Definition (semivariogram)]{.box-title}

For an intrinsically stationary field, the semivariogram is

$$
\gamma(h)=\frac12\operatorname{Var}\{Z(s+h)-Z(s)\}.
$$

When a stationary covariance exists, \(\gamma(h)=C(0)-C(h)\). More generally, a valid variogram is conditionally negative definite rather than positive definite.
:::

Variograms remove an unknown constant mean through increments and can remain meaningful when the field has no finite stationary variance. Confusing a variogram with a covariance reverses the geometry: small variogram means strong similarity, while large covariance means strong similarity.

## Kriging as constrained kernel prediction {#kriging}

Suppose \(y_i=Z(s_i)+\varepsilon_i\) with independent noise variance \(\tau^2\). A linear predictor at \(s_0\) has the form \(\widehat Z(s_0)=w^\top y\). Under a known constant mean, ordinary kriging minimizes prediction variance subject to \(1^\top w=1\).

:::: {.theorem #thm-ordinary-kriging}
[Theorem (ordinary kriging system)]{.box-title}

Let \(K_{ij}=C(s_i,s_j)+\tau^2\mathbf 1\{i=j\}\) and \(k_0=(C(s_i,s_0))_i\). The minimum-variance unbiased linear predictor solves

$$
\begin{pmatrix}K&1\\1^\top&0\end{pmatrix}
\begin{pmatrix}w\\\eta\end{pmatrix}
=
\begin{pmatrix}k_0\\1\end{pmatrix}.
$$

**Assumptions.** The covariance and noise model are second-order valid; the displayed saddle system is nonsingular; the mean is an unknown constant. **Proof status.** Proved by minimizing the quadratic prediction variance with a Lagrange multiplier for unbiasedness [@stein1999].
::::

Universal kriging replaces the constant by a trend \(m(s)=p(s)^\top\beta\) and imposes \(P^\top w=p(s_0)\). This is the spatial analogue of the null-space treatment in [[ch:smoothing-splines-and-additive-rkhs]]. With Gaussian assumptions, the kriging predictor and variance are the Gaussian-process posterior mean and variance. Without Gaussianity, they remain best linear unbiased quantities, but not a complete posterior distribution.

::: {.example #example-spatial-validation}
[Example (random validation can conceal spatial failure)]{.box-title}

Suppose a monitoring network contains dense local clusters and sparse remote regions. A random train-test split places neighbors of most test sites in training, so interpolation appears excellent. Holding out entire spatial blocks instead measures transfer across distance and may reveal bias from a misspecified range or trend. The correct split follows the intended deployment geometry.

**Verification artifact.** checks/example-ch-spatial-example-spatial-validation.json records the example source hash and verification scope.
:::

## Anisotropy and nonstationarity {#spatial-anisotropy}

Geometric anisotropy replaces Euclidean distance by

$$
r_A(s,t)=\lVert A(s-t)\rVert,
$$

where \(A\) rotates and rescales directions. If \(c(r)\) is a valid isotropic covariance, then \(c(r_A)\) remains valid. This models direction-dependent ranges but not location-dependent behavior.

Nonstationary models may use spatially varying length scales, local mixtures, deformation maps, or process convolutions. A deformation \(u:\mathcal D\to\mathbb R^p\) gives \(C(s,t)=c\{u(s)-u(t)\}\), which is valid by composition. It can nevertheless fold distant locations together or become weakly identifiable. Maps and local range estimates should be inspected directly.

Compactly supported kernels from approximation theory create sparse covariance matrices when observation pairs beyond a range have zero covariance. Sparsity is valuable, but a hard support radius is a scientific assumption. Tapering an existing covariance alters both prediction and uncertainty and needs sensitivity analysis [@matern1960; @stein1999].

## Matérn fields and the SPDE connection {#spatial-spde}

The Matérn covariance links smoothness, dimension, and differential operators. In Euclidean space, a Matérn field can be characterized formally through an SPDE of the form

$$
(\kappa^2-\Delta)^{\alpha/2}Z=\mathcal W,
$$

where \(\mathcal W\) is spatial white noise and \(\alpha\) controls smoothness relative to dimension. Discretizing the weak equation in a local finite-element basis produces a Gaussian Markov random field with a sparse precision matrix [@lindgren2011spde].

::: {.proposition #prop-spde-sparsity}
[Proposition (local basis yields sparse precision)]{.box-title}

When the weak SPDE is discretized in compactly supported basis functions, precision-matrix entries vanish for basis pairs whose supports do not interact through the local differential operator.

**Assumptions.** The operator is local, the weak formulation is well defined, and the chosen basis functions have compact support. **Proof status.** Follows by inspection of the mass and stiffness integrals; nonoverlapping local supports contribute zero [@lindgren2011spde].
:::

This changes the computational object from a dense covariance to a sparse precision. It does not make inference exact automatically: mesh resolution, boundary treatment, numerical quadrature, and hyperparameter approximation introduce additional errors.

## Space-time covariance design {#spatiotemporal-kernels}

A separable space-time covariance has

$$
C\{(s,t),(s',t')\}=C_S(s,s')C_T(t,t').
$$

It is valid and computationally convenient, especially on a complete grid where Kronecker algebra applies. It assumes that temporal correlation has the same shape at every spatial scale and that spatial correlation evolves identically at every lag. Those restrictions can be unrealistic.

Nonseparable constructions couple spatial and temporal lag while preserving positive definiteness. The Gneiting family is a standard example in which temporal separation modifies both spatial range and marginal scale [@gneiting2002space]. Other constructions arise from advection-diffusion equations, latent dynamic factors, and state-space models.

Irregularly sampled time series need not form a rectangular grid. Matrix-free covariance products, inducing points, state-space representations, or sparse precision approximations can exploit structure without filling missing grid cells with invented observations.

## Multivariate fields and change of support {#multivariate-spatial}

Several physical quantities observed at the same locations require a matrix-valued covariance. Linear models of coregionalization use

$$
K(s,t)=\sum_r c_r(s,t)B_r,
$$

which is the spatial version of the operator-valued construction in [[ch:vector-and-operator-valued-kernels]]. Cross-covariances cannot be chosen independently; the entire block kernel must be positive definite.

Spatial observations may represent points, pixel averages, administrative regions, or sensor footprints. If an observation is an average over region \(A\), its covariance with an average over \(B\) is obtained by integrating the point covariance over both supports. Treating region averages as point values at centroids understates this change-of-support uncertainty.

:::: {.algorithm #algo-spatial-workflow}
[Algorithm (spatial kernel workflow)]{.box-title}

**Input.** Georeferenced observations, support geometry, trend covariates, measurement-error information, and a deployment region.

**Output.** Predictions and uncertainty validated at the deployment scale.

1. Plot the sampling geometry, supports, missingness, and response against coordinates and time.
2. Fit the trend without using test regions, then inspect directional residual variograms.
3. Compare stationary, anisotropic, and scientifically motivated nonstationary kernels.
4. Estimate covariance parameters with likelihood or restricted likelihood, recording bounds and numerical jitter.
5. Validate with spatial or space-time blocks whose separation matches deployment.
6. Report predictive scores, interval coverage, residual spatial dependence, condition estimates, and sensitivity to mesh or inducing resolution.

Dense factorization is cubic in sample size. Sparse precision, compact support, Kronecker structure, and iterative solves can reduce cost, but each changes the diagnostics that must be reported.
::::

## Common mistakes and practical implications {#spatial-practice}

- Random cross-validation usually overstates performance when nearby observations share latent conditions.
- A nugget can represent measurement error, microscale variation, or both; those interpretations are not interchangeable.
- Fitting a flexible mean and a long-range covariance simultaneously can be weakly identifiable.
- Latitude and longitude should not be treated as planar coordinates over large regions.
- A positive-definite spatial kernel need not remain valid after an arbitrary space-time modification.
- Posterior standard deviations omit covariance-selection and mesh uncertainty unless those are explicitly integrated.

Spatial kernels are most useful when geometry, support, and deployment are specified before model selection. Maps of uncertainty and residual dependence are primary diagnostics, not decorative outputs.

## Summary and further reading {#spatial-summary}

Spatial kernel models combine a trend, a valid covariance or variogram, and a validation design that respects dependence. Kriging is constrained kernel prediction and coincides with GP posterior prediction under Gaussian assumptions. Anisotropy, nonstationarity, space-time coupling, and change of support require explicit modeling. Matérn SPDE methods replace dense covariance algebra with sparse precision algebra. Classical treatments include [@matern1960] and [@stein1999], with the SPDE construction in [@lindgren2011spde] and nonseparable space-time kernels in [@gneiting2002space].

## Exercises {#exercises}

1. [warm-up]{.ex-tag} Explain the sign and definiteness difference between a covariance function and a semivariogram.
2. [computation]{.ex-tag} Solve an ordinary-kriging saddle system for a supplied covariance matrix and target covariance vector, then compute the prediction variance.
3. [proof]{.ex-tag} Prove that an anisotropic covariance obtained by applying a fixed linear map to spatial differences remains positive definite.
4. [challenge]{.ex-tag} Derive the covariance between two region-average observations from a point-level covariance and state a numerical quadrature strategy.
5. [synthesis]{.ex-tag} Design a blocked evaluation for a space-time environmental monitoring problem, including trend selection, covariance alternatives, support geometry, uncertainty scores, and scaling choices.
