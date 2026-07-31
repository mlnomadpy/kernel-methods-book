---
id: ch-spatial
slug: spatial-and-spatiotemporal-kernels
title: Spatial and Spatiotemporal Kernel Models
part: VI · Designing Kernels
order: 31
tier: advanced
prerequisites:
  - kernel-families
  - gaussian-processes-and-rvm
  - geometric-and-equivariant-kernels
objectives:
  - 'Distinguish covariance, generalized covariance, and variogram models.'
  - Derive ordinary and universal kriging and their prediction variances.
  - >-
    Reconstruct the Matérn SPDE finite-element precision and its approximation
    boundary.
  - Prove validity of the Gneiting nonseparable space-time covariance class.
  - Design change-of-support and blocked validation analyses.
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
  - rasmussen2006
example_code_policy: visible-for-executable
narrative_link_policy: exact
---
# Spatial and Spatiotemporal Kernel Models

<p class="lead">Two rain gauges a kilometer apart do not deliver two independent readings; they sit under the same storm. Spatial data breaks the exchangeable-sample story twice: observations are correlated through shared weather, geology, and measurement conditions, and the prediction that matters is usually at an unmonitored map location rather than at another random draw. A random train-test split can therefore flatter a model by placing a test station beside its training neighbors. This chapter turns the dependence itself into the object of study. It derives kriging as constrained kernel projection, reconstructs how the Matérn SPDE turns a dense field into a sparse finite-element model, proves a major class of nonseparable space-time kernels, and builds validation around the geometry of deployment.</p>

## Random fields, covariance, and variograms {#spatial-random-fields}

Let \(\mathcal D\subseteq\mathbb R^d\) and let
\(\{Z(s):s\in\mathcal D\}\) be a second-order random field:
\(\mathbb E Z(s)^2\lt\infty\) for every \(s\). Define

$$
m(s)=\mathbb EZ(s),
\qquad
C(s,t)=
\mathbb E[\{Z(s)-m(s)\}\{Z(t)-m(t)\}].
$$

For any sites \(s_1,\ldots,s_n\) and coefficients \(a\),

$$
\sum_{i,j}a_i a_jC(s_i,s_j)
=\operatorname{Var}\left\{\sum_i a_iZ(s_i)\right\}\ge0.
$$

A covariance is therefore a positive-semidefinite kernel. Weak stationarity
means \(m(s)=\mu\) and \(C(s,t)=c(s-t)\). Isotropy further restricts
\(c(h)\) to a function of \(\lVert h\rVert\).

Not every spatial model has a stationary finite variance. Increment geometry
leads to a broader object.

::: {.definition #def-semivariogram}
[Definition (semivariogram and intrinsic stationarity)]{.box-title}

A field is intrinsically stationary when
\(\mathbb E\{Z(s+h)-Z(s)\}=0\) and

$$
\gamma(h)=\frac12
\operatorname{Var}\{Z(s+h)-Z(s)\}
$$

depends only on \(h\). A valid variogram is conditionally negative definite:

$$
\sum_{i,j}a_i a_j\gamma(s_i-s_j)\le0
\quad\text{whenever}\quad
\sum_i a_i=0.
$$
:::

If a stationary covariance exists,
\(\gamma(h)=C(0)-C(h)\). The conditional sign follows because the constant
\(C(0)\) vanishes for zero-sum coefficients and the covariance quadratic form
is nonnegative. A variogram is not a covariance matrix with a reversed color
scale. It acts on a constrained coefficient space.

Three small-scale effects must also be separated:

- measurement error adds independent variance to observed values but not to
  the latent field;
- a discontinuity of the latent covariance at the origin represents
  microscale variation;
- numerical jitter is an algorithmic stabilization and is not a scientific
  variance component.

Calling all three a nugget obscures what is being predicted.

The previous chapter's null-space lesson now becomes a mean-model lesson.
[[ch:smoothing-splines-and-additive-rkhs|Smoothing splines]] separated an
unpenalized trend from a penalized fluctuation. Universal kriging makes the
same algebraic split, but gives the fluctuation a second-order stochastic
meaning. Confusing those meanings produces a common error: a numerically valid
kernel solve is reported as a calibrated spatial probability model even
though neither the trend nor the covariance has survived spatial validation.

## Kriging as constrained kernel projection {#kriging}

Suppose

$$
y_i=Z(s_i)+\varepsilon_i,
\qquad
\operatorname{Cov}(\varepsilon)=\Sigma_\varepsilon,
$$

with measurement error independent of \(Z\). Let

$$
K_{ij}=C(s_i,s_j)+(\Sigma_\varepsilon)_{ij},
\qquad
k_0=(C(s_i,s_0))_{i=1}^n,
\qquad
C_{00}=C(s_0,s_0).
$$

Assume the mean lies in the trend space

$$
m(s)=p(s)^\top\beta,
$$

where \(p(s)\in\mathbb R^q\), \(P_{ia}=p_a(s_i)\), and
\(p_0=p(s_0)\). A linear predictor \(w^\top y\) is unbiased for every
\(\beta\) exactly when

$$
P^\top w=p_0.
$$
{#eq-spatial-1}

:::: {.theorem #thm-universal-kriging}
[Theorem (universal kriging system and variance)]{.box-title}

**Assumptions.** The four second-order, definiteness, rank, and estimability assumptions stated below are all in force.

Assume:

1. \(Z\) is second order with covariance \(C\);
2. \(K\) is positive definite;
3. \(P\) has column rank \(q\);
4. \(p_0\) lies in the estimable trend space.

Then the unique minimum-variance linear unbiased predictor of \(Z(s_0)\) is
\(\widehat Z(s_0)=w^\top y\), where

$$
\begin{pmatrix}
K&P\\
P^\top&0
\end{pmatrix}
\begin{pmatrix}w\\\eta\end{pmatrix}
=
\begin{pmatrix}k_0\\p_0\end{pmatrix}.
$$
{#eq-spatial-2}

Its mean-squared prediction error is

$$
\sigma_K^2(s_0)
=C_{00}-w^\top k_0-p_0^\top\eta.
$$
{#eq-spatial-3}

Equivalently, with \(B=P^\top K^{-1}P\),

$$
w=K^{-1}k_0+
K^{-1}PB^{-1}(p_0-P^\top K^{-1}k_0)
$$
{#eq-spatial-4}

and

$$
\sigma_K^2(s_0)
=C_{00}-k_0^\top K^{-1}k_0+
(p_0-P^\top K^{-1}k_0)^\top
B^{-1}
(p_0-P^\top K^{-1}k_0).
$$
{#eq-spatial-5}

**Proof status.** Complete below. The projection and covariance analysis are
classical; see [@stein1999].
::::

:::: {.proof}
[Proof]{.box-title}

Unbiasedness follows from

$$
\mathbb E(w^\top y-Z(s_0))
=\beta^\top(P^\top w-p_0),
$$

which vanishes for every \(\beta\) exactly under [[eq:eq-spatial-1]]. The prediction-error
variance is

$$
V(w)=C_{00}-2w^\top k_0+w^\top Kw.
$$

Minimize \(V(w)\) subject to [[eq:eq-spatial-1]]. The Lagrangian

$$
\mathcal L(w,\eta)
=\tfrac12w^\top Kw-w^\top k_0
+\eta^\top(P^\top w-p_0)
$$

has first-order equations \(Kw+P\eta=k_0\) and
\(P^\top w=p_0\), giving [[eq:eq-spatial-2]]. Positive definiteness of \(K\) makes
\(V\) strictly convex, so the constrained minimizer is unique.

Multiplying \(Kw+P\eta=k_0\) by \(w^\top\) gives
\(w^\top Kw=w^\top k_0-p_0^\top\eta\). Substitution in \(V(w)\)
yields [[eq:eq-spatial-3]]. Solving the first block equation for \(w\), substituting into
the constraint, and solving for \(\eta\) gives [[eq:eq-spatial-4]]. Substituting [[eq:eq-spatial-4]] into
[[eq:eq-spatial-3]] yields [[eq:eq-spatial-5]]. The matrix \(B\) is positive definite because \(K\) is
positive definite and \(P\) has full column rank. [\(\square\)]{.qed}
::::

Ordinary kriging is the case \(p(s)=1\). Simple kriging treats the mean as
known and omits the constraint. Under joint Gaussianity, [[eq:eq-spatial-4]] and [[eq:eq-spatial-5]] are
the posterior mean and variance after integrating a diffuse linear trend.
Without Gaussianity, they remain best linear unbiased quantities, not a full
posterior law.

**RKHS projection and what variance means.** {#kriging-rkhs}

If \(C\) is an RKHS kernel and observations are noise free, simple kriging
chooses coefficients minimizing

$$
\left\lVert
C(s_0,\cdot)-\sum_iw_iC(s_i,\cdot)
\right\rVert_{\mathcal H_C}^2.
$$

The squared residual norm is
\(C_{00}-k_0^\top K^{-1}k_0\), the same power function studied in
[[ch:kernel-interpolation-and-approximation]]. Universal kriging adds
polynomial-reproduction constraints, just as a conditionally positive-definite
spline adds a null space.

This identity unifies two calculations but not two guarantees. The RKHS power
function is a worst-case deterministic error multiplier for targets in an RKHS
ball. Kriging variance is a model-based mean-square error under a covariance
model. Neither establishes frequentist coverage after estimating covariance
parameters.

:::: {.example #example-spatial-validation}
[Example (the price of an unknown mean)]{.box-title}

Take two noiseless observations at \(s_1=0\) and \(s_2=2\), predict at
\(s_0=1\), and use the exponential covariance

$$
C(s,t)=e^{-\lvert s-t\rvert}.
$$

Then

$$
K=
\begin{pmatrix}1&e^{-2}\\e^{-2}&1\end{pmatrix},
\qquad
k_0=
\begin{pmatrix}e^{-1}\\e^{-1}\end{pmatrix}.
$$

For ordinary kriging, symmetry and \(1^\top w=1\) give
\(w=(1/2,1/2)^\top\). With the multiplier convention in [[eq:eq-spatial-2]],

$$
\eta=e^{-1}-\frac{1+e^{-2}}2\approx-0.1998.
$$

For observations \(y=(0,2)^\top\), the prediction is \(1\), and

$$
\sigma_{\mathrm{OK}}^2
=1-e^{-1}-\eta
\approx0.8319.
$$

If the mean is known to be zero, simple kriging uses

$$
w_{\mathrm{SK}}
=K^{-1}k_0
=\frac{e^{-1}}{1+e^{-2}}
\begin{pmatrix}1\\1\end{pmatrix}
\approx
\begin{pmatrix}0.3240\\0.3240\end{pmatrix},
$$

and

$$
\sigma_{\mathrm{SK}}^2
=1-\frac{2e^{-2}}{1+e^{-2}}
=\tanh(1)
\approx0.7616.
$$

Ordinary kriging pays extra variance to remain unbiased for every constant
mean. It also forces weights to sum to one, while simple kriging shrinks toward
its known mean. If the zero-mean assumption is false, the smaller simple
kriging variance is not evidence of a better predictor.

The following code solves both systems without an explicit inverse and checks
the unbiasedness constraint and the displayed variances. That distinction
matters: the ordinary predictor spends variance to protect against an unknown
constant mean, whereas the simple predictor spends no such budget because its
zero mean is an assumption.

```python
import numpy as np

K = np.array([[1.0, np.exp(-2.0)], [np.exp(-2.0), 1.0]])
k0 = np.full(2, np.exp(-1.0))
y = np.array([0.0, 2.0])

augmented = np.block([[K, np.ones((2, 1))],
                      [np.ones((1, 2)), np.zeros((1, 1))]])
solution = np.linalg.solve(augmented, np.r_[k0, 1.0])
w_ok, eta = solution[:2], solution[2]
w_sk = np.linalg.solve(K, k0)
var_ok = 1.0 - w_ok @ k0 - eta
var_sk = 1.0 - k0 @ w_sk

assert np.allclose(w_ok, [0.5, 0.5])
assert np.isclose(w_ok.sum(), 1.0)
assert np.isclose(w_ok @ y, 1.0)
assert np.isclose(var_ok, 0.8319074)
assert np.isclose(var_sk, np.tanh(1.0))
print(w_ok, w_sk, var_ok, var_sk)
```

For validation, a random split of a dense monitoring network often places a
nearby station beside each held-out site. A spatial block split asks the harder
and operationally relevant question represented here: what happens when the
target is separated from every training location?

**Verification scope.** Every displayed weight and variance follows from the
two-by-two system. The example verifies algebra, not covariance-model
adequacy.
::::

## Anisotropy, nonstationarity, and identifiability {#spatial-anisotropy}

Geometric anisotropy replaces Euclidean distance by

$$
r_A(s,t)=\lVert A(s-t)\rVert.
$$

If \(c(u-v)\) is positive definite on \(\mathbb R^r\), then
\(C_A(s,t)=c\{A(s-t)\}\) is positive definite because it is the pullback of
\(c\) under \(s\mapsto As\). A singular \(A\) remains valid, but it collapses
some spatial directions and destroys identifiability along them.

Nonstationarity lets covariance geometry vary with location. A deformation
\(u:\mathcal D\to\mathbb R^r\) gives

$$
C(s,t)=c\{u(s)-u(t)\},
$$

which is valid by the same pullback argument. It can still be a poor model:
folding can map distant sites together, a flexible trend can compete with a
long covariance range, and sparse sampling can make local length scales
unidentifiable.

Compactly supported covariances and covariance tapering produce sparse Gram
matrices. Their zeros are modeling statements, not free computation.
Sensitivity to the support radius belongs in the analysis [@matern1960;
@stein1999].

<figure class="viz" data-figure="anisotropic-covariance" data-alt="Three contour maps show covariance around a marked location. Circular contours represent an isotropic range, elongated contours represent directional anisotropy, and curved asymmetric contours represent location-dependent geometry."><figcaption>Range changes the size of a correlation neighborhood, anisotropy changes its direction, and nonstationarity lets that neighborhood change with location. Each construction must preserve positive definiteness for every set of sites.</figcaption></figure>

## Paper module: the Matérn SPDE and sparse precision {#spatial-spde}

The Matérn covariance on \(\mathbb R^d\) is

$$
C_\nu(h)=
\sigma^2\frac{2^{1-\nu}}{\Gamma(\nu)}
(\kappa\lVert h\rVert)^\nu
K_\nu(\kappa\lVert h\rVert),
\qquad \nu,\kappa\gt0.
$$
{#eq-spatial-6}

Its spectral density is proportional to

$$
(\kappa^2+\lVert\omega\rVert^2)^{-(\nu+d/2)}.
$$

Lindgren, Rue, and Lindström begin with the SPDE

$$
(\kappa^2-\Delta)^{\alpha/2}x=\mathcal W,
\qquad
\alpha=\nu+\frac d2,
$$
{#eq-spatial-7}

where \(\mathcal W\) is Gaussian spatial white noise. Fourier transformation
multiplies by
\((\kappa^2+\lVert\omega\rVert^2)^{\alpha/2}\), so the solution spectrum is
proportional to
\((\kappa^2+\lVert\omega\rVert^2)^{-\alpha}\), exactly the Matérn spectrum.
This is the conceptual link between a covariance family and a local
differential operator [@lindgren2011spde].

The computational contribution is to approximate the weak solution in local
finite-element basis functions:

$$
x_h(s)=\sum_{j=1}^m\psi_j(s)w_j.
$$
{#eq-spatial-8}

For piecewise-linear basis functions on a triangulation, define the mass and
stiffness matrices

$$
C_{ij}=\int_{\mathcal D}\psi_i\psi_j,
\qquad
G_{ij}=\int_{\mathcal D}\nabla\psi_i^\top\nabla\psi_j,
\qquad
K_\kappa=\kappa^2C+G.
$$
{#eq-spatial-9}

:::: {.theorem #thm-spde-fem-precision}
[Theorem (finite-element precision for \(\alpha=2\))]{.box-title}

**Assumptions.** The domain, boundary condition, finite-element basis, white-noise discretization, and mass treatment stated below are fixed.

Let \(\mathcal D\) be a bounded Lipschitz domain with the Neumann boundary
condition used in the finite-element construction. Let
\(\psi_1,\ldots,\psi_m\in H^1(\mathcal D)\) be linearly independent local
basis functions, and suppose \(C\) and \(K_\kappa\) in [[eq:eq-spatial-9]] are nonsingular.
The Galerkin approximation to

$$
(\kappa^2-\Delta)x=\mathcal W
$$

has Gaussian coefficient vector \(w\) with precision

$$
Q=K_\kappa^\top C^{-1}K_\kappa.
$$
{#eq-spatial-10}

Replacing \(C\) by the diagonal lumped mass matrix
\(\widetilde C\), with
\(\widetilde C_{ii}=\int\psi_i\), gives the sparse approximation

$$
\widetilde Q=K_\kappa^\top\widetilde C^{-1}K_\kappa.
$$
{#eq-spatial-11}

**Proof status.** Complete derivation below. The precision formula in [[eq:eq-spatial-10]] is Theorem 2(a)
and equations (7)-(10) of [@lindgren2011spde].
::::

:::: {.proof}
[Proof]{.box-title}

Test the SPDE against each \(\psi_i\). Green's identity and the Neumann
condition give

$$
\langle\psi_i,(\kappa^2-\Delta)x_h\rangle
=\sum_j
\left\{\kappa^2\langle\psi_i,\psi_j\rangle+
\langle\nabla\psi_i,\nabla\psi_j\rangle\right\}w_j
=(K_\kappa w)_i.
$$

Let \(\xi_i=\langle\psi_i,\mathcal W\rangle\). White noise is defined by

$$
\operatorname{Cov}(\langle f,\mathcal W\rangle,
\langle g,\mathcal W\rangle)=\langle f,g\rangle_{L^2},
$$

so \(\xi\sim N(0,C)\). The Galerkin equations are

$$
K_\kappa w=\xi.
$$

Hence

$$
\operatorname{Cov}(w)
=K_\kappa^{-1}C K_\kappa^{-\top},
$$

whose inverse is [[eq:eq-spatial-10]]. Local basis functions make \(C\), \(G\), and
\(K_\kappa\) sparse, but \(C^{-1}\) is generally dense. Replacing \(C\) by
the diagonal \(\widetilde C\) preserves locality in [[eq:eq-spatial-11]], which completes
the derivation. [\(\square\)]{.qed}
::::

For integer \(\alpha\gt2\), the paper obtains recursive precision formulas by
repeated application of the second-order operator. Its Theorem 3 proves weak
convergence for \(\alpha=1,2\) when the finite-element spaces become dense in
the relevant \(H^1\) space; Theorem 4 propagates convergence through the
integer recursion [@lindgren2011spde].

**What sparsity costs.** {#spatial-spde-boundary}

The SPDE route does not turn a dense Matérn calculation into an exact sparse
copy.

- The finite-element field is an approximation to the weak SPDE solution.
- Mass lumping changes \(C\) and therefore changes the precision.
- Noninteger \(\alpha\) does not yield the same finite local Markov recursion;
  the reciprocal spectrum is not a polynomial.
- Boundary conditions alter covariance near the edge. The Neumann condition
  used in the paper can inflate boundary variance.
- A coarse or badly shaped mesh can dominate statistical error.
- Hyperparameter and mesh uncertainty do not appear in a conditional
  posterior standard deviation unless explicitly integrated.

The correct comparison is therefore dense covariance error versus
discretization, boundary, and sparse-inference error at a fixed computational
budget.

## Paper module: Gneiting's nonseparable covariance class {#spatiotemporal-kernels}

A separable space-time covariance

$$
C\{(s,t),(s',t')\}
=C_S(s-s')C_T(t-t')
$$

is valid and enables Kronecker algebra on a complete grid. It also says that
the spatial correlation shape is the same at every temporal lag. Gneiting's
question was how to couple the lags while retaining a proof of positive
definiteness in every spatial dimension [@gneiting2002space].

::: {.definition #def-completely-monotone}
[Definition (completely monotone and Bernstein functions)]{.box-title}

A function \(\varphi:[0,\infty)\to[0,\infty)\) is completely monotone when

$$
(-1)^r\varphi^{(r)}(v)\ge0
$$

for every integer \(r\ge0\) and \(v\gt0\). A positive function
\(\psi\) is a Bernstein function when \(\psi'\) is completely monotone.
:::

:::: {.theorem #thm-gneiting-class}
[Theorem (Gneiting nonseparable class)]{.box-title}

**Assumptions.** The complete-monotonicity and Bernstein-function hypotheses stated below are both required.

Let \(\varphi\) be completely monotone and bounded at zero, and let
\(\psi:[0,\infty)\to(0,\infty)\) have completely monotone derivative.
For \(h\in\mathbb R^d\) and \(u\in\mathbb R\),

$$
C(h,u)=
\frac{\sigma^2}{\{\psi(u^2)\}^{d/2}}
\varphi\left(
\frac{\lVert h\rVert^2}{\psi(u^2)}
\right)
$$
{#eq-spatial-12}

is a stationary covariance function on \(\mathbb R^d\times\mathbb R\).

**Proof status.** Reconstructed below from Criterion 2 and its appendix proof
in [@gneiting2002space].
::::

:::: {.proof}
[Proof]{.box-title}

Bernstein's theorem represents the completely monotone function as a
nonnegative mixture:

$$
\varphi(r)=\int_0^\infty e^{-ar}\,dF(a).
$$

It is enough to prove validity for each mixture component and then integrate.
For \(a\gt0\), take the spatial Fourier transform of

$$
\{\psi(u^2)\}^{-d/2}
\exp\left\{-a\lVert h\rVert^2/\psi(u^2)\right\}.
$$

Up to a positive constant depending on \(a\) and \(d\), the transform at
frequency \(\omega\) is

$$
\exp\left\{
-\frac{\lVert\omega\rVert^2}{4a}\psi(u^2)
\right\}.
$$
{#eq-spatial-13}

Because \(u\mapsto u^2\) is conditionally negative definite and \(\psi\) is a
Bernstein function, \(u\mapsto\psi(u^2)\) is conditionally negative
definite. Schoenberg's theorem then makes [[eq:eq-spatial-13]] a temporal covariance for
every \(\omega\). The partial-Fourier criterion used in the paper implies that
the original function is a covariance on space-time.

Nonnegative integration over \(a\) preserves positive definiteness, yielding
[[eq:eq-spatial-12]]. Nonintegrable edge cases follow by multiplying by an integrable
completely monotone factor and passing to the pointwise limit, as in the
paper's appendix. [\(\square\)]{.qed}
::::

A useful parametric specialization is

$$
C(h,u)=
\frac{\sigma^2}
{(1+a\lvert u\rvert^{2\alpha})^\tau}
\exp\left[
-\frac{c\lVert h\rVert^{2\gamma}}
{(1+a\lvert u\rvert^{2\alpha})^{\beta\gamma}}
\right],
$$
{#eq-spatial-14}

with \(a,c\ge0\), \(0\lt\alpha,\gamma\le1\),
\(0\le\beta\le1\), and \(\tau\ge\beta d/2\). The parameter
\(\beta=0\) gives a separable model; larger \(\beta\) lets the spatial
correlation shape change with temporal lag.

Take \(d=2\), \(a=c=\alpha=\gamma=\beta=\tau=1\). Then

$$
C(h,u)=\frac{1}{1+u^2}
\exp\left\{-\frac{\lVert h\rVert^2}{1+u^2}\right\}.
$$

At unit spatial and temporal lag,

$$
C(1,0)=e^{-1}\approx0.3679,\qquad
C(0,1)=\tfrac12,\qquad
C(1,1)=\tfrac12e^{-1/2}\approx0.3033.
$$

If the covariance were separable and normalized at the origin, the last value
would equal \(C(1,0)C(0,1)\approx0.1839\). The difference is a numerical
signature of the interaction, not a validity test. Validity comes from the
theorem.

**Failure boundary and model comparison.** {#spatiotemporal-boundary}

The Gneiting class is stationary and fully symmetric:

$$
C(h,u)=C(-h,u)=C(h,-u).
$$

It cannot represent directional advection in which a plume at positive time
lag is displaced downwind. A covariance can be nonseparable and still miss
the dominant physics. Separable, Gneiting, and evolution-equation models
should be compared in common currency:

| Model | Interaction | Main computational structure | Characteristic failure |
|---|---|---|---|
| separable | none | Kronecker products on grids | spatial range cannot vary with lag |
| Gneiting | symmetric lag coupling | dense unless combined with another approximation | cannot express directional propagation |
| dynamic/SPDE | operator-driven | sparse precision or state-space recursion | discretization and operator misspecification |

Random pointwise cross-validation tends to favor whichever model interpolates
dense neighborhoods. Held-out spatial regions and future time windows are
needed to test the interaction that matters.

**Multivariate fields and change of support.** {#multivariate-spatial}

For \(p\) jointly observed variables, a matrix-valued covariance must make the
entire block Gram matrix positive semidefinite. A linear model of
coregionalization uses

$$
K(s,t)=\sum_{r=1}^R c_r(s,t)B_r,
\qquad B_r\succeq0.
$$

For vectors \(a_i\in\mathbb R^p\),

$$
\sum_{i,j}a_i^\top K(s_i,s_j)a_j
=\sum_r\sum_{i,j}
c_r(s_i,s_j)
(B_r^{1/2}a_i)^\top(B_r^{1/2}a_j)\ge0.
$$

Choosing each cross-covariance separately does not provide this certificate.

Spatial data also arrive on different supports. For normalized weight
functions \(w_A,w_B\), define

$$
Z_A=\int_Aw_A(s)Z(s)\,ds,
\qquad
Z_B=\int_Bw_B(t)Z(t)\,dt.
$$

Then

$$
\operatorname{Cov}(Z_A,Z_B)
=\int_A\int_Bw_A(s)C(s,t)w_B(t)\,ds\,dt.
$$
{#eq-spatial-15}

Point data, pixel averages, and administrative totals are therefore different
linear functionals of the same field. Replacing a region average by its
centroid is accurate only when the covariance and mean vary little across the
support. Quadrature resolution should be refined until both integrated
covariances and final predictions stabilize.

**Estimation, validation, and computational choices.** {#spatial-workflow}

Covariance parameters are often estimated by Gaussian likelihood or restricted
maximum likelihood. The distinction matters because flexible trends absorb
low-frequency variation that a long-range covariance could otherwise explain.
REML removes the fitted trend degrees of freedom from covariance estimation,
but it does not repair a misspecified trend, support, or covariance family.

:::: {.algorithm #algo-spatial-workflow}
[Algorithm (auditable spatial kernel workflow)]{.box-title}

**Input.** Georeferenced observations, support geometry, trend covariates,
measurement-error information, and a deployment region and horizon.

**Output.** Predictions and uncertainty evaluated at the deployment scale.

1. Project coordinates appropriately and plot sites, supports, missingness,
   response, and time.
2. Reserve complete spatial regions and future time windows before fitting.
3. Fit candidate trends on training blocks and inspect directional residual
   variograms.
4. Compare stationary, anisotropic, nonstationary, separable, nonseparable,
   and operator-driven covariances that have scientific justification.
5. Record likelihood bounds, numerical jitter, factorization or iterative
   residuals, and covariance-parameter identifiability.
6. For SPDE fits, refine the mesh and outer boundary until the operational
   predictions stabilize.
7. Score RMSE, log predictive density, interval coverage and width, and
   residual dependence by distance, region, and horizon.
8. Report sensitivity to support quadrature, trend specification, covariance
   range, mesh, and validation-block size.
::::

Dense kriging is cubic in the number of observations. Compact support,
Kronecker products, inducing variables, iterative covariance solves, and SPDE
precision models reduce cost by preserving different objects. They should not
be compared only by wall time. Prediction, likelihood, uncertainty, and
residual dependence can respond differently to the same approximation.

This is where the chapter's narrative closes its loop. A covariance
certificate made the finite matrix legal; the kriging constraint made the
trend estimable; the SPDE or structured approximation made the computation
feasible. None of those steps decides whether the result transfers across
geography. A blocked residual that grows with distance is evidence against the
operational model even when likelihood optimization converged. Conversely, a
slightly worse random-split score can accompany a better blocked score because
the latter measures the actual interpolation gap. Model selection should use
the same support, separation distance, and forecast horizon that define the
deployment decision.

## Common mistakes and operational implications {#spatial-practice}

- Treating latitude and longitude as planar coordinates over a large region.
- Reporting a variogram as if it were a positive-semidefinite covariance.
- Using the same nugget symbol for measurement error, microscale variation,
  and jitter.
- Fitting a flexible trend and a long covariance range without checking
  identifiability.
- Declaring a covariance valid because its plotted contours look plausible.
- Treating an SPDE mesh as a harmless implementation detail.
- Using random splits when deployment requires transfer across distance or
  time.
- Reporting conditional kriging variance as if covariance selection and mesh
  uncertainty had been integrated.

Spatial kernels are useful when geometry, support, and deployment are
specified before model selection. Uncertainty maps and residual dependence
plots are diagnostics, not decoration.

The Gaussian-process formulas in
[[ch:gaussian-processes-and-rvm|the GP chapter]] give the corresponding
conditional distribution when joint Gaussianity is assumed [@rasmussen2006].
The next reliability question is stronger: whether intervals remain useful
when the spatial covariance or deployment distribution is wrong. That is the
handoff to
[[ch:distribution-shift-robustness-and-conformal-prediction|the reliability
chapter]], where model-based variance and empirical coverage are deliberately
kept apart.

## Summary and further reading {#spatial-summary}

Kriging is a constrained projection problem whose trend constraints determine
both weights and variance [@stein1999]. The Matérn family links interpretable
smoothness to a rational spectrum [@matern1960]. The SPDE construction of
[@lindgren2011spde] replaces dense covariance calculations with an approximate
weak finite-element field and a sparse precision after mass lumping; its mesh,
boundary, and integer-order assumptions are part of the estimator.
[@gneiting2002space] constructs nonseparable space-time covariances from
complete monotonicity and proves validity through partial Fourier transforms.
The class models symmetric interaction, not directional dynamics. Every
spatial analysis must finally confront support, covariance estimation, and
validation at the distance and horizon where the prediction will be used.

## Exercises {#exercises}

1. [warm-up]{.ex-tag} Prove the conditional negative-definiteness statement for \(\gamma(h)=C(0)-C(h)\), and explain why the coefficient constraint is essential.
2. [proof]{.ex-tag} Derive the universal-kriging system in [[eq:eq-spatial-2]] and its equivalent mean-squared-error forms through [[eq:eq-spatial-5]], including the invertibility argument for \(P^\top K^{-1}P\).
3. [computation]{.ex-tag} Reproduce the two-site worked example, then add independent measurement noise variance \(0.1\) and recompute simple and ordinary kriging weights and variances.
4. [proof]{.ex-tag} Prove that a fixed linear anisotropy map preserves positive definiteness, including the singular-map case, and state what becomes unidentifiable.
5. [proof]{.ex-tag} Starting from the weak SPDE equations, derive \(Q=K_\kappa^\top C^{-1}K_\kappa\). Explain precisely why sparse \(C\) does not imply sparse \(C^{-1}\), and what mass lumping changes.
6. [synthesis]{.ex-tag} Verify the nonseparability calculation following [[eq:eq-spatial-14]], then design a covariance witness that distinguishes a fully symmetric Gneiting model from directional advection.
7. [challenge]{.ex-tag} Derive [[eq:eq-spatial-15]] for two region averages, propose a convergent quadrature calculation, and describe a counterexample where centroid substitution is badly biased.
8. [exploration]{.ex-tag} Design a blocked evaluation comparing dense Matérn kriging, a Matérn SPDE approximation, and a Gneiting space-time model. Specify trend treatment, deployment blocks, mesh or solver budgets, uncertainty scores, residual diagnostics, and failure criteria.
