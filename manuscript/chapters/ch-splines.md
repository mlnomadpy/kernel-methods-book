---
id: ch-splines
slug: smoothing-splines-and-additive-rkhs
title: Smoothing Splines and Additive RKHS Models
part: VI · Designing Kernels
order: 30
tier: core
prerequisites:
  - kernels-and-rkhs
  - kernel-ridge-and-friends
  - mercer-and-rates
objectives:
  - Derive a smoothing spline from a differential seminorm and its null space.
  - >-
    Reconstruct the Kimeldorf-Wahba finite representation and coefficient
    system.
  - >-
    Derive leave-one-out and generalized cross-validation criteria and their
    limits.
  - Construct identifiable additive and tensor-product interaction spaces.
  - >-
    Connect spline, Green-function, KRR, penalized-likelihood, and
    Gaussian-process views.
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
  - scholkopf2001
example_code_policy: visible-for-executable
narrative_link_policy: exact
---
# Smoothing Splines and Additive RKHS Models

<p class="lead">Long before the phrase kernel machine became standard, statisticians faced a dilemma this book keeps meeting: a curve that passes through every observation can oscillate violently between the points, while a curve rigid enough to be stable can erase the signal. Their answer was to make roughness expensive. The resulting variational problem looks infinite dimensional, yet its minimizer is a finite spline whose free polynomial part determines extrapolation and whose penalized part is an RKHS expansion. This chapter reconstructs that result rather than merely quoting it. It then follows the next question the original theory forced into view: how should the amount of smoothing be selected from noisy data? The answers lead from the Kimeldorf-Wahba representer theorem to Craven-Wahba generalized cross-validation, penalized likelihood, smoothing-spline ANOVA, and uncertainty.</p>

## The variational problem and its hidden null space {#roughness-to-estimator}

Let \(I=[0,1]\), let \(x_1,\ldots,x_n\in I\) be distinct, and suppose

$$
y_i=f^\star(x_i)+\varepsilon_i.
$$

For an integer \(m\ge 1\), define

$$
W_2^m(I)=
\left\{f:f^{(m-1)}\text{ is absolutely continuous and }f^{(m)}\in L^2(I)\right\}.
$$

The order-\(m\) smoothing spline minimizes

$$
\mathcal Q_\lambda(f)
=\frac1n\sum_{i=1}^n\{y_i-f(x_i)\}^2
+\lambda J_m(f),
\qquad
J_m(f)=\int_0^1\{f^{(m)}(t)\}^2\,dt,
\tag{51.1}
$$

over \(W_2^m(I)\), with \(\lambda\gt0\). The penalty is a seminorm because

$$
\mathcal N_m=\{f:J_m(f)=0\}
=\operatorname{span}\{1,x,\ldots,x^{m-1}\}.
$$

These directions are not weakly penalized. They are not penalized at all. As
\(\lambda\to\infty\), the fit therefore approaches least squares in \(\mathcal N_m\),
not the zero function. For \(m=2\), the limiting fit is affine.

This is the semiparametric version of the projection argument inherited from
[[ch:kernel-tricks|the representer-theorem chapter]]: observations can see only
finitely many directions, but the usual RKHS proof must be repaired because
\(J_m\) vanishes on an entire polynomial space. The repair is not to add a
small penalty and hope. It is to carry the null space explicitly, identify it
through the design, and apply the projection argument only to the penalized
complement [@scholkopf2001].

Choose a complementary Hilbert space

$$
\mathcal H_m=
\left\{g\in W_2^m(I):g^{(r)}(0)=0,\ 0\le r\le m-1\right\},
\qquad
\langle g,h\rangle_{\mathcal H_m}
=\int_0^1g^{(m)}h^{(m)}.
$$

Every \(f\in W_2^m(I)\) has a unique decomposition \(f=p+g\) with
\(p\in\mathcal N_m\) and \(g\in\mathcal H_m\). Repeated integration gives

$$
g(x)=\int_0^1\frac{(x-u)_+^{m-1}}{(m-1)!}\,g^{(m)}(u)\,du,
$$

so evaluation is bounded on \(\mathcal H_m\). Its reproducing kernel is

$$
R_m(s,t)=
\int_0^1
\frac{(s-u)_+^{m-1}(t-u)_+^{m-1}}{\{(m-1)!\}^2}\,du.
\tag{51.2}
$$

This explicit construction is important. The kernel is manufactured by the
differential penalty and the boundary convention. Changing either changes the
function space and the fit.

::: {.definition #def-spline-null-space}
[Definition (penalty null space and identifiable design)]{.box-title}

For a nonnegative quadratic penalty \(J\), its null space is
\(\mathcal N_J=\{f:J(f)=0\}\). If \(p_1,\ldots,p_q\) span \(\mathcal N_J\), the
design matrix \(P\in\mathbb R^{n\times q}\) has entries \(P_{ia}=p_a(x_i)\).
The design identifies the null space when \(P\) has column rank \(q\).
:::

If \(P\) is rank deficient, a function in the null space can vanish at every
observation while costing no penalty. The fitted values may still be unique, but
the trend coefficients are not. This is the first failure boundary: a seminorm
problem requires both a kernel and an identified null space.

## Paper module: Kimeldorf and Wahba's finite reduction {#spline-kimeldorf-wahba}

The central question of Kimeldorf and Wahba was broader than pointwise
regression. Let \(\mathcal V=\mathcal N\oplus\mathcal H\), where
\(\mathcal N=\operatorname{span}\{p_1,\ldots,p_q\}\) is finite dimensional and
\(\mathcal H\) is an RKHS with kernel \(R\). Let
\(\ell_1,\ldots,\ell_n\) be bounded linear functionals on \(\mathcal V\). They may
be point evaluations, derivative measurements, integrals, or mixtures of these.
Write \(r_i\in\mathcal H\) for the representer of the restriction of \(\ell_i\) to
\(\mathcal H\):

$$
\ell_i(g)=\langle g,r_i\rangle_{\mathcal H}.
$$

For point evaluation, \(r_i=R(x_i,\cdot)\). The paper's Section 5 asks for the
minimizer of a positive-definite quadratic data discrepancy plus the squared
norm of the penalized projection [@kimeldorf1971].

:::: {.theorem #thm-spline-representer}
[Theorem (semiparametric spline representer)]{.box-title}

**Assumptions.** The four boundedness, finite-null-space, rank, and regularization assumptions stated below are all in force.

Let \(W\in\mathbb R^{n\times n}\) be positive definite and let

$$
\mathcal Q(f)=
\{\ell(f)-y\}^{\top}W\{\ell(f)-y\}
+\lambda\lVert g\rVert_{\mathcal H}^2,
\qquad f=p+g\in\mathcal N\oplus\mathcal H.
\tag{51.3}
$$

Assume:

1. every \(\ell_i\) is bounded on \(\mathcal V\);
2. \(\mathcal N\) is finite dimensional;
3. \(P_{ia}=\ell_i(p_a)\) has column rank \(q\);
4. \(\lambda\gt0\).

Then the minimizer exists, is unique as an element of
\(\mathcal N\oplus\mathcal H\), and has the form

$$
\widehat f
=\sum_{a=1}^q d_a p_a+\sum_{i=1}^n c_i r_i.
\tag{51.4}
$$

For squared pointwise loss with \(W=n^{-1}I\), the coefficients may be chosen to
satisfy

$$
\begin{pmatrix}
K+n\lambda I&P\\
P^\top&0
\end{pmatrix}
\begin{pmatrix}c\\d\end{pmatrix}
=
\begin{pmatrix}y\\0\end{pmatrix},
\qquad
K_{ij}=R(x_i,x_j).
\tag{51.5}
$$

**Proof status.** Complete below. This is a modernized specialization of
Section 5, Lemma 5.1 and Theorem 5.1 of [@kimeldorf1971].
::::

:::: {.proof}
[Proof]{.box-title}

Let \(\mathcal S=\operatorname{span}\{r_1,\ldots,r_n\}\subseteq\mathcal H\).
For any \(f=p+g\), decompose \(g=g_\parallel+g_\perp\) with
\(g_\parallel\in\mathcal S\) and \(g_\perp\perp\mathcal S\). Since

$$
\ell_i(g_\perp)=\langle g_\perp,r_i\rangle_{\mathcal H}=0
$$

for every \(i\), replacing \(g\) by \(g_\parallel\) leaves the data term in
\(\mathcal Q\) unchanged. Orthogonality gives

$$
\lVert g\rVert_{\mathcal H}^2
=\lVert g_\parallel\rVert_{\mathcal H}^2
+\lVert g_\perp\rVert_{\mathcal H}^2.
$$

Every minimizer must therefore have \(g_\perp=0\), proving the finite form
(51.4).

For pointwise squared loss, write the fitted vector as \(Pd+Kc\). The finite
objective is

$$
\frac1n\lVert y-Pd-Kc\rVert^2+\lambda c^\top Kc.
$$

At a minimum, variation in \(d\) gives
\(P^\top(y-Pd-Kc)=0\). Variation in the represented function \(g\), or
equivalently in a coefficient representation modulo the null space of \(K\),
gives \(y-Pd-Kc=n\lambda c\). Combining these equations yields
\((K+n\lambda I)c+Pd=y\) and \(P^\top c=0\), which is (51.5).

The quadratic objective is coercive on \(\mathcal H\), and the full-rank
condition on \(P\) controls \(\mathcal N\). Thus a minimizer exists. If two
minimizers existed, strict convexity in the fitted values and the
\(\mathcal H\)-norm would force identical observations and identical penalized
components; full rank of \(P\) would then force identical null-space
coefficients. [\(\square\)]{.qed}
::::

The difficult move is not solving (51.5). It is recognizing that the
orthogonal complement of the observation representers is invisible to the
data and visible to the penalty. That move extends immediately to derivative
and integral data, provided the corresponding functionals are bounded.

**What the theorem does not license.** {#spline-representer-boundary}

The assumptions mark real boundaries.

- If a derivative functional is unbounded in the chosen Sobolev space, it has
  no RKHS representer there.
- If \(W\) is indefinite, the data term need not be convex.
- If \(\lambda=0\), existence and uniqueness become interpolation questions.
- If \(P\) lacks full rank, the free trend is not identified.
- If the penalty or its kernel is learned jointly with unrestricted
  parameters, the fixed-space orthogonal-decomposition proof is insufficient.

The paper's contribution was therefore not the slogan that splines are finite.
It was a typed reduction for bounded observations and a fixed roughness space.

## Green functions, natural boundaries, and KRR {#green-spline-krr}

The kernel in (51.2) is a one-sided integrated Green kernel for the anchored
space \(\mathcal H_m\). Once the null-space term is restored, the solution of
(51.1) is a polynomial of degree \(2m-1\) between adjacent knots. For \(m=2\),
the first variation in a knot-free interval gives

$$
\int \widehat f''h''=0
$$

for every compactly supported perturbation \(h\). Integrating twice in the
distributional sense yields \(\widehat f^{(4)}=0\) there, so the fit is piecewise
cubic. Perturbations at the boundary yield the natural conditions
\(\widehat f''(0)=\widehat f''(1)=0\).

This route explains three equivalent descriptions:

1. a minimizer of squared error plus integrated squared curvature;
2. a natural cubic spline with knots at the observed inputs;
3. a semiparametric KRR estimator with an affine null space.

After projecting away the null space, the penalized coordinates obey ordinary
KRR shrinkage. Before projection, writing only
\(k_x^\top(K+n\lambda I)^{-1}y\) silently changes the estimator because it
penalizes or fixes the trend.

<figure class="viz" data-figure="spline-decomposition" data-alt="Three aligned panels show an affine null-space trend, a wavy penalized component, and their sum. The final spline overlays the affine trend to show which part remains when the rough component is strongly shrunk."><figcaption>A smoothing spline contains a free trend and a penalized departure. The trend determines the large-penalty limit and boundary extrapolation; the kernel component controls local curvature.</figcaption></figure>

:::: {.example #example-spline-nullspace}
[Example (one curvature direction)]{.box-title}

Three observations make the null-space geometry visible without software. Take

$$
(x_1,x_2,x_3)=(0,\tfrac12,1),
\qquad
y=(0,1,0)^\top,
$$

and penalize \(\int_0^1(f'')^2\). The natural cubic interpolant is

$$
s(x)=
\begin{cases}
3x-4x^3,&0\le x\le\tfrac12,\\
3(1-x)-4(1-x)^3,&\tfrac12\le x\le1.
\end{cases}
$$

Its second derivative is \(-24x\) on the first half and
\(-24(1-x)\) on the second, so

$$
J_2(s)=2\int_0^{1/2}(24x)^2\,dx=48.
$$

At the data sites, decompose

$$
y=\frac13\mathbf1+\frac13q,
\qquad
q=(-1,2,-1)^\top.
$$

The two-dimensional affine null space is left unchanged. There is only one
remaining data direction, the curvature contrast \(q\). Let \(z=3s-1\), whose
site values equal \(q\) and whose roughness is \(J_2(z)=9J_2(s)=432\). Every
candidate fit is

$$
f_\theta(x)=\frac13+\theta z(x),
$$

and (51.1) reduces exactly to

$$
2(\theta-\tfrac13)^2+432\lambda\theta^2.
$$

Therefore

$$
\widehat\theta_\lambda=
\frac{1/3}{1+216\lambda},
\qquad
\widehat y=
\frac13\mathbf1+
\frac{1}{3(1+216\lambda)}q.
\tag{51.6}
$$

At \(\lambda=0.01\), the curvature shrinkage factor is \(0.3165\) and
\(\widehat y\approx(0.2278,0.5443,0.2278)^\top\). The smoother has eigenvalues
\(1,1,0.3165\), so its effective degrees of freedom are \(2.3165\).

The entire calculation is reproduced below. The code deliberately constructs
the interpolant and integrates its squared second derivative numerically,
rather than inserting the value \(48\), so it checks the bridge from the
variational definition to the scalar shrinkage problem.

```python
import numpy as np

lam = 0.01
grid = np.linspace(0.0, 1.0, 200_001)
s2 = np.where(grid <= 0.5, -24.0 * grid, -24.0 * (1.0 - grid))
roughness = np.trapezoid(s2**2, grid)

q = np.array([-1.0, 2.0, -1.0])
shrinkage = 1.0 / (1.0 + 216.0 * lam)
fitted = np.ones(3) / 3.0 + shrinkage * q / 3.0
effective_df = 2.0 + shrinkage

assert np.isclose(roughness, 48.0, atol=1e-8)
assert np.allclose(fitted, [0.2278481, 0.5443038, 0.2278481])
assert np.isclose(effective_df, 2.3164557)
print(roughness, shrinkage, fitted, effective_df)
```

This example also exposes a selection failure. With \(n=3\), exactly two
unpenalized directions, and only one penalized direction, the GCV score is
constant for every finite \(\lambda\). There is not enough residual geometry to
choose a smoothing level. A tuning criterion cannot manufacture information
that the design does not contain.

**Verification scope.** The interpolation formula, roughness \(48\), shrinkage
factor, fitted values, and degrees of freedom are hand-checkable consequences
of (51.1). No probabilistic coverage claim is made.
::::

## Paper module: Craven and Wahba's generalized cross-validation {#spline-gcv}

Once the infinite problem has become a linear smoother, the next problem is
choosing \(\lambda\). For fixed design and fixed \(\lambda\),

$$
\widehat y=A_\lambda y.
$$

In the Demmler-Reinsch basis, the null-space directions have smoother
eigenvalue one and each penalized direction has eigenvalue

$$
a_j(\lambda)=\frac{1}{1+n\lambda\rho_j},
$$

where \(\rho_j\) is its roughness eigenvalue. Thus

$$
\operatorname{df}(\lambda)=\operatorname{tr}(A_\lambda)
$$

measures sensitivity of the fitted vector to the data. It is not the number of
basis coefficients.

::: {.proposition #prop-spline-loocv}
[Proposition (exact leave-one-out residual)]{.box-title}

**Assumptions.** Quadratic loss, fixed design, fixed smoothing parameter, and a well-posed deleted-data fit are assumed.

Suppose \(A_\lambda\) is the fitted-value matrix of a penalized least-squares
linear smoother and is independent of \(y\). If \(A_{\lambda,ii}\ne1\), the
residual obtained by refitting after deleting observation \(i\) is

$$
y_i-\widehat y_i^{(-i)}
=\frac{y_i-\widehat y_i}{1-A_{\lambda,ii}}.
\tag{51.7}
$$

**Assumptions.** Quadratic loss, fixed \(\lambda\), fixed design and weights,
and a well-posed deleted-data fit. **Proof status.** Complete below; the
identity underlies Section 1 and equation (1.11) of [@cravenwahba1979].
:::

::: {.proof}
[Proof]{.box-title}

Represent deletion by giving observation \(i\) a provisional response \(z\)
equal to its unknown deleted-fit prediction. The full-data smoother applied to
\(y+(z-y_i)e_i\) must reproduce \(z\) at coordinate \(i\):

$$
z=\widehat y_i+A_{\lambda,ii}(z-y_i).
$$

Solving gives

$$
z=\frac{\widehat y_i-A_{\lambda,ii}y_i}
{1-A_{\lambda,ii}},
$$

and subtracting from \(y_i\) yields (51.7). The argument is valid because the
zero residual attached to the provisional observation gives the same normal
equations as deletion. [\(\square\)]{.qed}
:::

Ordinary leave-one-out cross-validation is therefore

$$
\operatorname{CV}(\lambda)
=\frac1n\sum_{i=1}^n
\left\{\frac{y_i-\widehat y_i}{1-A_{\lambda,ii}}\right\}^2.
\tag{51.8}
$$

Craven and Wahba replace the individual leverages by their average:

$$
\operatorname{GCV}(\lambda)
=
\frac{n^{-1}\lVert(I-A_\lambda)y\rVert^2}
{\bigl[n^{-1}\operatorname{tr}(I-A_\lambda)\bigr]^2}.
\tag{51.9}
$$

The replacement is not an arbitrary shortcut. Section 3 of
[@cravenwahba1979] diagonalizes the symmetric smoother and applies ordinary
cross-validation after an orthogonal rotation that equalizes the diagonal
leverages. This gives GCV its rotation invariance.

:::: {.theorem #thm-spline-gcv-efficiency}
[Theorem (Craven-Wahba asymptotic efficiency, simplified)]{.box-title}

**Assumptions.** The regular-design, error, leverage, and candidate-smoothing conditions stated below are all in force.

Assume the regression model \(y_i=g(t_i)+\varepsilon_i\) on \([0,1]\), with
independent errors of mean zero and common variance \(\sigma^2\);
\(g\in W_2^m\); and a regular triangular array of sites satisfying

$$
\int_0^{t_{i,n}}w(u)\,du=\frac{i}{n}
$$

for a continuous density \(w\) bounded above and away from zero. Let
\(\lambda_n^\star\) minimize the expected in-sample squared error and let
\(\bar\lambda_n\) be a suitable minimizer of expected GCV. Under the
eigenvalue regularity conditions used in the paper,

$$
\frac{\mathbb E R(\bar\lambda_n)}
{\mathbb E R(\lambda_n^\star)}
\longrightarrow1.
\tag{51.10}
$$

**Proof status.** Proof skeleton only. The exact result is Theorem 4.3 of
[@cravenwahba1979].
::::

The proof writes expected risk as squared bias plus
\(\sigma^2n^{-1}\operatorname{tr}(A_\lambda^2)\), then shows that expected GCV
minus \(\sigma^2\) approximates this risk uniformly near the optimal smoothing
regime. The key spectral estimates imply
\(\lambda\to0\) but \(n\lambda^{1/(2m)}\to\infty\): the estimator admits more
detail with more data, but not so quickly that noise passes unfiltered.

**Failure boundary and comparison of selectors.** {#spline-selection-boundary}

Equation (51.10) is not a distribution-free finite-sample guarantee. It does
not cover arbitrary dependence, strong heteroscedasticity, adaptive choice
among many model structures, or a design with too little penalized residual
space. The criteria answer different questions:

| Selector | Quantity used | Main appeal | Main failure |
|---|---|---|---|
| LOOCV | individual \(A_{ii}\) | exact deleted residuals | unstable when \(A_{ii}\) is near one |
| GCV | average leverage | cheap and rotation invariant | can undersmooth under dependence or uneven leverage |
| blocked CV | deployment-shaped holdouts | tests transfer geometry | fewer effective folds and higher variance |
| marginal likelihood | full Gaussian model | jointly estimates variance components | model-based and sensitive to mean/covariance misspecification |

For serial or spatial residuals, random deletion leaves correlated neighbors in
training. The result can look precise while selecting a curve that fails at the
deployment horizon. The validation unit must match the dependence unit.

**Penalized likelihood as repeated spline smoothing.** {#spline-penalized-irls}

Squared error is not the only observation model. For independent exponential
family responses with linear predictor \(\eta_i=f(x_i)\), consider

$$
-\frac1n\sum_{i=1}^n\ell_i(\eta_i)+\frac\lambda2J(f).
\tag{51.11}
$$

At an iterate \(\eta^{(r)}\), let

$$
w_i^{(r)}=-\ell_i''(\eta_i^{(r)}),
\qquad
z_i^{(r)}
=\eta_i^{(r)}
+\frac{\ell_i'(\eta_i^{(r)})}{w_i^{(r)}}.
$$

The second-order surrogate is a weighted smoothing-spline problem:

$$
\min_f\;
\frac{1}{2n}\sum_iw_i^{(r)}
\{z_i^{(r)}-f(x_i)\}^2
+\frac\lambda2J(f).
\tag{51.12}
$$

Thus iteratively reweighted least squares changes the local responses and
weights, while the representer span and null space remain fixed. The quadratic
construction follows the likelihood geometry studied by [@green1984]; its use
inside the RKHS penalty is the chapter's explicit synthesis.

This extension has its own boundary. The matrix called a smoother now depends
on the current fitted values and therefore on \(y\). The fixed-matrix LOOCV
identity (51.7) is no longer exact without an influence approximation. If
weights vanish, explode, or become negative because the local model is
ill-conditioned, the quadratic step can cease to identify a stable update.

## Smoothing-spline ANOVA {#spline-anova}

For \(x=(x_1,\ldots,x_d)\), a full multivariate smooth is difficult to estimate
and difficult to explain. Smoothing-spline ANOVA decomposes

$$
f(x)=\mu+\sum_j f_j(x_j)+\sum_{j\lt k}f_{jk}(x_j,x_k)+\cdots.
\tag{51.13}
$$

Let \(\mu_j\) be a declared reference measure for coordinate \(j\). Define a
centering operator

$$
(\mathcal C_jf)(x_j)
=f(x_j)-\int f(u)\,d\mu_j(u).
$$

If \(k_j\) is a kernel, its centered version is

$$
k_j^\circ(x,z)
=k_j(x,z)-\int k_j(x,u)d\mu_j(u)-\int k_j(v,z)d\mu_j(v)
+\iint k_j(v,u)d\mu_j(v)d\mu_j(u).
\tag{51.14}
$$

The main-effect RKHS has kernel \(k_j^\circ\), and the \(j,k\) interaction has
kernel \(k_j^\circ k_k^\circ\). Expanding

$$
\prod_{j=1}^d(1+k_j^\circ)
$$

generates the constant, main-effect, and interaction spaces. The constraints

$$
\int f_j\,d\mu_j=0,\qquad
\int f_{jk}(x_j,x_k)\,d\mu_j(x_j)=
\int f_{jk}(x_j,x_k)\,d\mu_k(x_k)=0
$$

prevent an interaction from hiding a main effect.

::: {.proposition #prop-spline-anova-identifiable}
[Proposition (uniqueness of the two-factor ANOVA decomposition)]{.box-title}

**Assumptions.** The component spaces and centering constraints stated below define a direct identifiable decomposition.

Suppose \(\mu=\mu_1\otimes\mu_2\) and
\(f=\mu_0+f_1+f_2+f_{12}\), where the components satisfy the centering
constraints above. Then the four components are unique in \(L^2(\mu)\).

**Proof status.** Complete.
:::

::: {.proof}
[Proof]{.box-title}

Integrating \(f\) over both coordinates gives \(\mu_0\). Integrating over
\(x_2\) gives \(\mu_0+f_1(x_1)\), because the centered \(f_2\) and the
\(x_2\)-marginal of \(f_{12}\) vanish. Hence \(f_1\) is identified. The same
argument identifies \(f_2\), and subtraction identifies \(f_{12}\).
[\(\square\)]{.qed}
:::

The product-measure assumption is not harmless. With strongly dependent
covariates, empirical and population centering produce different decompositions,
and an interaction may be weakly identified where the design has little joint
support. Interpretability is a property of the declared measure and constraints,
not merely of the formula (51.13).

:::: {.algorithm #algo-spline-anova-fit}
[Algorithm (auditable additive RKHS fit)]{.box-title}

**Input.** Training observations, reference measures, component kernels,
null-space bases, candidate penalties, and a deployment-shaped validation
design.

**Output.** An identified additive predictor with componentwise diagnostics.

1. Scale covariates and estimate every centering operation from training data
   only.
2. Construct null-space and penalized bases for each main effect.
3. Construct only scientifically defensible interactions and impose both
   marginal constraints.
4. Solve the penalized system by a null-space-aware factorization or a
   converged matrix-free method.
5. Select smoothing parameters with nested or blocked validation, GCV only
   when its error assumptions are credible, or a declared likelihood model.
6. Report componentwise effective degrees of freedom, residual structure,
   condition estimates, and sensitivity to the interaction hierarchy.
::::

**Bayesian interpretation and uncertainty.** {#spline-bayesian-view}

Put a zero-mean Gaussian process prior on the penalized component \(g\) with
covariance \(\sigma_g^2R\), retain \(p(x)^\top d\) as a diffuse trend, and use
independent Gaussian noise of variance \(\sigma_\varepsilon^2\). The posterior
mean solves (51.1) after the ratio
\(n\lambda=\sigma_\varepsilon^2/\sigma_g^2\) is matched to the coefficient
convention. Section 7 of [@kimeldorf1971] establishes the corresponding
minimum-variance linear prediction interpretation.

The algebraic equality does not merge all inferential meanings:

- the spline variational statement is deterministic conditional on the data;
- the mixed-model or GP statement is probabilistic under a covariance model;
- a Bayesian credible band averages under that model;
- a frequentist confidence band must account for smoothing bias and,
  ideally, smoothing-parameter selection.

Pointwise intervals are not simultaneous bands, and neither automatically
covers after selecting interactions. When coverage is operationally important,
the conformal methods in
[[ch:distribution-shift-robustness-and-conformal-prediction]] provide a
separate layer with separate assumptions.

**Practice, diagnostics, and model choice.** {#spline-practice}

A defensible spline analysis reports more than a fitted line.

Common mistakes are to leave the null space unidentified, interpret GCV as an
independent test error, ignore a leverage denominator near zero, or report
pointwise Bayesian intervals as simultaneous frequentist bands.

- State the differential operator, boundary convention, and null space.
- Plot the smoother eigenvalues or report effective degrees of freedom.
- Check leverage denominators \(1-A_{ii}\), not only mean GCV.
- Compare residual dependence with the independence assumptions used for
  tuning.
- Separate interpolation inside the observed range from extrapolation beyond
  it.
- For ANOVA models, report the centering measure and where joint covariate
  support is sparse.
- For penalized likelihood, monitor weight ranges, step acceptance, and final
  score residuals.
- Treat posterior bands as model-based until coverage is checked externally.

Use a spline when the roughness operator and null space express meaningful
scientific behavior. Use a generic kernel when similarity is easier to specify
than derivatives. Use both only after explaining which part of the geometry
each contributes.

That decision hands the reader to
[[ch:spatial-and-spatiotemporal-kernels|the spatial-kernel chapter]], where the
same saddle system reappears as universal kriging: the free spline trend
becomes the spatial mean model, and the penalized Green kernel becomes a
covariance or generalized covariance. The analogy is useful only if the
semantics remain separate: roughness minimization is not, by itself, a random
field model.

## Summary and further reading {#spline-summary}

The smoothing spline is a semiparametric kernel estimator: an unpenalized
finite-dimensional trend plus an RKHS component generated by a differential
penalty. Kimeldorf and Wahba's finite reduction applies to bounded linear
observations and exposes the rank assumptions that the slogan "representer
theorem" hides [@kimeldorf1971]. Craven and Wahba turn the resulting linear
smoother into a data-driven selection rule and prove asymptotic risk efficiency
under regular design and independent homoscedastic noise [@cravenwahba1979].
[@wahba1990] develops the broader spline and ANOVA theory. Penalized IRLS
extends the computational pattern beyond Gaussian responses, but its
data-dependent weights invalidate naive reuse of fixed-smoother diagnostics
[@green1984].

## Exercises {#exercises}

1. [warm-up]{.ex-tag} For \(J_m(f)=\int_0^1\{f^{(m)}(t)\}^2dt\), identify the null space and explain both the large-\(\lambda\) limit and the rank condition required of its design matrix.
2. [proof]{.ex-tag} Starting from repeated integration, verify that (51.2) reproduces evaluation on the anchored space \(\mathcal H_m\).
3. [proof]{.ex-tag} Extend the proof of Theorem 51.1 to bounded derivative or integral observations \(\ell_i\), and state exactly what fails if one \(\ell_i\) is unbounded.
4. [computation]{.ex-tag} Reproduce every quantity in the three-point worked example, including \(J_2(s)=48\), the shrinkage factor in (51.6), the fitted values at \(\lambda=0.01\), and the effective degrees of freedom.
5. [proof]{.ex-tag} Prove the exact leave-one-out identity (51.7) directly from the normal equations or a rank-one update, and identify the instability as \(A_{ii}\to1\).
6. [synthesis]{.ex-tag} Explain why GCV is constant in the three-point example. Construct a four-point contrast with two penalized eigenvalues and write its GCV score in the smoother eigenbasis.
7. [challenge]{.ex-tag} Derive the two-factor centered ANOVA kernel from (51.14), prove the marginal constraints, and explain how dependent covariates alter the interpretation.
8. [exploration]{.ex-tag} Design a comparison among a natural cubic spline, RBF KRR, and a Matérn GP for boundary extrapolation. Specify null-space or mean assumptions, tuning protocol, uncertainty scores, conditioning diagnostics, and one failure test.
