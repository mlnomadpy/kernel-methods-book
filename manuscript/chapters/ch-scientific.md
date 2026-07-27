---
id: ch-scientific
slug: scientific-computing-and-operator-learning
title: Kernels for Scientific Computing and Operator Learning
part: X · Dynamics and Scientific Learning
order: 52
tier: advanced
prerequisites:
  - mercer-and-rates
  - vector-and-operator-valued-kernels
  - inverse-learning-and-spectral-regularization
  - gaussian-processes-and-rvm
objectives:
  - >-
    Derive kernel estimators from bounded differential, boundary, integral, and
    point-evaluation information.
  - >-
    Establish when symmetric collocation is positive definite and when a small
    sampled residual controls solution error.
  - >-
    Reconstruct Gaussian-process differential-equation learning and Bayesian
    probabilistic numerics under their exact assumptions.
  - >-
    Separate approximation, discretization, algebraic, observation, and model
    errors in a scientific computation.
  - >-
    Compare kernel operator regression and Fourier neural operators in
    discretization-independent function spaces.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-scientific.yml
verification_date: null
bibliography:
  - wendland2005
  - raissi2017gpde
  - cockayne2019probnum
  - li2020fno
---
# Kernels for Scientific Computing and Operator Learning

<p class="lead">A curve that passes through every measurement can still be physically impossible. It can violate the equation governing the field, contradict a boundary condition, or conserve the wrong quantity. Scientific learning therefore asks more than prediction at another row of a data table. The unknown may be a function constrained by a differential operator, a coefficient visible only through a forward solve, or an operator mapping one field to another on meshes never seen during training. Kernels can address all three, but only after the continuous problem is stated precisely. This chapter develops the shared mechanism, bounded linear information in a function space, and then follows it through collocation, differential-equation Gaussian processes, probabilistic numerics, and operator learning. At every stage we keep five errors separate: approximation, information discretization, linear-algebra error, observation noise, and model discrepancy.</p>

## The continuous problem comes before the matrix {#scientific-continuous-problem}

Let \(\Omega\subset\mathbb R^d\) be a bounded domain and let \(X\) and \(Y\) be normed spaces of functions or distributions. A linear boundary-value problem has the abstract form

$$
\mathcal T u=(\mathcal A u,\mathcal B u)=(f,g)\in Y,
$$

where \(\mathcal A\) acts in \(\Omega\) and \(\mathcal B\) encodes boundary information. Before choosing sites or a kernel, four questions must have answers.

1. **Existence.** Does at least one \(u^\dagger\in X\) satisfy the equation?
2. **Uniqueness.** Is \(\ker\mathcal T=\{0\}\), or has a null space been fixed by side conditions?
3. **Stability.** Is there a constant \(C_{\mathrm{stab}}\) such that
   $$
   \lVert v\rVert_X\le C_{\mathrm{stab}}\lVert\mathcal T v\rVert_Y
   $$
   for all admissible \(v\)?
4. **Information.** Are the functionals used by the numerical method bounded on the selected hypothesis space?

These are logically separate. A collocation matrix can be invertible although the continuous problem is unstable in the target norm. Conversely, a well-posed PDE may lead to a singular discrete system because two information functionals agree on the chosen RKHS.

::: {.definition #def-scientific-error-ledger}
[Definition (five-part scientific error ledger)]{.box-title}

For an estimator \(\widetilde u\), record:

$$
\begin{aligned}
e_{\mathrm{app}}&=\text{error from restricting the target to the hypothesis space},\\
e_{\mathrm{disc}}&=\text{error from replacing continuous information by finitely many functionals},\\
e_{\mathrm{alg}}&=\text{error from terminating or perturbing the numerical solve},\\
e_{\mathrm{obs}}&=\text{propagated measurement noise},\\
e_{\mathrm{model}}&=\text{error from a wrong operator, boundary condition, or constitutive law}.
\end{aligned}
$$

A posterior standard deviation or collocation residual usually addresses only part of this ledger.
:::

## Bounded linear information and its representers {#scientific-linear-information}

A thermocouple reading, a measured flux, a cell average, and a linear PDE residual all extract one scalar from \(u\). Write the observations as

$$
y_i=L_i u+\varepsilon_i,\qquad L_i\in\mathcal H_k^\ast.
$$

Point evaluation belongs to this class by the definition of an RKHS. Derivative evaluation does not follow automatically. If \(L_i u=D^\alpha u(x_i)\), the map \(L_i:\mathcal H_k\to\mathbb R\) must be bounded. For a sufficiently regular kernel its Riesz representer is

$$
r_i(\cdot)=L_i^{(z)}k(\cdot,z),
$$

and the functional Gram matrix is

$$
G_{ij}=\langle r_j,r_i\rangle_{\mathcal H_k}
      =L_i^{(x)}L_j^{(z)}k(x,z).
$$

The order of differentiation matters. Two second-order PDE functionals require mixed derivatives of total order four across the two kernel arguments.

:::: {.theorem #thm-scientific-functional-representer}
[Theorem (representer theorem for linear scientific information)]{.box-title}

Let \(\mathcal H_k\) be a real RKHS, let \(L_1,\ldots,L_m\in\mathcal H_k^\ast\), and let

$$
J(u)=\Phi(L_1u,\ldots,L_mu)+\Omega(\lVert u\rVert_{\mathcal H_k}),
$$

where \(\Phi:\mathbb R^m\to(-\infty,\infty]\) is arbitrary and \(\Omega:[0,\infty)\to\mathbb R\) is strictly increasing. If \(J\) has a minimizer, every minimizer has the form

$$
\widehat u=\sum_{i=1}^m c_i r_i.
$$

If \(\Omega\) is merely nondecreasing, at least one minimizer has this form, but other minimizers may contain an invisible orthogonal component.

**Assumptions.** Fixed real RKHS, bounded linear information, existence of a minimizer, and the stated monotonicity of the radial penalty.

**Proof status.** Proved below by orthogonal decomposition.
::::

:::: {.proof}
Let \(S=\operatorname{span}\{r_1,\ldots,r_m\}\), a closed finite-dimensional subspace. Decompose any \(u\in\mathcal H_k\) as \(u=u_S+u_\perp\), where \(u_\perp\perp S\). Since

$$
L_i u_\perp=\langle u_\perp,r_i\rangle_{\mathcal H_k}=0,
$$

the data term is identical for \(u\) and \(u_S\). Pythagoras gives

$$
\lVert u\rVert_{\mathcal H_k}^2
=\lVert u_S\rVert_{\mathcal H_k}^2+\lVert u_\perp\rVert_{\mathcal H_k}^2.
$$

If \(u_\perp\ne0\), strict increase of \(\Omega\) makes \(J(u_S)\lt J(u)\). Thus every minimizer lies in \(S\). Under nondecrease, deleting \(u_\perp\) cannot increase the objective, which proves existence of a span-valued minimizer but not uniqueness. \(\square\)
::::

The proof identifies the exact boundary of the result. A nonlinear residual \(u\mapsto\mathcal N(u)(x)\) is not a bounded linear functional, and a kernel family whose parameters are optimized jointly with \(u\) is not one fixed RKHS. Both require a different argument.

## Symmetric collocation and discrete solvability {#kernel-collocation}

For interior sites \(X_A=\{x_i\}_{i=1}^{m_A}\) and boundary sites \(X_B=\{z_j\}_{j=1}^{m_B}\), define

$$
L_i^A u=(\mathcal A u)(x_i),\qquad
L_j^B u=(\mathcal B u)(z_j).
$$

Stack these as \(L_1,\ldots,L_m\) and solve the minimum-norm interpolation problem

$$
\min_{u\in\mathcal H_k}\lVert u\rVert_{\mathcal H_k}
\quad\text{subject to}\quad
L_i u=y_i.
$$

The representer theorem gives \(\widehat u=\sum_jc_jr_j\) and

$$
Gc=y,\qquad G_{ij}=L_i^{(x)}L_j^{(z)}k(x,z).
$$

This is called symmetric collocation because every row and every basis function is generated by the same information family. It differs from unsymmetric Kansa-type collocation, which applies the operator only in the testing argument.

::: {.proposition #prop-scientific-collocation}
[Proposition (positive definiteness and uniqueness)]{.box-title}

Assume \(L_1,\ldots,L_m\) are bounded on \(\mathcal H_k\). Then \(G\) is positive semidefinite. It is positive definite if and only if the functionals are linearly independent as elements of \(\mathcal H_k^\ast\). In that case the minimum-norm interpolant exists and is unique for every \(y\in\mathbb R^m\).

**Assumptions.** Fixed RKHS, bounded linear functionals, and exact functional evaluation.

**Proof status.** Proved below by the Gram representation.
:::

:::: {.proof}
For \(a\in\mathbb R^m\),

$$
a^\top Ga
=\sum_{i,j}a_i a_j\langle r_j,r_i\rangle_{\mathcal H_k}
=\left\lVert\sum_i a_i r_i\right\rVert_{\mathcal H_k}^2\ge0.
$$

Equality holds exactly when \(\sum_i a_i r_i=0\), equivalently when
\(\sum_i a_iL_i=0\) on \(\mathcal H_k\). Thus \(G\) is positive definite precisely when the functionals are independent. The linear system then has one coefficient vector, and the representer theorem gives the unique minimum-norm interpolant. \(\square\)
::::

Regularized least-squares collocation replaces exact constraints by

$$
\sum_i w_i\{L_i u-y_i\}^2+\lambda\lVert u\rVert_{\mathcal H_k}^2.
$$

Weights are not cosmetic. Interior residuals, boundary values, and physical measurements have different units and counts. A factor that changes when the mesh is refined can silently change the continuous objective.

### From sampled residual to solution error {#scientific-residual-stability}

Suppose \(\mathcal T:X\to Y\) is injective and stable:

$$
\lVert v\rVert_X\le C_{\mathrm{stab}}\lVert\mathcal T v\rVert_Y.
$$

If \(u^\dagger\) and \(\widehat u\) satisfy the same exact boundary conditions, then

$$
\lVert u^\dagger-\widehat u\rVert_X
\le C_{\mathrm{stab}}
\lVert\mathcal A u^\dagger-\mathcal A\widehat u\rVert_Y.
$$

This simple inequality is the bridge from residual to state error. Collocation provides only residual values at finitely many sites. To control the \(Y\)-norm between sites, one additionally needs a sampling inequality involving fill distance, target smoothness, and a norm bound. Native-space analysis supplies such inequalities for appropriate kernels and domains [@wendland2005]. No theorem permits the replacement

$$
\lVert r\rVert_Y\quad\text{by}\quad\max_{x_i\in X_A}|r(x_i)|
$$

without assumptions on \(r\) and the site geometry.

::: {.remark}
Flat or extremely smooth radial kernels can make \(e_{\mathrm{app}}\) small while making \(e_{\mathrm{alg}}\) enormous through ill-conditioning. Approximation order and floating-point stability must be reported separately.
:::

## A complete collocation calculation {#scientific-worked-poisson}

Consider

$$
-u''(x)=2,\quad 0\lt x\lt1,\qquad u(0)=u(1)=0,
$$

whose solution is \(u^\dagger(x)=x(1-x)\). Use the degree-two polynomial kernel

$$
k(x,z)=(1+xz)^2=1+2xz+x^2z^2.
$$

Its RKHS is the three-dimensional polynomial space generated by
\((1,\sqrt2x,x^2)\). Take the information functionals

$$
L_0u=u(0),\qquad L_1u=u(1),\qquad L_2u=-u''(1/2),
$$

with target vector \(y=(0,0,2)^\top\). Their representers are

$$
r_0(x)=1,\qquad
r_1(x)=(1+x)^2,\qquad
r_2(x)=-2x^2.
$$

Therefore

$$
G=
\begin{bmatrix}
1&1&0\\
1&4&-2\\
0&-2&4
\end{bmatrix},
\qquad
Gc=
\begin{bmatrix}0\\0\\2\end{bmatrix}.
$$

Solving gives

$$
c=\left(-\frac12,\frac12,\frac34\right)^\top,
$$

and hence

$$
\widehat u(x)
=-\frac12+\frac12(1+2x+x^2)-\frac32x^2
=x-x^2.
$$

The exact solution is recovered because it lies in the RKHS and the three independent functionals identify all three polynomial coefficients. The same calculation also exposes three failure boundaries.

- Removing either boundary functional leaves a nontrivial null direction.
- Replacing \(k\) by a degree-one polynomial kernel makes second-derivative information identically zero.
- Perturbing the forcing to a nonconstant function leaves the discrete equation satisfied at \(x=1/2\) but does not make the global residual vanish.

::: {.example #example-scientific-boundary}
[Example (boundary constraints change the admissible space)]{.box-title}

Two kernel solvers may use the same interior differential residuals but different boundary functionals. Their solutions can agree at every interior collocation site and still represent different boundary-value problems. Boundary conditions are part of the operator and function space, not an afterthought to be patched onto an unconstrained regression fit.
:::

<figure class="viz" data-figure="collocation-residual" data-alt="The left panel compares the exact sine solution with a values-only RBF fit and a fit constrained by values, differential-equation residuals, and zero boundary conditions. The right panel compares equation residuals at independent locations."><figcaption>Different information functionals produce different representers. A smaller independent residual supports the discretized solver, but a continuous error claim still needs stability and a sampling inequality.</figcaption></figure>

<figure class="viz" data-figure="scientific-collocation-convergence" data-alt="A log-log plot shows both independent differential-equation residual and solution L2 error decreasing as the number of collocation sites increases, on distinct scales.">
<figcaption>Refining the information set improves both the discrete equation certificate and the recovered function in this Poisson example, but the curves are not interchangeable. Turning residual decay into solution-error decay still requires the stability argument developed above.</figcaption>
</figure>

## Paper module I: Gaussian processes for linear differential equations {#scientific-module-gpde}

### The question and exact setting {#scientific-gpde-setting}

Raissi, Perdikaris, and Karniadakis asked whether the unknown coefficients of a linear differential equation could be learned jointly with a latent solution from scarce, noisy observations [@raissi2017gpde]. Let

$$
f=\mathcal L^\phi u,
$$

where \(\mathcal L^\phi\) is linear in \(u\) but may depend on a finite-dimensional parameter \(\phi\). Place

$$
u\sim\mathcal{GP}(m,k_\theta).
$$

Assume every functional needed to evaluate \(u\) and \(\mathcal L^\phi u\) is mean-square bounded. A sufficient condition for a differential operator of order \(r\) is that the required mixed derivatives of \(k_\theta\) exist and are continuous.

Observe noisy values at two site sets:

$$
y_u=u(X_u)+\varepsilon_u,\qquad
y_f=f(X_f)+\varepsilon_f,
$$

with independent Gaussian noises of variances \(\sigma_u^2\) and \(\sigma_f^2\). The paper's new move is not merely differentiating a kernel. It treats the differential equation as a map between jointly Gaussian latent fields and learns \((\theta,\phi,\sigma_u,\sigma_f)\) through their joint marginal likelihood.

### Covariance derivation and executable object {#scientific-gpde-derivation}

The construction is easiest to audit one block at a time. Start with the prior
covariance of the latent field, then apply the observation operator to the
argument associated with each observed quantity. Linearity gives

$$
\begin{aligned}
k_{uu}(x,z)&=k_\theta(x,z),\\
k_{uf}(x,z)&=\mathcal L_z^\phi k_\theta(x,z),\\
k_{fu}(x,z)&=\mathcal L_x^\phi k_\theta(x,z),\\
k_{ff}(x,z)&=\mathcal L_x^\phi\mathcal L_z^\phi k_\theta(x,z).
\end{aligned}
$$

Thus the observation covariance is

$$
K_y=
\begin{bmatrix}
K_{uu}(X_u,X_u)+\sigma_u^2I & K_{uf}(X_u,X_f)\\
K_{fu}(X_f,X_u) & K_{ff}(X_f,X_f)+\sigma_f^2I
\end{bmatrix}.
$$

For example, if \(\mathcal L^\phi=\phi\,\partial_x\), the upper-right entry is
\(\phi\,\partial_z k_\theta(x,z)\), while the lower-right entry is
\(\phi^2\partial_x\partial_z k_\theta(x,z)\). This two-observation check catches
the most common implementation error: differentiating both arguments in an
off-diagonal block. A second invariant is symmetry,
\(K_{fu}(X_f,X_u)=K_{uf}(X_u,X_f)^\top\); its failure usually means that an
operator was applied to the wrong argument or that row and column site orderings
disagree.

For a test functional \(M\), define

$$
k_{My}=
\begin{bmatrix}
M_xk_\theta(x,X_u)&
M_x\mathcal L_z^\phi k_\theta(x,X_f)
\end{bmatrix}.
$$

The posterior mean and variance are

$$
\begin{aligned}
\mathbb E[Mu\mid y]&=Mm+k_{My}K_y^{-1}(y-m_y),\\
\operatorname{Var}(Mu\mid y)&=
M_xM_zk_\theta(x,z)-k_{My}K_y^{-1}k_{yM}.
\end{aligned}
$$

In implementation these expressions use a Cholesky factorization, never an explicit inverse. Parameters are commonly estimated by minimizing

$$
\frac12(y-m_y)^\top K_y^{-1}(y-m_y)
+\frac12\log\det K_y+\frac{n}{2}\log(2\pi).
$$

The first term rewards fit; the second penalizes covariance volume. When \(\phi\) enters only through \(\mathcal L^\phi\), both terms carry information about the differential equation.

### What is proved, what is inherited, and what fails {#scientific-gpde-boundary}

The joint Gaussian formulas are exact under the stated prior and linear information model. They do not prove that marginal-likelihood optimization identifies \(\phi\). Identification fails if two parameter values induce the same covariance on the observed sites, if the latent field has insufficient excitation, or if \(\phi\) can be traded against a kernel length scale. The construction also changes qualitatively for a nonlinear operator because \(\mathcal N(u)\) is generally not Gaussian.

In the zero-noise, fixed-parameter case, the posterior mean equals the minimum-RKHS-norm functional interpolant and the posterior variance equals the squared functional power function. This connects the paper to collocation. The interpretations differ:

| Object | Deterministic reading | GP reading |
|---|---|---|
| Posterior mean | Minimum-norm functional interpolant | Conditional mean under the prior |
| Posterior variance | Worst-case squared residual representer norm | Conditional prior variance |
| Hyperparameter fit | Kernel and scale selection | Empirical Bayes |
| Failure diagnostic | Residual, stability, fill distance | Prior predictive and coverage checks |

Neither reading automatically includes operator misspecification or numerical differentiation error.

## Paper module II: Bayesian probabilistic numerical methods {#probabilistic-numerics}

### Numerical tasks as inverse problems {#scientific-pn-setting}

Cockayne, Oates, Sullivan, and Girolami separate three maps [@cockayne2019probnum]:

$$
u\in\mathcal U,\qquad
A:\mathcal U\to\mathcal A,\qquad
Q:\mathcal U\to\mathcal Q.
$$

The latent object is \(u\), the numerical procedure obtains information \(a=A(u)\), and the desired answer is \(Q(u)\). Assume \(\mathcal U,\mathcal A,\mathcal Q\) are standard Borel spaces, \(\mu\) is a probability measure on \(\mathcal U\), and a regular conditional distribution \(\mu^a\) of \(u\) given \(A(u)=a\) exists. A probabilistic numerical method returns a probability distribution on \(\mathcal Q\). It is Bayesian for \(\mu\) when that output is

$$
Q_\#\mu^a,
$$

the pushforward of the conditional distribution through \(Q\).

This definition is the paper's central contribution. It prevents a distribution attached to a deterministic answer from being called Bayesian unless it is generated by conditioning a coherent prior through the actual information operator.

### Gaussian linear case and Bayes-risk derivation {#scientific-pn-derivation}

Let \(\mathcal U\) be a separable Hilbert space, \(u\sim\mathcal N(m,C)\), and let \(A:\mathcal U\to\mathbb R^m\) and \(Q:\mathcal U\to\mathbb R^q\) be bounded linear maps. With additive Gaussian information noise \(\eta\sim\mathcal N(0,\Gamma)\),

$$
a=Au+\eta.
$$

Then \(Q(u)\mid a\) is Gaussian with

$$
\begin{aligned}
m_{Q\mid a}
&=Qm+QCA^\ast(ACA^\ast+\Gamma)^{-1}(a-Am),\\
C_{Q\mid a}
&=QCQ^\ast-QCA^\ast(ACA^\ast+\Gamma)^{-1}ACQ^\ast.
\end{aligned}
$$

These are the same equations as functional GP conditioning, written without coordinates. The probabilistic-numerical object is the whole conditional law, not just \(m_{Q\mid a}\).

::: {.proposition #prop-scientific-bayes-risk}
[Proposition (what posterior variance calibrates on average)]{.box-title}

Assume \(Q(u)\in L^2(\mu;\mathbb R^q)\), with expectation also taken over the information noise when it is present. Among all measurable estimators \(\delta(a)\), the posterior mean \(\delta^\ast(a)=\mathbb E[Q(u)\mid a]\) minimizes integrated squared error, and

$$
\mathbb E\lVert Q(u)-\delta^\ast(a)\rVert_2^2
=\mathbb E\,\operatorname{tr}\operatorname{Cov}(Q(u)\mid a).
$$

**Assumptions.** Square-integrable quantity of interest, measurable information and decision rules, and a regular conditional distribution.

**Proof status.** Proved below by conditional orthogonal decomposition in \(L^2\).
:::

:::: {.proof}
For any \(\delta(a)\), condition on \(a\) and write

$$
Q(u)-\delta
=\{Q(u)-\mathbb E(Q(u)\mid a)\}
+\{\mathbb E(Q(u)\mid a)-\delta\}.
$$

The conditional cross term is zero because the first bracket has conditional mean zero. Therefore

$$
\mathbb E[\lVert Q(u)-\delta\rVert^2\mid a]
=\operatorname{tr}\operatorname{Cov}(Q(u)\mid a)
+\lVert\mathbb E(Q(u)\mid a)-\delta\rVert^2.
$$

The second term is minimized by the posterior mean. Taking expectation proves the identity. \(\square\)
::::

### Failure boundary and afterlife {#scientific-pn-boundary}

The proposition is an average statement under \(\mu\). It is not pointwise frequentist calibration for a fixed \(u^\dagger\), and it can be badly misleading when \(\mu\) assigns too little mass near the true solution or when \(A\) is implemented approximately but treated as exact. A narrow conditional distribution can coexist with a large \(e_{\mathrm{model}}\) or \(e_{\mathrm{alg}}\).

Probabilistic mesh refinement can choose the next functional by expected reduction in posterior loss. Deterministic residual refinement chooses it by an error indicator. The two policies should be compared on actual target error, not on their own internal uncertainty criteria.

## Inverse scientific problems {#scientific-inverse-problems}

In an inverse problem the coefficient \(a\) is unknown and the measured field is generated through a forward solution operator:

$$
y=\mathcal O\mathcal G(a)+\varepsilon.
$$

Kernel regularization may be placed on \(a\), on the state \(u=\mathcal G(a)\), or on a discrepancy term. The decomposition

$$
y-\mathcal O\widetilde{\mathcal G}(a)
=\varepsilon
+\mathcal O\{\mathcal G(a)-\widetilde{\mathcal G}(a)\}
$$

shows why surrogate error belongs in the likelihood. Treating \(\widetilde{\mathcal G}\) as exact makes the inverse posterior overconfident.

Identifiability comes before regularization. If

$$
\mathcal O\mathcal G(a_1)=\mathcal O\mathcal G(a_2)
$$

for \(a_1\ne a_2\), a kernel penalty can select one representative but cannot recover the missing information. Local diagnostics examine the singular values of the Fréchet derivative \(D(\mathcal O\mathcal G)(a)\); global nonidentifiability may remain even when the derivative is injective locally.

## Kernel operator regression {#scientific-kernel-operator-regression}

Let \(\mathcal A\) be an input function space and \(\mathcal U\) a separable Hilbert output space. An operator-valued positive kernel

$$
K:\mathcal A\times\mathcal A\to\mathcal L(\mathcal U)
$$

defines an RKHS of maps \(F:\mathcal A\to\mathcal U\). Given pairs \((a_i,u_i)\), vector-valued ridge regression solves

$$
\min_F\frac1n\sum_{i=1}^n\lVert F(a_i)-u_i\rVert_{\mathcal U}^2
+\lambda\lVert F\rVert_{\mathcal H_K}^2.
$$

The representer theorem yields

$$
\widehat F(a)=\sum_{i=1}^nK(a,a_i)c_i,\qquad c_i\in\mathcal U.
$$

For \(K(a,b)=k(a,b)I_{\mathcal U}\), expansion in any orthonormal output basis decouples the output coordinates while sharing the same input Gram matrix. A finite grid is only a coordinate system if the norms, sampling maps, and reconstruction maps are defined at the function-space level.

Generalization error and discretization error are different:

$$
\lVert \widehat F_h(a_h)-\mathcal G(a)\rVert_{\mathcal U}
\le
\underbrace{\lVert \widehat F_h(a_h)-P_h\widehat F(a)\rVert}_{\text{implementation and discretization}}
+
\underbrace{\lVert P_h\{\widehat F(a)-\mathcal G(a)\}\rVert}_{\text{operator learning}}
+
\underbrace{\lVert P_h\mathcal G(a)-\mathcal G(a)\rVert}_{\text{output projection}}.
$$

Testing on a finer grid addresses only part of this inequality.

## Paper module III: Fourier neural operators {#scientific-module-fno}

### Architecture and derivation {#scientific-fno-derivation}

Li and collaborators target repeated solution of a parametric PDE by learning the operator \(a\mapsto u\), rather than one discretized vector-to-vector map [@li2020fno]. On a periodic domain, a neural-operator layer has the form

$$
v_{t+1}(x)=\sigma\left(
W_tv_t(x)+
\int_\Omega\kappa_t(x-y)v_t(y)\,dy
\right).
$$

Translation invariance makes the integral a convolution. The Fourier convolution theorem gives

$$
\mathcal F(\kappa_t\ast v_t)(\xi)
=\widehat\kappa_t(\xi)\widehat v_t(\xi).
$$

The FNO keeps modes \(\xi\in\Lambda_M\), learns complex matrices \(R_t(\xi)\), and computes

$$
v_{t+1}
=\sigma\left(
W_tv_t+
\mathcal F^{-1}\{R_t(\xi)\widehat v_t(\xi)\mathbf1_{\xi\in\Lambda_M}\}
\right).
$$

On \(N\) grid points, FFTs cost \(O(N\log N)\) per channel, while the learned spectral multiplication costs \(O(|\Lambda_M|c^2)\) for channel width \(c\). The same spectral parameters can be evaluated on another uniform grid if that grid resolves the retained modes.

### A heat-operator calculation {#scientific-fno-heat}

The heat semigroup supplies a useful oracle because its exact multiplier is
known before any network is trained. On the one-dimensional torus, the heat equation

$$
\partial_tu=\nu\partial_{xx}u,\qquad u(\cdot,0)=a
$$

has solution operator

$$
\widehat{\mathcal G_ta}(m)
=e^{-4\pi^2\nu m^2t}\widehat a(m).
$$

A linear Fourier layer can represent the projection onto modes \(|m|\le M\) exactly by setting

$$
R(m)=e^{-4\pi^2\nu m^2t}.
$$

Take \(a(x)=\sin(2\pi x)+\tfrac12\sin(6\pi x)\). At time \(t\), the two
amplitudes must become \(e^{-4\pi^2\nu t}\) and
\(\tfrac12e^{-36\pi^2\nu t}\). A learned layer that instead preserves their
ratio has fitted an identity-like map, not diffusion. This hand-checkable
two-mode input should therefore precede aggregate test error: it detects a
wrong FFT convention, an incorrect frequency grid, or a multiplier attached
to the wrong mode.

If \(a\in H^s(\mathbb T)\), Parseval gives

$$
\begin{aligned}
\lVert\mathcal G_ta-P_M\mathcal G_ta\rVert_{L^2}^2
&=\sum_{|m|\gt M}e^{-8\pi^2\nu m^2t}|\widehat a(m)|^2\\
&\le \{1+(2\pi M)^2\}^{-s}\lVert a\rVert_{H^s}^2.
\end{aligned}
$$

This is a truncation bound, not a learned-operator generalization bound. It explains why resolution transfer is plausible when the relevant spectrum is resolved, and why discontinuities or unresolved turbulence are a failure boundary. In practice, report the retained spectral energy and repeat the two-mode oracle on every evaluation grid; a small training loss cannot certify resolution transfer when the target energy lies above the cutoff.

### Comparison under one currency {#scientific-operator-comparison}

The FNO paper contributes a mesh-independent parameterization and experiments on Burgers, Darcy, and Navier-Stokes problems, including evaluation at higher resolution than training. It does not establish that arbitrary grid refinement reduces the error of a trained model. Grid transfer can expose the same learned low-mode operator more finely without reducing its approximation or statistical error.

<figure class="viz" data-figure="operator-resolution-transfer" data-alt="Output error is plotted against evaluation-grid resolution. A fixed low-mode operator reaches a nonzero error plateau even as the grid becomes finer, whereas an operator containing the missing Fourier mode has negligible error.">
<figcaption>A mesh-independent parameterization can be evaluated on a finer grid without learning any missing frequencies. Once the grid resolves the target, the red plateau is approximation error in the operator, not discretization error in the evaluation.</figcaption>
</figure>

| Method | Optimization | Main inductive bias | Natural certificate | Typical failure |
|---|---|---|---|---|
| Kernel operator ridge | Convex for fixed \(K\) | Similar input functions have similar outputs | Objective residual and RKHS norm | Kernel misspecification and \(n\times n\) scaling |
| FNO | Nonconvex | Translation-compatible spectral mixing | Validation error and spectral diagnostics | Aliasing, geometry mismatch, unresolved modes |
| Classical solver | Usually no training | Governing equation and discretization | A posteriori numerical estimator | Per-instance cost and model error |

An honest benchmark uses the same training trajectories, boundary treatment, physical units, target norms, test parameter distribution, and compute accounting. It includes an established numerical solver because a surrogate is useful only relative to the cost and accuracy of the solver it replaces.

## An auditable scientific-learning workflow {#scientific-workflow}

:::: {.algorithm #algo-scientific-kernel-workflow}
[Algorithm (continuous-to-discrete scientific audit)]{.box-title}

**Input.** A continuous problem, information or simulation pairs, a kernel or operator architecture, and an operational target norm.

**Output.** An estimator with a five-part error ledger.

1. State the domain, function spaces, units, operator, null space, and boundary conditions.
2. Establish well-posedness or state precisely which part is assumed.
3. Verify boundedness of every point, derivative, trace, flux, and integral functional.
4. Choose the discretization and record fill distance, mesh geometry, quadrature, and reconstruction maps.
5. Solve with scaling, stable factorizations or preconditioning, and algebraic residual monitoring.
6. Evaluate data error, independent equation residual, boundary residual, target-norm error, and conditioning separately.
7. For probabilistic output, run prior-predictive, coverage, and misspecification checks.
8. For operator learning, test unseen parameters, finer and irregular meshes, shifted forcing, and unresolved-frequency cases.
9. Compare with a trusted numerical method under matched accuracy and hardware budgets.

Stop refinement when the operational error and an independent diagnostic stabilize, not merely when a training residual or posterior variance becomes small.
::::

## Common mistakes, failure boundaries, and practical implications {#scientific-practice}

- A kernel that is smooth in its displayed formula may still lack the mixed derivatives required by the information functionals.
- Positive semidefiniteness of the collocation matrix does not imply invertibility.
- A small sampled residual does not imply small solution error without continuous stability and a sampling inequality.
- A physics penalty with the wrong boundary condition can confidently solve the wrong problem.
- GP variance conditions on the prior, kernel, operator, and exact information model. It is not automatic frequentist coverage.
- A Bayesian probabilistic numerical method is calibrated on average under its prior, not uniformly over all admissible functions.
- A surrogate in an inverse problem must propagate approximation error into the likelihood.
- Resolution transfer is not convergence to the continuum.
- FNO spectral truncation can hide high-frequency error even when the output grid is fine.
- A faster surrogate comparison is incomplete if data generation and retraining costs are omitted.

## Summary and further reading {#scientific-summary}

Bounded linear information turns differential equations and boundary conditions into RKHS representers. Symmetric collocation inherits positive semidefiniteness from their Gram geometry, while continuous error control additionally requires PDE stability and a sampling inequality [@wendland2005]. Linear differential-equation GPs construct exact covariance blocks by applying operators to kernel arguments and can learn equation parameters only when those parameters are identifiable [@raissi2017gpde]. Bayesian probabilistic numerics defines uncertainty through conditioning and pushforward, with calibration interpreted relative to the prior and information model [@cockayne2019probnum]. Kernel operator regression and FNOs learn maps between functions through different geometries; neither mesh transfer nor low training loss removes discretization, model, or distribution-shift error [@li2020fno].

## Exercises {#exercises}

1. [warm-up]{.ex-tag} For \(k\in C^{2r}(\Omega\times\Omega)\), state a sufficient condition for derivative evaluations \(u\mapsto D^\alpha u(x)\), \(|\alpha|\le r\), to be bounded on \(\mathcal H_k\). Explain why differentiability in only one kernel argument is insufficient for the functional Gram matrix.
2. [computation]{.ex-tag} Reproduce the polynomial-kernel Poisson calculation in Section [A complete collocation calculation](#scientific-worked-poisson). Verify \(G\), solve for \(c\), and show which matrix row becomes zero if \(k(x,z)=1+xz\).
3. [proof]{.ex-tag} Prove the functional representer theorem for a nondecreasing norm penalty and construct a case where a minimizer outside the representer span exists.
4. [proof]{.ex-tag} Assume \(\mathcal T:X\to Y\) is injective and \(\lVert v\rVert_X\le C_{\mathrm{stab}}\lVert\mathcal Tv\rVert_Y\). Derive a state-error bound for an approximate solution with nonzero boundary residual. Identify the additional inequality needed to replace a continuous residual norm by sampled residuals.
5. [computation]{.ex-tag} For \(u\sim\mathcal{GP}(0,k)\) and \(f=u'+\phi u\), derive \(k_{uf}\) and \(k_{ff}\). Add independent value and equation noise, then write the log marginal likelihood used to estimate \(\phi\).
6. [synthesis]{.ex-tag} Show that zero-noise GP conditioning on linearly independent bounded functionals has the same mean as minimum-norm functional interpolation. Explain why equality of the mean does not make the deterministic power function a frequentist coverage guarantee.
7. [proof]{.ex-tag} Prove the posterior-mean Bayes-risk identity in Proposition [what posterior variance calibrates on average](#prop-scientific-bayes-risk), then give a two-point prior example showing that small average Bayes risk need not bound error at a fixed point outside the prior support.
8. [challenge]{.ex-tag} Design a matched benchmark for kernel operator ridge regression, an FNO, and a classical solver on a parametric heat or Darcy problem. Specify input and output spaces, train and test measures, meshes, norms, compute accounting, uncertainty diagnostics, and at least two failure shifts.
