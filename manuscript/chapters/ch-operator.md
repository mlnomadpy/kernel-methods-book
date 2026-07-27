---
id: ch-operator
slug: vector-and-operator-valued-kernels
title: Vector- and Operator-Valued Kernels
part: VI · Designing Kernels
order: 39
tier: advanced
prerequisites:
  - kernels-and-rkhs
  - kernel-tricks
objectives:
  - >-
    Certify an operator-valued kernel from its block quadratic form or a feature
    operator.
  - >-
    Prove the vector-valued representer theorem and derive the regularized block
    system.
  - >-
    Read separable kernels through simultaneous input and output
    eigendirections.
  - Explain when matrix regularizers admit a shared-span representer theorem.
  - >-
    Construct functional-response and differentially constrained vector-field
    kernels.
  - 'Diagnose discretization error, ill-conditioning, and negative transfer.'
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-operator.yml
verification_date: null
bibliography:
  - micchelli2005vv
  - argyriou2009
  - alvarez2012vv
---
# Vector- and Operator-Valued Kernels

<p class="lead">A clinician predicting treatment response rarely wants one number. Biomarker trajectories, adverse-event risks, and dose response are different outputs of one physiology. A force field must return an entire vector, and its components cannot violate the differential laws that couple them. Fitting one scalar model per coordinate discards this structure, but sharing information indiscriminately can make every task worse. Operator-valued kernels make the transfer mechanism explicit: at two inputs the kernel returns an operator that transports an output direction at one location into an output direction at the other. This chapter develops that statement from the Hilbert-space construction through the representer theorem, block solvers, matrix regularization, functional responses, and constrained vector fields. The central question is not whether tasks can share. It is which directions should share, under what assumptions, and how we detect when sharing becomes negative transfer.</p>

## Paper module I: vector-valued RKHS learning {#operator-paper-micchelli}

Micchelli and Pontil's starting obstacle was computational as much as mathematical. If the response belongs to a Hilbert space \(\mathcal Y\), then a predictor belongs to a function space whose values are themselves vectors or functions. Scalar RKHS theory does not say what object reproduces a directional evaluation \(\langle f(x),y\rangle_{\mathcal Y}\), nor why a regularized problem over infinitely many output-valued functions should reduce to finitely many coefficients. Their construction answers both questions [@micchelli2005vv].

**Exact setting.** Let \(\mathcal X\) be a nonempty set and let \(\mathcal Y\) be a real separable Hilbert space. Write \(\mathcal L(\mathcal Y)\) for the bounded linear operators on \(\mathcal Y\), and \(A^*\) for the adjoint of \(A\). No topology on \(\mathcal X\) is needed for the algebraic RKHS construction. Continuity, measurability, or integral-risk statements require additional structure and are not consequences of positive definiteness.

::: {.definition #def-operator-kernel}
[Definition (operator-valued positive definite kernel)]{.box-title}

A map \(K:\mathcal X\times\mathcal X\to\mathcal L(\mathcal Y)\) is an operator-valued positive definite kernel when

$$
K(x,z)=K(z,x)^*
$$

and, for every \(n\), every \(x_1,\ldots,x_n\in\mathcal X\), and every \(y_1,\ldots,y_n\in\mathcal Y\),

$$
\sum_{i,j=1}^n
\langle y_i,K(x_i,x_j)y_j\rangle_{\mathcal Y}\ge 0.
$$

The condition tests the full block operator. Checking that each scalar entry of a matrix-valued kernel is positive definite is neither necessary nor sufficient.
:::

The output direction matters. For \(x\in\mathcal X\) and \(y\in\mathcal Y\), define the section \(K_xy=K(\cdot,x)y\). On finite sums

$$
f=\sum_{i=1}^nK_{x_i}y_i,
\qquad
g=\sum_{j=1}^mK_{z_j}v_j,
$$

set

$$
\langle f,g\rangle_0
=
\sum_{i=1}^n\sum_{j=1}^m
\langle y_i,K(x_i,z_j)v_j\rangle_{\mathcal Y}.
$$

Positive definiteness makes this form nonnegative. Quotienting by its null space and completing gives a Hilbert space \(\mathcal H_K\). The identity

$$
\langle f,K_xy\rangle_{\mathcal H_K}
=\langle f(x),y\rangle_{\mathcal Y}
$$

first holds on finite sums and then extends by continuity. Moreover,

$$
\|K_xy\|_{\mathcal H_K}^2
=\langle y,K(x,x)y\rangle_{\mathcal Y}
\le \|K(x,x)\|_{\mathrm{op}}\|y\|_{\mathcal Y}^2.
$$

Thus boundedness of \(K(x,x)\) makes every directional evaluation bounded. This is the operator-valued Moore-Aronszajn construction, not an analogy with the scalar result.

::: {.theorem #thm-operator-correspondence}
[Theorem (kernel and vector-valued RKHS correspondence)]{.box-title}

Every operator-valued positive definite kernel \(K\) determines, up to an isometric isomorphism fixing evaluations, a unique RKHS \(\mathcal H_K\) of functions from \(\mathcal X\) to \(\mathcal Y\). Conversely, the bounded evaluation operators of every such RKHS determine an operator-valued positive definite kernel.

**Assumptions.** \(\mathcal Y\) is a Hilbert space, \(K(x,z)\) is bounded for each pair, and the displayed Hermitian block-positivity condition holds. **Proof status.** The forward construction is derived above. For the converse, if \(E_xf=f(x)\), then \(K(x,z)=E_xE_z^*\), and the block quadratic form is the squared norm of \(\sum_iE_{x_i}^*y_i\). This is the correspondence developed in Sections 2 and 3 of [@micchelli2005vv].
:::

The factorization \(K(x,z)=\Phi(x)^*\Phi(z)\), with \(\Phi(x):\mathcal Y\to\mathcal F\), is therefore a certificate:

$$
\sum_{i,j}\langle y_i,K(x_i,x_j)y_j\rangle
=
\left\|\sum_i\Phi(x_i)y_i\right\|_{\mathcal F}^2.
$$

It is often the shortest way to validate a nonseparable construction.

## A complete vector-valued representer proof {#operator-representer-proof}

The main finite-reduction result is stronger when its quantifiers are visible. Let \(x_1,\ldots,x_n\) be fixed training inputs. Let

$$
\mathcal S
=\operatorname{span}\{K_{x_i}y:1\le i\le n,\ y\in\mathcal Y\}.
$$

If \(\mathcal Y\) is infinite dimensional, \(\mathcal S\) need not be finite dimensional, but it is generated by only \(n\) output-valued sections. We use its closure when taking an orthogonal projection.

:::: {.theorem #thm-operator-representer}
[Theorem (vector-valued representer theorem)]{.box-title}

Let \(K\) be an operator-valued positive definite kernel and let \(\mathcal H_K\) be its RKHS. Consider

$$
\inf_{f\in\mathcal H_K}
\Phi\{f(x_1),\ldots,f(x_n)\}
+\Omega(\|f\|_{\mathcal H_K}),
$$

where \(\Phi:\mathcal Y^n\to\mathbb R\cup\{+\infty\}\) is arbitrary and \(\Omega:[0,\infty)\to\mathbb R\) is strictly increasing. If a minimizer exists, every minimizer belongs to \(\overline{\mathcal S}\). When \(\mathcal Y=\mathbb R^q\), it has the finite form

$$
f(\cdot)=\sum_{i=1}^nK(\cdot,x_i)c_i,
\qquad c_i\in\mathbb R^q.
$$

**Assumptions.** Point evaluations are bounded, the objective depends on \(f\) only through the displayed evaluations and norm, \(\Omega\) is strictly increasing, and a minimizer exists. Convexity is not required for the representation, although it is useful for existence and uniqueness. **Proof status.** Proved below by orthogonal decomposition, following the regularization argument in [@micchelli2005vv].
::::

**Proof.** Decompose any \(f\in\mathcal H_K\) as \(f=f_\parallel+f_\perp\), where \(f_\parallel\in\overline{\mathcal S}\) and \(f_\perp\perp\overline{\mathcal S}\). For every \(i\) and every \(y\in\mathcal Y\), reproduction gives

$$
\langle f_\perp(x_i),y\rangle_{\mathcal Y}
=
\langle f_\perp,K_{x_i}y\rangle_{\mathcal H_K}
=0.
$$

Since this holds for all \(y\), \(f_\perp(x_i)=0\). Therefore \(f\) and \(f_\parallel\) have identical data terms. Pythagoras gives

$$
\|f\|_{\mathcal H_K}^2
=\|f_\parallel\|_{\mathcal H_K}^2
+\|f_\perp\|_{\mathcal H_K}^2.
$$

If \(f_\perp\ne0\), strict monotonicity of \(\Omega\) makes \(f_\parallel\) strictly better. Hence a minimizer must have \(f_\perp=0\). For \(\mathcal Y=\mathbb R^q\), \(\mathcal S\) is spanned by the \(nq\) sections \(K_{x_i}e_a\), so the displayed finite expansion follows. \(\square\)

**Failure boundary.** If \(\Omega\) is merely nondecreasing, at least one minimum-norm minimizer lies in the span, but other minimizers may carry an invisible perpendicular component. If the objective includes a derivative, integral, or deployment evaluation not represented among the observations, the span must include the corresponding adjoint representers. If the kernel itself is optimized over a parameterized family, this fixed-space proof does not justify a joint finite expansion across all parameter values.

## Squared loss as a block operator equation {#operator-block-system}

Take \(\mathcal Y=\mathbb R^q\) and the normalized objective

$$
\mathcal J(f)
=
\frac1n\sum_{i=1}^n\|y_i-f(x_i)\|_2^2
+\lambda\|f\|_{\mathcal H_K}^2,
\qquad \lambda\gt0.
$$

Stack \(c=(c_1^\top,\ldots,c_n^\top)^\top\) and \(y\) in \(\mathbb R^{nq}\). Let the block Gram matrix \(\mathbf K\) have block \(\mathbf K_{ij}=K(x_i,x_j)\). For \(f=\sum_iK_{x_i}c_i\),

$$
f_X=\mathbf Kc,
\qquad
\|f\|_{\mathcal H_K}^2=c^\top\mathbf Kc.
$$

The coefficient objective is

$$
\frac1n\|y-\mathbf Kc\|_2^2+\lambda c^\top\mathbf Kc.
$$

Its stationarity condition is

$$
\mathbf K(\mathbf K+n\lambda I)c=\mathbf Ky.
$$

The convenient equation

$$
(\mathbf K+n\lambda I)c=y
$$

always supplies a minimizer because \(\mathbf K+n\lambda I\) is strictly positive definite. When \(\mathbf K\) is singular, other coefficient vectors can represent the same function, but the fitted values and RKHS minimizer are unique. This distinction matters: ridge regularization guarantees a unique function, not necessarily a unique unregularized coordinate representation.

For general bounded sampling operators \(S_i:\mathcal Y\to\mathbb R^{m_i}\), the observations are \(S_if(x_i)\). Their representers are \(K_{x_i}S_i^*a\), and the observed Gram block is

$$
G_{ij}=S_iK(x_i,x_j)S_j^*.
$$

This formulation handles missing output coordinates and irregularly sampled response curves without first filling in nonexistent data.

## Worked example: transfer with missing outputs {#operator-worked-transfer}

::: {.example #example-operator-two-task}
[Example (two tasks, two sites, and one response per site)]{.box-title}

Let

$$
k_X=
\begin{pmatrix}
1&r\\
r&1
\end{pmatrix},
\qquad
B=
\begin{pmatrix}
1&\rho\\
\rho&1
\end{pmatrix},
\qquad
K(x,z)=k(x,z)B.
$$

Only task 1 is observed at \(x_1\), and only task 2 is observed at \(x_2\). Both observations equal one. The two sampling operators are \(S_1=e_1^\top\) and \(S_2=e_2^\top\), so the observed Gram matrix is

$$
G=
\begin{pmatrix}
1&r\rho\\
r\rho&1
\end{pmatrix}.
$$

With ridge \(\tau=n\lambda\), symmetry gives coefficients

$$
a_1=a_2=\frac{1}{1+\tau+r\rho}.
$$

Choose \(r=1/4\), \(\rho=3/4\), and \(\tau=1/2\). Then \(B\) has eigenvalues \(7/4\) and \(1/4\), so it is positive definite, and \(a_1=a_2=16/27\). The predictions at \(x_1\) are

$$
f_1(x_1)=\frac{19}{27}\approx0.7037,
\qquad
f_2(x_1)=\frac{16}{27}\approx0.5926.
$$

For the independent-task kernel \(\rho=0\), they become \(2/3\) and \(1/6\). Coupling therefore moves the scarcely observed second task at \(x_1\) from \(0.1667\) to \(0.5926\). If the true second response at \(x_1\) has the same sign, this is useful transfer. If it has the opposite sign, the same algebra is negative transfer.

**Verification.** The fractions follow from a \(2\times2\) solve and can be checked by direct substitution. The calculation verifies this finite example only; it does not establish a population advantage for task coupling.
:::

<figure class="viz" data-figure="operator-valued-field" data-alt="Two panels show the vector response to a unit observation in output one. With an identity output matrix only output one responds; with positive off-diagonal coupling, output two receives a smaller dashed response with the same input shape."><figcaption>An operator-valued kernel separates two mechanisms. The input kernel determines where influence travels, while the output operator determines which response directions receive it. A wrong off-diagonal coupling produces negative transfer by the same mechanism that produces useful borrowing.</figcaption></figure>

<figure class="viz" data-figure="operator-kernel-coupling-transfer" data-alt="Prediction error for a second task is plotted against output-coupling strength. Error decreases when the two outputs are aligned but rises sharply when they have opposite signs.">
<figcaption>Output coupling is an assumption about task geometry. Increasing it helps when the tasks share direction and creates negative transfer when they oppose each other, so learning or validating \(B\) matters as much as choosing the scalar input kernel.</figcaption>
</figure>

This example exposes all ten parts of the first paper module. The question is finite reduction for vector outputs; the setting is a Hilbert-valued predictor; the contribution is the operator-valued RKHS and representer form; the theorem and proof are explicit; the block solve is executable; the sign-reversed task is the failure boundary; independent regression is the comparison; missing-output operators show the extension; and the source location is Sections 2 through 4 of [@micchelli2005vv].

## Separable kernels and their spectral anatomy {#operator-separable}

The simplest valid construction is

$$
K(x,z)=k(x,z)B,
$$

where \(k\) is scalar positive definite and \(B\succeq0\). If \(B=A^*A\), then

$$
\sum_{i,j}\langle y_i,k(x_i,x_j)By_j\rangle
=
\sum_{i,j}k(x_i,x_j)\langle Ay_i,Ay_j\rangle\ge0,
$$

because expanding \(Ay_i\) in an orthonormal basis produces a sum of scalar-kernel quadratic forms.

For training inputs, \(\mathbf K=K_X\otimes B\). If

$$
K_X=U\Lambda U^\top,
\qquad
B=V\Sigma V^\top,
$$

then

$$
\mathbf K
=(U\otimes V)(\Lambda\otimes\Sigma)(U\otimes V)^\top.
$$

The joint eigenmodes are products of an input mode and an output mode. With squared loss, the mode \((i,j)\) is shrunk by

$$
\frac{\lambda_i\sigma_j}{\lambda_i\sigma_j+n\lambda}.
$$

This formula is the precise meaning of borrowing strength in a separable model. Large \(\sigma_j\) preserves the corresponding output combination; a null direction of \(B\) cannot be learned at all.

::: {.proposition #prop-operator-task-rotation}
[Proposition (task rotation decouples separable KRR)]{.box-title}

For complete \(n\times q\) responses and isotropic squared loss, rotating the data matrix as \(\widetilde Y=U^\top YV\) turns separable vector-valued KRR into \(nq\) scalar equations

$$
\widetilde C_{ij}
=
\frac{\widetilde Y_{ij}}{\lambda_i\sigma_j+n\lambda}.
$$

**Assumptions.** \(K_X\) and \(B\) are symmetric positive semidefinite, responses are observed on a common design, and the ridge is isotropic in output coordinates. **Proof status.** Verified by substituting the Kronecker eigendecomposition into \((K_X\otimes B+n\lambda I)c=y\).
:::

**Failure boundary.** The decoupling disappears with task-specific designs, coordinate-dependent losses, or sums \(\sum_rK_r\otimes B_r\) whose factors do not commute. A separable kernel also forces every output eigendirection to use the same input length scale. That is a modeling restriction, not merely a computational convenience.

## Paper module II: when matrix regularization has a representer form {#operator-paper-argyriou}

Operator-valued kernels fix the output geometry before optimization. Argyriou, Micchelli, and Pontil ask a complementary question: which regularizers on a task-parameter matrix preserve a data-span reduction at all? Their answer shows that the representer theorem is not automatic once a familiar penalty is replaced by an arbitrary matrix penalty [@argyriou2009].

Let \(W=[w_1,\ldots,w_q]\in\mathbb R^{d\times q}\). Task \(t\) observes pairs \((x_{ti},y_{ti})\) and predicts \(w_t^\top x_{ti}\). Let

$$
\mathcal L=\operatorname{span}\{x_{ti}:1\le t\le q,\ 1\le i\le m_t\}.
$$

A shared-span representer theorem says that each optimal \(w_t\) may be chosen in \(\mathcal L\), so every task can use input directions observed by every other task. This is broader than the taskwise span produced by a Frobenius penalty.

::: {.theorem #thm-operator-matrix-characterization}
[Theorem (matrix-monotone characterization)]{.box-title}

Suppose \(d\ge2q\) and \(\Omega:\mathbb R^{d\times q}\to\mathbb R\) is differentiable. The orthogonal-monotonicity property

$$
W^\top P=0
\quad\Longrightarrow\quad
\Omega(W+P)\ge\Omega(W)
$$

holds exactly when

$$
\Omega(W)=h(W^\top W)
$$

for a function \(h\) that is nondecreasing in Loewner order:

$$
0\preceq A\preceq C
\quad\Longrightarrow\quad
h(A)\le h(C).
$$

Such a regularizer admits a shared-span solution for interpolation and for the corresponding regularized problems covered by the paper.

**Assumptions.** Finite-dimensional real matrices, differentiability, and \(d\ge2q\) for the necessity direction. The loss or constraints depend on \(W\) through the task measurements. **Proof status.** Proposition 13 of [@argyriou2009] equates the orthogonal-monotonicity property with the shared-span representer property, and Theorem 15 gives the displayed functional characterization. The sufficient projection step and the trace-norm case are proved below. The paper's necessity proof uses the dimension condition to construct orthogonal matrix perturbations.
:::

Let \(\Pi\) be the orthogonal projector onto \(\mathcal L\), and decompose \(W=\overline W+P\) with \(\overline W=\Pi W\). Every training prediction is unchanged because \(x_{ti}^\top P e_t=0\). Also \(\overline W^\top P=0\), so

$$
W^\top W
=\overline W^\top\overline W+P^\top P
\succeq\overline W^\top\overline W.
$$

Matrix monotonicity gives \(\Omega(W)\ge\Omega(\overline W)\). Projection therefore never worsens the objective, and a shared-span minimizer exists.

The nuclear norm supplies a substantial example. Since

$$
\|W\|_*
=\operatorname{tr}\{(W^\top W)^{1/2}\},
$$

and eigenvalues are monotone under Loewner order,

$$
\|\overline W+P\|_*
=
\operatorname{tr}\{(\overline W^\top\overline W+P^\top P)^{1/2}\}
\ge
\operatorname{tr}\{(\overline W^\top\overline W)^{1/2}\}
=\|\overline W\|_*.
$$

This proves the trace-norm shared-span result corresponding to Theorem 12 of [@argyriou2009]. The nuclear norm is not just a sparsity slogan. It penalizes the singular values of \(W\), encouraging task vectors to lie in a low-dimensional shared feature subspace.

**Failure boundary.** The necessity statement above does not cover nondifferentiable penalties, even though important nondifferentiable examples such as the trace norm satisfy the sufficient monotonicity property directly. The dimension bound belongs to the characterization proof and must not be dropped silently. An entrywise \(\ell_1\) penalty depends on the chosen coordinate basis rather than only on \(W^\top W\); a projection onto the data span can increase it, so the shared-span conclusion is not generally licensed.

**Comparison and afterlife.** Fixed \(K=kB\) chooses task coupling in output space; spectral matrix regularization learns a low-dimensional shared input subspace. These are different inductive biases even when both produce low-rank matrices. Later output-kernel learning methods combine them by estimating a positive semidefinite \(B\), while spectral methods regularize \(W^\top W\). The review [@alvarez2012vv] places these constructions in the broader multi-output and Gaussian-process literature.

## A common currency for multi-output models {#operator-comparison}

| Model | Shared object | Finite system | Main strength | Failure boundary |
|---|---|---|---|---|
| Independent scalar kernels | Nothing across tasks | \(q\) scalar systems | No negative transfer from coupling | Wastes genuine shared structure |
| Separable \(kB\) | Output eigendirections | One Kronecker block system | Interpretable and fast | Same input geometry in each output mode |
| Sum \(\sum_r k_rB_r\) | Several input-output modes | Structured block system | Direction-specific length scales | Identifiability and tuning burden |
| Spectral matrix penalty | Shared input subspace | Matrix optimization | Learns task features | Linear or explicitly featured input model |
| Differential kernel | Physical range constraint | Nonseparable block system | Constraint holds everywhere | Structural bias when physics is approximate |

Comparisons should use per-task held-out risk, joint risk, output calibration when meaningful, and compute at matched tolerances. Reporting only average risk can hide severe degradation on a low-resource task.

## Functional responses without pretending a grid is truth {#operator-functional}

Let \(\mathcal Y=L^2(T,\nu)\). A separable operator-valued kernel may act as

$$
[K(x,z)g](t)
=
k(x,z)\int_Tb(t,s)g(s)\,d\nu(s),
$$

where the integral operator \(B\) with kernel \(b\) is bounded, self-adjoint, and positive semidefinite. If \((\phi_r,\sigma_r)\) are its eigenpairs, then

$$
K(x,z)g
=
k(x,z)\sum_r\sigma_r\langle g,\phi_r\rangle\phi_r.
$$

Every response mode \(\phi_r\) is a scalar KRR problem with strength \(\sigma_r\). Trace-class \(B\) gives a compact output coupling, but trace class is stronger than the boundedness needed for the RKHS construction.

In real data the response curve is often observed at irregular points. Let \(S_i:\mathcal Y\to\mathbb R^{m_i}\) be a bounded sampling or averaging operator. Point evaluation is not bounded on bare \(L^2(T)\), so a model using literal point samples must choose a smoother output space, use averaged measurements, or explicitly discretize. Treating an interpolated curve as fully observed hides this domain issue.

With a basis \(\phi_1,\ldots,\phi_R\), three errors must remain separate:

1. response discretization or basis truncation;
2. statistical estimation from \(n\) input-response pairs;
3. iterative or low-rank error in the block solve.

A convergence claim must state which of \(n\), \(R\), and output rank grows, and in which norm the response error is measured.

## Differentially constrained vector fields {#operator-physical-fields}

Some output couplings are laws. Let \(x,z\in\mathbb R^d\) and let \(\psi(x,z)\) be a scalar kernel smooth enough that derivative evaluation is bounded in its RKHS. Define

$$
K_{\mathrm{cf}}(x,z)
=\nabla_x\nabla_z^\top\psi(x,z).
$$

For coefficient vectors \(c_i\),

$$
\sum_{i,j}c_i^\top K_{\mathrm{cf}}(x_i,x_j)c_j
=
\left\|
\sum_i\sum_{a=1}^dc_{ia}\,
\partial_{x_a}\psi(x_i,\cdot)
\right\|_{\mathcal H_\psi}^2
\ge0.
$$

Thus \(K_{\mathrm{cf}}\) is positive semidefinite. Every finite expansion has the form

$$
f(z)=\sum_iK_{\mathrm{cf}}(z,x_i)c_i
=\nabla_z
\left\{
\sum_i c_i^\top\nabla_{x_i}\psi(x_i,z)
\right\},
$$

so it is a gradient field and is curl-free on simply connected regions where the derivatives commute. In \(\mathbb R^d\), the complementary construction

$$
K_{\mathrm{df}}(x,z)
=
\{-\Delta_x I+\nabla_x\nabla_x^\top\}\psi(x,z)
$$

can generate divergence-free fields under the required smoothness and stationarity conventions [@alvarez2012vv].

**Failure boundary.** A whole-space divergence-free kernel does not enforce no-flow or no-slip boundary conditions on a bounded domain. Derivatives of a nonsmooth kernel may not define bounded evaluation functionals. Hard constraints reduce variance when exact, but create irreducible bias under unresolved forcing or approximate conservation. A soft residual penalty is safer when the law is uncertain.

## Learning output geometry and controlling scale {#operator-learning-output}

For a separable kernel, estimating \(B\) from the same data used to fit \(f\) changes the function space. A generic objective is

$$
\min_{B\succeq0,\ f\in\mathcal H_{kB}}
\sum_i\ell\{y_i,f(x_i)\}
+\lambda\|f\|_{\mathcal H_{kB}}^2
+\rho\Omega(B).
$$

The scale of \(B\) and the function norm can trade off, so a trace constraint, trace penalty, or fixed diagonal is needed for identifiability. The parameterization \(B=LL^\top+\delta I\) guarantees positivity and exposes rank, but optimization in \(L\) is nonconvex. Optimization over \(B\) may be convex for a carefully chosen formulation without being cheap.

The spectrum of \(B\) is a diagnostic, not an oracle. A nearly rank-one estimate says the fitted model used one output combination; it does not prove the tasks share one true latent mechanism. Validate:

- every task against its independent scalar baseline;
- a diagonal-\(B\) ablation;
- sensitivity to output standardization;
- stability of the learned eigenspace across resamples;
- performance when one high-resource task is removed.

Negative transfer is established task by task. An average gain does not cancel a material loss on a safety-critical output.

## Block computation without forming the block matrix {#operator-computation}

For \(K_X\otimes B\) and a coefficient matrix \(C\in\mathbb R^{q\times n}\),

$$
(K_X\otimes B)\operatorname{vec}(C)
=
\operatorname{vec}(BC K_X^\top).
$$

The product costs the two smaller matrix multiplications and avoids \(O(n^2q^2)\) storage. A direct dense factorization of the full block matrix costs \(O(n^3q^3)\), while the eigendecomposition of a single separable kernel costs \(O(n^3+q^3)\) plus transforms. For sums of separable kernels, matrix-free conjugate gradients preserve fast products even when one joint diagonalization is unavailable.

:::: {.algorithm #algo-operator-model-selection}
[Algorithm (multi-output modeling and audit)]{.box-title}

**Input.** Multi-output observations, sampling operators for missing coordinates, candidate input kernels, candidate output ranks, and a labeled validation split.

**Output.** A valid operator-valued predictor and a transfer report.

1. Standardize each output from training data and preserve the inverse transform.
2. Fit independent scalar baselines with the same validation budget.
3. Fit a diagonal and then a full or low-rank separable \(B\), constraining \(B\succeq0\).
4. Record eigenvalues of \(B\), block residuals, solver iterations, and regularization.
5. Test sums of separable kernels only if residual patterns show different input scales by output mode.
6. Report every task's risk and uncertainty metric, not only a pooled score.
7. Stress the coupling by task removal, output rescaling, and a diagonal-\(B\) ablation.

Stop an iterative solve using the residual of the original regularized block system. A stable low-rank factor does not imply that the linear system has been solved.
::::

Entrywise approximations of a block kernel can destroy positive semidefiniteness. Operator-valued random features or Nyström factors should approximate a feature operator so the resulting block matrix remains a Gram matrix.

## Common mistakes and practical implications {#operator-practice}

- Pointwise nonnegative matrix entries do not certify operator-valued positive definiteness.
- A representer theorem requires a fixed space and the correct observation representers.
- \(B\succeq0\) permits negative off-diagonal entries; it constrains eigenvalues, not pairwise signs.
- A low-rank output matrix can suppress a real output direction permanently.
- Point sampling is not bounded on \(L^2\) without additional response regularity.
- A physically constrained whole-space kernel may violate boundary conditions.
- Better pooled risk can coexist with harmful transfer on one task.

Begin with independent models and \(kB\). Move to sums, learned output geometry, or nonseparable physical kernels only when a held-out failure pattern identifies the missing structure.

## Summary and further reading {#operator-summary}

An operator-valued kernel is certified by one block quadratic form and induces an RKHS through directional evaluation representers. The vector-valued representer theorem then reduces regularized learning to output-valued coefficients, and squared loss becomes a block ridge system. Micchelli and Pontil supply this functional foundation [@micchelli2005vv]. Argyriou, Micchelli, and Pontil show that matrix regularizers admit a shared-span reduction only under a precise orthogonal monotonicity condition, characterized in the differentiable case by Loewner-monotone functions of \(W^\top W\) [@argyriou2009]. The broader construction map, including multi-output Gaussian-process and differential kernels, is developed in [@alvarez2012vv].

The chapter's practical boundary is equally important: coupling is an assumption. Its success must be established against independent fits, and its algebra must preserve positive definiteness, bounded observations, and the correct output norm.

## Exercises {#exercises}

1. [warm-up]{.ex-tag} Prove directly that \(K(x,z)=k(x,z)B\) is operator-valued positive definite when \(k\) is a scalar positive definite kernel and \(B\succeq0\).
2. [computation]{.ex-tag} Let \(K_X=\begin{psmallmatrix}1&1/2\\1/2&1\end{psmallmatrix}\) and \(B=\begin{psmallmatrix}1&1/2\\1/2&1\end{psmallmatrix}\). Compute all four eigenvalues of \(K_X\otimes B\), then give the four ridge shrinkage factors when \(n\lambda=1/2\).
3. [proof]{.ex-tag} Prove the vector-valued representer theorem by orthogonal decomposition, including the step that \(f_\perp(x_i)=0\) as an equality in \(\mathcal Y\).
4. [computation]{.ex-tag} Reproduce the missing-output worked example with \(r=1/4\), \(\rho=3/4\), and \(\tau=1/2\). Then change the second observation to \(-1\) and compute both predictions at \(x_1\).
5. [proof]{.ex-tag} Let \(\Pi\) project onto the span of all task inputs and write \(W=\Pi W+(I-\Pi)W\). Prove that every Loewner-monotone regularizer \(h(W^\top W)\) admits a shared-span minimizer.
6. [synthesis]{.ex-tag} Compare a separable kernel \(kB\) with a nuclear-norm penalty on a task matrix. Identify the space in which each method shares information, one computational advantage, and one failure mode.
7. [proof]{.ex-tag} Assuming derivative evaluation is bounded in \(\mathcal H_\psi\), prove that \(K_{\mathrm{cf}}(x,z)=\nabla_x\nabla_z^\top\psi(x,z)\) is positive semidefinite and that every finite expansion is curl-free on a simply connected domain.
8. [exploration]{.ex-tag} Design a multi-output validation protocol that can reveal negative transfer hidden by average risk. Include missing outputs, task scaling, coupling ablations, and a decision rule for rejecting the joint model.
