---
id: ch-rkbs
slug: reproducing-kernel-banach-and-variation-spaces
title: Reproducing-Kernel Banach and Variation Spaces
part: XI · Learning the Representation
order: 56
tier: research
prerequisites:
  - kernels-and-rkhs
  - vector-and-operator-valued-kernels
  - inverse-learning-and-spectral-regularization
objectives:
  - Identify exactly which representer arguments require Hilbert geometry.
  - >-
    Construct reproducing-kernel Banach spaces through dual pairs, semi-inner
    products, and duality mappings.
  - >-
    Prove a smooth Banach representer theorem and distinguish its dual expansion
    from a linear primal kernel expansion.
  - >-
    State sparse RKBS representer results with their predual, compactness,
    extreme-point, and norm assumptions.
  - >-
    Derive the finite-width ridge-spline representation of variation-norm
    minimizers and explain its optimization boundary.
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

<p class="lead">A kernel ridge solution is finite, but it is rarely sparse. Its centers are fixed at the observations, and a squared Hilbert norm spreads energy across correlated directions. A shallow neural network asks for something different: choose the directions and offsets of its atoms, and often choose only a few. Replacing the Hilbert norm can deliver that geometry, but it also removes orthogonality, the Riesz identification of a space with its dual, and the familiar linear coefficient formula. This chapter rebuilds the theory without smuggling those conveniences back in. Smooth reproducing-kernel Banach spaces produce finite representations in the dual. Nonsmooth RKBS and measure spaces produce sparse extreme solutions through weak-star compactness and convex geometry. A Radon-domain variation norm then turns the infinite search over ridge functions into a finite-width neural ridge spline, under assumptions that make the width bound true.</p>

## Three representer questions, not one {#rkbs-three-questions}

The phrase “representer theorem” hides three different conclusions.

1. **Hilbert representation.**
   $$
   \widehat f=\sum_{i=1}^n\alpha_i k(x_i,\cdot).
   $$
   The primal solution is a linear combination of fixed data sections.
2. **Smooth Banach representation.**
   $$
   J(\widehat f)=\sum_{i=1}^n\alpha_i\delta_{x_i}.
   $$
   The duality image is finite; applying \(J^{-1}\) can make the primal dependence nonlinear.
3. **Atomic or variation representation.**
   $$
   \widehat f=\sum_{j=1}^M a_j\varphi_{\theta_j}+q.
   $$
   A sparse extreme minimizer exists, the atom parameters \(\theta_j\) may be learned continuously, and a null-space term \(q\) may remain.

The first conclusion does not imply the second without a duality identification, and the second does not imply sparsity. Smoothness makes the duality map single valued; sparsity usually comes from the corners that destroy smoothness.

## What Hilbert geometry supplied {#rkbs-hilbert-structure}

Let \(\mathcal H\) be an RKHS and let \(E:\mathcal H\to\mathbb R^n\) be the evaluation map

$$
Ef=(f(x_1),\ldots,f(x_n)).
$$

The standard proof uses:

- the Riesz map \(\mathcal H\simeq\mathcal H^\ast\);
- an orthogonal decomposition
  \(\mathcal H=\operatorname{ran}E^\ast\oplus\ker E\);
- the Pythagorean identity;
- the linear gradient of \(\tfrac12\lVert f\rVert_{\mathcal H}^2\).

A general Banach space \(\mathcal B\) has an annihilator

$$
(\ker E)^\perp
=\{\nu\in\mathcal B^\ast:\nu(h)=0\text{ for all }h\in\ker E\},
$$

but no canonical orthogonal complement inside \(\mathcal B\). The finite-dimensional identity that survives is

$$
(\ker E)^\perp=\operatorname{ran}E^\ast
=\operatorname{span}\{\delta_{x_1},\ldots,\delta_{x_n}\}.
$$

This identity lives in the dual and is the backbone of smooth Banach representer results.

## Paper module I: the RKBS construction of Zhang, Xu, and Zhang {#rkbs-module-zhang}

### Exact dual-pair setting {#rkbs-dual-pair}

Zhang, Xu, and Zhang define an RKBS more narrowly than “any Banach space with bounded point evaluation” [@zhang2009rkbs]. Let \(\mathcal X\) be an arbitrary set. A real version of their construction consists of:

- a reflexive Banach space \(\mathcal B\) of pointwise-defined functions on \(\mathcal X\);
- an isometric identification of \(\mathcal B^\ast\) with another Banach space \(\mathcal B^\#\) of functions on \(\mathcal X\);
- bounded point evaluation on both \(\mathcal B\) and \(\mathcal B^\#\);
- the dual pairing \(\langle f,g\rangle_{\mathcal B,\mathcal B^\#}\).

The insistence that elements are pointwise-defined matters. Ordinary \(L^p\) contains equivalence classes, so point evaluation is not well-defined without choosing representatives or adding regularity.

::: {.definition #def-rkbs-dual-pair}
[Definition (two-sided reproducing-kernel Banach space)]{.box-title}

A dual pair \((\mathcal B,\mathcal B^\#)\) as above has a two-sided reproducing kernel \(K:\mathcal X\times\mathcal X\to\mathbb R\) when

$$
\begin{aligned}
K(x,\cdot)&\in\mathcal B^\#,
&f(x)&=\langle f,K(x,\cdot)\rangle,\\
K(\cdot,y)&\in\mathcal B,
&g(y)&=\langle K(\cdot,y),g\rangle.
\end{aligned}
$$

The kernel need not be symmetric or positive definite in the Hilbert-space sense.
:::

That final sentence is a structural warning. In a general dual pair, \(K(x,y)\) is a bilinear pairing of two feature maps rather than an inner product of one feature map with itself.

### Semi-inner products and the duality map {#rkbs-duality}

A normed space can admit a semi-inner product \([\cdot,\cdot]_{\mathcal B}\) that is linear in the first argument and satisfies

$$
[f,f]_{\mathcal B}=\lVert f\rVert_{\mathcal B}^2,\qquad
|[f,g]_{\mathcal B}|\le\lVert f\rVert_{\mathcal B}\lVert g\rVert_{\mathcal B},
$$

but need not be symmetric or additive in the second argument. If \(\mathcal B\) is smooth, each nonzero \(f\) has a unique norming functional. The normalized duality map \(J:\mathcal B\to\mathcal B^\ast\) is defined by

$$
\langle f,J(f)\rangle=\lVert f\rVert_{\mathcal B}^2,\qquad
\lVert J(f)\rVert_{\mathcal B^\ast}=\lVert f\rVert_{\mathcal B},
$$

and \([g,f]_{\mathcal B}=\langle g,J(f)\rangle\).

For \(1\lt p\lt\infty\), with conjugate exponent \(q=p/(p-1)\), the normalized map on \(\ell_p^d\) is

$$
J_p(x)_j
=\lVert x\rVert_p^{\,2-p}|x_j|^{p-2}x_j.
$$

It maps into \(\ell_q^d\), is nonlinear unless \(p=2\), and satisfies \(J_q(J_p(x))=x\). Uniform convexity gives uniqueness of best approximants; uniform Fréchet differentiability gives a well-behaved unique semi-inner product. These are genuine assumptions, not Banach-space defaults.

### The dual representer theorem {#rkbs-smooth-representer}

Let \(E:\mathcal B\to\mathbb R^n\) be a bounded linear information map, not necessarily point evaluation. Consider

$$
\min_{f\in\mathcal B}
\Phi(Ef)+\lambda\Psi(\lVert f\rVert_{\mathcal B}),
\qquad \lambda\gt0.
$$

The paper develops interpolation and regularization results in uniformly convex, uniformly Fréchet differentiable RKBSs. The finite representation appears for the dual element \(f^\ast=J(f)\), as in its minimum-norm interpolation Theorem 19 and regularization results in Section 5 [@zhang2009rkbs].

:::: {.theorem #thm-rkbs-dual-representer}
[Theorem (smooth Banach representer principle)]{.box-title}

Let \(\mathcal B\) be reflexive, strictly convex, and smooth. Let \(E:\mathcal B\to\mathbb R^n\) be bounded. Suppose \(\Phi:\mathbb R^n\to\mathbb R\) is convex and differentiable, \(\Psi:[0,\infty)\to\mathbb R\) is differentiable with \(\Psi'(t)\gt0\) for \(t\gt0\), and a nonzero minimizer \(\widehat f\) exists. Then

$$
J(\widehat f)\in\operatorname{ran}E^\ast.
$$

For point evaluations this means

$$
J(\widehat f)=\sum_{i=1}^n\alpha_i\delta_{x_i}.
$$

If \(J^{-1}\) exists, the primal solution is

$$
\widehat f=J^{-1}\left(\sum_i\alpha_i\delta_{x_i}\right),
$$

which is generally nonlinear in \(\alpha\).

**Assumptions.** Reflexivity and smoothness of \(\mathcal B\), bounded finite-dimensional information, a nonzero existing minimizer, and a positive radial derivative at every positive norm.

**Proof status.** Proved below through the annihilator identity \((\ker E)^\perp=\operatorname{ran}E^\ast\).
::::

:::: {.proof}
Set \(z=E\widehat f\) and let \(h\in\ker E\). The curve \(\widehat f+th\) leaves the data term unchanged. First-order optimality at \(t=0\) therefore gives

$$
0
=\lambda\Psi'(\lVert\widehat f\rVert)
\frac{\langle h,J(\widehat f)\rangle}{\lVert\widehat f\rVert}.
$$

The radial derivative assumption gives \(\Psi'(\lVert\widehat f\rVert)\gt0\), so

$$
\langle h,J(\widehat f)\rangle=0
\quad\text{for every }h\in\ker E.
$$

Hence \(J(\widehat f)\in(\ker E)^\perp\). Because the range of \(E\) is finite-dimensional,

$$
(\ker E)^\perp=\operatorname{ran}E^\ast.
$$

Thus \(J(\widehat f)=E^\ast\alpha\) for some \(\alpha\in\mathbb R^n\). For evaluation information,
\(E^\ast\alpha=\sum_i\alpha_i\delta_{x_i}\). \(\square\)
::::

The proof also shows what changes for a nonsmooth loss or norm: gradients become subdifferentials. If \(\Psi'\) can vanish, the norming functional need not be identified by the data. Reflexivity and coercivity help establish existence; strict convexity gives primal uniqueness; smoothness gives a single-valued \(J\). None of these properties implies sparse coefficients.

## A finite-dimensional Banach calculation {#rkbs-worked-lp}

The nonlinear inverse duality map is visible in three coordinates. Minimize

$$
\lVert x\rVert_p
\quad\text{subject to}\quad
x_1+x_3=1,\qquad x_2+x_3=1.
$$

Symmetry gives \(x_1=x_2=a\), \(x_3=b\), and \(a+b=1\).

For \(p=2\), minimizing \(2a^2+b^2\) yields

$$
(a,b)=\left(\frac13,\frac23\right),
\qquad
x^{(2)}=\left(\frac13,\frac13,\frac23\right).
$$

For \(p=4\), minimizing \(2a^4+b^4\) gives

$$
2a^3=b^3,\qquad
a=\frac{1}{1+2^{1/3}},\qquad
b=\frac{2^{1/3}}{1+2^{1/3}}.
$$

Both solutions are dense, but they differ because the unit balls have different curvature. For \(p=4\), the dual element \(J_4(x^{(4)})\) lies in the span of the two constraint normals

$$
(1,0,1),\qquad(0,1,1),
$$

while \(x^{(4)}\) itself is obtained by the nonlinear map \(J_{4/3}\). This is the smooth RKBS conclusion in its smallest form.

## Nonsmooth geometry, preduals, and weak-star existence {#rkbs-nonsmooth-geometry}

To obtain exact sparsity, move from smooth balls to balls with exposed faces. Let \(\mathcal B=\mathcal B_\ast^\ast\) be a dual Banach space with predual \(\mathcal B_\ast\), and equip bounded subsets of \(\mathcal B\) with the weak-star topology. Banach-Alaoglu makes the closed unit ball weak-star compact. If the measurement functionals belong to \(\mathcal B_\ast\), they are weak-star continuous.

For \(f\ne0\), the subdifferential of the norm is

$$
\partial\lVert\cdot\rVert_{\mathcal B}(f)
=\{\nu\in\mathcal B^\ast:
\lVert\nu\rVert_{\mathcal B^\ast}=1,\;
\langle\nu,f\rangle=\lVert f\rVert_{\mathcal B}\}.
$$

In a smooth space this set has one element. In \(\ell_1\) or a measure space it can have a face of norming functionals. Extreme points of that face identify candidate active atoms.

::: {.definition #def-rkbs-atomic-norm}
[Definition (atomic norm and measurement operator)]{.box-title}

For a centrally symmetric atom set \(\mathcal A\subset\mathcal B\), define the gauge

$$
\lVert f\rVert_{\mathcal A}
=\inf\left\{\sum_j|a_j|:
f=\sum_j a_j\varphi_j,\ \varphi_j\in\mathcal A\right\}.
$$

Let \(L:\mathcal B\to\mathbb R^n\) collect weak-star continuous measurements. The minimum-norm interpolation problem is

$$
\inf\{\lVert f\rVert_{\mathcal A}:Lf=y\}.
$$
:::

Finite measurements do not by themselves guarantee a finite atomic solution. One also needs existence, compactness in a suitable topology, and control of the extreme points that can support a minimizer.

## Paper module II: sparse representers in RKBS {#rkbs-module-wang}

### The paper's question and exact assumptions {#rkbs-wang-setting}

Wang, Xu, and Yan ask when an RKBS promotes a genuinely sparse kernel expansion, rather than only a finite dual representation [@wang2024sparserkbs]. Their main setting is a real dual Banach space \(\mathcal B\) with predual \(\mathcal B_\ast\), weak-star continuous linearly independent measurements \(\nu_1,\ldots,\nu_n\in\mathcal B_\ast\), and

$$
Lf=(\langle\nu_1,f\rangle,\ldots,\langle\nu_n,f\rangle).
$$

For \(y\ne0\), let

$$
S(y)=\operatorname*{argmin}\{\lVert f\rVert_{\mathcal B}:Lf=y\}.
$$

Weak-star compactness and lower semicontinuity ensure the solution set is nonempty and compact under the paper's hypotheses. The paper first characterizes extreme points of \(S(y)\) through a data-dependent norming face, then imposes two stronger assumptions to turn those extreme points into sparse kernel sections.

For a suitable dual certificate \(\widehat\nu\in\operatorname{span}\{\nu_i\}\), the assumptions can be stated as:

- **A1, atomic norming face.** There is a finite set \(X_{\widehat\nu}^0\) such that every extreme point of
  \(\partial\lVert\cdot\rVert_{\mathcal B_\ast}(\widehat\nu)\)
  is a signed kernel section \(\pm K(\cdot,x)\), \(x\in X_{\widehat\nu}^0\).
- **A2, exact \(\ell_1\) geometry.** For distinct \(x_j\in X_{\widehat\nu}^0\),
  $$
  \left\lVert\sum_j\alpha_jK(\cdot,x_j)\right\rVert_{\mathcal B}
  =C\lVert\alpha\rVert_1
  $$
  for a fixed \(C\gt0\).

A1 makes the exposed face atomic. A2 prevents cancellation from hiding coefficient mass.

### Sparse extreme-point theorem and derivation {#rkbs-wang-theorem}

Define the finite matrix

$$
(L_{\widehat\nu})_{ij}
=\langle\nu_i,K(\cdot,x_j)\rangle,
\qquad x_j\in X_{\widehat\nu}^0.
$$

The paper's Theorem 10 gives the following conclusion.

:::: {.theorem #thm-rkbs-sparse-extreme}
[Theorem (sparse kernel representation of extreme minimizers)]{.box-title}

Under the predual, existence, certificate, A1, and A2 assumptions above, every extreme point \(\widehat f\in\operatorname{ext}S(y)\) admits

$$
\widehat f=\sum_{j=1}^M\alpha_jK(\cdot,x_j),
\qquad
M\le\operatorname{rank}(L_{\widehat\nu})\le n,
$$

with nonzero \(\alpha_j\), \(x_j\in X_{\widehat\nu}^0\), and

$$
\sum_{j=1}^M|\alpha_j|
=\frac{\lVert\widehat\nu\rVert_{\mathcal B_\ast}}{C}.
$$

**Assumptions.** The paper's predual and weak-star existence conditions, a valid data-dependent certificate, and Assumptions A1 and A2.

**Proof status.** Reconstructed below from the paper's Proposition 7, Proposition 8, Lemma 9, and Theorem 10.
::::

:::: {.proof}
A1 reduces the data-dependent extreme norming face to finitely many signed kernel sections. Therefore any extreme minimum-norm solution can be represented using those sections. A2 turns the infinite-dimensional norm minimization into

$$
\min_{\alpha}\lVert\alpha\rVert_1
\quad\text{subject to}\quad
L_{\widehat\nu}\alpha=y.
$$

Let \(\widehat\alpha\) be an extreme minimizer. If its support columns were linearly dependent, there would be a nonzero vector \(h\) supported on the same coordinates with
\(L_{\widehat\nu}h=0\). Basis-pursuit optimality supplies a dual certificate \(c\) satisfying
\(L_{\widehat\nu,S}^\top c=\operatorname{sign}(\widehat\alpha_S)\) on the active set \(S\). Hence

$$
\operatorname{sign}(\widehat\alpha_S)^\top h_S
=c^\top L_{\widehat\nu,S}h_S=0.
$$

For sufficiently small \(t\gt0\), the signs of
\(\widehat\alpha\pm th\) stay fixed, so both perturbations are feasible and have the same \(\ell_1\) norm. They are distinct minimizers whose midpoint is \(\widehat\alpha\), contradicting extremality. Thus the active columns are independent and

$$
|\operatorname{supp}\widehat\alpha|
\le\operatorname{rank}(L_{\widehat\nu}).
$$

A2 and the norming-certificate identity give the displayed coefficient sum. \(\square\)
::::

The proof reconstructs the rank argument, but the passage from the RKBS problem to the finite \(\ell_1\) problem depends entirely on A1 and A2. The paper shows that \(\ell_1(\mathbb N)\) and certain measure RKBSs satisfy them. It also shows why \(\ell_p(\mathbb N)\), \(1\lt p\lt\infty\), does not inherit this sparse conclusion: its norm is curved, its norming faces are not coordinate faces, and A2 fails.

### Worked sparse certificate {#rkbs-worked-l1}

Return to the constraints

$$
x_1+x_3=1,\qquad x_2+x_3=1,
$$

but minimize \(\lVert x\rVert_1\). Writing \(x=(1-t,1-t,t)\) gives

$$
\lVert x\rVert_1=2|1-t|+|t|.
$$

The minimum is attained uniquely at \(t=1\), so

$$
x^{(1)}=(0,0,1).
$$

The dual problem is

$$
\max_{c\in\mathbb R^2}(1,1)^\top c
\quad\text{subject to}\quad
\left\lVert
\begin{bmatrix}
1&0\\0&1\\1&1
\end{bmatrix}c
\right\rVert_\infty\le1.
$$

The certificate \(c=(1/2,1/2)\) has dual objective \(1\) and correlations
\((1/2,1/2,1)\). Only the third atom saturates the dual constraint, explaining the one-coordinate primal solution. The measurement matrix has rank two, so the theorem allows at most two atoms; the exposed face sharpens the actual solution to one.

::: {.example #example-rkbs-dense-sparse}
[Example (quadratic and atomic regularization select different geometry)]{.box-title}

An RKHS squared norm spreads a solution across correlated directions because energy adds quadratically. An atomic norm pays for the total magnitude of selected atoms and can prefer a small set of extreme directions. Both estimators can interpolate the same observations, yet one expresses smooth distributed energy while the other expresses sparse learned features.
:::

<figure class="viz" data-figure="hilbert-vs-variation" data-alt="Two fitted curves nearly overlap a target made from two localized bumps. A coefficient plot shows the quadratic solution spreading weight across many correlated atoms while the atomic-norm solution uses a small active set."><figcaption>Prediction error alone does not reveal representation geometry. A Hilbert penalty distributes energy; an atomic penalty can expose a low-dimensional face and select a sparse certificate.</figcaption></figure>

<figure class="viz" data-figure="rkbs-sparsity-path" data-alt="A regularization path plots training root-mean-square error against the number of active atoms. Larger atomic penalties move left toward smaller supports while increasing residual error."><figcaption>Atomic regularization does not produce sparsity for free. Its path exposes the exact exchange: increasing \(\lambda\) removes active atoms, eventually paying in fit. The useful operating point is a frontier choice, not the sparsest endpoint by default.</figcaption></figure>

### Regularization and its boundary {#rkbs-wang-regularization}

For

$$
\min_f Q_y(Lf)+\lambda\phi(\lVert f\rVert_{\mathcal B}),
$$

the sparse result requires lower semicontinuity and coercivity for existence, convexity to control the solution set, and strict increase of \(\phi\) to connect each regularized minimizer with a minimum-norm interpolant at its fitted value. The paper's Theorem 12 shows that nonzero extreme regularized solutions inherit a rank-bounded kernel expansion under A1 and A2 [@wang2024sparserkbs].

This does not say that every minimizer is sparse. A convex solution set may contain nonsparse convex combinations of sparse extreme points. It also does not make the atom search easy: \(X_{\widehat\nu}^0\) is data-dependent and may be expensive to identify.

## Paper module III: neural networks as ridge splines {#variation-ridge-splines}

### Radon-domain variation space {#rkbs-parhi-setting}

Parhi and Nowak seek a function-space problem whose solutions are finite-width single-hidden-layer networks [@parhi2021banach]. For integer \(m\ge2\), let \(R\) denote the Radon transform, let \(\Lambda^{d-1}\) be the ramp filter, and define

$$
R_m=c_d\partial_t^m\Lambda^{d-1}R.
$$

They work in the growth-restricted native space

$$
\mathcal F_m=
\left\{
f\in L^{\infty,m-1}(\mathbb R^d):
R_mf\in\mathcal M(S^{d-1}\times\mathbb R)
\right\},
$$

where \(\mathcal M\) is the space of finite Radon measures. The regularizer is

$$
\lVert R_mf\rVert_{\mathcal M}.
$$

Its null space \(\mathcal N_m\) is finite-dimensional under the growth restriction and consists of polynomials of degree less than \(m\). A measurement operator
\(V:\mathcal F_m\to\mathbb R^N\) must be continuous, linear, and surjective, and it must identify the null space:

$$
Vq_1=Vq_2,\ q_1,q_2\in\mathcal N_m
\quad\Longrightarrow\quad q_1=q_2.
$$

The last condition is the analogue of the polynomial side constraints in spline interpolation.

### Finite-width theorem and proof skeleton {#rkbs-parhi-theorem}

:::: {.theorem #thm-rkbs-ridge-spline}
[Theorem (finite-width polynomial ridge-spline minimizer)]{.box-title}

Let \(G:\mathbb R^N\to\mathbb R\) be strictly convex, coercive, and lower semicontinuous. Under the assumptions on \(V\), \(\mathcal F_m\), and \(\mathcal N_m\) above, the problem

$$
\min_{f\in\mathcal F_m}
G(Vf)+\lVert R_mf\rVert_{\mathcal M}
$$

has a minimizer of the form

$$
s(x)=\sum_{k=1}^K v_k\rho_m(w_k^\top x-b_k)+q(x),
$$

where

$$
\rho_m(t)=\frac{(t)_+^{m-1}}{(m-1)!},
\quad w_k\in S^{d-1},
\quad q\in\mathcal N_m,
\quad K\le N-\dim\mathcal N_m.
$$

This is Theorem 1 of Parhi and Nowak [@parhi2021banach].

**Assumptions.** Growth-restricted native space, continuous linear surjective measurements, identification on the polynomial null space, and strictly convex coercive lower-semicontinuous data fit.

**Proof status.** Proof skeleton reconstructed below; the stable right-inverse and Banach-space construction are cited to the primary paper.
::::

The difficult step is not Carathéodory alone. The proof:

1. constructs a stable right inverse of \(R_m\);
2. equips \(\mathcal F_m\) with a Banach direct-sum topology separating the measure component from \(\mathcal N_m\);
3. reduces the problem to total-variation minimization over Radon measures;
4. invokes an extreme-point representer theorem for the measure problem;
5. maps each Dirac measure at \((w,b)\) back to the ridge atom \(\rho_m(w^\top x-b)\);
6. restores the polynomial null-space term.

The width bound subtracts \(\dim\mathcal N_m\) because that many measurements are needed to identify the unpenalized polynomial component.

### From variation norm to path norm and weight decay {#rkbs-parhi-regularizer}

For a finite network

$$
f_\theta(x)=\sum_{k=1}^K v_k\rho_m(w_k^\top x-b_k)+q(x),
$$

the Radon-domain variation seminorm becomes

$$
\lVert R_mf_\theta\rVert_{\mathcal M}
=\sum_{k=1}^K|v_k|\lVert w_k\rVert_2^{m-1}
$$

after accounting for the homogeneity of \(\rho_m\). The paper's Theorem 8 relates the continuous variational problem to finite-width training once \(K\) is large enough and shows an equivalent balanced penalty

$$
\frac12\sum_{k=1}^K
\left(|v_k|^2+\lVert w_k\rVert_2^{2m-2}\right).
$$

For \(m=2\), these correspond to a ReLU path-type penalty and balanced weight decay. The equivalence concerns global minimizers. It does not prove that gradient descent finds one, and it does not say that every parameterization of the same function has the same cost before balancing.

### A one-knot ridge-spline calculation {#rkbs-worked-ridge-spline}

Take \(d=1\), \(m=2\), and exact data

$$
f(0)=0,\qquad f(1)=1,\qquad f(2)=0.
$$

The null space is affine functions. The function

$$
s(x)=x-2(x-1)_+
$$

has one ReLU atom plus the affine term \(x\), interpolates all three values, and has distributional second derivative

$$
s''=-2\delta_1.
$$

Therefore \(\lVert s''\rVert_{\mathrm{TV}}=2\). To see optimality, any admissible \(f\) has average slope \(1\) on \([0,1]\) and average slope \(-1\) on \([1,2]\). The total variation of its first derivative must be at least the difference between those two averages, namely \(2\). The tent function attains the bound.

Here \(N=3\), \(\dim\mathcal N_2=2\), and the theorem predicts a minimizer with at most one atom. The hand calculation reaches that bound exactly.

### Failure boundary and afterlife {#rkbs-parhi-boundary}

- If \(V\) is not injective on \(\mathcal N_m\), the polynomial part can drift without penalty.
- If point evaluation is not continuous on \(\mathcal F_m\), scattered-data fitting is not covered.
- The theorem asserts existence of a sparse minimizer, not uniqueness.
- Finite width is bounded by measurements, not by an intrinsic truth width.
- The convex function-space problem can induce a nonconvex atom-location problem.
- Ordinary neural-network training is not automatically an algorithm for the variation problem.

The paper's value is a precise bridge: finite networks are not merely analogies to splines; they are representers of a specific continuous-domain variational problem.

## Algorithms and certificates {#rkbs-algorithms}

For a finite dictionary \(\{\varphi_j\}_{j=1}^M\), atomic regularization becomes a lasso-like problem. For a continuous dictionary, conditional-gradient or exchange algorithms maintain an active set.

:::: {.algorithm #algo-rkbs-column-generation}
[Algorithm (certified atomic column generation)]{.box-title}

**Input.** Measurements \(L\), loss \(Q_y\), atom family \(\mathcal A\), regularization \(\lambda\), and oracle tolerance \(\epsilon_{\mathrm{LMO}}\).

**Output.** A finite atomic predictor and a decomposed optimality certificate.

1. Solve the convex coefficient problem on the current active atoms.
2. Form a dual vector \(c\) from the loss subgradient.
3. Call the linear minimization oracle
   $$
   \varphi^\star\in\arg\max_{\varphi\in\mathcal A}
   |\langle L^\ast c,\varphi\rangle|.
   $$
4. If the maximum is at most the dual feasibility threshold plus
   \(\epsilon_{\mathrm{LMO}}\), stop.
5. Add \(\varphi^\star\), refit all active coefficients, and optionally remove zero atoms.
6. Report the primal objective, dual bound, active-set size, oracle tolerance, and numerical residual separately.
::::

The outer master problem can be convex while the oracle is nonconvex. A small restricted-master gap is not a global certificate unless the oracle has found, or bounded, the most violating atom.

## One comparison table {#rkbs-comparison}

| Geometry | Finite object | Sparsity mechanism | Main assumptions | Main computational obstacle |
|---|---|---|---|---|
| RKHS | Primal section expansion | Usually none for squared norm | Bounded evaluation, fixed PSD kernel | Dense Gram solve |
| Smooth RKBS | Dual evaluation expansion | None implied | Reflexivity, convexity, smoothness, existence | Nonlinear inverse duality map |
| Sparse RKBS | Extreme kernel expansion | Exposed norming face and exact \(\ell_1\) geometry | Predual, weak-star compactness, A1, A2 | Data-dependent atom set |
| Variation space | Ridge atoms plus null space | TV measure extreme points | Stable right inverse, continuous surjective measurements, null-space identification | Continuous nonconvex atom oracle |

The statistical capacity measures also differ. RKHS analysis naturally uses eigenvalues and effective dimension. Atomic spaces use dual norms, metric entropy, variation bounds, or sparsity. A smaller active set lowers evaluation cost but does not by itself imply lower prediction risk.

## Common mistakes, failure boundaries, and practical implications {#rkbs-practice}

- Bounded evaluation alone does not supply the two-sided RKBS structure used by Zhang, Xu, and Zhang.
- A general RKBS kernel need not be symmetric or positive definite.
- A dual representer is not a linear primal expansion unless the inverse duality map is linear.
- Smoothness and exact sparsity pull in opposite geometric directions.
- Existence in a nonreflexive space usually needs weak-star compactness and measurements from a predual.
- A1 and A2 are sufficient sparse-kernel conditions, not universal properties of Banach spaces.
- An extreme sparse minimizer may coexist with nonsparse minimizers.
- A width bound is an existence result, not a guarantee for gradient descent.
- Null-space terms must be included and identified by the measurements.
- An approximate atom oracle weakens the global dual certificate.

## Summary and further reading {#rkbs-summary}

The RKBS foundation uses a dual pair of pointwise function spaces, and smooth semi-inner-product geometry puts the representer in the dual [@zhang2009rkbs]. Sparse RKBS theory instead studies weak-star compact solution sets and their extreme points; kernel sparsity follows only when data-dependent norming faces are atomic and the selected sections carry exact \(\ell_1\) geometry [@wang2024sparserkbs]. Radon-domain variation spaces turn total-variation measure minimizers into finite sums of truncated-power ridge atoms plus a polynomial null-space term, with width controlled by the number of measurements [@parhi2021banach]. These are related representer principles, but their assumptions, conclusions, and algorithms are not interchangeable.

## Exercises {#exercises}

1. [warm-up]{.ex-tag} For each of the three representer forms in Section [Three representer questions, not one](#rkbs-three-questions), identify whether the finite expansion is primal or dual, whether its atoms are fixed or learned, and whether sparsity follows.
2. [computation]{.ex-tag} Verify the \(\ell_4^3\) minimizer in Section [A finite-dimensional Banach calculation](#rkbs-worked-lp). Compute its normalized duality image and exhibit coefficients placing that image in the span of the two constraint normals.
3. [proof]{.ex-tag} Extend Theorem [smooth Banach representer principle](#thm-rkbs-dual-representer) to a convex nondifferentiable data term by using subgradients. State the condition needed on the radial norm penalty at a nonzero minimizer.
4. [computation]{.ex-tag} Derive the dual of the \(\ell_1^3\) interpolation problem in Section [Worked sparse certificate](#rkbs-worked-l1), verify the certificate \(c=(1/2,1/2)\), and use complementary slackness to recover the active coordinate.
5. [proof]{.ex-tag} Prove that an extreme solution of \(\min\{\lVert\alpha\rVert_1:A\alpha=y\}\) has at most \(\operatorname{rank}(A)\) nonzero coordinates. Explain why this finite-dimensional fact does not by itself prove the RKBS sparse representer theorem.
6. [proof]{.ex-tag} Prove that every function interpolating \((0,0),(1,1),(2,0)\) with distributional second derivative a finite measure satisfies \(\lVert f''\rVert_{\mathrm{TV}}\ge2\). Verify that \(s(x)=x-2(x-1)_+\) attains equality.
7. [synthesis]{.ex-tag} Compare \(\ell_2\), \(\ell_p\) for \(1\lt p\lt\infty\), and \(\ell_1\) regularization in terms of existence, uniqueness, duality-map smoothness, coefficient sparsity, and sensitivity to correlated atoms.
8. [challenge]{.ex-tag} Design a continuous-dictionary regression experiment comparing kernel ridge regression, a fixed-dictionary lasso, and variation-norm ridge splines. Specify the atom oracle, primal and dual certificates, matched capacity tuning, extrapolation tests, compute accounting, and a case where the oracle fails.
