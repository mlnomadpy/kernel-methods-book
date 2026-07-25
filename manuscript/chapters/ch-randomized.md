---
id: ch-randomized
slug: random-features-sketches-and-randomized-kernel-linear-algebra
title: 'Random Features, Sketches, and Randomized Kernel Linear Algebra'
part: III · Optimization and Scaling
order: 13
tier: core
prerequisites:
  - large-scale-kernels
objectives:
  - >-
    Distinguish matrix, regularized-spectral, optimization, and statistical
    approximation guarantees and identify the implications that are valid
    between them.
  - >-
    Derive Random Maclaurin and TensorSketch features for dot-product and
    polynomial kernels, including unbiasedness, variance, and computational
    cost.
  - >-
    Compare pointwise, uniform, spectral, and risk guarantees for random Fourier
    features under explicit assumptions.
  - >-
    Explain ridge-leverage sampling, structured and orthogonal features,
    quasi-Monte Carlo features, and data-dependent landmark selection.
  - >-
    Derive randomized kernel linear-algebra tools for sketched ridge regression,
    trace estimation, log determinants, and streaming updates.
  - >-
    Use lower bounds and failure diagnostics to choose an approximation by the
    downstream quantity it must preserve.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-randomized.yml
verification_date: null
bibliography:
  - rahimi2007
  - rudi2017
  - bach2017quadrature
  - le2013fastfood
  - yu2016orf
  - avron2016
  - alaoui2015
  - kar2012random
  - pham2013tensor
  - avron2017rff
  - li2021rff
  - musco2017nystrom
  - harbrecht2012cholesky
  - li2016dpp
  - yang2017sketch
  - hutchinson1989
  - ubaru2017slq
  - zhang2019incremental
  - song2021tensor
  - han2015logdet
  - wenger2022preconditioning
---
# Random Features, Sketches, and Randomized Kernel Linear Algebra

<p class="lead">A million-point kernel problem does not have one approximation problem. It has several. A feature map can approximate every kernel value and still distort the regularized directions that kernel ridge regression uses. A low-rank matrix can have a small Frobenius error and still be a poor preconditioner. An iterative solve can reach a tiny residual while the resulting predictor has the wrong risk. Randomization becomes reliable only after we name the object that must survive it. This chapter develops that discipline from end to end. We construct random features for dot-product and stationary kernels, compress polynomial tensors without forming them, select data-adaptive frequencies and landmarks, sketch ridge systems, estimate traces and log determinants, and maintain approximations in a stream. Each paper module states its setting, central move, guarantee, executable object, and failure boundary. The common language is a set of four error currencies that prevents matrix accuracy, numerical accuracy, and statistical accuracy from being mistaken for one another.</p>

## Four error currencies {#rand-four-currencies}

Let \(x_1,\ldots,x_n\) be fixed inputs, let \(K \in \mathbb R^{n\times n}\) be their positive semidefinite Gram matrix, and let \(\widetilde K \succeq 0\) be a randomized approximation. Write

$$
A_\gamma = K+\gamma I,
\qquad
\widetilde A_\gamma=\widetilde K+\gamma I,
\qquad
\gamma=n\lambda\gt0.
$$

The scaling convention \(\gamma=n\lambda\) matches the kernel ridge system \(A_\gamma\alpha=y\). Other chapters sometimes absorb \(n\) into \(K\) or \(\lambda\); every feature-count theorem must be converted to the convention actually used.

::: {.definition #def-rand-currencies}
[Definition (four approximation currencies)]{.box-title}

For a matrix approximation \(\widetilde K\), define:

1. **matrix error**
   $$
   E_{\mathrm{mat},2}=\lVert K-\widetilde K\rVert_2,
   \qquad
   E_{\mathrm{mat},F}=\lVert K-\widetilde K\rVert_F;
   $$
2. **regularized-spectral error**
   $$
   E_{\mathrm{reg}}(\gamma)
   =
   \left\lVert
   A_\gamma^{-1/2}(K-\widetilde K)A_\gamma^{-1/2}
   \right\rVert_2;
   $$
3. **optimization error**, such as
   $$
   E_{\mathrm{opt}}
   =
   \frac{\lVert\widehat\alpha-\widetilde\alpha\rVert_{A_\gamma}}
        {\lVert\widehat\alpha\rVert_{A_\gamma}},
   \quad
   \widehat\alpha=A_\gamma^{-1}y,
   \quad
   \widetilde\alpha=\widetilde A_\gamma^{-1}y;
   $$
4. **statistical error**, such as excess prediction risk
   $$
   E_{\mathrm{stat}}=R(\widetilde f)-R(f_\rho)
   \quad\text{or}\quad
   R(\widetilde f)-R(\widehat f_{\mathrm{KRR}}).
   $$

The norm \(\lVert v\rVert_{A_\gamma}=(v^\top A_\gamma v)^{1/2}\) measures coefficient error in the geometry of the regularized system.
:::

A fifth quantity, computational error, belongs beside these four: the residual of an iterative solve, roundoff, and stochastic estimation error. It is not interchangeable with approximation error. In particular, solving \(\widetilde A_\gamma\widetilde\alpha=y\) exactly says nothing by itself about how close \(\widetilde\alpha\) is to \(A_\gamma^{-1}y\).

::: {.proposition #prop-rand-reg-to-solve}
[Proposition (regularized spectral approximation controls the ridge solution)]{.box-title}

Assume \(K,\widetilde K\succeq0\), \(\gamma\gt0\), and

$$
\left\lVert
A_\gamma^{-1/2}(\widetilde K-K)A_\gamma^{-1/2}
\right\rVert_2
\leq \varepsilon\lt1.
$$

Then

$$
(1-\varepsilon)A_\gamma
\preceq
\widetilde A_\gamma
\preceq
(1+\varepsilon)A_\gamma
$$

and, for every \(y\),

$$
\frac{\lVert\widetilde\alpha-\widehat\alpha\rVert_{A_\gamma}}
     {\lVert\widehat\alpha\rVert_{A_\gamma}}
\leq
\frac{\varepsilon}{1-\varepsilon}.
$$

**Assumptions.** Fixed finite Gram matrices, common ridge \(\gamma\gt0\), symmetric PSD approximation, and \(\varepsilon\lt1\). **Proof status.** complete.
:::

::: {.proof}
[Proof]{.box-title}

Set

$$
E=A_\gamma^{-1/2}(\widetilde K-K)A_\gamma^{-1/2}.
$$

Then \(\widetilde A_\gamma=A_\gamma^{1/2}(I+E)A_\gamma^{1/2}\). The condition \(\lVert E\rVert_2\leq\varepsilon\) places every eigenvalue of \(I+E\) in \([1-\varepsilon,1+\varepsilon]\), which proves the Loewner inequalities. Moreover,

$$
A_\gamma^{1/2}(\widetilde\alpha-\widehat\alpha)
=
\big[(I+E)^{-1}-I\big]A_\gamma^{-1/2}y.
$$

The eigenvalues of \((I+E)^{-1}-I\) are \(-e_j/(1+e_j)\), whose magnitudes are at most \(\varepsilon/(1-\varepsilon)\). Taking Euclidean norms gives the claimed energy-norm bound because \(\lVert A_\gamma^{-1/2}y\rVert_2=\lVert\widehat\alpha\rVert_{A_\gamma}\). [\(\square\)]{.qed}
:::

This implication is deliberately one-way. A small solution error for one right-hand side does not imply a spectral approximation for every right-hand side. A small unregularized matrix error implies

$$
E_{\mathrm{reg}}(\gamma)
\leq
\frac{\lVert K-\widetilde K\rVert_2}{\gamma},
$$

but this may be useless when \(\gamma\) is small. Statistical risk needs additional assumptions about the data-generating distribution, loss, target regularity, and noise. Avron et al. make the regularized spectral condition the bridge from random Fourier features to KRR guarantees [@avron2017rff, Sections 2--4]; Yang, Pilanci, and Wainwright instead characterize when a sketch preserves the statistically relevant eigenspace [@yang2017sketch, Sections 2--3].

::: {.example #example-rand-currencies}
[Example (the four currencies disagree)]{.box-title}

Take

$$
K=\operatorname{diag}(9,1,0.04),
\qquad
\widetilde K=\operatorname{diag}(9,0,0),
\qquad
\gamma=1.
$$

The rank-one approximation has \(E_{\mathrm{mat},2}=1\). Its regularized error is only

$$
E_{\mathrm{reg}}(1)
=
\max\left\{\frac{1}{1+1},\frac{0.04}{0.04+1}\right\}
=0.5.
$$

For \(y=(0,1,1)^\top\),

$$
\widehat\alpha=(0,0.5,0.9615)^\top,
\qquad
\widetilde\alpha=(0,1,1)^\top.
$$

The relative \(A_\gamma\)-norm error is approximately \(0.5858\), below the proposition's conservative bound \(0.5/(1-0.5)=1\). The example also shows why rank alone is not a statistical verdict. Discarding the eigenvalue \(0.04\) is harmless when ridge \(1\) already suppresses it, but discarding the eigenvalue \(1\) changes a direction the estimator still uses. No statement about population risk follows until the response and sampling model are specified.

**Verification artifact.** checks/example-ch-randomized-example-rand-currencies.json records the example source hash and verification scope.
:::

## Random Maclaurin features {#rand-maclaurin}

Random Fourier features begin with translation invariance. Random Maclaurin features begin with a different structure: a dot-product kernel

$$
k(x,y)=f(\langle x,y\rangle),
\qquad
f(t)=\sum_{p=0}^{\infty}a_p t^p,
\qquad
a_p\geq0.
$$

On a ball \(\lVert x\rVert,\lVert y\rVert\leq R\), assume the series converges absolutely for \(|t|\leq R^2\). The nonnegative coefficients certify positive definiteness because each \(\langle x,y\rangle^p\) is the inner product of \(x^{\otimes p}\) and \(y^{\otimes p}\). The obstacle is that \(x^{\otimes p}\) has \(d^p\) coordinates.

Kar and Karnick's move is to sample both a degree and a rank-one tensor probe [@kar2012random, Sections 3--4]. Choose a probability mass function \(\rho_p\gt0\) whenever \(a_p\gt0\). Draw \(P\sim\rho\), then draw independent Rademacher vectors \(\omega_1,\ldots,\omega_P\in\{-1,+1\}^d\), and define

$$
\phi(x)
=
\sqrt{\frac{a_P}{\rho_P}}
\prod_{r=1}^{P}\langle\omega_r,x\rangle.
$$

For \(P=0\), the empty product is \(1\). An \(m\)-dimensional map uses independent copies:

$$
z(x)=\frac{1}{\sqrt m}\big(\phi_1(x),\ldots,\phi_m(x)\big)^\top.
$$

::: {.theorem #thm-rand-maclaurin}
[Theorem (unbiased Random Maclaurin features and their second moment)]{.box-title}

Under the convergence assumptions above,

$$
\mathbb E\,\langle z(x),z(y)\rangle=k(x,y).
$$

For one feature, let

$$
M(x,y)
=
\mathbb E_\omega
\left[\langle\omega,x\rangle^2\langle\omega,y\rangle^2\right]
=
\lVert x\rVert^2\lVert y\rVert^2
+2\langle x,y\rangle^2
-2\sum_{j=1}^{d}x_j^2y_j^2.
$$

If \(\sum_p a_p^2M(x,y)^p/\rho_p\lt\infty\), then

$$
\operatorname{Var}\!\left[\langle z(x),z(y)\rangle\right]
=
\frac{1}{m}
\left[
\sum_{p=0}^{\infty}\frac{a_p^2}{\rho_p}M(x,y)^p
-k(x,y)^2
\right].
$$

**Assumptions.** Independent Rademacher probes, independent features, absolute convergence, and finite displayed second moment. **Proof status.** complete.
:::

::: {.proof}
[Proof]{.box-title}

For one Rademacher vector,

$$
\mathbb E\big[\langle\omega,x\rangle\langle\omega,y\rangle\big]
=
\sum_{j,\ell}x_jy_\ell\,\mathbb E[\omega_j\omega_\ell]
=
\langle x,y\rangle.
$$

Conditioning on \(P=p\) and using independence across the \(p\) probes gives

$$
\mathbb E[\phi(x)\phi(y)\mid P=p]
=
\frac{a_p}{\rho_p}\langle x,y\rangle^p.
$$

Averaging over \(P\) yields \(\sum_pa_p\langle x,y\rangle^p=k(x,y)\). For the second moment, the same conditioning gives

$$
\mathbb E[\phi(x)^2\phi(y)^2\mid P=p]
=
\frac{a_p^2}{\rho_p^2}M(x,y)^p.
$$

Multiplying by \(\rho_p\), summing over \(p\), and subtracting the squared mean gives the one-feature variance. Averaging \(m\) independent copies divides it by \(m\). To obtain the displayed formula for \(M\), expand four sums and retain index patterns in which every Rademacher variable appears an even number of times. [\(\square\)]{.qed}
:::

The theorem exposes the design problem hidden by the word "random." The degree law \(\rho\) changes both runtime and variance. Large degrees are expensive, but making \(\rho_p\) too small multiplies the second moment by \(1/\rho_p\). For a fixed pair \((x,y)\), the second-moment term is minimized by \(\rho_p\propto a_pM(x,y)^{p/2}\), but that distribution depends on the pair one is trying to approximate. A practical law must dominate the worst relevant input norms or be tuned on a training distribution.

```text
Algorithm: Random Maclaurin map
input: x in R^d, feature count m, coefficients a_p, degree law rho
for j = 1,...,m:
    sample P_j from rho
    value = sqrt(a_{P_j} / rho_{P_j})
    for r = 1,...,P_j:
        sample a Rademacher vector omega_{jr}
        value = value * dot(omega_{jr}, x)
    z_j(x) = value / sqrt(m)
return z(x)
```

The expected evaluation cost is \(O(md\,\mathbb E P)\) for dense inputs. Hash-generated signs avoid storing all probes. The resulting map is unbiased, but its coordinates can be heavy-tailed even on a bounded input ball because they multiply several random projections.

::: {.example #example-rand-maclaurin}
[Example (unbiased can still mean unusably noisy)]{.box-title}

For the degree-two kernel \(k(x,y)=\langle x,y\rangle^2\), take \(x=(1,2)\) and \(y=(3,-1)\). The exact kernel value is \(1\). With deterministic degree \(P=2\), \(\omega_1=(1,1)\), and \(\omega_2=(1,-1)\), one sampled product is

$$
\phi(x)\phi(y)
=
(3)(-1)(2)(4)
=-24.
$$

Unbiasedness is an expectation, not a positivity guarantee for one sampled inner product. Here

$$
M(x,y)=5\cdot10+2\cdot1^2-2(1^2 3^2+2^2(-1)^2)=26,
$$

so a single feature has second moment \(26^2=676\) and variance \(675\). Averaging \(m\) copies reduces the variance to \(675/m\), but the example explains why Random Maclaurin can require many coordinates on high-norm data. Input normalization is part of the method, not cosmetic preprocessing.

**Verification artifact.** checks/example-ch-randomized-example-rand-maclaurin.json records the example source hash and verification scope.
:::

**Failure boundary.** Negative Taylor coefficients invalidate this construction as a PSD mixture. A divergent series on the input range invalidates the interchange of expectation and summation. Even with nonnegative coefficients, large input norms or degree tails can make the variance enormous. TensorSketch attacks the \(dP\) multiplication cost and tensor dimension, but it does not erase the degree dependence.

## TensorSketch and CountSketch polynomial features {#rand-tensorsketch}

For the homogeneous polynomial kernel \(k_p(x,y)=\langle x,y\rangle^p\), the exact feature vector is \(u(x)=x^{\otimes p}\in\mathbb R^{d^p}\). CountSketch compresses a vector \(u\in\mathbb R^D\) into \(m\) buckets. Choose a hash \(h:[D]\to\{0,\ldots,m-1\}\) and signs \(s:[D]\to\{-1,+1\}\), then set

$$
[C(u)]_b=\sum_{j:h(j)=b}s(j)u_j.
$$

TensorSketch uses independent coordinate hashes \(h_r:[d]\to\{0,\ldots,m-1\}\) and signs \(s_r:[d]\to\{-1,+1\}\), then defines on a tensor index \(i=(i_1,\ldots,i_p)\)

$$
h(i)=\left(\sum_{r=1}^{p}h_r(i_r)\right)\bmod m,
\qquad
s(i)=\prod_{r=1}^{p}s_r(i_r).
$$

The convolution theorem is the computational breakthrough:

$$
\operatorname{TS}_p(x)
=
\operatorname{FFT}^{-1}
\left(
\prod_{r=1}^{p}
\operatorname{FFT}(C_r(x))
\right),
$$

where \(C_r(x)\) is the degree-one CountSketch using \(h_r,s_r\). This computes the sketch of \(x^{\otimes p}\) without materializing \(d^p\) entries, in \(O(p(d+m\log m))\) time and \(O(pm)\) working memory [@pham2013tensor, Section 3].

::: {.theorem #thm-rand-tensorsketch}
[Theorem (TensorSketch unbiasedness)]{.box-title}

Assume the sign functions are independent across tensor factors and sufficiently independent within each factor so that every nontrivial sign product in the proof has zero expectation. Then, for fixed \(x,y\in\mathbb R^d\),

$$
\mathbb E
\left[
\left\langle
\operatorname{TS}_p(x),\operatorname{TS}_p(y)
\right\rangle
\right]
=
\langle x,y\rangle^p.
$$

**Assumptions.** Fixed degree \(p\), common random sketch for \(x\) and \(y\), uniform hashes, independent symmetric signs, and exact arithmetic for the identity. **Proof status.** complete for unbiasedness. Stronger variance and subspace-embedding bounds are cited rather than silently inferred.
:::

::: {.proof}
[Proof]{.box-title}

Write \(u=x^{\otimes p}\) and \(v=y^{\otimes p}\), indexed by tensor multi-indices. Expanding the sketched inner product gives

$$
\langle C(u),C(v)\rangle
=
\sum_{i,j}
\mathbf 1\{h(i)=h(j)\}s(i)s(j)u_i v_j.
$$

When \(i=j\), the sign product is \(1\), so the diagonal contribution is \(\sum_i u_iv_i=\langle u,v\rangle\). When \(i\neq j\), at least one tensor coordinate differs. Independence and symmetry of the factor signs make \(\mathbb E[s(i)s(j)]=0\), so every off-diagonal collision vanishes in expectation. Therefore

$$
\mathbb E\langle C(u),C(v)\rangle
=
\langle u,v\rangle
=
\langle x,y\rangle^p.
$$

The FFT construction computes exactly this same bucketed tensor sketch because circular convolution adds the component hashes modulo \(m\) and multiplies their signs. [\(\square\)]{.qed}
:::

The hard part of TensorSketch theory is not unbiasedness. It is controlling correlated collisions well enough to preserve an entire polynomial-feature subspace. The original paper derives degree-dependent variance and uses repeated sketches to obtain concentration [@pham2013tensor, Sections 3--4]. Later work improves the dependence of the sketch dimension and runtime for high polynomial degree [@song2021tensor, Theorems 1--3]. Those results should not be replaced by the ordinary CountSketch variance formula: the tensor hash has extra dependencies.

```text
Algorithm: TensorSketch for degree p
input: x in R^d, output dimension m
for r = 1,...,p:
    initialize c_r = zeros(m)
    for coordinate j = 1,...,d:
        c_r[h_r(j)] += s_r(j) * x_j
    f_r = FFT(c_r)
return real(IFFT(product_r f_r))
```

For the inhomogeneous polynomial kernel \((c+\langle x,y\rangle)^p\), augment \(x\) by a constant coordinate \(\sqrt c\) and sketch the homogeneous tensor of the augmented vector. For an analytic dot-product kernel, combine degree sampling from Random Maclaurin with a TensorSketch at the sampled degree.

**Failure boundary.** TensorSketch is attractive when \(p\) is modest and \(x\) is sparse. Increasing \(p\) increases collision structure, numerical dynamic range, and the dimension required by subspace guarantees. The FFT formula uses circular convolution, so inconsistent hash conventions produce a silently wrong map. A fresh sketch at test time is also wrong: every point must use the same sampled hashes and signs.

## Random Fourier features: four different guarantees {#rand-rff-guarantees}

Let \(k(x,y)=\kappa(x-y)\) be a continuous shift-invariant kernel on \(\mathbb R^d\), normalized by \(k(0)=1\). Bochner's theorem gives a probability measure \(p\) such that

$$
\kappa(x-y)
=
\mathbb E_{\omega\sim p}
\cos\bigl(\omega^\top(x-y)\bigr).
$$

Using paired sine and cosine coordinates, define

$$
z(x)
=
\frac{1}{\sqrt m}
\bigl(
\cos(\omega_1^\top x),\sin(\omega_1^\top x),
\ldots,
\cos(\omega_m^\top x),\sin(\omega_m^\top x)
\bigr)^\top.
$$

Then \(z(x)^\top z(y)\) is the average of \(m\) bounded kernel samples. Rahimi and Recht introduced this explicit map and its pointwise and uniform analyses [@rahimi2007].

::: {.theorem #thm-rand-rff-pointwise}
[Theorem (pointwise random Fourier concentration)]{.box-title}

For fixed \(x,y\in\mathbb R^d\) and i.i.d. \(\omega_j\sim p\),

$$
\mathbb P\left(
\left|z(x)^\top z(y)-k(x,y)\right|\geq\varepsilon
\right)
\leq
2\exp\left(-\frac{m\varepsilon^2}{2}\right).
$$

Consequently \(m\geq2\varepsilon^{-2}\log(2/\delta)\) suffices for pointwise error at most \(\varepsilon\) with probability at least \(1-\delta\).

**Assumptions.** Fixed pair independent of the sampled frequencies, normalized stationary kernel, and i.i.d. frequencies. **Proof status.** complete.
:::

::: {.proof}
[Proof]{.box-title}

For \(\Delta=x-y\), each summand

$$
X_j=\cos(\omega_j^\top\Delta)
$$

lies in \([-1,1]\) and has mean \(\kappa(\Delta)\). The feature inner product is \(m^{-1}\sum_jX_j\). Hoeffding's inequality for independent variables with range length \(2\) gives

$$
\mathbb P\left(
\left|\frac1m\sum_jX_j-\mathbb EX_j\right|\geq\varepsilon
\right)
\leq
2\exp\left(-\frac{2m^2\varepsilon^2}{4m}\right),
$$

which is the stated bound. [\(\square\)]{.qed}
:::

The theorem says nothing uniform over a region and nothing about a learned predictor. The literature supplies four distinct levels.

### Pointwise and uniform function approximation {#rand-rff-pointwise-uniform}

Pointwise concentration fixes \((x,y)\) before features are drawn. Uniform approximation asks for

$$
\sup_{x,y\in\mathcal X}
\left|z(x)^\top z(y)-k(x,y)\right|
\leq\varepsilon.
$$

::: {.theorem #thm-rand-rff-uniform}
[Theorem (a compact-domain RFF bound)]{.box-title}

Let \(\mathcal X\subset\mathbb R^d\) be compact, let \(k(x,y)=\kappa(x-y)\) be normalized as above, and assume the spectral measure has finite second moment

$$
\sigma_p^2=\mathbb E_{\omega\sim p}\lVert\omega\rVert^2\lt\infty.
$$

For the \(m\)-frequency map and every \(\varepsilon\gt0\),

$$
\mathbb P\left[
\sup_{x,y\in\mathcal X}
\left|z(x)^\top z(y)-k(x,y)\right|
\geq\varepsilon
\right]
\leq
2^8
\left(
\frac{\sigma_p\operatorname{diam}(\mathcal X)}{\varepsilon}
\right)^2
\exp\left(
-\frac{m\varepsilon^2}{4(d+2)}
\right).
$$

**Assumptions.** Compact Euclidean domain, normalized real stationary kernel, i.i.d. spectral draws, finite spectral second moment, and the paired sine-cosine normalization used above. **Proof status.** reconstructed proof sketch; the stated bound is from [@rahimi2007, Claim 1].
:::

::: {.proof}
[Proof sketch]{.box-title}

Cover the difference set \(\mathcal X-\mathcal X\) by an \(r\)-net. Apply the pointwise Hoeffding bound at every net point and union-bound over the covering number. The random approximation is differentiable in the difference variable, and its expected squared Lipschitz constant is controlled by \(\sigma_p^2\). Markov's inequality controls the event that this Lipschitz constant is too large. On the complementary event, accuracy on the net extends to the full difference set. Choosing \(r\) to balance the net cardinality with the between-net variation yields the displayed constants and exponent. [\(\square\)]{.qed}
:::

The dimension, diameter, and spectral moment are the price of the supremum. If \(p\) has tails too heavy for the required Lipschitz control, the displayed compact-domain bound does not apply. Uniform kernel approximation is stronger than pointwise approximation, but still does not identify which directions of the sample Gram matrix matter after regularization.

### Spectral approximation {#rand-rff-spectral}

For sample points \(x_1,\ldots,x_n\), let \(\varphi_\omega\in\mathbb C^n\) have entries \(e^{i\omega^\top x_j}\). Then

$$
K=\mathbb E_{\omega\sim p}
\left[\varphi_\omega\varphi_\omega^*\right].
$$

The random-feature Gram matrix is an empirical average of these rank-one PSD matrices. A regularized spectral guarantee asks for

$$
(1-\varepsilon)(K+\gamma I)
\preceq
ZZ^*+\gamma I
\preceq
(1+\varepsilon)(K+\gamma I).
$$

This is the condition needed by the [regularized spectral approximation proposition](#prop-rand-reg-to-solve). Avron et al. derive feature budgets for it, prove a Gaussian-kernel lower bound, and show how a leverage-weighted frequency distribution improves the budget [@avron2017rff, Theorems 7--10].

### Optimization and risk {#rand-rff-risk}

Once the regularized system is controlled, one can bound the discrepancy between exact and approximate ridge solutions. Population risk requires more. For square loss, a typical analysis decomposes

$$
R(\widetilde f)-R(f_\rho)
=
\underbrace{R(f_\lambda)-R(f_\rho)}_{\text{regularization bias}}
+
\underbrace{R(\widehat f_\lambda)-R(f_\lambda)}_{\text{sampling error}}
+
\underbrace{R(\widetilde f_\lambda)-R(\widehat f_\lambda)}_{\text{feature error}}.
$$

The first term needs a source or approximation condition, the second needs a sampling and noise model, and the third needs a feature budget tied to \(\lambda\) and capacity. Rudi and Rosasco prove that a feature count much smaller than \(n\) can retain optimal learning rates under explicit capacity and source assumptions [@rudi2017]. Li et al. unify square-loss and Lipschitz-loss analyses and express budgets through effective degrees of freedom [@li2021rff, Theorems 9--12]. Neither result says that a small pointwise kernel error automatically yields a small excess risk.

The four levels can be remembered as a quantifier ladder:

| Guarantee | Quantifier | Main currency | What it can support |
|---|---|---|---|
| Pointwise | one fixed pair | scalar error | kernel evaluation |
| Uniform | every pair in a domain | sup-norm error | geometry on that domain |
| Spectral | every vector on one sample | regularized Loewner order | solves and preconditioning |
| Risk | new random observations | population loss | statistical prediction |

Each row adds assumptions. None can be obtained by renaming the row above it.

## Ridge-leverage and data-adaptive features {#rand-adaptive-features}

Plain RFF samples frequencies according to the kernel's spectral density \(p\). It treats every Fourier component as equally important before seeing the data. Regularized ridge leverage asks a sharper question: how much can one sampled feature influence the directions that survive \(K+\gamma I\)?

Define the feature leverage function

$$
\tau_\gamma(\omega)
=
p(\omega)\,
\varphi_\omega^*(K+\gamma I)^{-1}\varphi_\omega,
$$

and its integral

$$
\int\tau_\gamma(\omega)\,d\omega
=
\operatorname{tr}\bigl(K(K+\gamma I)^{-1}\bigr)
=
d_{\mathrm{eff}}(\gamma).
$$

The normalized density

$$
q_\gamma(\omega)
=
\frac{\tau_\gamma(\omega)}{d_{\mathrm{eff}}(\gamma)}
$$

oversamples frequencies that are important in the regularized sample geometry. To retain unbiasedness, a sample from \(q_\gamma\) is reweighted by \(\sqrt{p(\omega)/q_\gamma(\omega)}\).

::: {.theorem #thm-rand-leverage-features}
[Theorem (ideal leverage sampling yields effective-dimension spectral complexity)]{.box-title}

Let \(K=\int p(\omega)\varphi_\omega\varphi_\omega^*\,d\omega\), let \(\gamma\gt0\), assume \(d_{\mathrm{eff}}(\gamma)\gt0\), and fix \(\varepsilon,\delta\in(0,1)\). Draw \(\omega_1,\ldots,\omega_m\) independently from \(q_\gamma\), and define

$$
\widetilde K
=
\frac1m\sum_{j=1}^{m}
\frac{p(\omega_j)}{q_\gamma(\omega_j)}
\varphi_{\omega_j}\varphi_{\omega_j}^*.
$$

There is a universal constant \(C\) such that

$$
m
\geq
C\,
\frac{d_{\mathrm{eff}}(\gamma)}{\varepsilon^2}
\log\left(\frac{2\max\{1,d_{\mathrm{eff}}(\gamma)\}}{\delta}\right)
$$

suffices for

$$
\left\lVert
(K+\gamma I)^{-1/2}(\widetilde K-K)(K+\gamma I)^{-1/2}
\right\rVert_2
\leq\varepsilon
$$

with probability at least \(1-\delta\).

**Assumptions.** Exact access to \(q_\gamma\), independent samples, finite \(d_{\mathrm{eff}}(\gamma)\), and the finite-sample feature decomposition above. **Proof status.** proof sketch; constants depend on the chosen matrix concentration form.
:::

::: {.proof}
[Proof sketch]{.box-title}

Whiten one reweighted feature:

$$
X(\omega)
=
(K+\gamma I)^{-1/2}
\frac{p(\omega)}{q_\gamma(\omega)}
\varphi_\omega\varphi_\omega^*
(K+\gamma I)^{-1/2}.
$$

It is rank one and PSD. Under \(q_\gamma=\tau_\gamma/d_{\mathrm{eff}}\),

$$
\lVert X(\omega)\rVert_2
=
\frac{p(\omega)}{q_\gamma(\omega)}
\varphi_\omega^*(K+\gamma I)^{-1}\varphi_\omega
=
d_{\mathrm{eff}}(\gamma).
$$

Its expectation is

$$
\mathbb EX
=(K+\gamma I)^{-1/2}K(K+\gamma I)^{-1/2},
$$

whose norm is at most \(1\) and trace is \(d_{\mathrm{eff}}(\gamma)\). A matrix Bernstein or Chernoff inequality applied to \(m^{-1}\sum_jX(\omega_j)\) gives the stated effective-dimension scaling, up to universal constants and logarithmic conventions. [\(\square\)]{.qed}
:::

The theorem identifies both the contribution and the obstacle. Ideal leverage sampling replaces a worst-case feature bound by effective dimension, but computing \(\tau_\gamma\) appears to require the inverse one hoped to avoid. Practical methods therefore estimate leverage from a pilot sketch, use an analytically tractable upper envelope, or learn frequencies jointly with a downstream objective. Bach connects optimized feature sampling to quadrature and proves upper and lower bounds governed by integral-operator eigenvalues [@bach2017quadrature, Sections 3--4]. Avron et al. construct a Gaussian-kernel proposal that upper-bounds the leverage function in a restricted regime [@avron2017rff, Section 4]. Li et al. give an approximate empirical leverage procedure and analyze its risk [@li2021rff, Sections 5--6].

```text
Algorithm: two-stage approximate leverage RFF
input: training inputs X, ridge gamma, pilot size m0, final size m
draw m0 ordinary Fourier features and form pilot matrix Z0
use Z0 to approximate diagonal feature leverage or an upper envelope
construct a normalized proposal q_hat
draw m frequencies from q_hat
reweight each feature by sqrt(p(omega) / q_hat(omega))
fit the downstream regularized linear model
report pilot cost, final cost, and the ridge level used by q_hat
```

**Failure boundary.** The sampling law is tied to the inputs and ridge level. Reusing it after a large distribution shift or changing \(\lambda\) can remove its advantage. Approximate leverage scores must be upper bounds, or accurate in the direction required by the concentration proof; an arbitrary heuristic frequency score does not inherit the theorem. Labels can improve task adaptation, but then the map is no longer a label-free kernel approximation and must be evaluated for selection bias.

## Structured, orthogonal, and quasi-Monte Carlo features {#rand-structured-features}

The feature count \(m\) is not the whole cost. Dense Gaussian frequencies require \(O(md)\) storage and work per example. Three families change the sampling mechanism while trying to retain the same kernel integral.

### Fastfood {#rand-fastfood}

Fastfood replaces a dense Gaussian matrix by products of diagonal random matrices, Hadamard transforms, and permutations. For dimensions padded to a power of two, a block has the schematic form

$$
W
\approx
\frac{1}{\sigma\sqrt d}
S H G \Pi H B,
$$

where \(B,G,S\) are diagonal, \(\Pi\) is a permutation, and \(H\) is the Walsh-Hadamard transform. The map costs \(O(d\log d)\) per block and \(O(d)\) storage rather than \(O(d^2)\) [@le2013fastfood, Section 3]. The contribution is computational structure, not a claim that the rows are independent Gaussian samples. Dependence between rows changes the variance proof.

### Orthogonal random features {#rand-orf}

Orthogonal random features draw frequency directions orthogonally and then apply radial scaling so that each row has the desired Gaussian marginal. The marginals preserve unbiasedness for the Gaussian kernel, while negative dependence reduces variance in important regimes [@yu2016orf, Theorems 1--2]. Orthogonality is available only in blocks of at most \(d\) directions; for \(m\gt d\), multiple independent blocks are required. The gain can shrink when the spectrum is anisotropic or when radial rescaling dominates angular redundancy.

### Quasi-Monte Carlo features {#rand-qmc}

Quasi-Monte Carlo replaces i.i.d. uniform points in the inverse-CDF representation of the spectral integral by a low-discrepancy sequence. For a sufficiently regular integrand on a fixed-dimensional cube, discrepancy bounds can improve integration error beyond the root-\(m\) Monte Carlo rate. Avron et al. adapt this idea to shift-invariant kernels and analyze the dependence on the transformed integrand [@avron2016, Sections 3--5].

The failure boundary is dimension and smoothness. Inverse Gaussian transforms create boundary singularities, effective dimension may be high, and a deterministic low-discrepancy sequence does not supply ordinary i.i.d. confidence intervals. Randomized QMC restores replication-based error assessment, but only if independent scramblings are used.

| Method | Kernel sample | Build/apply cost | Main gain | Main failure boundary |
|---|---|---:|---|---|
| i.i.d. RFF | independent spectral draws | \(O(md)\) | simple concentration | Monte Carlo variance |
| Fastfood | structured dependent block | \(O(m\log d)\) near \(m\asymp d\) | memory and transform speed | padding, dependence, hardware constants |
| ORF | orthogonal directions in blocks | \(O(md)\), faster structured variants exist | variance reduction | block size \(d\), anisotropy |
| QMC | low-discrepancy spectral nodes | \(O(md)\) | integration accuracy for smooth low-dimensional problems | dimension, transform singularities, no i.i.d. error bar |

These methods must be compared at equal wall time and memory, not only at equal \(m\). A structured transform can lose to a dense matrix multiplication on a GPU at moderate dimension because the latter is better optimized.

## Pivoted Cholesky, greedy Nyström, and DPP landmarks {#rand-column-selection}

Random features sample basis functions before or around the data. Column methods choose basis functions \(k(x_i,\cdot)\) from the observed sample. For an index set \(S\), the Nyström approximation is

$$
\widetilde K_S
=
K_{:S}K_{SS}^{\dagger}K_{S:}.
$$

When \(K_{SS}\) is nonsingular, this is the Gram matrix of projections onto the span of the selected kernel sections. The residual

$$
R_S=K-\widetilde K_S
$$

is PSD. Its diagonal is not merely an error heuristic.

Landmark selection should be judged in the currency required downstream. The comparison below uses the same ranks for uniform and regularized leverage-weighted landmark designs, then measures both relative Gram error and the error of the resulting ridge predictor. The two panels need not rank methods by identical margins, which is precisely why matrix and predictor accuracy must be reported separately.

<figure class="viz" data-figure="nystrom-sampling-comparison" data-alt="Relative Nyström Gram-matrix and kernel-ridge predictor errors versus landmark count for uniform and ridge-leverage-weighted landmark selection."><figcaption>Nyström quality has more than one currency. Ridge-leverage-weighted landmarks reduce both errors in this nonuniform design, but matrix error and predictor error remain distinct quantities and should never be substituted for one another.</figcaption></figure>

::: {.proposition #prop-rand-pchol-power}
[Proposition (pivoted Cholesky residuals are sample power functions)]{.box-title}

Let \(K\) be a Gram matrix and let \(S\) index a nonsingular principal submatrix. For every sample index \(i\),

$$
[R_S]_{ii}
=
K_{ii}-K_{iS}K_{SS}^{-1}K_{Si}
=
P_S(x_i)^2,
$$

where \(P_S\) is the kernel interpolation power function for the selected sites. Moreover \(R_S\succeq0\). Greedy diagonal pivoting therefore selects the point of maximum current sample power.

**Assumptions.** PSD Gram matrix and invertible selected principal block; use a pseudoinverse and quotient-space interpretation otherwise. **Proof status.** complete.
:::

::: {.proof}
[Proof]{.box-title}

Reorder indices so \(S\) comes first and write

$$
K=
\begin{pmatrix}
K_{SS} & K_{S\bar S}\\
K_{\bar S S} & K_{\bar S\bar S}
\end{pmatrix}.
$$

The Schur complement

$$
K_{\bar S\bar S}
-K_{\bar S S}K_{SS}^{-1}K_{S\bar S}
$$

is PSD because \(K\succeq0\) and \(K_{SS}\succ0\). This is the nonzero block of \(R_S\), proving \(R_S\succeq0\). Its \(i\)-th diagonal is the displayed expression. The power-function identity from [[ch:kernel-interpolation-and-approximation]] gives exactly the same expression as the squared norm of the residual representer \(k(x_i,\cdot)-\Pi_Sk(x_i,\cdot)\). [\(\square\)]{.qed}
:::

Pivoted Cholesky repeatedly chooses \(i=\arg\max_j[R_S]_{jj}\), appends a normalized residual column to the factor \(L\), and downdates the diagonal. It uses \(O(nm)\) kernel entries and \(O(nm^2)\) arithmetic for \(m\) pivots in a straightforward implementation. Harbrecht, Peters, and Schneider control trace-norm error and prove exponential convergence under sufficiently fast eigenvalue decay [@harbrecht2012cholesky].

```text
Algorithm: greedy pivoted Cholesky
input: kernel diagonal d, target rank m, kernel column oracle
L = zeros(n,m)
for r = 1,...,m:
    pivot i = argmax_j d_j
    stop if d_i is below tolerance
    residual column c = K[:,i] - L[:,:r-1] L[i,:r-1]^T
    L[:,r] = c / sqrt(d_i)
    d = d - L[:,r]^2
    clip only roundoff-sized negative entries and record the clipping
return L, with K approximately L L^T
```

The residual trace measures total unexplained kernel variance, while the largest residual diagonal is the worst remaining sample power function. The greedy sequence below drives both downward and reveals where the algorithm spends its pivots: it first covers separated regions, then fills the largest remaining geometric gaps.

<figure class="viz" data-figure="pivoted-cholesky-residual" data-alt="Relative trace residual and maximum residual diagonal decrease over twenty pivoted-Cholesky steps, alongside the spatial order in which input sites are selected."><figcaption>Pivoted Cholesky is greedy interpolation in matrix form. Each pivot removes the largest residual diagonal, so the maximum sample uncertainty and the total trace residual decrease together as selected sites spread across the domain.</figcaption></figure>

Greedy selection targets the maximum diagonal residual. Ridge-leverage Nyström targets regularized spectral error and can be implemented recursively without first computing exact leverage scores [@musco2017nystrom, Theorem 3 and Algorithm 2]. A fixed-size determinantal point process selects \(S\) with probability proportional to \(\det K_{SS}\), rewarding volume and therefore diversity. Li, Jegelka, and Sra derive Nyström and KRR error consequences for DPP landmarks and analyze a faster sampler under stated mixing conditions [@li2016dpp, Sections 3--4].

::: {.example #example-rand-pchol}
[Example (one pivot exposes what remains)]{.box-title}

Consider

$$
K=
\begin{pmatrix}
1&0.8&0.2\\
0.8&1&0.3\\
0.2&0.3&1
\end{pmatrix}.
$$

Break the initial diagonal tie by choosing point \(1\). The rank-one Cholesky/Nyström approximation uses the first column, and the residual is

$$
R_{\{1\}}
=
\begin{pmatrix}
0&0&0\\
0&0.36&0.14\\
0&0.14&0.96
\end{pmatrix}.
$$

The remaining squared power values are \(0.36\) and \(0.96\), so greedy pivoting chooses point \(3\) next. Uniform sampling would treat points \(2\) and \(3\) alike; the residual geometry does not. A DPP also disfavors selecting points \(1\) and \(2\) together because their \(2\times2\) determinant is \(1-0.8^2=0.36\), smaller than the determinant \(1-0.2^2=0.96\) for points \(1\) and \(3\).

**Verification artifact.** checks/example-ch-randomized-example-rand-pchol.json records the example source hash and verification scope.
:::

**Failure boundary.** Greedy maximum-diagonal selection optimizes a local residual criterion, not downstream risk. DPP sampling promotes diversity but can be expensive to sample exactly, and diversity is not the same as label relevance. Ridge leverage depends on \(\gamma\); a landmark set chosen for one ridge can be inefficient for another. Near-duplicate points can make \(K_{SS}\) singular, so stable implementations use pivot tolerances and triangular solves rather than an explicit inverse.

## Oblivious sketches and sketched KRR {#rand-oblivious-krr}

Column sampling is data-aware. An oblivious sketch chooses a random matrix \(S\in\mathbb R^{m\times n}\) independently of the data, using Gaussian entries, a subsampled randomized Hadamard transform, or a sparse hashing transform. A generalized Nyström approximation is

$$
\widetilde K
=
KS^\top(SK S^\top)^\dagger SK.
$$

The central paper question is not whether \(\widetilde K\) is close in Frobenius norm. It is how small \(m\) can be while the sketched KRR estimator retains the minimax prediction rate.

Let \(K/n=U\operatorname{diag}(\widehat\mu_1,\ldots,\widehat\mu_n)U^\top\). For a target regularization scale \(\lambda\), split

$$
U=[U_1\ U_2],
\qquad
\widehat\mu_j\gt\lambda \text{ on } U_1,
\qquad
\widehat\mu_j\leq\lambda \text{ on } U_2.
$$

A useful sketch must embed the high-eigenvalue space \(U_1\) and keep the sketched low-eigenvalue tail controlled. In one common formulation, there are constants \(c_1,c_2\) such that

$$
\lVert (SU_1)^\top SU_1-I\rVert_2\leq c_1
\quad\text{and}\quad
\lVert SU_2\operatorname{diag}(\widehat\mu_{2})^{1/2}\rVert_2
\leq c_2\sqrt\lambda.
$$

Yang, Pilanci, and Wainwright call the corresponding condition \(K\)-satisfiability and prove that Gaussian and randomized Hadamard sketches with dimension proportional to statistical dimension, up to logarithmic factors for the latter, preserve the minimax KRR rate under fixed-design sub-Gaussian regression assumptions [@yang2017sketch, Theorems 1--2]. Their lower bound is statistical: sketches below the critical dimension cannot generally retain the rate.

The paper's contribution is a two-scale argument. The sketch must be nearly isometric where the empirical kernel has eigenvalues above the critical radius; below that radius, contraction is acceptable because regularization and noise already dominate those directions. This is why an unregularized subspace embedding of the entire rank can be excessive.

```text
Algorithm: sketched KRR through generalized Nystrom
input: Gram operator K, response y, ridge gamma, sketch S in R^{m x n}
form B = K S^T and W = S B
factor W with a rank-revealing Cholesky or eigendecomposition
construct a factor Z such that Z Z^T = B W^dagger B^T
solve (Z Z^T + gamma I) alpha_tilde = y
    using Woodbury or conjugate gradients
check the true residual against the approximated system
on a small reference subset, compare with the exact ridge solution
```

The optimization implication is supplied by the [regularized spectral approximation proposition](#prop-rand-reg-to-solve) if the sketch yields a regularized spectral approximation. The statistical implication additionally needs the regression model and critical-radius argument. A sketch can therefore be:

- a good matrix compressor but statistically wasteful;
- a good statistical sketch but a poor unregularized approximation;
- a good preconditioner even when it is too crude to replace the kernel in the final predictor.

A preconditioner succeeds by clustering the spectrum of the linear system, not necessarily by replacing the kernel accurately entry by entry. On the same ill-conditioned ridge system, diagonal scaling gives a modest improvement while a spectral low-rank preconditioner collapses the residual in far fewer Krylov iterations. The plotted residual is computed against the original system, which prevents an approximate solve from certifying itself.

<figure class="viz" data-figure="preconditioned-cg-convergence" data-alt="Relative residual histories for ordinary conjugate gradients, diagonally preconditioned conjugate gradients, and a spectral low-rank preconditioner on the same kernel system."><figcaption>Preconditioning changes optimization geometry. A spectral approximation can be too crude to replace \(K\) as a predictor yet still cluster the system spectrum enough to accelerate conjugate gradients; convergence must be checked with the true residual.</figcaption></figure>

**Failure boundary.** Obliviousness protects against choosing the sketch after seeing the matrix, but not against an adaptive data stream that reacts to the sketch. Sparse sketches can require larger \(m\) for coherent eigenspaces. A rank-\(m\) approximation cannot preserve more than \(m\) independent high-energy directions, regardless of the sketch distribution.

## Stochastic traces and Lanczos log determinants {#rand-trace-logdet}

Low-rank features accelerate prediction. Gaussian-process training introduces another target:

$$
\log\det A_\theta
=
\operatorname{tr}\log A_\theta,
\qquad
A_\theta=K_\theta+\sigma^2I\succ0.
$$

Neither a solve residual nor a low-rank Frobenius error directly controls this spectral sum. Stochastic trace estimation supplies the outer randomization; Lanczos quadrature approximates each matrix-function quadratic form.

::: {.theorem #thm-rand-hutchinson}
[Theorem (Hutchinson trace estimator)]{.box-title}

Let \(B\in\mathbb R^{n\times n}\) be symmetric and let \(z\) have independent Rademacher entries. Then

$$
\mathbb E[z^\top Bz]=\operatorname{tr}B,
\qquad
\operatorname{Var}(z^\top Bz)
=
2\sum_{i\neq j}B_{ij}^2.
$$

For \(s\) independent probes, the average has the same mean and variance divided by \(s\).

**Assumptions.** Fixed symmetric matrix independent of the probes and independent Rademacher coordinates. **Proof status.** complete.
:::

::: {.proof}
[Proof]{.box-title}

Expand

$$
z^\top Bz
=
\sum_iB_{ii}
+2\sum_{i\lt j}B_{ij}z_iz_j.
$$

Since \(\mathbb E[z_iz_j]=0\) for \(i\neq j\), the expectation is \(\operatorname{tr}B\). In the variance, cross-products vanish unless the unordered index pairs coincide, so

$$
\operatorname{Var}(z^\top Bz)
=
4\sum_{i\lt j}B_{ij}^2
=
2\sum_{i\neq j}B_{ij}^2.
$$

Independence of probes divides the variance of their average by \(s\). [\(\square\)]{.qed}
:::

The original estimator was introduced for influence-matrix traces in smoothing splines [@hutchinson1989]. For a log determinant, set \(B=\log A\), but do not form \(\log A\). Starting from \(q_1=z/\lVert z\rVert\), run \(r\) Lanczos steps on \(A\) to obtain an orthonormal basis \(Q_r\) and a symmetric tridiagonal \(T_r=Q_r^\top A Q_r\). Then

$$
z^\top\log(A)z
\approx
\lVert z\rVert^2 e_1^\top\log(T_r)e_1.
$$

::: {.proposition #prop-rand-lanczos-exact}
[Proposition (Lanczos quadrature polynomial exactness)]{.box-title}

In exact arithmetic, after \(r\) Lanczos steps without breakdown,

$$
z^\top p(A)z
=
\lVert z\rVert^2e_1^\top p(T_r)e_1
$$

for every polynomial \(p\) of degree at most \(2r-1\).

**Assumptions.** Symmetric \(A\), nonzero \(z\), exact arithmetic, and no premature breakdown; if breakdown occurs because the Krylov space becomes invariant, the formula is exact for every function defined on the reached eigenvalues. **Proof status.** proof sketch.
:::

::: {.proof}
[Proof sketch]{.box-title}

The spectral measure induced by \(A\) and \(z\) is

$$
d\mu(t)=\sum_j |u_j^\top z|^2\,\delta_{\lambda_j}(t),
$$

so \(z^\top p(A)z=\int p(t)\,d\mu(t)\). Lanczos recurrence coefficients form the Jacobi matrix \(T_r\) for orthogonal polynomials under \(\mu\). The eigenvalues of \(T_r\) and squared first components of its eigenvectors are the nodes and weights of the \(r\)-point Gauss rule, which is exact for polynomials through degree \(2r-1\). [\(\square\)]{.qed}
:::

For \(f(t)=\log t\), quadrature error is controlled by polynomial or rational approximation on an interval containing the spectrum. The condition number matters because \(\log t\) varies most rapidly near zero. Ubaru, Chen, and Saad combine stochastic probes with Lanczos and derive bounds for \(\operatorname{tr}f(A)\) [@ubaru2017slq, Sections 3--4]. Han, Malioutov, and Shin use stochastic traces with Chebyshev expansions and make the condition-number dependence explicit [@han2015logdet]. Preconditioning can reduce both Krylov iterations and stochastic variance if the preconditioner's log determinant is tractable [@wenger2022preconditioning, Theorems 1--3].

```text
Algorithm: stochastic Lanczos log determinant
input: SPD matrix-vector product A(v), probes s, Lanczos steps r
estimate = 0
for j = 1,...,s:
    draw Rademacher z_j
    run r-step Lanczos from z_j / norm(z_j), producing T_j
    q_j = norm(z_j)^2 * e_1^T log(T_j) e_1
    estimate += q_j / s
return estimate
```

Two errors must be reported separately:

1. **probe error**, the Monte Carlo error in approximating \(\operatorname{tr}\log A\);
2. **quadrature error**, the bias in approximating each \(z^\top\log(A)z\).

Increasing probes does not repair too few Lanczos steps, and increasing Lanczos steps does not reduce probe variance.

::: {.example #example-rand-slq}
[Example (probe error and quadrature error are different)]{.box-title}

Let

$$
A=
\begin{pmatrix}
2&0.5\\
0.5&1
\end{pmatrix},
\qquad
\log\det A=\log(1.75)=0.5596.
$$

There are two Rademacher directions up to sign: \(z_+=(1,1)^\top\) and \(z_-=(1,-1)^\top\). With one Lanczos step, the Gauss estimates are

$$
\lVert z_+\rVert^2\log\left(\frac{z_+^\top Az_+}{\lVert z_+\rVert^2}\right)
=2\log2=1.3863
$$

and

$$
\lVert z_-\rVert^2\log\left(\frac{z_-^\top Az_-}{\lVert z_-\rVert^2}\right)
=2\log1=0.
$$

Averaging both gives \(0.6931\), so using every possible probe direction has removed probe error but not one-step quadrature bias. With two Lanczos steps, the Krylov space is all of \(\mathbb R^2\), each quadratic form is exact, and averaging the two signs gives \(\operatorname{tr}\log A=\log\det A=0.5596\).

**Verification artifact.** checks/example-ch-randomized-example-rand-slq.json records the example source hash and verification scope.
:::

**Failure boundary.** The logarithm requires \(A\succ0\). Tiny eigenvalues increase approximation difficulty and make a silently clipped spectrum unacceptable. Reusing probes across nearby hyperparameters can reduce noise in objective differences, but the dependence must be retained in uncertainty estimates. Finite-precision Lanczos loses orthogonality; residual and repeated-run checks are needed.

## Streaming and mergeable sketches {#rand-streaming}

A batch approximation can revisit all data. A stream cannot. At time \(t\), a usable state must fit a fixed memory budget and update without reconstructing the past.

Random features are naturally streamable when their seed is fixed. For squared-loss ridge regression with \(z_t=z(x_t)\in\mathbb R^m\), maintain

$$
G_t=\sum_{i=1}^{t}z_iz_i^\top,
\qquad
b_t=\sum_{i=1}^{t}z_i y_i.
$$

Then the exact ridge solution in the random-feature space is

$$
w_t=(G_t+t\lambda I)^{-1}b_t.
$$

The state costs \(O(m^2)\), not \(O(m)\). First-order online updates reduce state to \(O(m)\) but introduce optimization and step-size error. A mergeable linear sketch satisfies

$$
\mathcal S(D_1\cup D_2)
=
\mathcal S(D_1)+\mathcal S(D_2)
$$

for additive sufficient statistics, so independent shards can combine without raw-data exchange.

Incremental Nyström and data-adaptive sketches are harder because the basis changes. Zhang and Liao maintain sparse-transform and sampling sketches through rank-one updates, construct a time-varying explicit map, and prove unbiased product approximation, relative matrix error, and sublinear regret under their online protocol [@zhang2019incremental, Theorems 1--3]. The paper's central move is to update the sketch algebraically rather than recomputing it from the growing kernel matrix.

```text
Algorithm: fixed-seed streaming random-feature ridge
input: feature seed, dimension m, ridge schedule lambda_t
initialize G = zeros(m,m), b = zeros(m)
for each arriving pair (x_t,y_t):
    z = feature_map(x_t, seed)
    G = G + z z^T
    b = b + z y_t
    periodically solve (G + t lambda_t I) w = b
    record prequential loss before updating on y_t
return state (seed,G,b,w)
```

The failure boundary is adaptivity. Classical concentration assumes the data or queries do not adversarially react to the sketch. A fixed feature map can become stale under concept drift; a refreshed map invalidates old sufficient statistics unless they are replayed or transported. Mergeability also does not imply privacy. The sketch may leak information and requires a separate privacy analysis.

## Lower bounds and no-free-lunch boundaries {#rand-lower-bounds}

Randomization can reduce work by exploiting spectrum, regularization, distribution, or acceptable failure probability. It cannot repeal dimension.

::: {.theorem #thm-rand-rank-lower}
[Theorem (rank is an unavoidable matrix-approximation budget)]{.box-title}

Let \(K\succeq0\) have eigenvalues \(\lambda_1\geq\cdots\geq\lambda_n\geq0\). For every matrix \(\widetilde K\) of rank at most \(m\),

$$
\lVert K-\widetilde K\rVert_2\geq\lambda_{m+1},
\qquad
\lVert K-\widetilde K\rVert_F^2
\geq
\sum_{j\gt m}\lambda_j^2.
$$

Equality is achieved by the truncated eigendecomposition.

**Assumptions.** Finite matrix and rank constraint only. **Proof status.** complete.
:::

::: {.proof}
[Proof]{.box-title}

The null space of \(\widetilde K\) has dimension at least \(n-m\). The span of the top \(m+1\) eigenvectors of \(K\) has dimension \(m+1\), so it intersects that null space in a nonzero vector \(v\). For a unit vector in the intersection,

$$
\lVert(K-\widetilde K)v\rVert
=
\lVert Kv\rVert
\geq
\lambda_{m+1},
$$

which proves the spectral bound. The Frobenius statement is the Eckart-Young-Mirsky theorem: the squared singular values discarded by any rank-\(m\) approximation sum to at least the eigenvalue tail shown. [\(\square\)]{.qed}
:::

For regularized spectral error, the corresponding unavoidable tail is governed by

$$
\frac{\lambda_{m+1}}{\lambda_{m+1}+\gamma},
$$

not \(\lambda_{m+1}\) alone. This is why regularization can make a low-rank approximation useful even when unregularized matrix error remains large.

Three further limitations matter:

1. Avron et al. prove a Gaussian-kernel lower bound showing that ordinary RFF can require a large feature budget for a regularized spectral approximation; the upper-bound dependence is not merely a proof artifact [@avron2017rff, Theorem 8].
2. Bach gives eigenvalue-based lower bounds for feature and quadrature approximation that apply to any selected points or features in the stated integral-operator setting [@bach2017quadrature, Propositions 2--3].
3. Yang, Pilanci, and Wainwright show that a sketch below the statistical dimension can fail to preserve the minimax KRR rate on admissible regression problems [@yang2017sketch, Theorem 2].

These lower bounds use different currencies and are not interchangeable. A matrix lower bound does not prove a risk lower bound. A minimax risk lower bound does not say every dataset is hard. A lower bound for i.i.d. Fourier sampling does not rule out data-adaptive features.

## A decision table for randomized kernel approximation {#rand-decision-table}

The right method follows from the target, available structure, and failure mode.

<figure class="viz" data-figure="approximation-decision-map" data-alt="A decision map separates exact and matrix-free methods, Nyström approximations, random features, and structured products by spectral concentration and pressure from streaming or distribution constraints.">
<figcaption>Approximation is a systems decision as well as a theorem choice. Data-adaptive column methods gain most from a concentrated observed spectrum; portable random features gain most when a reusable seed is easier to carry than landmarks or a growing matrix.</figcaption>
</figure>

| Need | Preferred starting point | Budget diagnostic | Verify | Avoid when |
|---|---|---|---|---|
| portable map for a stationary kernel | RFF, then ORF/Fastfood/QMC | feature variance and wall time | held-out risk plus Gram check | spectral density unavailable |
| dot-product or polynomial kernel | Random Maclaurin or TensorSketch | degree, input norm, collision variance | unbiasedness and repeated seeds | high degree and unnormalized inputs |
| data-adaptive low rank | ridge-leverage or recursive Nyström | \(d_{\mathrm{eff}}(\gamma)\) | regularized spectral error | landmarks cannot be retained |
| interpolation geometry | pivoted Cholesky | maximum residual diagonal | power function and trace residual | risk is driven by low-residual label directions |
| diverse landmarks | fixed-size DPP | determinant and sampling cost | Nyström error and task loss | exact sampling dominates training |
| fast ridge solve | approximation as preconditioner | preconditioned condition number | true residual and solution error | approximation is substituted without validation |
| minimax fixed-design KRR | oblivious statistical sketch | critical radius/statistical dimension | prediction error under model assumptions | adaptive adversarial stream |
| GP log determinant | Hutchinson plus Lanczos | probe variance and spectral interval | probe and quadrature errors separately | matrix is not safely positive definite |
| one-pass stream | fixed-seed features or incremental sketch | memory, update latency, drift | prequential loss and replay check | feature basis changes without state transport |

A complete experiment should compare methods at equal peak memory and equal total construction plus training time. Report at least one metric in every relevant currency:

$$
\frac{\lVert K-\widetilde K\rVert_F}{\lVert K\rVert_F},
\qquad
E_{\mathrm{reg}}(\gamma),
\qquad
\frac{\lVert A_\gamma\widehat\alpha-y\rVert}{\lVert y\rVert},
\qquad
R_{\mathrm{test}}(\widetilde f)-R_{\mathrm{test}}(\widehat f).
$$

If uncertainty is used, add calibration or coverage. A single kernel RMSE curve is not an end-to-end approximation study.

## Common mistakes and practical implications {#rand-practice}

**Calling unbiasedness accuracy.** Random Maclaurin, TensorSketch, and RFF are unbiased under their sampling assumptions. Their variance can still make a finite map useless.

**Using the wrong ridge normalization.** A theorem written for \(K+\lambda I\) cannot be inserted unchanged into code solving \(K+n\lambda I\). Effective dimension and feature budgets change with the scale.

**Reporting only Frobenius error.** Frobenius norm weights all entries or eigen-directions according to matrix energy, not according to a regularized solve or prediction task.

**Fitting the approximation budget on the test set.** Rank, feature count, frequency proposal, and stopping tolerance are hyperparameters. Selecting them on test risk leaks information.

**Ignoring approximation construction.** Exact leverage scores, exact DPP samples, and a dense pilot Gram matrix can cost more than the solve they replace.

**Forming inverses.** Use triangular solves, QR, eigendecompositions of small cores, Woodbury identities, and matrix-free products. An expression containing \(K_{SS}^{-1}\) is mathematics, not implementation advice.

**Hiding stochastic variability.** Use fixed seeds for reproducibility and multiple independent seeds for uncertainty. One seed cannot quantify feature variance.

**Clipping without diagnosis.** Tiny negative residual diagonals from roundoff may be clipped after recording their scale. Materially negative values indicate a broken PSD assumption, inconsistent kernel evaluations, or unstable updates.

## Summary and further reading {#rand-summary}

Randomized kernel computation is a collection of typed approximations. Random Maclaurin samples an analytic dot-product expansion. TensorSketch hashes a tensor product and computes the hash by FFT convolution. RFF samples a Fourier integral, while ridge-leverage features adapt that integral to the regularized sample spectrum. Fastfood, ORF, and QMC change the cost or variance of spectral sampling. Pivoted Cholesky, leverage Nyström, and DPPs choose data columns according to residual uncertainty, regularized importance, or diversity. Oblivious sketches preserve a statistically relevant eigenspace. Hutchinson and Lanczos target spectral sums rather than predictors. Streaming methods preserve only what can be updated and carried forward.

The practical sequence is:

1. name the downstream quantity;
2. choose the matching error currency;
3. state the regularization and sampling convention;
4. derive the approximation and its computational cost;
5. verify the theorem's assumptions;
6. measure construction, numerical, and statistical errors separately;
7. test the method at its failure boundary.

The original Random Maclaurin, TensorSketch, RFF, Nyström leverage, statistical sketching, trace-estimation, and streaming papers are [@kar2012random; @pham2013tensor; @rahimi2007; @avron2017rff; @alaoui2015; @yang2017sketch; @hutchinson1989; @zhang2019incremental]. Their later refinements are best understood as answers to specific limitations: variance, data blindness, transform cost, landmark redundancy, expensive leverage computation, ill-conditioned spectral functions, and nonstationary memory.

## Exercises {#exercises}

::: {.exercises}
1.  [warm-up]{.ex-tag} For each statement, identify the strongest error currency it establishes: (a) \(\lVert K-\widetilde K\rVert_F/\lVert K\rVert_F\leq0.01\); (b) \((1-\varepsilon)(K+\gamma I)\preceq\widetilde K+\gamma I\preceq(1+\varepsilon)(K+\gamma I)\); (c) the approximate linear system has relative residual \(10^{-8}\); (d) the approximate predictor's test risk is within \(0.002\) of exact KRR on an independent test set. For each, name one conclusion that does not follow without extra assumptions.
2.  [computation]{.ex-tag} Reproduce the Random Maclaurin example for \(k(x,y)=\langle x,y\rangle^2\), \(x=(1,2)\), and \(y=(3,-1)\). Compute the exact kernel, \(M(x,y)\), the one-feature variance, and the variance after \(m=2700\) independent features. Use Chebyshev's inequality to upper-bound the probability that the absolute kernel error exceeds \(1\). Explain why the bound is conservative but still diagnostically useful.
3.  [proof]{.ex-tag} Let \(C\) be an ordinary CountSketch with a uniform pairwise-independent hash and independent four-wise-independent Rademacher signs. For fixed \(u,v\in\mathbb R^D\), prove \(\mathbb E\langle Cu,Cv\rangle=\langle u,v\rangle\). Then show that the variance is controlled by off-diagonal collision terms and derive the safe bound \(\operatorname{Var}(\langle Cu,Cv\rangle)\leq 2\lVert u\rVert^2\lVert v\rVert^2/m\). Finally explain why this ordinary CountSketch bound cannot simply be asserted for TensorSketch without checking the dependencies induced by the tensor hash.
4.  [proof]{.ex-tag} Prove the [regularized spectral approximation proposition](#prop-rand-reg-to-solve) again by using the resolvent identity \(\widetilde A_\gamma^{-1}-A_\gamma^{-1}=-\widetilde A_\gamma^{-1}(\widetilde K-K)A_\gamma^{-1}\). Your final bound should be in the \(A_\gamma\)-energy norm. Identify exactly where \(\varepsilon\lt1\) is needed.
5.  [computation]{.ex-tag} For the matrix in the [pivoted Cholesky example](#example-rand-pchol), carry out the second pivoted-Cholesky step after selecting points \(1\) and \(3\). Compute the new residual matrix, its remaining diagonal entry, and the rank-two Nyström approximation. Verify that the residual is PSD and that its trace equals its only nonzero eigenvalue.
6.  [synthesis]{.ex-tag} Suppose ordinary RFF needs a feature bound governed by a worst-case whitened feature norm \(L\), while ideal leverage sampling makes that norm \(d_{\mathrm{eff}}(\gamma)\). Explain why the leverage proposal is optimal for equalizing the whitened rank-one sample norms. Then give two reasons an approximate leverage implementation may fail to realize the theoretical saving, one computational and one statistical.
7.  [computation]{.ex-tag} For \(A=\begin{pmatrix}2&0.5\\0.5&1\end{pmatrix}\), enumerate the four Rademacher probes, compute the exact Hutchinson estimate \(z^\top\log(A)z\) for the two distinct directions up to sign, and verify that their average is \(\log\det A=\log1.75\). Compare this exact two-step Lanczos result with the one-step values \(2\log2\) and \(0\). State which discrepancy is probe error and which is quadrature error.
8.  [proof]{.ex-tag} Let \(A\succ0\) have only \(r\) distinct eigenvalues that receive nonzero weight from the starting vector \(z\). Show that Lanczos terminates after at most \(r\) steps and that \(z^\top f(A)z=\lVert z\rVert^2e_1^\top f(T_r)e_1\) for every function \(f\) defined on those eigenvalues. Relate this finite-support argument to the polynomial exactness proposition.
9.  [exploration]{.ex-tag} Design a streaming kernel-ridge system under a strict memory budget. Compare (a) fixed-seed RFF with sufficient statistics \(G_t,b_t\), (b) first-order online updates in the same features, and (c) an incremental Nyström dictionary. For each, report asymptotic memory, update cost, what can be merged across shards, how drift is detected, and what guarantee is lost if the feature basis changes without replay.
10. [challenge]{.ex-tag} Let \(K\) have eigenvalues \(\lambda_j=j^{-2}\), let \(\gamma=10^{-3}\), and require regularized tail error \(\lambda_{m+1}/(\lambda_{m+1}+\gamma)\leq0.1\). Find the smallest admissible \(m\). Compare it with the rank required for unregularized spectral error \(\lambda_{m+1}\leq10^{-3}\). Then explain why neither calculation alone determines the feature count needed for a population-risk guarantee.
:::
