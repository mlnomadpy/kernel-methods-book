---
narrative_link_policy: exact
id: ch-gp
slug: gaussian-processes-and-rvm
title: Gaussian Processes and the RVM
part: IX · Gaussian Processes and Sequential Decisions
order: 49
tier: core
prerequisites:
  - kernels-and-deep-learning
objectives:
  - >-
    Read a positive definite kernel as a covariance law over functions and
    compute a GP posterior.
  - >-
    Derive the identity between the GP posterior mean and kernel ridge
    regression.
  - >-
    Separate latent-function uncertainty, observation noise, and model
    misspecification in a prediction.
  - >-
    Distinguish finite-dimensional Gaussian laws, sample-path regularity, and
    the RKHS associated with the covariance kernel.
  - >-
    Optimize marginal likelihood while detecting weak identifiability and
    numerically unstable covariance matrices.
  - >-
    Compare Laplace classification, RVM sparsity, and inducing-point
    approximations by the model each one changes.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-gp.yml
verification_date: null
bibliography:
  - rasmussen2006
  - williams1996
  - neal1996
  - tipping2001
  - mackay1992
  - scholkopf2002
  - williams1998classification
  - quinonero2005
  - snelson2006fitc
  - titsias2009svgp
  - hensman2013bigdata
  - wu2024alternatinggp
  - lin2025latentkronecker
  - rathore2025adasap
  - hoffbauer2025kernelmatmul
  - wilson2013
  - kernelbook-code-ch-gp-ex1
  - kernelbook-code-ch-gp-ex2
  - kernelbook-code-ch-gp-ex3
  - damianou2013deepgp
  - jylanki2011robustgp
  - kuss2005gpc
  - szabo2015coverage
  - vanderVaart2008gpcontraction
  - zhang2004microergodic
example_code_policy: visible-for-executable
---
# Gaussian Processes and the RVM

<p class="lead">A forecast that reads 0.8 could mean 0.8 give or take 0.01, or 0.8 give or take 1.0; the decisions those two numbers justify are entirely different, and nothing built so far can tell them apart. Every method to this point writes down a loss, adds a regularizer, and minimizes, returning a single curve that is silent about uncertainty under its assumptions. The Bayesian route reaches the same algorithms from the opposite side: it places a probability distribution over functions before any data arrive, then updates that distribution with Bayes' rule once the data are in. The answer is no longer a single curve but a posterior distribution. Its intervals are conditional statements about a chosen covariance, likelihood, and hyperparameters, not automatic coverage guarantees, and learning to read that qualification is part of learning Gaussian processes. The centerpiece of this chapter is a single identity: the Gaussian process posterior mean is exactly the kernel ridge regression fit of [[ch:kernel-ridge-and-friends|the ridge chapter]], with the noise variance playing the role of the regularization strength. Around that identity we develop the distinction between sample paths and RKHS functions, marginal-likelihood learning, Gaussian process classification, and the sparse Bayesian prior behind the Relevance Vector Machine.</p>

## The Bayesian view: a prior over functions {#bayesian-view}

The framework of risk minimization asks which function best fits the data under a complexity penalty. The Bayesian framework asks a different question: given a prior belief about which functions are plausible, and given a model of how observations are generated from a function, how plausible is each function after we see the data? The two ingredients are a likelihood and a prior, and the machinery that combines them is Bayes' rule. This account follows [@scholkopf2002; @mackay1992].

Suppose we observe inputs \(X=(x_1,\dots,x_m)\) and targets \(Y=(y_1,\dots,y_m)\), and we posit a hypothesis \(f\) together with a model \(P(y\mid x,f(x))\) of how each label is produced from the underlying value \(f(x)\). The prototypical case is additive noise, \(y=f(x)+\xi\), so that \(P(y\mid x, f(x))=P(y-f(x))\). Under the usual independence assumption the likelihood factorizes,

$$P(Y\mid X,f)=\prod_{i=1}^m P(y_i\mid x_i,f(x_i)).$$

The likelihood tells us how probable the observed sample is if \(f\) is responsible for it. The prior \(p(f)\) encodes what we believe before seeing data: a preference for smooth functions, for small values, or for a particular correlation structure among the values \(f(x_i)\).

Bayes' rule turns the likelihood around. Since \(p(Y\mid f,X)p(f)=p(f\mid X,Y)\,p(Y)\), and \(p(Y)\) does not depend on \(f\), the posterior over hypotheses is

$$p(f\mid X,Y)\ \propto\ p(Y\mid f,X)\,p(f).$$

To predict at a new location \(x\) we integrate the label model against the posterior,

$$p(y\mid X,Y,x)=\int p(y\mid f,x)\,p(f\mid X,Y)\,df,$$

which yields not a point estimate but a predictive distribution. When this integral is intractable, a common shortcut is the maximum a posteriori (MAP) estimate, the mode of the posterior,

$$f_{\mathrm{MAP}}=\arg\min_f\big[-\ln p(Y\mid f,X)-\ln p(f)\big].$$

Read the two terms: the negative log likelihood is a data-fit term and the negative log prior is a penalty independent of the data. This is precisely the regularized risk functional \(R_{\mathrm{emp}}[f]+\Omega[f]\), with the loss identified as the negative log likelihood and the regularizer as the negative log prior. Squared loss corresponds to Gaussian noise, and a quadratic RKHS penalty corresponds to a Gaussian prior. The correspondence is exact at the level of the mode; the Bayesian side additionally offers the mean and the variance, which risk minimization has no counterpart for.

## The Gaussian process {#gaussian-processes}

To put a prior on a function we cannot write down a density on an infinite-dimensional space directly. The trick, developed in the process and neural-network literature [@neal1996; @williams1996], is to specify the prior only through the joint distribution of the function values at any finite set of points, and to make that joint distribution Gaussian. A collection of function values that is jointly Gaussian for every finite subset of locations is a Gaussian process.

:::: {.definition #def-40-1}
[Definition (Gaussian process)]{.box-title}

A stochastic process \(t(x)\) indexed by \(x\in\mathcal X\) is a *Gaussian process* if, for every finite set \(x_1,\dots,x_m\in\mathcal X\), the vector \((t(x_1),\dots,t(x_m))\) is normally distributed. It is determined by a mean function \(\mu(x)=\mathbb E[t(x)]\) and a covariance function

$$k(x,x')=\operatorname{cov}\big(t(x),t(x')\big),\qquad K_{ij}=k(x_i,x_j).$$

We write \((t(x_1),\dots,t(x_m))\sim\mathcal N(\mu,K)\), and take \(\mu\equiv 0\) unless stated otherwise.
::::

The covariance function is not an arbitrary symmetric function. For any coefficients \(c\in\mathbb R^m\) the variance of the linear combination \(\sum_i c_i t(x_i)\) is nonnegative,

$$0\le\operatorname{Var}\Big(\sum_i c_i t(x_i)\Big)=\sum_{i,j}c_i c_j\operatorname{cov}\big(t(x_i),t(x_j)\big)=c^\top K c,$$

so \(K\) is positive semidefinite for every choice of points. Conversely, a symmetric positive-semidefinite kernel defines a consistent family of finite-dimensional Gaussian laws; the Kolmogorov extension theorem then supplies a process having those laws. This is the first bridge: a covariance kernel is a PSD kernel, but continuity or differentiability of its sample paths requires additional hypotheses.

When \(K\) is positive definite, the zero-mean law of \(t=(t(x_1),\dots,t(x_m))\) has the density

$$p(t)=(2\pi)^{-m/2}(\det K)^{-1/2}\exp\!\Big(-\tfrac12 t^\top K^{-1}t\Big),$$

and \(-\ln p(t)=\tfrac12 t^\top K^{-1}t+\text{const}\). If \(K\) is singular, the Gaussian law is supported on \(\operatorname{range}(K)\) and has no density with respect to \(m\)-dimensional Lebesgue measure; the correct quadratic form uses \(K^\dagger\) on that support. That boundary case matters whenever inputs are duplicated or the kernel has finite rank.

### Finite values, sample paths, and the RKHS are different objects {#gp-paths-rkhs}

The vector \(t\), a random function \(f\), and an element of the RKHS \(\mathcal H_k\) are related but not interchangeable. On a finite design, the minimum-\(\mathcal H_k\)-norm function satisfying \(g(x_i)=t_i\) has squared norm \(t^\top K^\dagger t\), provided \(t\in\operatorname{range}(K)\). This is the precise sense in which the Gaussian quadratic form and the RKHS penalty agree. It does **not** follow that a draw \(f\sim\mathcal{GP}(0,k)\) belongs to \(\mathcal H_k\). If the covariance operator has infinitely many positive eigenvalues, a GP sample lies outside its own RKHS with probability one: in Mercer coordinates a draw has coefficients \(\sqrt{\lambda_j}Z_j\), so its formal RKHS norm is

$$\sum_{j:\lambda_j\gt 0}\frac{\lambda_j Z_j^2}{\lambda_j}=\sum_j Z_j^2=\infty\qquad\text{almost surely}.$$

The RKHS instead describes finite-energy directions along which the Gaussian measure can be shifted; it contains the posterior mean, but typically not the prior sample paths. This distinction, developed in GPML's discussion of RKHSs and regularization [@rasmussen2006, ch. 6], prevents a common but serious error: “the prior puts mass on smooth RKHS functions” is generally false as stated, even when the sample paths are very smooth.

Regularity of the random function is read from increments. For a centered process,

$$\mathbb E\!\left[(f(x)-f(x'))^2\right]
=k(x,x)+k(x',x')-2k(x,x').$$

Thus covariance continuity controls mean-square continuity. If the mixed derivative \(\partial_x\partial_{x'}k(x,x')\) exists with suitable continuity, the mean-square derivative exists and has covariance

$$\operatorname{cov}\!\left(\partial_x f(x),\partial_{x'}f(x')\right)
=\partial_x\partial_{x'}k(x,x').$$

Mean-square differentiability is still not the same as almost-sure differentiability; the latter needs stronger increment bounds. GPML uses this hierarchy to explain why a Matérn smoothness parameter changes path roughness, while the squared-exponential covariance produces exceptionally smooth paths [@rasmussen2006, §4.1.1]. The kernel therefore controls three distinct geometries: finite covariance matrices, an RKHS of admissible shifts and estimators, and the regularity of random sample paths.

::::: {.example #example-40-4}
[Example (duplicate inputs expose three different assumptions)]{.box-title}

```python
import numpy as np

K = np.ones((2, 2))
y = np.array([1.0, -1.0])
A = K + 0.01 * np.eye(2)
alpha = np.linalg.solve(A, y)
condition = np.linalg.cond(A)
assert np.allclose(alpha, [100.0, -100.0])
assert np.isclose(condition, 201.0)
print(np.linalg.eigvalsh(K), alpha, condition, np.ones(2) @ alpha)
```

:::: wex
::: wex-setup
Take two observations at the same input \(x_1=x_2=x_0\), normalize
\(k(x_0,x_0)=1\), and let \(y=(1,-1)^\top\). This is a deliberately
contradictory dataset. The calculations are reproduced by
`checks/ch-gp-depth.py`.
:::

1.  [Inspect the prior law.]{.wex-op} The latent Gram matrix is
    \(K=\begin{pmatrix}1&1\\1&1\end{pmatrix}\), with eigenvalues \(2\) and
    \(0\). The prior enforces \(f(x_1)=f(x_2)\) almost surely, so it has no
    two-dimensional density.
2.  [Try the noise-free observation.]{.wex-op} The target \(y\) lies entirely
    in \(\ker(K)\), since \(Ky=0\). No latent function can interpolate both
    labels at one input. A pseudoinverse does not repair incompatible data:
    \(K^\dagger y=0\).
3.  [Add modeled noise.]{.wex-op} With \(\sigma^2=0.01\),
    \(A=K+0.01I\) has eigenvalues \(2.01\) and \(0.01\), condition number
    \(201\), and \(\alpha=A^{-1}y=(100,-100)^\top\). Yet the posterior mean at
    the duplicated site is \((1,1)\alpha=0\). The large opposite
    coefficients encode the contradiction in the noise direction; they do
    not create two latent values at one input.
4.  [Separate noise from jitter.]{.wex-op} If \(0.01\) is observation noise,
    the predictive distribution says the labels genuinely vary around one
    latent value. If it is merely numerical jitter, it authorizes no such
    data-generating interpretation. The matrices are identical, but the
    claims are not.

**Reading.** PSD validity, consistency of the observed labels with a
noise-free prior, and numerical solvability are three separate questions.
Duplicate inputs force the reader to answer all three.
::::
:::::

### Matérn smoothness can be read from one increment {#gp-matern-regularity}

The Matérn family makes the regularity hierarchy concrete. In one input
dimension, set \(r=|x-x'|\). At \(\nu=\tfrac12\),

$$k_{1/2}(r)=\exp(-r/\ell),\qquad
\mathbb E[(f(x+h)-f(x))^2]
=2\big(1-e^{-|h|/\ell}\big)
\sim \frac{2|h|}{\ell}.$$

The variance of the difference quotient therefore grows like
\(2/(\ell|h|)\); no mean-square derivative exists. At \(\nu=\tfrac32\),

$$k_{3/2}(r)=\left(1+\frac{\sqrt3r}{\ell}\right)
\exp\left(-\frac{\sqrt3r}{\ell}\right),$$

and a Taylor expansion gives

$$2\big(k_{3/2}(0)-k_{3/2}(|h|)\big)
=\frac{3h^2}{\ell^2}+O(|h|^3).$$

Now the difference quotients have a finite limiting variance
\(3/\ell^2\), consistent with one mean-square derivative. The
squared-exponential covariance has derivatives of every order at the
diagonal and is mean-square differentiable to every order. This comparison
does more than rank kernels as “rough” or “smooth”: it shows the exact
calculation that the smoothness label abbreviates. The detailed Matérn
construction and its spectral density remain in
[[ch:kernel-families|the kernel-families chapter]] and
[[ch:spatial-and-spatiotemporal-kernels|the spatial chapter]].

::::: {.example #example-40-5}
[Example (the same tiny displacement separates rough from differentiable)]{.box-title}

```python
import numpy as np

h = 1e-3
increment_half = 2.0 * (1.0 - np.exp(-h))
a = np.sqrt(3.0)
increment_three_half = 2.0 * (1.0 - (1.0 + a * h) * np.exp(-a * h))
print(increment_half, increment_half / h**2)
print(increment_three_half, increment_three_half / h**2)
assert np.isclose(increment_half / h**2, 1999.00033325)
assert np.isclose(increment_three_half / h**2, 2.99653815)
```

:::: wex
::: wex-setup
Set \(\ell=1\) and \(h=10^{-3}\). Compare the exact increment variance with
its leading asymptotic term. The calculation is reproduced by
`checks/ch-gp-depth.py`.
:::

1.  [Matérn \(\nu=\tfrac12\).]{.wex-op}
    \(2(1-e^{-0.001})=0.0019990003\), against the leading term
    \(2|h|=0.002\). Dividing by \(h^2\) gives difference-quotient variance
    \(1999.0003\).
2.  [Matérn \(\nu=\tfrac32\).]{.wex-op}
    \(2[1-(1+\sqrt3h)e^{-\sqrt3h}]=2.9965381\times10^{-6}\), against
    \(3h^2=3\times10^{-6}\). Dividing by \(h^2\) gives \(2.9965\), approaching
    the derivative variance \(3\).

**Reading.** Both kernels are continuous and both increments are small.
Only the rate at which the increment variance vanishes decides whether a
mean-square derivative exists.
::::
:::::

### General linear observations reuse the same conditioning theorem {#gp-linear-observations}

Point evaluation is only one bounded linear observation. Let
\(L_1,\ldots,L_n\) be linear functionals for which the covariance actions
below exist, and observe

$$y_i=L_i f+\varepsilon_i,\qquad
\varepsilon\sim\mathcal N(0,\Sigma_\varepsilon).$$

Then the observation covariance and test cross-covariance are

$$[K_{LL}]_{ij}=L_i^{(x)}L_j^{(x')}k(x,x'),\qquad
[k_{*L}]_i=L_i^{(x)}k(x_*,x),$$

so the posterior is obtained by replacing \(K+\sigma^2I\) with
\(K_{LL}+\Sigma_\varepsilon\). Derivative observations use derivatives of
the kernel; integral observations integrate it; averaged sensor readings
apply the averaging functionals on both arguments. The required hypothesis
is not merely that \(k\) is PSD. Each \(L_i\) must be well defined in
mean square, and the resulting block covariance must be finite. This is the
probabilistic counterpart of the generalized representer theorem in
[[ch:kernel-tricks|the kernel-tricks chapter]] and the differential
observation machinery in
[[ch:scientific-computing-and-operator-learning|the scientific-computing
chapter]].

:::: {.proposition #prop-40-4}
[Proposition (posterior under finite linear observations)]{.box-title}

Let \(f\sim\mathcal{GP}(m,k)\). Let \(L=(L_1,\ldots,L_n)\) and
\(M=(M_1,\ldots,M_r)\) be finite collections of linear functionals whose
joint actions on \(m\) and \(k\) exist and are finite. Observe
\(y=Lf+\varepsilon\), independently of \(f\), with
\(\varepsilon\sim\mathcal N(0,\Sigma_\varepsilon)\), and assume
\(K_{LL}+\Sigma_\varepsilon\) is positive definite. Then

$$Mf\mid y\sim\mathcal N\left(
Mm+K_{ML}(K_{LL}+\Sigma_\varepsilon)^{-1}(y-Lm),\;
K_{MM}-K_{ML}(K_{LL}+\Sigma_\varepsilon)^{-1}K_{LM}
\right).$$

**Scope.** Point values, derivatives, and integrals are special cases only
when their covariance actions are well defined. Unbounded functionals
cannot be inserted formally.
**Assumptions.** The stated joint covariance actions are finite,
\(\Sigma_\varepsilon\) is PSD, the noise is independent of \(f\), and
\(K_{LL}+\Sigma_\varepsilon\) is positive definite.
**Proof status.** Proved immediately below.
::::

::: {.proof}
[Proof]{.box-title}

Linearity of \(L\) and \(M\) makes \((Lf,Mf)\) jointly Gaussian, with means
\((Lm,Mm)\) and covariance blocks
\((K_{LL},K_{LM};K_{ML},K_{MM})\). Independent Gaussian noise adds
\(\Sigma_\varepsilon\) to the observation block and changes no
cross-covariance. Apply the finite-dimensional Gaussian conditioning
identity. [\(\square\)]{.qed}
:::

## Gaussian process regression {#gp-regression}

In regression we do not observe the latent function directly. We observe it through additive Gaussian noise, \(y_i=t(x_i)+\xi_i\) with \(\xi_i\sim\mathcal N(0,\sigma^2)\) independent. Because a sum of independent Gaussians is Gaussian, the observed vector \(y\) and any future latent value are jointly Gaussian, and everything we need follows from one fact about Gaussians: conditioning a joint Gaussian on part of its coordinates produces another Gaussian, with closed-form mean and covariance.

Write \(k_*=(k(x_1,x_*),\dots,k(x_m,x_*))^\top\) for the vector of covariances between the training points and a test point \(x_*\), and \(k_{**}=k(x_*,x_*)\). Since \(y_i=t(x_i)+\xi_i\), the training observations have covariance \(K+\sigma^2 I\), while the latent test value \(f_*=t(x_*)\) has covariance \(k_*\) with the training values and variance \(k_{**}\) with itself. The joint law is

$$\begin{pmatrix} y\\ f_*\end{pmatrix}\sim\mathcal N\!\left(0,\ \begin{pmatrix} K+\sigma^2 I & k_*\\ k_*^\top & k_{**}\end{pmatrix}\right).$$

The Gaussian conditioning formula, that \((a\mid b)\) has mean \(\Sigma_{ab}\Sigma_{bb}^{-1}b\) and covariance \(\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}\), gives the predictive distribution of \(f_*\) given the data at once.

:::: {.theorem #thm-40-2}
[Theorem (GP predictive equations)]{.box-title}

Let \(k\) be a symmetric PSD kernel, let \(X=(x_1,\ldots,x_n)\) be fixed, and let \(\sigma^2\gt 0\). Under the prior \(f\sim\mathcal{GP}(0,k)\) and the observation model \(y_i=f(x_i)+\varepsilon_i\), with independent \(\varepsilon_i\sim\mathcal N(0,\sigma^2)\), the posterior over the latent value at a test point \(x_*\) is Gaussian, \(f_*\mid X,y,x_*\sim\mathcal N(\bar m(x_*),v(x_*))\), with

$$\bar m(x_*)=k_*^\top (K+\sigma^2 I)^{-1}y,\qquad v(x_*)=k_{**}-k_*^\top (K+\sigma^2 I)^{-1}k_*.$$

The predictive distribution of a noisy observation \(y_*\) adds \(\sigma^2\) to \(v(x_*)\).

**Assumptions.** \(k\) is symmetric and PSD, \(\sigma^2\gt0\), the
observation noise is independent Gaussian noise, and the design and kernel
hyperparameters are conditioned upon.
**Scope.** The variance is conditional on \(X\), the chosen kernel and its hyperparameters, and the Gaussian noise model. It is not by itself a frequentist coverage guarantee.
**Proof status.** Proved immediately below.
::::

::: {.proof}
[Proof]{.box-title}

Apply the conditioning identity with \(a=f_*\) and \(b=y\), reading the blocks off the joint covariance: \(\Sigma_{bb}=K+\sigma^2 I\), \(\Sigma_{ab}=k_*^\top\), \(\Sigma_{aa}=k_{**}\). The conditional mean is \(\Sigma_{ab}\Sigma_{bb}^{-1}b=k_*^\top(K+\sigma^2 I)^{-1}y\) and the conditional variance is \(\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}=k_{**}-k_*^\top(K+\sigma^2 I)^{-1}k_*\). For a noisy observation \(y_*=f_*+\xi_*\) with independent \(\xi_*\sim\mathcal N(0,\sigma^2)\), the variance of \(f_*\) and \(\xi_*\) add, giving \(v(x_*)+\sigma^2\). [\(\square\)]{.qed}
:::

Three features of these equations deserve emphasis. The mean is a linear combination of kernel functions centered at the training points, \(\bar m(x_*)=\sum_i \alpha_i k(x_i,x_*)\) with \(\alpha=(K+\sigma^2 I)^{-1}y\), so the posterior mean lives in the span of the training kernels exactly as the representer theorem predicts. Conditional on fixed hyperparameters, the variance does not depend on the observed targets \(y\); it measures how the covariance regards the test functional as explained by the observed functionals. “Near a training point” is only a useful shorthand for local stationary kernels, not a theorem for arbitrary kernels. Finally, the modeled noise variance \(\sigma^2\) makes \(K+\sigma^2I\) positive definite. Numerical jitter may also be added in an implementation, but jitter is a stabilization device and must not silently be interpreted as observation noise [@rasmussen2006].

:::: {.algorithm #algo-40-1}
[Algorithm (GP regression)]{.box-title}

::: algo-io
[Input]{.algo-lab} Training data \(X,y\); covariance function \(k\) with hyperparameters \(\theta\); noise variance \(\sigma^2\); test point \(x_*\).

[Output]{.algo-lab} Predictive mean \(\bar m(x_*)\), variance \(v(x_*)\), and (optionally) a hyperparameter gradient step.
:::

1.  Form the Gram matrix \(K\) with \(K_{ij}=k(x_i,x_j)\) and set \(A=K+\sigma^2 I\).
2.  Compute the Cholesky factor \(A=LL^\top\) (numerically stabler than inverting \(A\)).
3.  Solve \(A\alpha=y\) by two triangular solves against \(L\).
4.  Form \(k_*=(k(x_1,x_*),\dots,k(x_m,x_*))^\top\); return \(\bar m(x_*)=k_*^\top\alpha\).
5.  Solve \(Lw=k_*\); return \(v(x_*)=k(x_*,x_*)-w^\top w\).
6.  To learn hyperparameters, compute the log marginal likelihood \(\mathcal L(\theta,\sigma^2)\) and its gradient \(\partial\mathcal L/\partial\theta_j=\tfrac12\,y^\top A^{-1}(\partial A/\partial\theta_j)A^{-1}y-\tfrac12\operatorname{tr}\!\big(A^{-1}\partial A/\partial\theta_j\big)\), take a gradient-ascent step, and repeat from step 1.
::::

::::: {.example #example-40-1}
[Example (GP posterior on three points)]{.box-title}

```python
import numpy as np

x = np.array([0.0, 1.0, 2.0])
y = np.array([1.0, 0.5, -0.5])
kernel = lambda a, b: np.exp(-((a[:, None] - b[None, :]) ** 2) / 2.0)
K = kernel(x, x)
A = K + 0.1 * np.eye(3)
kstar = kernel(x, np.array([0.5])).ravel()
alpha = np.linalg.solve(A, y)
mean = kstar @ alpha
variance = 1.0 - kstar @ np.linalg.solve(A, kstar)
_, logdet = np.linalg.slogdet(A)
lml = -0.5 * y @ alpha - 0.5 * logdet - 1.5 * np.log(2.0 * np.pi)
print(K, alpha, mean, variance, lml)
```

:::: wex
::: wex-setup
Training inputs \(x=(0,1,2)\), targets \(y=(1,\ 0.5,\ -0.5)\), Gaussian kernel \(k(x,x')=e^{-(x-x')^2/2}\) (length scale \(\ell=1\)), noise variance \(\sigma^2=0.1\). Predict at \(x_*=0.5\). The values are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-gp-ex1].
:::

1.  [Form the Gram matrix.]{.wex-op} With \(e^{-1/2}=0.6065\) and \(e^{-2}=0.1353\),

$$K=\begin{pmatrix}1&0.6065&0.1353\\0.6065&1&0.6065\\0.1353&0.6065&1\end{pmatrix},\qquad A=K+0.1\,I.$$
2.  [Solve for the coefficients.]{.wex-op} \(\alpha=A^{-1}y=(0.7321,\ 0.5046,\ -0.8228)\).
3.  [Evaluate the test covariances.]{.wex-op} \(k_*=(e^{-1/8},e^{-1/8},e^{-9/8})=(0.8825,\ 0.8825,\ 0.3247)\).
4.  [Read off the mean.]{.wex-op} \(\bar m(x_*)=k_*^\top\alpha=0.8242\).
5.  [Read off the variance.]{.wex-op} \(v(x_*)=k_{**}-k_*^\top A^{-1}k_*=1-0.9176=0.0824\), so the posterior standard deviation is \(0.287\).
6.  [Score the fit.]{.wex-op} The log marginal likelihood is \(-\tfrac12 y^\top A^{-1}y-\tfrac12\ln\det A-\tfrac{3}{2}\ln 2\pi=-3.2002\), using \(y^\top A^{-1}y=1.3958\) and \(\ln\det A=-0.509\).

**Reading.** The prediction at \(x_*=0.5\) is \(0.82\pm0.29\). The error bar is tight here because \(x_*\) sits close to two training points; it would widen in a region with no data, since the variance ignores \(y\) and tracks only the geometry of the inputs.
::::
:::::

That local calculation scales into the full posterior picture. The mean is pulled toward the observations, while the covariance shrinks wherever the kernel regards a test point as well covered. Far from data the mean returns toward its prior value and the uncertainty returns toward its prior scale.

<figure class="viz" data-figure="gp-posterior-anatomy" data-alt="A one-dimensional Gaussian process shows observations, posterior mean, posterior uncertainty, and the wider prior uncertainty. The posterior band narrows around observations and widens away from them.">
<figcaption>Covariance, observation noise, and input geometry determine where the posterior can be confident. The labels move the posterior mean, but posterior variance depends on where observations were made, not on their observed values.</figcaption>
</figure>

### The posterior is a process, not a collection of unrelated error bars {#gp-joint-posterior}

For test inputs \(X_*=(x^*_1,\ldots,x^*_r)\), define \(K_{*X}\in\mathbb R^{r\times n}\) and \(K_{**}\in\mathbb R^{r\times r}\) by evaluating the same covariance kernel. Conditioning gives the joint posterior

$$f_*\mid X,y,X_*\sim\mathcal N\!\left(
K_{*X}(K+\sigma^2I)^{-1}y,\;
K_{**}-K_{*X}(K+\sigma^2I)^{-1}K_{X*}
\right).$$

The off-diagonal entries of the posterior covariance matter. Pointwise intervals answer a marginal question at each \(x^*_j\); they do not give the probability that an entire curve lies inside all of those intervals. Nor can one sample a coherent posterior function by drawing each test coordinate independently. The joint covariance is what preserves slopes, oscillations, integrals, maxima, and other coupled functionals.

A nonzero prior mean changes the same formula by residualizing against it:

$$m_*(X_*)+K_{*X}(K+\sigma^2I)^{-1}\big(y-m(X)\big).$$

This small algebraic change has a large modeling consequence. A misspecified zero mean is not innocuous in extrapolation, because the posterior reverts to that mean where cross-covariances vanish. Explicit basis functions, trends, or a hierarchical mean model should be used when the application has a known baseline rather than asking a stationary covariance to manufacture it.

### Prediction becomes a decision only after a loss is named {#gp-decision-theory}

The posterior distribution does not select an action by itself. Under squared-error loss, the Bayes action is the posterior mean because

$$\mathbb E[(a-f_*)^2\mid y]
=(a-\mathbb E[f_*\mid y])^2+\operatorname{Var}(f_*\mid y),$$

whose first term is minimized at \(a=\mathbb E[f_*\mid y]\). Absolute-error loss selects a posterior median; asymmetric absolute loss selects a posterior quantile. If the downstream action is “inspect the system when the latent response exceeds \(c\),” the relevant output is \(P(f_*\gt c\mid y)\), not a mean with a symmetric interval. This decision-theoretic layer, explicit in GPML's regression development [@rasmussen2006, §2.4], explains why the same posterior can justify different summaries in different applications.

## Kernel ridge regression is the GP posterior mean {#krr-gp}

The coefficient vector \(\alpha=(K+\sigma^2 I)^{-1}y\) should look familiar. In [[ch:kernel-ridge-and-friends|the ridge chapter]] the kernel ridge solution was \(\alpha=(K+\lambda n I)^{-1}y\), where \(n\) is the number of training points and \(\lambda\) the regularization strength, and the fitted function was \(\hat f(x_*)=\sum_i\alpha_i k(x_i,x_*)=k_*^\top\alpha\). The GP posterior mean is the identical expression with \(\sigma^2\) in place of \(\lambda n\). This is not a coincidence; it is the MAP-equals-mean phenomenon made concrete.

:::: {.proposition #prop-40-3}
[Proposition (KRR / GP correspondence)]{.box-title}

For the convention
\(\frac1n\sum_{i=1}^n(y_i-f(x_i))^2+\lambda\|f\|_{\mathcal H_k}^2\),
kernel ridge regression with \(\lambda\gt 0\) produces the same predictor as the Gaussian process posterior mean with noise variance

$$\sigma^2=\lambda n.$$

Concretely, \(\hat f_{\mathrm{KRR}}(x_*)=k_*^\top(K+\lambda n I)^{-1}y=k_*^\top(K+\sigma^2 I)^{-1}y=\bar m(x_*)\) for every \(x_*\).

**Assumptions.** \(k\) is PSD, \(\lambda\gt0\), the KRR objective uses the
displayed \(1/n\) normalization, and the GP uses the same kernel with
independent Gaussian observation noise.
**Scope.** The scaling changes if the empirical loss is written without the factor \(1/n\). The identity concerns the posterior mean; it does not identify the frequentist sampling distribution of KRR with the GP posterior.
**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

With additive Gaussian noise the negative log posterior in coefficient space is, up to constants,

$$-\ln p(t\mid y)=\frac{1}{2\sigma^2}\|y-K\alpha\|^2+\frac12\alpha^\top K\alpha,$$

writing \(t=K\alpha\). This is the regularized least-squares objective whose stationarity condition is \(K[(K+\sigma^2 I)\alpha-y]=0\), solved by \(\alpha=(K+\sigma^2 I)^{-1}y\). The ridge objective \(\tfrac1n\|K\alpha-y\|^2+\lambda\,\alpha^\top K\alpha\) has stationarity condition \(K[(K+\lambda n I)\alpha-y]=0\), solved by \(\alpha=(K+\lambda n I)^{-1}y\). The two coincide iff \(\sigma^2=\lambda n\), and since the predictor is \(k_*^\top\alpha\) in both cases, the functions agree everywhere. Because the noise is Gaussian the posterior is Gaussian, so its mode and mean coincide, which is why the MAP fit (ridge) equals the posterior mean (GP). [\(\square\)]{.qed}
::::

::::: {.example #example-40-2}
[Example (the two fits agree)]{.box-title}

```python
import numpy as np

x = np.array([0.0, 1.0, 2.0])
y = np.array([1.0, 0.0, -1.0])
kernel = lambda a, b: np.exp(-((a[:, None] - b[None, :]) ** 2) / 2.0)
K = kernel(x, x)
ridge, n = 0.05, x.size
alpha_krr = np.linalg.solve(K + ridge * n * np.eye(n), y)
alpha_gp = np.linalg.solve(K + 0.15 * np.eye(n), y)
for query in [0.5, 1.5]:
    kstar = kernel(x, np.array([query])).ravel()
    print(query, kstar @ alpha_krr, kstar @ alpha_gp)
assert np.allclose(alpha_krr, alpha_gp)
```

:::: wex
::: wex-setup
Same kernel \(k(x,x')=e^{-(x-x')^2/2}\) on \(x=(0,1,2)\), now with \(y=(1,0,-1)\). Ridge parameter \(\lambda=0.05\), \(n=3\), so \(\lambda n=0.15\); set the GP noise variance to the matched value \(\sigma^2=0.15\). The values are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-gp-ex2].
:::

1.  [Solve the ridge system.]{.wex-op} \(\alpha_{\mathrm{KRR}}=(K+0.15\,I)^{-1}y=(0.9855,\ 0,\ -0.9855)\). The middle coefficient vanishes because both \(y\) and the geometry are antisymmetric about \(x=1\).
2.  [Solve the GP system.]{.wex-op} \(\alpha_{\mathrm{GP}}=(K+0.15\,I)^{-1}y=(0.9855,\ 0,\ -0.9855)\); the maximum coefficient difference is exactly \(0\).
3.  [Predict at two test points.]{.wex-op} At \(x_*=0.5\): \(\hat f_{\mathrm{KRR}}=\bar m=0.549782\). At \(x_*=1.5\): \(\hat f_{\mathrm{KRR}}=\bar m=-0.549782\). The differences are \(0\) to machine precision.

**Reading.** The two algorithms, derived from a loss-plus-penalty on one side and from Bayes' rule on the other, return the same point predictor to the last digit. The GP model also specifies joint posterior covariances, a marginal likelihood, and predictions for nonlinear functionals; those additions inherit the assumptions of the probabilistic model.
::::
:::::

### The spectral view explains both smoothing and uncertainty {#gp-equivalent-kernel}

The equality with kernel ridge regression is more informative after diagonalizing the training Gram matrix. Let \(K=U\operatorname{diag}(\lambda_1,\ldots,\lambda_n)U^\top\). At the training inputs, the posterior mean vector is

$$\bar m_X=K(K+\sigma^2I)^{-1}y
=U\operatorname{diag}\!\left(\frac{\lambda_j}{\lambda_j+\sigma^2}\right)U^\top y.$$

Each empirical eigenmode is multiplied by a shrinkage factor \(s_j=\lambda_j/(\lambda_j+\sigma^2)\). Large-variance modes survive; modes with \(\lambda_j\ll\sigma^2\) are suppressed. This is the equivalent-kernel perspective of GPML [@rasmussen2006, §§2.6, 7.1]: the posterior mean is a data- and noise-dependent smoother, not merely “a sum of kernels.” Its effective degrees of freedom are

$$\operatorname{df}_{\mathrm{eff}}
=\operatorname{tr}\!\left(K(K+\sigma^2I)^{-1}\right)
=\sum_{j=1}^n\frac{\lambda_j}{\lambda_j+\sigma^2},$$

the finite-sample counterpart of the effective dimension in [[ch:mercer-and-rates|the Mercer chapter]].

The latent posterior covariance at the training sites has the complementary filter

$$\Sigma_{X\mid y}
=K-K(K+\sigma^2I)^{-1}K
=U\operatorname{diag}\!\left(\frac{\lambda_j\sigma^2}{\lambda_j+\sigma^2}\right)U^\top.$$

Mean and uncertainty are therefore coupled mode by mode. A direction that the data can estimate relative to the noise is retained in the mean and reduced in posterior variance; a direction whose prior variance is tiny relative to the noise is shrunk in the mean, but it also had little absolute variance to begin with. This is why a small posterior variance is not synonymous with “the data learned this direction”: the prior may simply have ruled it out.

For a two-mode design with \((\lambda_1,\lambda_2)=(9,0.1)\) and \(\sigma^2=1\), the mean keeps \(90\%\) of the first mode but only \(1/11\) of the second. The corresponding posterior variances are \(0.9\) and \(1/11\). The second variance is numerically small, yet the data barely transmitted its observed coefficient into the posterior mean. A diagnostic must therefore inspect both the shrinkage factor and the prior scale, not the posterior variance alone.

## The marginal likelihood and hyperparameter learning {#marginal-likelihood}

A kernel comes with knobs: the length scale of a Gaussian covariance, the noise level \(\sigma^2\), an overall amplitude. In risk minimization these are set by cross-validation. The Bayesian framework offers an internal alternative. Collect the hyperparameters into \(\theta\) and integrate the latent function out of the joint distribution; because the prior on \(t\) is Gaussian and the noise is Gaussian, the marginal distribution of the data is Gaussian too, \(y\mid\theta\sim\mathcal N(0,\,K_\theta+\sigma^2 I)\). Its density, viewed as a function of \(\theta\), is the marginal likelihood, or evidence, and maximizing it is the type-II maximum likelihood estimate (MacKay 1992).

:::: {.definition #def-40-4}
[Definition (log marginal likelihood)]{.box-title}

With \(A=K_\theta+\sigma^2 I\), the log marginal likelihood is

$$\mathcal L(\theta)=\ln p(y\mid X,\theta)=-\frac12\,y^\top A^{-1}y-\frac12\ln\det A-\frac{n}{2}\ln(2\pi).$$
::::

The quadratic term \(-\tfrac12 y^\top A^{-1}y\) rewards fit in the covariance metric. The log-determinant term \(-\tfrac12\ln\det A\) measures the volume of distributions the covariance spreads probability over; relative to another model it often acts as an Occam factor. It is not a universal monotone penalty on “flexibility,” because changing a length scale can enlarge some eigenvalues while shrinking others. The useful statement is exact and spectral:

$$\mathcal L(\theta)
=-\frac12\sum_{j=1}^n\left[
\frac{(u_j^\top y)^2}{\lambda_j(\theta)+\sigma^2}
+\ln\big(\lambda_j(\theta)+\sigma^2\big)
+\ln(2\pi)\right].
$$

The evidence rewards a spectrum that allocates variance in directions where \(y\) actually has energy and charges unused variance through the logarithm. Differentiating with respect to a hyperparameter \(\theta_j\) gives, by the standard identities \(\partial\ln\det A=\operatorname{tr}(A^{-1}\partial A)\) and \(\partial A^{-1}=-A^{-1}(\partial A)A^{-1}\),

$$\frac{\partial\mathcal L}{\partial\theta_j}=\frac12\,y^\top A^{-1}\frac{\partial A}{\partial\theta_j}A^{-1}y-\frac12\operatorname{tr}\!\Big(A^{-1}\frac{\partial A}{\partial\theta_j}\Big),$$

which is the gradient step in the GP regression algorithm above. Gradient ascent or a Newton method then finds a (local) evidence maximum. The cost is dominated by the \(O(n^3)\) factorization of \(A\), which is why scalable approximations, sparse and low-rank, are the subject of [[ch:large-scale-kernels|the large-scale chapter]].

A refinement of this idea is automatic relevance determination. Give each input dimension its own length scale, \(k(x,x')=\exp\!\big(-\sum_{d}(x_d-x'_d)^2/2\ell_d^2\big)\), and optimize the evidence over them. At an evidence optimum, a large \(\ell_d\) can indicate that variation along coordinate \(d\) is unsupported by this dataset and model. It is not a theorem that every irrelevant variable will be removed: correlated inputs, local optima, mean misspecification, and an overly rigid covariance family can all distort the length scales. The same evidence principle, applied to basis-specific precisions, is what makes the Relevance Vector Machine sparse.

### Evidence is a model score, not an oracle {#gp-model-selection-diagnostics}

Hyperparameter learning adds an outer inference problem to the inner Gaussian conditioning problem. The posterior formulas above are exact only conditional on \(\theta\); plugging in the maximizer \(\hat\theta\) discards hyperparameter uncertainty. Three diagnostics keep the distinction visible.

First, inspect profiles or multiple optimization starts. Amplitude, noise, and length scale can trade off along shallow ridges, and a large gradient norm or an indefinite observed Hessian means the reported optimum is not yet interpretable. Second, compare evidence selection with predictive validation. Leave-one-out log predictive density asks how well each point is predicted without itself; marginal likelihood asks how plausible the entire observed vector is under the joint model. GPML develops both precisely because they answer different questions [@rasmussen2006, ch. 5]. Third, test calibration on held-out or time-ordered data. If standardized residuals

$$r_i=\frac{y_i-\bar m_{-i}(x_i)}
{\sqrt{v_{-i}(x_i)+\sigma^2}}$$

are too dispersed, too concentrated, or structured in \(x\), the fitted covariance or likelihood is missing something even when its evidence is locally maximal.

A fully Bayesian alternative places a prior on \(\theta\) and integrates it:

$$p(f_*\mid y)=\int p(f_*\mid y,\theta)\,p(\theta\mid y)\,d\theta.$$

This mixture is generally non-Gaussian and wider than the plug-in posterior in directions where \(\theta\) is weakly identified. Optimization, Laplace approximation in hyperparameter space, quadrature, or Monte Carlo are different approximations to this outer integral and should be named as such.

::::: {.example #example-40-7}
[Example (evidence and leave-one-out select different questions)]{.box-title}

```python
import numpy as np

x = np.arange(-2.0, 3.0)
y = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
for ell in [0.5, 1.0, 5.0]:
    K = np.exp(-((x[:, None] - x[None, :]) ** 2) / (2.0 * ell**2))
    A = K + 0.1 * np.eye(5)
    Q = np.linalg.inv(A)
    alpha = Q @ y
    lml = -0.5 * y @ alpha - 0.5 * np.linalg.slogdet(A)[1] \
          - 2.5 * np.log(2.0 * np.pi)
    loo_var = 1.0 / np.diag(Q)
    loo_mean = y - alpha / np.diag(Q)
    loo = np.sum(-0.5 * ((y - loo_mean)**2 / loo_var
                         + np.log(loo_var) + np.log(2.0 * np.pi)))
    print(ell, lml, loo)
```

:::: wex
::: wex-setup
Take \(x=(-2,-1,0,1,2)\), \(y=(0,0,1,0,0)\), a unit-amplitude
squared-exponential covariance, and \(\sigma^2=0.1\). Compare length scales
\(\ell\in\{0.5,1,5\}\). The calculation is reproduced by
`checks/ch-gp-depth.py`.
:::

For \(A=K+\sigma^2I\), write \(Q=A^{-1}\) and \(\alpha=Qy\). Exact
leave-one-out identities give

$$\mu_{-i}(x_i)=y_i-\frac{\alpha_i}{Q_{ii}},\qquad
\operatorname{Var}(y_i\mid y_{-i})=\frac1{Q_{ii}}.$$

| \(\ell\) | log marginal likelihood | summed LOO log predictive density |
|---:|---:|---:|
| \(0.5\) | \(-5.2711\) | \(-5.2547\) |
| \(1\) | **\(-5.1456\)** | \(-5.2561\) |
| \(5\) | \(-5.3871\) | **\(-5.1198\)** |

**Reading.** Evidence selects \(\ell=1\) among these candidates because it
scores the joint five-dimensional observation. LOO selects \(\ell=5\)
because it scores five conditional prediction problems. Neither criterion
is “wrong,” and neither result proves held-out calibration. The intended
deployment question decides which score is relevant.
::::
:::::

## Learning curves: where eigenvalue decay becomes statistical rate {#gp-learning-curves}

The finite spectral filter explains one dataset. A learning curve asks how
the prediction error changes with the information budget. The cleanest exact
calculation uses a Gaussian sequence experiment. Let

$$z_j=\theta_j+\frac{\sigma}{\sqrt n}\xi_j,\qquad
\xi_j\stackrel{\mathrm{iid}}{\sim}\mathcal N(0,1),\qquad
\theta_j\sim\mathcal N(0,\lambda_j).$$

This is GP regression after diagonalizing an idealized observation operator:
\(\lambda_j\) is prior variance in mode \(j\), while \(n/\sigma^2\) is the
precision supplied by the data. Conjugacy gives

$$\theta_j\mid z_j\sim\mathcal N\left(
\frac{n\lambda_j}{\sigma^2+n\lambda_j}z_j,\;
\frac{\lambda_j\sigma^2}{\sigma^2+n\lambda_j}
\right).$$

Under the stated prior and observation model, the integrated Bayes
mean-squared error of the posterior mean equals the sum of posterior
variances,

$$R_n=\sum_{j\ge1}
\frac{\lambda_j\sigma^2}{\sigma^2+n\lambda_j}
=\frac{\sigma^2}{n}\,
\mathcal N\left(\frac{\sigma^2}{n}\right),$$

where \(\mathcal N(\eta)=\sum_j\lambda_j/(\lambda_j+\eta)\) is the effective
dimension. This identity is exact for the sequence model. Transferring it to
random-design GP regression requires concentration of the empirical
covariance operator and is not automatic.

If \(\lambda_j\asymp j^{-2a}\) with \(a\gt\tfrac12\), modes up to
\(J_n\asymp(n/\sigma^2)^{1/(2a)}\) are data dominated. Splitting the sum at
\(J_n\) yields

$$R_n\asymp n^{-(2a-1)/(2a)}.$$

If \(\lambda_j\asymp e^{-cj}\), only \(O(\log n)\) modes exceed the noise
floor and \(R_n\asymp(\log n)/n\), up to constants and the noise scale.
These are average-case, well-specified sequence-model statements. They are
not worst-case minimax rates for every truth, nor do they prove calibrated
credible intervals. GPML's average learning-curve calculations
[@rasmussen2006, §7.3] and [[ch:mercer-and-rates|the Mercer chapter]] develop
the corresponding operator viewpoints.

::::: {.example #example-40-6}
[Example (two spectra, two learning curves)]{.box-title}

```python
import numpy as np

modes = np.arange(1.0, 10001.0)
spectra = {"polynomial": modes**-2, "exponential": np.exp(-modes / 5.0)}
for name, eigenvalues in spectra.items():
    for sample_size in [100.0, 1000.0, 10000.0]:
        risk = np.sum(eigenvalues / (1.0 + sample_size * eigenvalues))
        scaling = (risk * np.sqrt(sample_size) if name == "polynomial"
                   else risk * sample_size / np.log(sample_size))
        print(name, sample_size, risk, scaling)
```

:::: wex
::: wex-setup
Set \(\sigma^2=1\), truncate to \(10{,}000\) modes, and compute
\(R_n=\sum_j\lambda_j/(1+n\lambda_j)\). Compare the polynomial spectrum
\(\lambda_j=j^{-2}\) with the exponential spectrum
\(\lambda_j=e^{-j/5}\). The calculation is reproduced by
`checks/ch-gp-depth.py`.
:::

1.  [Polynomial spectrum.]{.wex-op} At
    \(n=(10^2,10^3,10^4)\), the risks are
    \((0.1520,0.0491,0.0156)\). Multiplying by \(n^{1/2}\) gives
    \((1.520,1.552,1.556)\), exposing the predicted \(n^{-1/2}\) rate.
2.  [Exponential spectrum.]{.wex-op} At the same budgets, the risks are
    \((0.2258,0.0340,0.00456)\). Multiplying by \(n/\log n\) gives
    \((4.903,4.928,4.946)\), approaching a constant as the
    \((\log n)/n\) calculation predicts.

**Reading.** “The GP learns faster” is incomplete. The rate comes from a
specified spectrum, noise scale, truth/prior relationship, loss, and
observation regime.
::::
:::::

## Fixed-domain asymptotics: prediction can succeed while parameters do not {#gp-microergodicity}

Asymptotics depend on how the design grows. In *increasing-domain*
asymptotics, observations spread over an expanding region and reveal
long-range dependence. In *fixed-domain* or *infill* asymptotics, points
become dense inside one bounded region. The second regime creates a
surprising identifiability boundary.

Consider a stationary Matérn covariance on a bounded subset of
\(\mathbb R^d\), \(d\le3\), with known smoothness \(\nu\), marginal variance
\(\sigma_f^2\), and range \(\rho\). Under a common Matérn parameterization,
Zhang (2004) showed that fixed-domain observations cannot consistently
estimate \(\sigma_f^2\) and \(\rho\) separately. The microergodic
combination

$$c=\frac{\sigma_f^2}{\rho^{2\nu}}$$

is consistently estimable, while distinct parameter pairs with the same
\(c\) can induce equivalent Gaussian measures on the bounded domain
[@zhang2004microergodic]. No estimator can reliably distinguish equivalent
measures from one increasingly dense realization.

This is not a counsel to abandon likelihood optimization. It changes what
the optimum is allowed to mean. A long ridge in \((\sigma_f^2,\rho)\) may be
the statistical geometry of the problem, not an optimizer failure. Moreover,
kriging predictors built from parameter pairs on the same microergodic
ridge can be asymptotically equivalent even though the individual parameter
estimates wander. Prediction consistency and parameter consistency are
different guarantees.

The failure witness is a common experimental design: one environmental field
measured at ever denser locations inside the same small plot. Reporting a
high-precision range estimate from that design ignores the asymptotic
regime. Replicated fields or an expanding spatial domain provide different
information. [[ch:spatial-and-spatiotemporal-kernels|The spatial chapter]]
returns to this distinction when covariance parameters have physical
interpretations.

## Posterior contraction is not credible-set coverage {#gp-contraction-coverage}

Let \(f_0\) be a fixed data-generating function and \(\Pi(\cdot\mid Y^{(n)})\)
the posterior. A sequence \(\varepsilon_n\downarrow0\) is a posterior
contraction rate in a norm \(\|\cdot\|\) if, for sufficiently large \(M\),

$$\Pi\left(f:\|f-f_0\|\gt M\varepsilon_n\mid Y^{(n)}\right)
\longrightarrow 0$$

in \(P_{f_0}\)-probability. The statement must name the observation model,
design, norm, prior scaling, and class containing \(f_0\). For Gaussian
priors a central quantity is the concentration function

$$\phi_{f_0}(\varepsilon)=
\inf_{\substack{h\in\mathcal H_k\\\|h-f_0\|\lt\varepsilon}}
\|h\|_{\mathcal H_k}^2
-\log P(\|W\|\lt\varepsilon),\qquad W\sim\mathcal{GP}(0,k).$$

The first term measures how expensively the RKHS approximates the truth; the
second measures prior small-ball mass. Bounds of the form
\(\phi_{f_0}(\varepsilon_n)\lesssim n\varepsilon_n^2\), together with testing
and model-specific conditions, lead to contraction rates
[@vanderVaart2008gpcontraction]. This formula exposes why “smooth kernel”
does not determine a rate by itself: the truth's approximation cost and the
prior's local mass both matter.

Now let \(C_n\) be a set with posterior probability \(1-\alpha\). That makes
\(C_n\) a Bayesian credible set. Frequentist coverage asks a different
question,

$$P_{f_0}(f_0\in C_n)\stackrel{?}{\longrightarrow}1-\alpha.$$

Contraction controls the radius of posterior mass around \(f_0\); it does
not ensure that a data-dependent center and radius cover \(f_0\) with the
nominal repeated-sampling frequency. Adaptive credible sets can have good
coverage over self-similar or polished-tail subclasses and fail badly for
other truths in the same broad smoothness scale
[@szabo2015coverage]. Inflation, undersmoothing, self-similarity conditions,
or external calibration may repair coverage, but each changes the claim.

Keep the four statements separate:

| Statement | Random object | Probability law | What it guarantees |
|---|---|---|---|
| posterior variance | \(f\mid Y\) | fitted GP model | conditional spread under the model |
| contraction | posterior mass around fixed \(f_0\) | repeated data from \(f_0\) | concentration rate in a named norm |
| credible-set probability | \(f\mid Y\) | fitted posterior | posterior mass of the reported set |
| frequentist coverage | data-dependent \(C_n(Y)\) | repeated data from \(f_0\) | long-run inclusion of the fixed truth |

The reliability chapter adds distribution shift and conformal guarantees;
neither can be inferred from a GP covariance alone.

## Gaussian process classification {#gp-classification}

For classification the target is a label \(y\in\{-1,+1\}\), and the additive-Gaussian story breaks: a Gaussian latent value cannot be a Bernoulli label. We keep the Gaussian process prior on a latent function \(t(x)\) but squash it through a non-Gaussian link, typically the logistic \(P(y=1\mid t)=\sigma(t)=(1+e^{-t})^{-1}\) or the probit \(P(y=1\mid t)=\Phi(t)\). The posterior over the latent values is now

$$p(t\mid X,Y)\ \propto\ \Big[\prod_{i=1}^m p(y_i\mid t(x_i))\Big]\exp\!\Big(-\tfrac12 t^\top K^{-1}t\Big),$$

which is no longer Gaussian, and the predictive integral has no closed form. Two approximations dominate.

The *Laplace approximation* (Williams and Rasmussen 1996) fits a Gaussian to the posterior at its mode. Writing the negative log posterior as \(\Psi(t)=-\sum_i\ln p(y_i\mid t(x_i))+\tfrac12 t^\top K^{-1}t\), one finds the mode by Newton's method: with \(c\) the gradient of the log likelihood and \(C=-\nabla^2\ln p(y\mid t)\) its (diagonal) negative Hessian, the update is

$$t_{\mathrm{new}}=(K^{-1}+C)^{-1}(C\,t_{\mathrm{old}}+c),$$

equivalent in coefficient form to \(\alpha_{\mathrm{new}}=(KC+I)^{-1}(KC\,\alpha_{\mathrm{old}}+c)\). The curvature \(C\) at the mode then supplies the covariance of the Gaussian approximation, from which predictive probabilities follow. Because the logistic likelihood is log-concave, \(\Psi\) is convex and the mode is unique. A second option replaces the intractable posterior with the best matching Gaussian in a moment-matching sense rather than a curvature sense; expectation propagation is the standard such scheme, and it is generally more accurate than Laplace for the probit link, at comparable cost. A variational method of Jaakkola and Jordan, which sandwiches the logistic between tractable quadratic bounds, is a third route (Schölkopf and Smola 2002). In every case the approximation reduces classification to a sequence of the same linear-algebra operations that regression required, an \(n\times n\) solve per iteration.

## Laplace mechanics: from mode to prediction {#laplace-mechanics}

The previous section stated the Laplace idea in one line, fit a Gaussian at the mode, and moved on. Since this approximation is the workhorse of Gaussian process classification, it deserves to be opened up: which Gaussian is being fitted, how the computation is organized so that it never forms \(K^{-1}\), and what the classifier finally reports. The program was carried through for GP classifiers, including the multiclass case with a softmax link [@williams1998classification]; the numerically stable formulation below follows [@rasmussen2006].

Start from the negative log posterior \(\Psi(t)=-\ln p(Y\mid t)+\tfrac12 t^\top K^{-1}t\), up to an additive constant, and let \(\hat t\) be its minimizer, found by the Newton iteration of the previous section. Because the gradient vanishes at the minimizer, the Taylor expansion of \(\Psi\) around \(\hat t\) has no linear term:

$$\Psi(t)=\Psi(\hat t)+\tfrac12\,(t-\hat t)^\top\big(K^{-1}+C\big)(t-\hat t)+O\big(\|t-\hat t\|^3\big),$$

where \(C=-\nabla^2\ln p(Y\mid\hat t)\) is the likelihood curvature at the mode, diagonal because the likelihood factorizes over points. Dropping the cubic remainder and exponentiating leaves a Gaussian, and that Gaussian is the Laplace approximation:

$$p(t\mid X,Y)\ \approx\ q(t\mid X,Y)=\mathcal N\big(\hat t,\ (K^{-1}+C)^{-1}\big).$$

The covariance adds two curvatures: the prior contributes \(K^{-1}\) and each observation contributes \(C_{ii}\) on the diagonal. For the logistic link \(C_{ii}=\pi_i(1-\pi_i)\le\tfrac14\) with \(\pi_i=\sigma(\hat t_i)\), so a single binary label is only mildly informative, and most informative exactly where the current prediction is undecided at \(\pi_i=\tfrac12\).

The stationarity condition \(\nabla\Psi(\hat t)=0\) reads \(\hat t=K\hat c\) with \(\hat c=\nabla\ln p(Y\mid\hat t)\): the mode is once more a kernel expansion over the training points, the classification counterpart of \(\alpha=(K+\sigma^2I)^{-1}y\) in regression. The coefficients are readable. For the logistic link, \(\hat c_i=y_i\,\sigma(-y_i\hat t_i)\), which is the probability the model currently assigns to the *wrong* class at point \(i\), signed by the label. Confidently fitted points contribute almost nothing to the expansion; uncertain and misfitted points carry it. The Laplace classifier has soft support vectors: the hard sparsity of the SVM is replaced by an exponential taper of influence.

Numerics next. The matrix \(K^{-1}+C\) should never be assembled from \(K^{-1}\), whose computation is exactly what ill-conditioned kernels punish. The right object is

$$B=I+C^{1/2}KC^{1/2},$$

whose eigenvalues lie in \([1,\ 1+\tfrac14\lambda_{\max}(K)]\) for the logistic link: bounded below by one no matter how close to singular \(K\) is. By the Woodbury identity, \((K^{-1}+C)^{-1}=K-KC^{1/2}B^{-1}C^{1/2}K\), so every Newton step and every predictive quantity reduces to Cholesky solves against \(B\), at \(O(n^3)\) per iteration and only a handful of iterations, since the log-concave likelihood makes \(\Psi\) convex.

Prediction assembles the same two Gaussian-conditioning moves as regression, now under the approximate posterior. The latent value at \(x_*\) has approximate mean and variance

$$\mu_*=k_*^\top\hat c,\qquad v_*=k_{**}-k_*^\top C^{1/2}B^{-1}C^{1/2}k_*,$$

the mean following from \(\mathbb E_q[f_*]=k_*^\top K^{-1}\hat t=k_*^\top\hat c\), the variance from \((K+C^{-1})^{-1}=C^{1/2}B^{-1}C^{1/2}\). The reported class probability then averages the link over the latent uncertainty,

$$\bar\pi_*=\int\sigma(f)\,\mathcal N\big(f\mid\mu_*,v_*\big)\,df,$$

a one-dimensional integral: closed form \(\bar\pi_*=\Phi\big(\mu_*/\sqrt{1+v_*}\big)\) for the probit link, a short quadrature for the logistic. The averaging is not decoration. Plugging \(\mu_*\) straight into the sigmoid ignores \(v_*\) and overstates confidence far from the data; averaging pulls \(\bar\pi_*\) toward \(\tfrac12\) exactly where the latent variance says the model does not know.

Finally, the same expansion prices the whole fit: integrating the Gaussian approximation gives an approximate evidence

$$\ln q(Y\mid X,\theta)=-\tfrac12\,\hat t^\top\hat c+\ln p(Y\mid\hat t)-\tfrac12\ln\det B,$$

using \(\hat t^\top K^{-1}\hat t=\hat t^\top\hat c\) at the mode. The three terms replay the regression marginal likelihood: a prior charge, a data fit, and a determinant acting as the Occam factor. Ascending this quantity tunes length scales and amplitudes for classification exactly as the exact evidence did for regression; the multiclass softmax makes \(C\) block structured but leaves the pipeline intact [@williams1998classification]. The same mechanics reappear wherever a Gaussian prior meets a factorized non-Gaussian likelihood, from robust regression to point-process intensity models.

:::: {.algorithm #algo-40-2}
[Algorithm (Laplace GP classification)]{.box-title}

::: algo-io
[Input]{.algo-lab} Gram matrix \(K\); labels \(y\in\{-1,+1\}^n\); link likelihood \(p(y_i\mid t_i)\); test point \(x_*\).

[Output]{.algo-lab} Predictive class probability \(\bar\pi_*\); approximate evidence \(\ln q(Y\mid X,\theta)\).
:::

1.  Initialize \(t=0\).
2.  Compute the gradient \(c=\nabla\ln p(Y\mid t)\) and curvature \(C=-\nabla^2\ln p(Y\mid t)\); form \(B=I+C^{1/2}KC^{1/2}\) and its Cholesky factor.
3.  Newton step: set \(b=Ct+c\) and update \(t\leftarrow K\big(b-C^{1/2}B^{-1}C^{1/2}Kb\big)\).
4.  Repeat steps 2 and 3 until \(\Psi\) stops decreasing; call the results \(\hat t\), \(\hat c\), \(\hat C\), \(\hat B\).
5.  Predict: \(\mu_*=k_*^\top\hat c\), \(v_*=k_{**}-k_*^\top\hat C^{1/2}\hat B^{-1}\hat C^{1/2}k_*\), and \(\bar\pi_*=\int\sigma(f)\,\mathcal N(f\mid\mu_*,v_*)\,df\) (probit: \(\Phi\big(\mu_*/\sqrt{1+v_*}\big)\)).
6.  Report \(\ln q(Y\mid X,\theta)=-\tfrac12\hat t^\top\hat c+\ln p(Y\mid\hat t)-\tfrac12\ln\det\hat B\) for hyperparameter ascent.
::::

### Expectation propagation matches moments instead of curvature {#gp-expectation-propagation}

Laplace asks what the posterior looks like infinitesimally near its mode.
Expectation propagation (EP) asks a different local question: which Gaussian
has the same first two moments as a one-factor correction of the current
approximation? Write

$$q(f)\propto\mathcal N(f\mid0,K)
\prod_{i=1}^n \tilde Z_i
\exp\left(-\tfrac12\tilde\tau_i f_i^2+\tilde\nu_i f_i\right).$$

To update site \(i\), divide its Gaussian factor out of the marginal
\(q(f_i)\), obtaining a cavity
\(q_{-i}(f_i)=\mathcal N(m_{-i},v_{-i})\). Form the tilted distribution

$$\widehat p_i(f_i)\propto
p(y_i\mid f_i)\,q_{-i}(f_i),$$

compute its normalization, mean, and variance, and choose
\((\tilde\tau_i,\tilde\nu_i)\) so the new marginal \(q(f_i)\) matches those
two moments. Cycle over sites until the moments stabilize.

For a probit likelihood \(p(y_i\mid f_i)=\Phi(y_if_i)\), set

$$z_i=\frac{y_im_{-i}}{\sqrt{1+v_{-i}}},\qquad
r_i=\frac{\phi(z_i)}{\Phi(z_i)}.$$

The tilted moments are closed form:

$$\widehat m_i=m_{-i}
+\frac{y_iv_{-i}}{\sqrt{1+v_{-i}}}r_i,$$

$$\widehat v_i=v_{-i}
-\frac{v_{-i}^2}{1+v_{-i}}r_i(r_i+z_i).$$

The new site natural parameters are
\(\tilde\tau_i=\widehat v_i^{-1}-v_{-i}^{-1}\) and
\(\tilde\nu_i=\widehat m_i/\widehat v_i-m_{-i}/v_{-i}\). Damping replaces
each new site by a convex combination with its old value. EP often gives
better posterior moments than Laplace for GP classification
[@kuss2005gpc], but “often” is empirical, not an ordering theorem. EP can
oscillate, produce an invalid cavity, or require negative site precision for
non-log-concave factors. A convergence cap, positive-cavity check, damping
policy, and comparison to quadrature or sampling on small problems belong
in the algorithm contract.

::::: {.example #example-40-8}
[Example (Laplace and EP against direct quadrature)]{.box-title}

```python
import math
import numpy as np

normal_cdf = np.vectorize(
    lambda z: 0.5 * (1.0 + math.erf(z / np.sqrt(2.0)))
)
x, y = np.array([0.0, 1.0]), np.array([-1.0, 1.0])
K = np.exp(-((x[:, None] - x[None, :]) ** 2) / (2.0 * 0.8**2))
nodes, weights = np.polynomial.hermite.hermgauss(60)
L = np.linalg.cholesky(K)
samples, posterior_weights = [], []
for i, a in enumerate(nodes):
    for j, b in enumerate(nodes):
        f = np.sqrt(2.0) * L @ np.array([a, b])
        weight = weights[i] * weights[j] / np.pi * np.prod(normal_cdf(y * f))
        samples.append(f)
        posterior_weights.append(weight)
samples = np.asarray(samples)
posterior_weights = np.asarray(posterior_weights)
posterior_weights /= posterior_weights.sum()
exact_mean = posterior_weights @ samples
centered = samples - exact_mean
exact_covariance = np.einsum("n,ni,nj->ij", posterior_weights, centered, centered)
normal_pdf = lambda z: np.exp(-0.5*z*z)/np.sqrt(2*np.pi)

# Laplace: Newton updates at the posterior mode.
Kinv = np.linalg.inv(K)
laplace_mean = np.zeros(2)
for _ in range(100):
    z = y*laplace_mean
    ratio = normal_pdf(z)/normal_cdf(z)
    W = ratio*(ratio+z)
    step = np.linalg.solve(
        Kinv+np.diag(W), y*ratio-Kinv@laplace_mean)
    laplace_mean += step
    if np.max(np.abs(step)) < 1e-13: break
laplace_cov = np.linalg.inv(Kinv+np.diag(W))

# EP: remove one Gaussian site, match the tilted probit moments, and damp.
site_tau=np.zeros(2); site_nu=np.zeros(2)
ep_cov=K.copy(); ep_mean=np.zeros(2)
for _ in range(200):
    old=ep_mean.copy()
    for i in range(2):
        cavity_tau=1/ep_cov[i,i]-site_tau[i]
        cavity_var=1/cavity_tau
        cavity_mean=(ep_mean[i]/ep_cov[i,i]-site_nu[i])/cavity_tau
        z=y[i]*cavity_mean/np.sqrt(1+cavity_var)
        ratio=normal_pdf(z)/normal_cdf(z)
        tilted_mean=cavity_mean+y[i]*cavity_var/np.sqrt(1+cavity_var)*ratio
        tilted_var=cavity_var-cavity_var**2/(1+cavity_var)*ratio*(ratio+z)
        new_tau=1/tilted_var-cavity_tau
        new_nu=tilted_mean/tilted_var-cavity_mean/cavity_var
        site_tau[i]=.7*new_tau+.3*site_tau[i]
        site_nu[i]=.7*new_nu+.3*site_nu[i]
        ep_cov=np.linalg.inv(Kinv+np.diag(site_tau))
        ep_mean=ep_cov@site_nu
    if np.max(np.abs(ep_mean-old)) < 1e-13: break

# Propagate all three posteriors to the same test-point probability.
kstar=np.exp(-((x-1.5)**2)/(2*.8**2)); beta=np.linalg.solve(K,kstar)
conditional_var=1-kstar@beta
def probability(mean,cov):
    mu=kstar@np.linalg.solve(K,mean)
    var=conditional_var+beta@cov@beta
    return float(normal_cdf(mu/np.sqrt(1+var)))
exact_probability=float(np.sum(
    posterior_weights*normal_cdf((samples@beta)/np.sqrt(1+conditional_var))))
print(exact_mean,np.diag(exact_covariance),exact_probability)
print(laplace_mean,np.diag(laplace_cov),probability(laplace_mean,laplace_cov))
print(ep_mean,np.diag(ep_cov),probability(ep_mean,ep_cov))
assert np.max(np.abs(ep_mean-exact_mean)) < 3e-5
```

:::: wex
::: wex-setup
Use two inputs \(x=(0,1)\), labels \(y=(-1,+1)\), a
squared-exponential covariance with \(\ell=0.8\), and a probit likelihood.
Predict \(P(y_*=+1)\) at \(x_*=1.5\). A \(60\times60\) Gauss-Hermite rule
provides a direct two-dimensional benchmark. The calculation is reproduced
by `checks/ch-gp-depth.py`.
:::

| Method | posterior mean at training points | marginal variances | \(P(y_*=+1)\) | log score for \(y_*=+1\) |
|---|---|---|---:|---:|
| direct quadrature | \((-0.3586,0.3586)\) | \((0.6265,0.6265)\) | \(0.6260\) | \(-0.4685\) |
| Laplace | \((-0.3265,0.3265)\) | \((0.6095,0.6095)\) | \(0.6162\) | \(-0.4842\) |
| EP | \((-0.3586,0.3586)\) | \((0.6268,0.6268)\) | \(0.6268\) | \(-0.4671\) |

**Reading.** In this tiny log-concave problem EP nearly matches the first
two exact moments; Laplace is tighter because it reads only curvature at the
mode. This example validates one problem, not a universal ranking. Its
purpose is to teach what the approximations preserve.
::::
:::::

### Multiclass likelihoods couple classes at each observation {#gp-multiclass}

For \(C\) classes, give each class a latent function \(f_c\) and use the
softmax likelihood

$$p(y_i=c\mid f_i)=
\frac{\exp(f_{ic})}{\sum_{r=1}^C\exp(f_{ir})}.$$

If \(\pi_i\) is the softmax probability vector, the negative Hessian of the
log likelihood for observation \(i\) is

$$W_i=\operatorname{diag}(\pi_i)-\pi_i\pi_i^\top.$$

This block is PSD, has null vector \(\mathbf1\), and couples every pair of
classes. Adding a constant to all \(C\) logits changes no probability, so
the likelihood alone cannot identify that direction; the prior or an
explicit reference-class constraint supplies the gauge. The full Laplace
curvature is \(K^{-1}+W\), with \(W\) block diagonal over observations but
not over classes. Treating \(C\) one-versus-rest classifiers as the same
model discards this coupling and need not produce probabilities summing to
one.

### A non-Gaussian likelihood changes both robustness and computation {#gp-robust-likelihood}

Gaussian observation noise gives every residual a score proportional to
its magnitude, so one extreme response can pull the entire posterior mean.
With a Student-\(t\) likelihood of degrees of freedom \(\nu\) and scale
\(s\),

$$\log p(y_i\mid f_i)
=\text{const}
-\frac{\nu+1}{2}
\log\left(1+\frac{(y_i-f_i)^2}{\nu s^2}\right),$$

the score with respect to \(f_i\) is

$$\frac{\partial}{\partial f_i}\log p(y_i\mid f_i)
=\frac{(\nu+1)(y_i-f_i)}
{\nu s^2+(y_i-f_i)^2}.$$

It tends to zero for an arbitrarily large residual, limiting outlier
influence. The price is that the posterior is no longer Gaussian and the
log likelihood need not be globally concave. Laplace can face multiple
modes or negative site curvature, so EP, variational bounds, or sampling
must be checked against the problem rather than inherited mechanically from
classification [@jylanki2011robustgp]. This is the promised bridge: exact GP
regression is a consequence of a Gaussian likelihood, not a property of the
prior alone.

## Sparse Bayesian priors and the Relevance Vector Machine {#laplacian-rvm}

The Gaussian process prior spreads its belief smoothly, and its posterior mean uses every training point: \(\alpha=(K+\sigma^2 I)^{-1}y\) is generically dense. Sometimes we want the opposite, a prediction supported on a handful of basis functions. Sparsity is a statement about the prior, and there are two ways to build it in.

The first keeps a single coefficient model and changes the prior's shape. A *Laplacian process* (Schölkopf and Smola 2002) places an independent Laplace prior on the expansion coefficients of \(f(x)=\sum_i\alpha_i k(x_i,x)\), so that \(-\ln p(\alpha)\propto\sum_i|\alpha_i|\). The MAP estimate then minimizes a data-fit term plus an \(\ell_1\) penalty, a linear or quadratic program whose solutions can be exactly sparse, in the spirit of basis pursuit and the LASSO. Because the prior depends on the data locations through the kernel expansion it is a data-dependent prior, a departure from a process prior specified independently on all finite index sets. The sparse MAP estimate alone does not provide posterior error bars; those require a separate approximation or sampling calculation under the non-Gaussian posterior.

The second way, and the one we develop in detail, is the Relevance Vector Machine [@tipping2001]. Rather than fix the prior variance of the coefficients, give every coefficient its own variance, controlled by its own hyperparameter, and let the evidence decide which variances to shrink to zero.

:::: {.definition #def-40-5}
[Definition (RVM prior)]{.box-title}

Model the target as \(t=K\alpha\) with independent Gaussian noise of variance \(\sigma^2\). Place on each coefficient an independent zero-mean Gaussian prior with its own precision \(s_i\gt 0\),

$$p(\alpha_i\mid s_i)=\sqrt{\tfrac{s_i}{2\pi}}\,\exp\!\Big(-\tfrac12 s_i\alpha_i^2\Big),\qquad p(\alpha\mid s)=\mathcal N(0,\,S^{-1}),\ \ S=\operatorname{diag}(s_1,\dots,s_m),$$

with a flat hyperprior on \(\ln s_i\) (or a broad Gamma prior). Each \(s_i\) is an automatic relevance determination hyperparameter for one basis function.
::::

The precision \(s_i\) is the inverse prior variance of \(\alpha_i\): a large \(s_i\) means \(\alpha_i\) is pinned tightly to zero, a small \(s_i\) lets it roam. With the noise variance \(\sigma^2\) and Gaussian prior fixed, the posterior over \(\alpha\) is Gaussian with covariance and mean

$$\Sigma=\big(\sigma^{-2}K^\top K+S\big)^{-1},\qquad \mu=\sigma^{-2}\Sigma K^\top y.$$

Learning the \(s_i\) proceeds by type-II maximum likelihood, exactly as for GP hyperparameters: the coefficients are integrated out to give the marginal likelihood \(p(y\mid s,\sigma^2)=\mathcal N(0,\ \sigma^2 I+K S^{-1}K^\top)\), which is then maximized over \(s\) and \(\sigma^2\). Setting derivatives to zero yields fixed-point updates.

:::: {.algorithm #algo-40-3}
[Algorithm (RVM update)]{.box-title}

::: algo-io
[Input]{.algo-lab} Design matrix \(K\) (\(K_{ij}=k(x_i,x_j)\)), targets \(y\); initial precisions \(s\), noise \(\sigma^2\).

[Output]{.algo-lab} An evidence stationary point and a pruned active set;
coefficients removed at the \(s_i\to\infty\) boundary are represented as
zero.
:::

1.  Form the posterior covariance \(\Sigma=(\sigma^{-2}K^\top K+S)^{-1}\) and mean \(\mu=\sigma^{-2}\Sigma K^\top y\).
2.  For each \(i\), compute the well-determinedness \(\gamma_i=1-s_i\Sigma_{ii}\in[0,1]\).
3.  Update each precision \(s_i\leftarrow\dfrac{\gamma_i}{\mu_i^2}=\dfrac{1-s_i\Sigma_{ii}}{\mu_i^2}\).
4.  Update the noise \(\sigma^2\leftarrow\dfrac{\|y-K\mu\|^2}{m-\sum_i\gamma_i}\).
5.  Whenever \(s_i\to\infty\) (numerically, exceeds a large threshold), prune basis \(i\): delete its row and column and continue. Repeat from step 1 until \(\mu\) stabilizes.

This fixed-point scheme follows the stationarity equations of the type-II
objective. It is not a globally convergent optimization theorem. An
implementation must report the evidence trajectory, active-set changes,
maximum iteration count, and sensitivity to initialization; pruning is
irreversible unless the algorithm explicitly supports re-entry.
::::

Why does this produce sparsity? The update \(s_i\leftarrow(1-s_i\Sigma_{ii})/\mu_i^2\) is the mechanism. If a basis function contributes little to explaining \(y\), its posterior mean \(\mu_i\) can become small while the evidence update increases \(s_i\), which shrinks the prior variance \(1/s_i\) and suppresses the coefficient further. In the limiting pruned state \(s_i\to\infty\), the coefficient's posterior mean and variance go to zero and the basis can be removed. This is a boundary limit of the optimization, not an ordinary finite fixed point, and convergence to it is initialization- and data-dependent because the evidence surface is nonconvex. The surviving bases are the relevance vectors [@tipping2001]. With a Gamma hyperprior, integrating out \(s_i\) produces a heavy-tailed marginal prior on \(\alpha_i\), which provides another view of the selective shrinkage. The effect resembles support-vector sparsity, but the mechanism and retained points are different; no general theorem says the RVM must be sparser than an SVM on every dataset.

The price of this elegance is that the marginal likelihood over the \(s_i\) is highly multimodal, so the fixed-point iteration finds a local optimum and the training is heavier than an SVM's on large data. The reward is a probabilistic, extremely sparse predictor that still carries the full apparatus of Bayesian error bars, and whose predictive mean and variance,

$$\bar y(x_*)=k_*^\top\mu,\qquad v(x_*)=\sigma^2+k_*^\top\Sigma k_*,$$

mirror the Gaussian process formulas of the regression section, now built on only the relevance vectors.

<figure class="viz" data-figure="gp-rvm-sparsity-comparison" data-alt="The left panel overlays a dense Gaussian-process fit and a close sparse kernel fit, marking seven retained relevance sites among forty-one observations. The right panel compares forty-one active GP centres with seven sparse centres.">
<figcaption>Sparsity concerns the predictive representation, not merely visual smoothness. Here a seven-centre expansion tracks the dense posterior mean closely, but that economy comes from a different prior and optimization problem, not from the ordinary GP posterior spontaneously becoming sparse.</figcaption>
</figure>

## Sparse and variational Gaussian processes {#sparse-variational-gp}

The Relevance Vector Machine made its predictor sparse because the prior wanted few active basis functions. A second, blunter pressure pushes in the same direction: cost. Every exact GP quantity in this chapter, the posterior mean, the variance, the evidence and its gradient, passes through linear solves or spectral functions of the \(n\times n\) matrix \(K+\sigma^2 I\). A dense Cholesky implementation spends \(O(n^3)\) time and \(O(n^2)\) memory, repeated at every step of hyperparameter learning, and becomes impractical beyond at most a few tens of thousands of points on ordinary hardware. The equations themselves do not stop there: matrix-free, subblock, and structured solvers can still target them at far larger \(n\), as the modern scaling section below explains. [[ch:large-scale-kernels|The large-scale chapter]] develops generic remedies, including random features and Nyström-type low-rank factorizations, which approximate the *matrix*. The Gaussian process literature developed a complementary line that approximates the *model*: replace the full process by one whose information about the data is carried by \(m\ll n\) well-placed points. The prize for staying inside the probabilistic frame is that variances and evidences survive with their meaning intact, and the quality of the approximation itself becomes a measurable quantity. Throughout this section \(n\) counts training points and \(m\) counts inducing points.

:::: {.definition #def-40-6}
[Definition (inducing points and the Nyström surrogate)]{.box-title}

Let \(Z=(z_1,\dots,z_m)\) be *inducing inputs* in \(\mathcal X\) and let \(u=(t(z_1),\dots,t(z_m))\) be the latent process evaluated there. Write \(K_{uu}\), \(K_{uf}\), \(K_{ff}\) for the kernel matrices among inducing and training inputs (so \(K_{ff}=K\)), and define, for any two blocks of inputs \(a\) and \(b\),

$$Q_{ab}=K_{au}\,K_{uu}^{-1}\,K_{ub}.$$

\(Q_{ab}\) is the covariance that remains when all correlation between \(a\) and \(b\) is forced to pass through \(u\); as a matrix approximation it is exactly the Nyström approximation of \(K_{ab}\) built from the columns at \(Z\).
::::

Conditioning the joint Gaussian prior of \((f,u)\) gives \(p(f\mid u)=\mathcal N\big(K_{fu}K_{uu}^{-1}u,\ K_{ff}-Q_{ff}\big)\), by the same block formula that produced the predictive equations. The conditional covariance \(K_{ff}-Q_{ff}\) is a Schur complement, hence positive semidefinite, and its \(i\)th diagonal entry vanishes whenever \(x_i\) belongs to \(Z\): it measures exactly the part of the prior that the bottleneck \(u\) fails to carry. The classical sparse constructions all keep the exact \(p(u)=\mathcal N(0,K_{uu})\) and alter only this conditional [@quinonero2005]. The *subset of regressors* (SoR) approximation pretends the conditional is deterministic at training and test points alike, \(f=K_{fu}K_{uu}^{-1}u\); the model degenerates to a rank-\(m\) GP with kernel \(Q\), and its variances collapse wherever \(Q\) does. The *deterministic training conditional* (DTC) keeps the deterministic rule at the training points but restores the exact conditional at test points, repairing the predictive variance. The *fully independent training conditional* (FITC) keeps every training point's exact conditional variance and severs only their correlations, replacing \(K_{ff}-Q_{ff}\) by its diagonal [@snelson2006fitc]. Each variant is exact Gaussian inference in its modified model, and all share one predictive template. Setting

$$\Lambda_{\mathrm{SoR}}=\Lambda_{\mathrm{DTC}}=\sigma^2 I,\qquad \Lambda_{\mathrm{FITC}}=\operatorname{diag}\big(K_{ff}-Q_{ff}\big)+\sigma^2 I,$$

and \(\Sigma=\big(K_{uu}+K_{uf}\Lambda^{-1}K_{fu}\big)^{-1}\), the predictive law at \(x_*\) is \(\mathcal N(\mu_*,v_*)\) with

$$\mu_*=k_{u*}^\top\,\Sigma\,K_{uf}\Lambda^{-1}y,\qquad v_*=k_{**}-Q_{**}+k_{u*}^\top\,\Sigma\,k_{u*},$$

where \(k_{u*}\) collects the covariances between \(Z\) and \(x_*\) and \(Q_{**}=k_{u*}^\top K_{uu}^{-1}k_{u*}\). SoR alone replaces the honest \(k_{**}\) by \(Q_{**}\), which is what deflates its error bars away from \(Z\). The cost is \(O(nm^2)\) once, then \(O(m)\) per test mean and \(O(m^2)\) per test variance: linear in \(n\), with \(m\) ours to choose.

:::: {.algorithm #algo-40-4}
[Algorithm (sparse GP prediction with inducing points)]{.box-title}

::: algo-io
[Input]{.algo-lab} Training data \(X,y\) (\(n\) points); inducing inputs \(Z\) (\(m\ll n\)); kernel \(k\); noise \(\sigma^2\); family (DTC or FITC).

[Output]{.algo-lab} Predictive mean and variance at any \(x_*\); a training objective for \(\theta\), \(\sigma^2\), \(Z\).
:::

1.  Form \(K_{uu}\) (\(m\times m\)) and \(K_{uf}\) (\(m\times n\)); factor \(K_{uu}\) by Cholesky.
2.  With \(k_{ui}\) the \(i\)th column of \(K_{uf}\), compute the Nyström diagonal \(q_{ii}=k_{ui}^\top K_{uu}^{-1}k_{ui}\) for all \(i\) in \(O(nm^2)\); never form the full \(Q_{ff}\).
3.  Set \(\Lambda=\sigma^2 I\) (DTC) or \(\Lambda=\operatorname{diag}\big(k(x_i,x_i)-q_{ii}\big)+\sigma^2 I\) (FITC); \(\Lambda\) is diagonal.
4.  Factor \(\Sigma^{-1}=K_{uu}+K_{uf}\Lambda^{-1}K_{fu}\) (\(m\times m\)) and cache the weights \(w=\Sigma K_{uf}\Lambda^{-1}y\).
5.  At a test point form \(k_{u*}\) and report \(\mu_*=k_{u*}^\top w\), \(v_*=k_{**}-k_{u*}^\top K_{uu}^{-1}k_{u*}+k_{u*}^\top\Sigma k_{u*}\).
6.  To train, ascend the sparse evidence \(\ln\mathcal N\big(y\mid 0,\ Q_{ff}+\Lambda\big)\) or the variational bound \(\mathcal L_T\) below over \(\theta\), \(\sigma^2\), and the positions \(Z\); each gradient costs \(O(nm^2)\).
::::

### The variational answer: sparsity without changing the model {#titsias-bound}

DTC and FITC are honest algorithms with an awkward epistemology: each is exact inference in a model that is not the one we wrote down. Their evidences are evidences *of the surrogates*, so ascending them over \(Z\) rewards whichever surrogate most flatters the data, not the best approximation to the true posterior; the inducing positions act as extra kernel parameters, free to overfit. Variational inducing variables recast sparsity as approximate inference in the *exact* model [@titsias2009svgp]. Choose a variational posterior of the constrained form \(q(f,u)=p(f\mid u)\,q(u)\), which may reshape belief over the inducing values but must propagate it to \(f\) through the true conditional, and maximize a lower bound on the exact evidence. Maximizing over \(q(u)\) in closed form collapses the bound to a formula.

:::: {.proposition #prop-40-7}
[Proposition (Titsias, 2009)]{.box-title}

For any inducing inputs \(Z\), the collapsed variational bound

$$\mathcal L_T=\ln\mathcal N\big(y\mid 0,\ Q_{ff}+\sigma^2 I\big)-\frac{1}{2\sigma^2}\operatorname{tr}\big(K_{ff}-Q_{ff}\big)$$

satisfies \(\mathcal L_T\le\ln p(y\mid X)\), with equality whenever \(Q_{ff}=K_{ff}\), in particular when \(Z=X\).

**Assumptions.** \(K_{ff}\) is the Gram matrix of a PSD kernel,
\(K_{uu}\) is positive definite, \(Q_{ff}=K_{fu}K_{uu}^{-1}K_{uf}\),
\(\sigma^2\gt0\), and the observation likelihood is
\(\mathcal N(y\mid f,\sigma^2I)\). For singular \(K_{uu}\), the inducing
representation must be reduced to an independent span or formulated with a
pseudoinverse and its range conditions.
**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

Fix \(u\) and apply Jensen's inequality to the training conditional: \(\ln p(y\mid u)=\ln\mathbb E_{p(f\mid u)}[p(y\mid f)]\ \ge\ \mathbb E_{p(f\mid u)}[\ln p(y\mid f)]\). With \(p(f\mid u)=\mathcal N(a,\,K_{ff}-Q_{ff})\), \(a=K_{fu}K_{uu}^{-1}u\), and \(\ln p(y\mid f)=-\tfrac{1}{2\sigma^2}\|y-f\|^2-\tfrac n2\ln(2\pi\sigma^2)\), the expectation uses \(\mathbb E\|y-f\|^2=\|y-a\|^2+\operatorname{tr}(K_{ff}-Q_{ff})\), giving the pointwise inequality

$$p(y\mid u)\ \ge\ \mathcal N\big(y\mid a,\ \sigma^2 I\big)\,\exp\!\Big(-\tfrac{1}{2\sigma^2}\operatorname{tr}\big(K_{ff}-Q_{ff}\big)\Big).$$

The trace factor does not depend on \(u\). Averaging both sides over \(p(u)=\mathcal N(0,K_{uu})\) and using the linear-Gaussian marginalization \(\int\mathcal N\big(y\mid K_{fu}K_{uu}^{-1}u,\ \sigma^2 I\big)\,\mathcal N(u\mid 0,K_{uu})\,du=\mathcal N\big(y\mid 0,\ Q_{ff}+\sigma^2 I\big)\) gives \(p(y)\ge\mathcal N(y\mid 0,Q_{ff}+\sigma^2I)\,e^{-\operatorname{tr}(K_{ff}-Q_{ff})/(2\sigma^2)}\); take logarithms. If \(Q_{ff}=K_{ff}\) the trace vanishes and the first term is the exact log marginal likelihood, so the inequality is tight; \(Z=X\) gives \(Q_{ff}=K_{ff}K_{ff}^{-1}K_{ff}=K_{ff}\). [\(\square\)]{.qed}
::::

Read \(\mathcal L_T\) term by term. The first term is exactly the DTC evidence. The second charges it \(\tfrac{1}{2\sigma^2}\sum_i\operatorname{Var}[f_i\mid u]\), the total conditional variance the bottleneck throws away. That one trace changes the status of everything: the positions \(Z\), and \(m\) itself, become variational parameters, and improving \(\mathcal L_T\) provably tightens an approximation to one fixed model rather than drifting toward a more convenient model, with the bound only improving as inducing points are added (Titsias 2009). The optimal inducing posterior is available in closed form, \(q^*(u)=\mathcal N\big(\sigma^{-2}K_{uu}\Sigma K_{uf}\,y,\ K_{uu}\Sigma K_{uu}\big)\) with \(\Sigma=(K_{uu}+\sigma^{-2}K_{uf}K_{fu})^{-1}\), and the prediction it induces coincides with the DTC formulas above. Variationally trained sparse GPs therefore predict like DTC; what changes is the objective that places \(Z\) and tunes the hyperparameters.

::::: {.example #example-40-3}
[Example (two inducing points against the full GP)]{.box-title}

```python
import numpy as np

x = np.array([-2.0, -1.0, 0.5, 2.0])
y = np.array([-1.7, -0.8, 0.7, 1.5])
z, query, noise = np.array([-1.0, 0.5]), np.array([0.3]), 0.1
k = lambda a, b: np.exp(-((a[:, None] - b[None, :]) ** 2) / 8.0)
Kff, Kuu, Kfu = k(x, x), k(z, z), k(x, z)
Qff = Kfu @ np.linalg.solve(Kuu, Kfu.T)
diag_correction = np.diag(Kff - Qff)
Kus = k(z, query).ravel()
Sigma = np.linalg.inv(Kuu + k(z, x) @ k(x, z) / noise)
mean_dtc = Kus @ Sigma @ k(z, x) @ y / noise
variance_dtc = 1.0 - Kus @ np.linalg.solve(Kuu, Kus) + Kus @ Sigma @ Kus
Lambda = np.diag(diag_correction) + noise * np.eye(x.size)
Sigma_fitc = np.linalg.inv(Kuu + k(z, x) @ np.linalg.solve(Lambda, k(x, z)))
mean_fitc = Kus @ Sigma_fitc @ k(z, x) @ np.linalg.solve(Lambda, y)
print(diag_correction, mean_dtc, variance_dtc, mean_fitc)
```

:::: wex
::: wex-setup
Training inputs \(x=(-2,\,-1,\,0.5,\,2)\), targets \(y=(-1.7,\,-0.8,\,0.7,\,1.5)\), squared-exponential kernel \(k(x,x')=e^{-(x-x')^2/8}\) (length scale \(\ell=2\)), noise \(\sigma^2=0.1\). Inducing inputs \(Z=(-1,\,0.5)\), the two middle training points, so \(n=4\) and \(m=2\). Predict at \(x_*=0.3\). The values are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-gp-ex3].
:::

1.  [Fit the exact GP for reference.]{.wex-op}

$$K_{ff}=\begin{pmatrix}1&0.8825&0.4578&0.1353\\0.8825&1&0.7548&0.3247\\0.4578&0.7548&1&0.7548\\0.1353&0.3247&0.7548&1\end{pmatrix},$$

    and the full predictive equations give \(\bar m(x_*)=0.4672\), \(v(x_*)=0.0648\).
2.  [Assemble the Nyström surrogate.]{.wex-op} Since \(Z=(x_2,x_3)\), \(K_{uu}=\begin{pmatrix}1&0.7548\\0.7548&1\end{pmatrix}\) is a principal submatrix of \(K_{ff}\) and \(K_{uf}\) is its rows 2 and 3. Then

$$Q_{ff}=K_{fu}K_{uu}^{-1}K_{uf}=\begin{pmatrix}0.8797&0.8825&0.4578&0.0397\\0.8825&1&0.7548&0.3247\\0.4578&0.7548&1&0.7548\\0.0397&0.3247&0.7548&0.7095\end{pmatrix}.$$

    Rows 2 and 3 reproduce \(K_{ff}\) exactly; the corners degrade, the \(x_1\)-\(x_4\) covariance dropping from \(0.1353\) to \(0.0397\) because it must be carried through \(u\). The discarded diagonal is \(\operatorname{diag}(K_{ff}-Q_{ff})=(0.1203,\,0,\,0,\,0.2905)\), zero exactly at the inducing points, and \(Q_{**}=0.998\) against \(k_{**}=1\).
3.  [Predict with \(\Lambda=\sigma^2 I\) (DTC, equally the Titsias posterior).]{.wex-op} The template gives \(\mu_*=0.5054\) and, keeping \(k_{**}\), \(v_*=0.0443\). SoR substitutes \(Q_{**}\) and reports \(0.0423\).
4.  [Predict with FITC.]{.wex-op} \(\Lambda=\operatorname{diag}(0.2203,\,0.1,\,0.1,\,0.3905)\): the noise floor \(0.1\) plus the discarded variance, inflated exactly at the two poorly covered inputs. The template gives \(\mu_*=0.4364\) and \(v_*=0.0608\).
5.  [Score the sparsification with the variational bound.]{.wex-op} The fit term is \(\ln\mathcal N(y\mid 0,\ Q_{ff}+\sigma^2 I)=-4.982\) and the trace penalty \(\operatorname{tr}(K_{ff}-Q_{ff})/(2\sigma^2)=2.0544\) (the trace itself is \(0.4109\)), so \(\mathcal L_T=-4.982-2.0544=-7.0364\). The exact log marginal likelihood is \(-5.3762\): the bound holds with gap \(1.6602\), while the fit term alone, DTC's own evidence, overshoots the truth.

**Reading.** Half the data serving as inducing set moves the predictive means by a few hundredths, but the error bars tell the finer story: \(0.0423\) (SoR) \(\lt 0.0443\) (DTC) \(\lt 0.0608\) (FITC) against the exact \(0.0648\); the more of the discarded conditional variance a scheme keeps, the closer its uncertainty comes to the truth. The bound line is the operational one. \(\mathcal L_T\) sits \(1.6602\) below the exact evidence, all of it charged by the trace over the two badly covered inputs \(x_1\) and \(x_4\), so an optimizer ascending \(\mathcal L_T\) over \(Z\) is pushed to cover exactly those points, and at \(Z=X\) the bound closes.
::::
:::::

### Stochastic variational Gaussian processes {#svgp-big-data}

One computational ceiling remains: evaluating \(\mathcal L_T\) or its gradient sweeps all \(n\) points through the \(O(nm^2)\) sums, once per optimizer step. Stochastic variational inference removes it by refusing to collapse [@hensman2013bigdata]. Keep \(q(u)=\mathcal N(m_u,S)\) as explicit variational parameters; the uncollapsed evidence lower bound is

$$\mathcal L(m_u,S,Z,\theta)=\sum_{i=1}^{n}\mathbb E_{q(f_i)}\big[\ln p(y_i\mid f_i)\big]\;-\;\mathrm{KL}\big(q(u)\,\big\|\,p(u)\big),$$

where \(q(f_i)\) is Gaussian with mean \(k_{ui}^\top K_{uu}^{-1}m_u\) and variance \(k(x_i,x_i)-q_{ii}+k_{ui}^\top K_{uu}^{-1}SK_{uu}^{-1}k_{ui}\). For the Gaussian likelihood, maximizing over \((m_u,S)\) at fixed \(Z\) recovers \(\mathcal L_T\) exactly, so nothing is conceded. What is gained is the shape of the objective: the data enter only through a sum of independent per-point terms, so a random minibatch of size \(b\), rescaled by \(n/b\), gives an unbiased estimate of the sum, and stochastic gradient ascent applies jointly to \(m_u\), \(S\), \(Z\), and the kernel hyperparameters. One step costs \(O(bm^2+m^3)\), with no dependence on \(n\) at all; the bound moreover fits the framework of stochastic variational inference, where natural-gradient steps in the exponential-family parameterization of \(q(u)\) speed convergence. This is SVGP, demonstrated on hundreds of thousands of flight records with a few hundred inducing points [@hensman2013bigdata], and it is the template modern GP software implements for big data.

Two further dividends. Because each expectation in the sum is a one-dimensional Gaussian integral, any likelihood whose scalar expectations can be computed or quadratured slots straight in, so the same loop trains GP classifiers with the links of the classification section on data far beyond the Laplace approximation's reach. And the machinery composes: the sparse variational posterior is the substrate on which the deep constructions of the next section stand, and it is what makes Gaussian process surrogates affordable inside larger systems.

### Exact targets, approximate computation, and structured models {#modern-scalable-gp}

The inducing-variable section solved the cubic bottleneck by replacing the posterior family. That move is powerful, but it creates a new question: if the scientific decision genuinely needs the original covariance model, must scale force a different posterior? Not always. The matrix-free and subblock ideas of [[ch:large-scale-kernels|the large-scale chapter]] preserve a full linear-system target while changing how it is reached, and domain structure can sometimes change the cost of a kernel product without changing the structured model at all.

“Approximate GP” can therefore describe three mathematically different operations. An inducing-point method changes the inference family, or in older constructions changes the covariance model. An iterative solver may leave the model and posterior target unchanged but approximate a linear solve to a declared residual. A structured method may compute the exact posterior efficiently only for covariances and observation patterns with special algebra. These routes cannot be ranked by sample count alone:

| Route | What remains exact? | What is approximated or assumed? | Representative recent work |
|---|---|---|---|
| subblock iteration | the target system \( (K+\sigma^2I)\alpha=y \) | a finite iteration budget | alternating projection [@wu2024alternatinggp] |
| latent structure | the posterior for the structured model | a latent Cartesian-product representation | latent Kronecker GPs [@lin2025latentkronecker] |
| sketch-and-project | selected posterior-mean directions | randomized distributed projections | ADASAP [@rathore2025adasap] |
| inducing variables | the original GP prior in the variational formulation | posterior family \(q(f,u)\) | SVGP |
| sparse kernel products | the chosen iterative target | sparsity approximation and time-series structure | KernelMatmul [@hoffbauer2025kernelmatmul] |

Alternating projection accesses covariance subblocks and reports training and inference on datasets up to four million points. Latent-Kronecker inference uses an observation projection around a latent product grid and reports real applications with up to five million examples. ADASAP distributes approximate sketch-and-project updates and reports a problem with more than \(3\times10^8\) samples, but its primary guarantee concerns the posterior mean rather than the full covariance. KernelMatmul obtains linear time and memory for its large-time-series construction. These are substantial computational results, but each preserves a different object.

The deployment diagnostic follows directly. For an exact-target iterative method, report the true residual of every solve and the error of stochastic log-determinant or trace estimates used in hyperparameter learning. For a structured method, validate the structure rather than merely its speed. For a posterior approximation, test predictive means, variances, and calibration separately. A fast mean with a distorted covariance is not a fast Gaussian process for decisions that depend on uncertainty.

Consider the same million observations under two geometries. Daily measurements from \(1{,}000\) sensors over \(1{,}000\) common time points form a product index; a separable space-time kernel can expose Kronecker structure, so latent-grid algebra is a plausible model and computational strategy. One million unrelated molecular descriptors have the same sample count but no corresponding Cartesian product. Relabeling them onto a grid would alter the covariance model merely to obtain a fast algorithm. For the first dataset, test the separability and projection residuals. For the second, use a matrix-free, subblock, center, or feature method and report its own approximation target. The sample count never made that decision; the geometry did.

## Deep, multi-output, and spectral-mixture processes {#deep-multioutput-gp}

The prior we have used so far is restrictive in three separable ways: the kernel is picked from a small catalogue of fixed stationary forms, the process models a single scalar function, and the function is drawn in one shot rather than built out of simpler stages. Each restriction has a principled lift, and all three run on machinery this chapter has already built, evidence-driven hyperparameter learning and inducing-point inference.

### Learning the kernel: spectral mixtures {#spectral-mixture-kernels}

Bochner's theorem, developed in [[ch:kernel-families|the kernel-families chapter]], says a stationary kernel is the Fourier transform of a nonnegative spectral density: choosing a stationary covariance *is* choosing a distribution over frequencies. Seen through that lens the standard kernels are rigid commitments: the squared-exponential is a single Gaussian density centered at zero frequency, so it can say nothing beyond \"smooth at scale \(\ell\)\", and automatic relevance determination tunes one number per dimension without ever changing the shape. Spectral-mixture kernels put the flexibility where the theorem says it lives, modelling the spectral density as a mixture of \(Q\) Gaussians [@wilson2013]. In one dimension the resulting kernel is closed form,

$$k_{\mathrm{SM}}(\tau)=\sum_{q=1}^{Q}w_q\,\exp\big(-2\pi^2\sigma_q^2\tau^2\big)\cos\big(2\pi\mu_q\tau\big),\qquad \tau=x-x',$$

with weights \(w_q\), center frequencies \(\mu_q\), and bandwidths \(\sigma_q\). Each component is a squared-exponential envelope times a cosine: a quasi-periodic pattern at frequency \(\mu_q\) that persists over a range of order \(1/\sigma_q\); taking \(Q=1\) and \(\mu_1=0\) recovers the squared-exponential itself. Since mixtures of Gaussians can approximate any spectral density, spectral-mixture kernels are dense among stationary covariances, and every parameter is learned by ascending the marginal likelihood: kernel learning reduced to the hyperparameter learning of this chapter. Their practical payoff is extrapolation, with the evidence discovering periodic structure and continuing it beyond the data, where a fixed squared-exponential can only fall back to its mean [@wilson2013].

### Many outputs: coregionalization {#multi-output-gps}

Often the target is vector valued: \(p\) sensors on one machine, \(p\) related tasks, \(p\) outputs of a simulator. Modelling each coordinate with an independent GP wastes exactly what makes the problem interesting, that the outputs are correlated and data on one should sharpen predictions of another. A Gaussian prior over vector-valued functions needs a matrix-valued covariance, \(\operatorname{cov}\big(f_i(x),f_j(x')\big)\), positive semidefinite in the joint sense. The simplest separable choice, long used in geostatistics under the name co-kriging, is the *intrinsic coregionalization model*

$$\operatorname{cov}\big(f_i(x),f_j(x')\big)=B_{ij}\,k(x,x'),$$

with \(k\) an ordinary kernel on inputs and \(B\) a positive semidefinite \(p\times p\) *coregionalization matrix*; the joint Gram matrix is the Kronecker product \(B\otimes K\), positive semidefinite by the product closure rules. Writing \(B=AA^\top\) with \(A\) of rank \(r\) exposes the generative reading: the \(p\) outputs are linear mixtures of \(r\) shared latent GPs, and the *linear model of coregionalization* sums several such terms with different base kernels. The evidence learns \(B\), which is a task-similarity matrix read off from data, and transfer happens mechanically: an observation of output \(j\) enters the posterior of output \(i\) with weight proportional to \(B_{ij}\). This is the construction behind multi-task surrogates in [[ch:bayesian-optimization-and-bandits|Bayesian optimization]], where cheap evaluations of a related objective narrow the posterior over an expensive one, and Kronecker algebra plus the inducing-point machinery keeps the joint model tractable.

### Depth: compositions of processes {#deep-gps}

Stationarity is a strong commitment: one length scale for the whole input space. A function that undulates in one region and is nearly flat in another fits no stationary prior well. The classical fix warps the inputs through a fixed map \(h\) and models \(f(h(x))\); the radical fix is to learn the warp as a function too, and then there is no reason to stop at one stage. A *deep Gaussian process* composes layers \(f=f_L\circ\cdots\circ f_1\), each layer a GP prior over a map between adjacent feature spaces [@damianou2013deepgp]. The composite is no longer a Gaussian process: its marginals are non-Gaussian, can be multimodal, and its effective smoothness varies with position, which is precisely the expressive gain. The price is inference, since each layer's inputs are the previous layer's random outputs; inducing points at every layer make a nested variational approximation tractable, and later stochastic variants scale the same construction with minibatches. The contrast with infinite-width network limits is worth savoring: widening a single hidden layer can converge to one GP with an architecture-defined kernel [@neal1996], while composing GP layers keeps a non-Gaussian, data-updated hierarchy. That dialogue, between kernels fixed by an architecture and representations learned through it, is where this book is heading; [[ch:the-frontier|the frontier chapter]] takes it up in earnest through the infinite-width limits of deep networks and the kernels that grow depth.

## The GP journey through the rest of the book {#gp-cross-chapter-journey}

A deep chapter should create unresolved problems deliberately, then send the
reader to the result that resolves each one. The dependencies are:

| Problem encountered here | Why this chapter cannot settle it alone | Where the journey continues |
|---|---|---|
| Which functions are cheap under the covariance? | requires Mercer operators, source conditions, and effective dimension | [[ch:mercer-and-rates|Mercer spectra and rates]] |
| Does a covariance formula define the intended geometry? | requires Bochner, Schoenberg, mixtures, and closure proofs | [[ch:kernel-families|Kernel families]] |
| What if observations are derivatives, integrals, or vector outputs? | requires generalized representers and operator-valued PSD | [[ch:scientific-computing-and-operator-learning|Scientific computing]] and [[ch:vector-and-operator-valued-kernels|operator-valued kernels]] |
| Can a dense posterior target be computed without changing the model? | requires matrix-free solves, log determinants, and approximation contracts | [[ch:large-scale-kernels|Large-scale kernels]] |
| How should uncertainty choose the next experiment? | requires a utility, acquisition rule, and regret assumptions | [[ch:bayesian-optimization-and-bandits|Bayesian optimization]] |
| Does the interval cover under repeated deployment? | requires shift assumptions, calibration, and possibly conformal correction | [[ch:distribution-shift-robustness-and-conformal-prediction|Reliability and conformal prediction]] |
| How are integrals and differential equations inferred? | requires bounded functionals and numerical error decomposition | [[ch:kernel-quadrature-and-herding|Kernel quadrature]] and [[ch:scientific-computing-and-operator-learning|scientific computing]] |
| What changes when the feature map is learned? | covariance choice and representation learning become coupled | [[ch:deep-kernel-learning|Deep kernel learning]] |

The return points matter. After the large-scale chapter, revisit the
posterior covariance and ask which approximation target was preserved.
After the reliability chapter, revisit the word “uncertainty” and name its
probability law. After the scientific chapter, revisit the joint posterior
and replace point evaluations by operators. The book is not asking the
reader to memorize eight destinations; it is revisiting one object as new
failures force richer mathematics.

::::: {.example #example-40-9}
[Example (same pointwise intervals, different decisions)]{.box-title}

```python
import math
import numpy as np

Phi = lambda z: 0.5 * (1.0 + math.erf(z / np.sqrt(2.0)))
point_probability = 1.0 - Phi(1.0)
for rho in [0.95, 0.0]:
    nodes, weights = np.polynomial.legendre.leggauss(600)
    x = 0.5 * (1.0 + 9.0) * nodes + 0.5 * (1.0 - 9.0)
    joint_below = 5.0 * np.sum(
        weights * np.exp(-x**2 / 2.0) / np.sqrt(2.0 * np.pi)
        * np.vectorize(Phi)((1.0 - rho * x) / np.sqrt(1.0 - rho**2))
    )
    maximum_probability = 1.0 - joint_below
    average_sd = np.sqrt(0.04 * (1.0 + rho) / 2.0)
    average_probability = 1.0 - Phi(0.2 / average_sd)
    print(rho, point_probability, maximum_probability, average_probability)
```

:::: wex
::: wex-setup
Two fitted models predict latent values at sites \(A\) and \(B\). Both have
mean \((0.8,0.8)\), marginal variance \(0.04\), and therefore identical
pointwise \(95\%\) intervals. Model I has correlation \(0.95\); Model II has
correlation \(0\). The threshold for intervention is \(1\). The bivariate
normal probabilities are reproduced by `checks/ch-gp-depth.py`.
:::

1.  [Inspect either site separately.]{.wex-op} Both models report the same
    marginal exceedance probability \(P(f_A\gt1)=P(f_B\gt1)=0.1587\).
    No pointwise table can distinguish them.
2.  [Intervene if either site exceeds the threshold.]{.wex-op}
    Model I gives \(P(\max(f_A,f_B)\gt1)=0.1892\); Model II gives
    \(0.2921\). Independence creates more chances for at least one excursion.
3.  [Intervene if the average exceeds the threshold.]{.wex-op}
    Model I gives \(P((f_A+f_B)/2\gt1)=0.1556\); Model II gives
    \(0.0786\). Positive correlation makes a joint upward movement more
    plausible.
4.  [Add deployment evidence.]{.wex-op} If held-out standardized residuals
    show intervals that are too narrow, neither probability is operationally
    trustworthy until the likelihood, covariance, or calibration procedure
    is repaired.

**Reading.** A posterior mean and two marginal error bars are not a
decision-ready forecast. Joint covariance, the downstream event, and
external calibration can all reverse the conclusion.
::::
:::::

## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

The posterior interval is conditional on the kernel, likelihood, noise model, and fitted hyperparameters; it is not an unconditional promise of coverage. Keep latent-function variance separate from observation variance, and check calibration on held-out or time-ordered data. Marginal likelihood can prefer pathological length scales or noise levels when parameters are weakly identified, so inspect profiles or multiple starts rather than reporting one optimizer result.

Never invert a covariance matrix explicitly. Use a Cholesky factor or a monitored iterative solve, report jitter relative to the diagonal scale, and treat factorization failure or stalled residuals as modeling diagnostics. Sparse approximations require an additional declaration: subset-of-regressors changes the prior, variational inducing methods approximate inference in the original model, and structured or sketch-based solvers preserve still other targets. Their error bars are not interchangeable.

Do not use “smooth GP” as a substitute for a regularity statement. Name
mean-square, almost-sure, RKHS, or posterior-mean smoothness. Do not use
“consistent” without naming the asymptotic regime, estimand, and norm.
Fixed-domain parameter inconsistency can coexist with good prediction.
Posterior contraction can coexist with poor credible-set coverage. These are
not paradoxes; they are differently typed claims.

## Summary and further reading {#summary-and-further-reading}

A Gaussian process turns a PSD kernel into a consistent covariance law and
Bayes' rule into a joint posterior process. Its finite values, sample paths,
and RKHS live in different spaces. The posterior mean is kernel ridge
regression under a stated normalization; its spectral filter explains both
smoothing and effective dimension. Posterior covariance is conditional
model uncertainty, not automatic coverage. Marginal likelihood scores a
joint model, LOO scores conditional predictions, and neither eliminates
hyperparameter uncertainty. Learning curves require an eigenvalue regime;
parameter consistency requires an asymptotic regime; contraction and
coverage require separate theorems. Laplace and EP preserve different local
features of a non-Gaussian posterior, while RVM and inducing constructions
obtain sparsity by changing different objects. Modern iterative, structured,
and sketch-based systems extend the computational frontier only after
stating whether they preserve the model, target solve, posterior mean, or
full posterior. The durable reference is [@rasmussen2006].

## Exercises {#exercises}

1.  **(Conditioning a bivariate normal.)** Let \((f_1,f_2)\) be zero-mean Gaussian with covariance \(K=\begin{pmatrix}1&0.75\\0.75&1\end{pmatrix}\). Compute the conditional mean and variance of \(f_2\) given \(f_1=1\). Verify that observing \(f_1\) reduces the variance of \(f_2\) from \(1\) to \(1-0.75^2\).
2.  **(Noise-free limit and rank.)** Assume first that \(K\) is positive definite. Show that as \(\sigma^2\to 0\) the GP posterior mean interpolates the data exactly, \(\bar m(x_i)=y_i\), and \(v(x_i)\to 0\). Then let \(K\) be singular and decompose \(y=y_{\parallel}+y_{\perp}\) with \(y_{\parallel}\in\operatorname{range}(K)\) and \(y_{\perp}\in\ker(K)\). Which component is interpolated in the limit, and why is an arbitrary \(y_\perp\) incompatible with a noise-free degenerate prior?
3.  **(KRR equals GP, symbolically.)** Starting from the ridge objective \(\tfrac1n\|K\alpha-y\|^2+\lambda\alpha^\top K\alpha\), derive the normal equations and confirm the solution \(\alpha=(K+\lambda n I)^{-1}y\). State precisely why the GP posterior mean coincides with this fit but the GP additionally yields a variance. *Hint: identify the loss with the negative log Gaussian likelihood and the penalty with the negative log Gaussian prior.*
4.  **(Marginal-likelihood gradient.)** With \(A=K_\theta+\sigma^2 I\), verify the identities \(\partial\ln\det A=\operatorname{tr}(A^{-1}\partial A)\) and \(\partial A^{-1}=-A^{-1}(\partial A)A^{-1}\), and use them to derive \(\partial\mathcal L/\partial\theta_j\). Then specialize to \(\theta_j=\sigma^2\), where \(\partial A/\partial\sigma^2=I\). *Hint: for the determinant identity differentiate \(\ln\det A=\operatorname{tr}\ln A\).*
5.  **(Occam factor without a slogan.)** Take \(K_\theta=aK_1(\ell)\), diagonalize it, and rewrite the evidence as a sum over eigenmodes. Explain how the data projections \(u_j^\top y\), not “flexibility” alone, determine the competition between the quadratic and log-determinant terms. Construct a simple constant dataset for which the best length scale can lie at or near a boundary, disproving the claim that evidence must always choose an interior compromise. *Difficulty: medium.*
6.  **(RVM pruning as a boundary limit.)** Analyze the update \(s_i\leftarrow(1-s_i\Sigma_{ii})/\mu_i^2\) along a sequence for which \(s_i\to\infty\). Show that both the posterior variance and mean of coefficient \(i\) tend to zero under suitable boundedness of the remaining system. Explain why this is a boundary limit rather than a finite fixed point, and why the \(0/0\) form prevents the update alone from proving attraction or convergence. *Difficulty: hard.*
7.  **(Laplace vs. Gaussian prior.)** Contrast the MAP estimate under a Gaussian coefficient prior (\(-\ln p\propto\sum_i\alpha_i^2\)) with that under a Laplace prior (\(-\ln p\propto\sum_i|\alpha_i|\)). Explain why only the second yields exactly-zero coefficients, using the shape of the penalty near the origin. *Difficulty: medium.*
8.  **(GP classification curvature.)** For the logistic likelihood \(p(y\mid t)=\sigma(yt)\), compute \(c_i=\partial_{t_i}\ln p(y_i\mid t_i)\) and \(C_{ii}=-\partial_{t_i}^2\ln p(y_i\mid t_i)\), and confirm \(0\lt C_{ii}\le\tfrac14\). Explain why log-concavity guarantees the Laplace-approximation mode is unique. *Difficulty: hard. Hint: \(\sigma'=\sigma(1-\sigma)\).*
9.  **(The Nyström gap.)** Show that \(K_{ff}-Q_{ff}\) is the covariance of the conditional \(p(f\mid u)\), hence positive semidefinite, and that if a training input \(x_i\) belongs to the inducing set then the \(i\)th row and column of \(K_{ff}-Q_{ff}\) vanish. Check both claims against the worked sparse example, where \(Z=(x_2,x_3)\) gave \(\operatorname{diag}(K_{ff}-Q_{ff})=(0.1203,\,0,\,0,\,0.2905)\). Conclude that the FITC correction \(\Lambda_{\mathrm{FITC}}-\sigma^2 I\) never subtracts variance. *Difficulty: medium. Hint: a coordinate of \(u\) has zero variance given \(u\), and a positive semidefinite matrix with a zero diagonal entry has a zero row.*
10. **(A bound and a non-bound.)** Show that at \(Z=X\) one has \(Q_{ff}=K_{ff}\), so both the DTC evidence \(\ln\mathcal N(y\mid 0,\ Q_{ff}+\sigma^2 I)\) and the Titsias bound \(\mathcal L_T\) equal the exact log marginal likelihood. Explain why for general \(Z\) only \(\mathcal L_T\) is guaranteed to sit below the exact value, and confirm it on the worked example, where the DTC evidence \(-4.982\) exceeds the exact \(-5.3762\) while \(\mathcal L_T=-7.0364\) respects the bound with gap \(1.6602\). Which of the two is the safe objective for optimizing the positions \(Z\), and why? *Difficulty: medium.*
11. **(Averaging the probit.)** Prove the closed form used in the Laplace predictive step: for \(v\ge 0\),

$$\int\Phi(f)\,\mathcal N(f\mid\mu,v)\,df=\Phi\!\left(\frac{\mu}{\sqrt{1+v}}\right).$$

    Conclude that latent uncertainty always pulls the reported class probability toward \(\tfrac12\) compared with plugging the mean into the link, and that the pull grows with \(v\). *Difficulty: hard. Hint: write \(\Phi(f)=P(Z\le f)\) for \(Z\sim\mathcal N(0,1)\) independent of \(f\sim\mathcal N(\mu,v)\), and read the left side as \(P(Z-f\le 0)\) with \(Z-f\sim\mathcal N(-\mu,\,1+v)\).*
12. **(Prove, break, diagnose: path regularity.)** Prove the increment expansions for Matérn \(\nu=\tfrac12\) and \(\nu=\tfrac32\). Break the claim “a continuous covariance gives differentiable paths” using the difference-quotient variance. Diagnose a fitted one-dimensional process by estimating the log-log slope of empirical increment variance versus lag; state why this diagnostic cannot by itself identify almost-sure differentiability.
13. **(Prove, break, diagnose: singular observations.)** Prove Proposition 40.4 from the block Gaussian identity. Break the noise-free point-observation model with duplicated inputs and incompatible labels. Diagnose whether a diagonal term in software is modeled noise, known measurement variance, a nugget process, or numerical jitter; list the predictive claim permitted by each interpretation.
14. **(Prove, break, diagnose: model selection.)** Derive the exact LOO identities \(\mu_{-i}=y_i-\alpha_i/Q_{ii}\) and \(v_{-i}=1/Q_{ii}\) from a partitioned inverse. Break the claim that evidence and LOO must rank hyperparameters identically using Example 40.7. Diagnose a length-scale optimum with a profile plot, gradient norm, Hessian eigenvalues, and standardized LOO residuals.
15. **(Learning-curve rate.)** In the Gaussian sequence model with \(\lambda_j\asymp j^{-2a}\), split the risk sum at \(J_n\asymp n^{1/(2a)}\) and derive \(R_n\asymp n^{-(2a-1)/(2a)}\). Identify every step that fails if the empirical eigenvectors are random, the truth is fixed outside the prior, or the error is measured in a stronger norm.
16. **(Microergodic ridge.)** For fixed \(\nu\), sketch level curves of \(\sigma_f^2/\rho^{2\nu}\). Explain why dense observations on one bounded domain can sharpen prediction while leaving \(\sigma_f^2\) and \(\rho\) individually weakly identified. Contrast this with increasing-domain data and replicated independent fields.
17. **(Contraction is not coverage.)** Write the quantifiers in the definitions of posterior contraction, posterior credibility, and frequentist coverage. Construct a sequence of intervals centered at a biased estimator whose radii contract at the correct order but whose coverage tends to zero. State which additional bias or self-similarity control would be needed to repair the claim.
18. **(EP moment matching.)** Starting from a Gaussian cavity and a probit factor, derive the tilted mean and variance in the EP section. Reproduce Example 40.8, then reverse the site-update order and remove damping. Report convergence, cavity precisions, and predictive probability rather than assuming the same iteration path.
19. **(Softmax gauge.)** Prove that \(W_i=\operatorname{diag}(\pi_i)-\pi_i\pi_i^\top\) is PSD and annihilates \(\mathbf1\). Explain why this is both a probabilistic invariance and a numerical rank deficiency. Give two valid ways to fix the gauge.
20. **(Capstone: same marginals, different action.)** Reproduce Example 40.9. Then vary the correlation from \(-0.5\) to \(0.95\) and plot the probabilities of \(\max(f_A,f_B)\gt1\) and \((f_A+f_B)/2\gt1\). Explain why no collection of pointwise intervals can recover both curves.
