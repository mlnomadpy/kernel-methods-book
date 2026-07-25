---
id: ch-inverse
slug: inverse-learning-and-spectral-regularization
title: Inverse Learning and Spectral Regularization
part: 'IV · Generalization, Approximation, and Limits'
order: 17
tier: advanced
prerequisites:
  - mercer-and-rates
  - kernel-ridge-and-friends
objectives:
  - >-
    Formulate population and empirical kernel regression as linked inverse
    problems.
  - >-
    Derive spectral filters, residuals, qualification, and saturation from
    functional calculus.
  - >-
    Reconstruct source-capacity rates without hiding the sampling and noise
    assumptions.
  - >-
    Derive a data-dependent early-stopping rule from empirical kernel
    eigenvalues.
  - >-
    Compare regularizers by prediction error, inverse amplification, and
    computation.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-inverse.yml
verification_date: null
bibliography:
  - rosinverse2004
  - caponnetto2007
  - raskutti2014early
  - steinwart2012
  - blanchard2018
---
# Inverse Learning and Spectral Regularization

<p class="lead">Why does stopping gradient descent early prevent overfitting, and why can an interrupted optimizer behave like a ridge penalty? Kernel regression is an inverse problem in disguise. The population operator suppresses directions with small eigenvalues, while estimation tries to reconstruct them from noisy samples. Direct inversion divides by those eigenvalues and can turn a harmless perturbation into the largest component of the answer. Ridge, spectral cutoff, Landweber iteration, and conjugate gradients all decide how much of each direction may pass. Their spectral filters make that decision visible. Source conditions describe the target, effective dimension describes the number of noisy directions available to the estimator, and qualification describes the smoothness a filter can exploit before it saturates. This chapter develops those claims from their operator assumptions through rate calculations, stopping rules, failure witnesses, and a worked spectral audit.</p>

## The two inverse problems hidden in regression {#inverse-formulation}

Let \((\mathcal X,\mathcal A,P_X)\) be a probability space and let \(\mathcal H\) be a separable real RKHS with measurable kernel \(k\). Assume

$$
\sup_{x\in\mathcal X}k(x,x)\leq \kappa^2\lt\infty.
$$

The canonical inclusion \(S:\mathcal H\to L^2(P_X)\), \((Sf)(x)=f(x)\), is bounded because
\(\lVert Sf\rVert_{L^2}\leq\kappa\lVert f\rVert_{\mathcal H}\). Its adjoint is

$$
S^\ast g=\int_{\mathcal X}g(x)k_x\,dP_X(x),
\qquad k_x=k(x,\cdot),
$$

where the Bochner integral exists under the bounded-kernel assumption. Two operators must be kept separate:

$$
T=S^\ast S:\mathcal H\to\mathcal H,
\qquad
L=SS^\ast:L^2(P_X)\to L^2(P_X).
$$

Both are positive and self-adjoint. The operator \(T\) is trace class, with
\(\operatorname{tr}T=\int k(x,x)\,dP_X(x)\leq\kappa^2\). The nonzero spectra of \(T\) and \(L\) agree, but their eigenvectors live in different spaces.

For square-loss regression with \(f_\rho(x)=\mathbb E[Y\mid X=x]\), the population risk is

$$
\mathcal E(f)=\mathbb E\{Y-f(X)\}^2.
$$

If the risk minimizer over \(\overline{S\mathcal H}\) has a minimum-norm preimage \(f_{\mathcal H}\in\mathcal H\), its normal equation is

$$
Tf_{\mathcal H}=S^\ast f_\rho.
$$

The prediction target is \(Sf_{\mathcal H}\), not necessarily \(f_\rho\). A component of \(f_\rho\) orthogonal to \(\overline{S\mathcal H}\) is irreducible approximation error. A component in \(\ker S\) is invisible in prediction and is removed by choosing the minimum-norm preimage.

Given data \((x_i,y_i)_{i=1}^n\), define the sampling operator

$$
S_n f=\frac{1}{\sqrt n}\bigl(f(x_1),\ldots,f(x_n)\bigr)^\top,
\qquad
T_n=S_n^\ast S_n=\frac1n\sum_{i=1}^n k_{x_i}\otimes k_{x_i},
$$

and \(g_n=S_n^\ast y/\sqrt n=n^{-1}\sum_i y_i k_{x_i}\). The empirical normal equation is \(T_n f=g_n\). Learning therefore perturbs both the right-hand side and the operator. Treating it as a fixed-operator inverse problem misses the random discretization error \(T_n-T\).

### Paper module: Rosasco et al. make the analogy literal {#inverse-paper-rosasco}

Before [@rosinverse2004], regularized learning and inverse problems had similar formulas, but the direct operator and the stochastic analogue of observation noise were often left implicit. Their move was to take the RKHS inclusion \(S\) as the forward operator and the empirical sampling map \(S_n\) as its random discretization. This turns regularized least squares into Tikhonov regularization of a perturbed normal equation.

The exact setting in their Sections 4 and 5 uses a continuous bounded kernel, square loss, bounded responses, and i.i.d. sampling. The stochastic perturbations are

$$
\delta_{1,n}=\lVert g_n-S^\ast f_\rho\rVert_{\mathcal H},
\qquad
\delta_{2,n}=\lVert T_n-T\rVert_{\mathcal L(\mathcal H)}.
$$

The contribution is not the formula for ridge. It is the decomposition of learning error into a deterministic regularization argument, expressed in \((\delta_{1,n},\delta_{2,n},\lambda)\), followed by a probabilistic control of the two perturbations. This separates the functional-analytic question from concentration.

For the empirical Tikhonov estimator

$$
\widehat f_\lambda=(T_n+\lambda I)^{-1}g_n,
$$

the resolvent identity gives

$$
\widehat f_\lambda-f_\lambda
=(T_n+\lambda I)^{-1}(g_n-g)
+(T_n+\lambda I)^{-1}(T-T_n)f_\lambda,
$$

where \(g=S^\ast f_\rho\) and \(f_\lambda=(T+\lambda I)^{-1}g\). Since
\(\lVert(T_n+\lambda I)^{-1}\rVert\leq\lambda^{-1}\),

$$
\lVert\widehat f_\lambda-f_\lambda\rVert_{\mathcal H}
\leq \frac{\delta_{1,n}+\delta_{2,n}\lVert f_\lambda\rVert_{\mathcal H}}{\lambda}.
$$

This bound is deliberately crude, but it exposes the mechanism: \(\lambda\) must tend to zero to remove approximation bias, yet not so quickly that empirical perturbations divided by \(\lambda\) explode.

**Failure boundary.** The construction does not identify components in \(\ker S\), does not cover arbitrary losses without a different nonlinear operator, and does not make the inverse bounded. Regularization selects a stable approximation; it cannot recover information absent from the range of the forward operator.

## Spectral filters and what they preserve {#inverse-filter-catalog}

By the spectral theorem, a bounded Borel function \(g_\lambda\) defines
\(g_\lambda(T)\). This makes regularization a scalar decision repeated over every eigendirection.

::: {.definition #def-spectral-filter}
[Definition (spectral regularization family)]{.box-title}

For \(0\lt\lambda\leq1\), a family \(g_\lambda:[0,\kappa^2]\to\mathbb R\) has fitted-value filter

$$
q_\lambda(\mu)=\mu g_\lambda(\mu)
$$

and residual \(r_\lambda(\mu)=1-q_\lambda(\mu)\). It is an admissible spectral regularizer if constants \(D,E,\gamma\) independent of \(\lambda\) satisfy

$$
\sup_\mu |q_\lambda(\mu)|\leq D,\qquad
\sup_\mu |g_\lambda(\mu)|\leq \frac{E}{\lambda},\qquad
\sup_\mu |r_\lambda(\mu)|\leq\gamma,
$$

and \(r_\lambda(\mu)\to0\) for every \(\mu\gt0\) as \(\lambda\to0\).
:::

The population estimator is \(f_\lambda=g_\lambda(T)g\). If \(g=Tf_\star\), then

$$
f_\lambda-f_\star=-r_\lambda(T)f_\star.
$$

This identity is the bias calculation. It says nothing about the empirical operator or noise.

| Method | \(g_\lambda(\mu)\) or \(g_t(\mu)\) | Residual | Main boundary |
|---|---:|---:|---|
| Tikhonov | \((\mu+\lambda)^{-1}\) | \(\lambda/(\mu+\lambda)\) | qualification one |
| Spectral cutoff | \(\mu^{-1}\mathbf1\{\mu\geq\lambda\}\) | \(\mathbf1\{\mu\lt\lambda\}\) | discontinuous under spectral perturbation |
| \(m\)-fold Tikhonov | \(\mu^{-1}\{1-(\lambda/(\mu+\lambda))^m\}\) | \((\lambda/(\mu+\lambda))^m\) | \(m\) solves or factorizations |
| Landweber | \(\{1-(1-\eta\mu)^t\}/\mu\) | \((1-\eta\mu)^t\) | stopping time and step size are part of estimator |
| Gradient flow | \(\{1-e^{-t\mu}\}/\mu\) | \(e^{-t\mu}\) | continuous-time idealization |

## Source conditions, qualification, and saturation {#inverse-source}

Qualification is visible as a slope limit. If a source condition contributes smoothness exponent \(\nu\), an ideal cutoff can continue converting larger \(\nu\) into faster bias decay, whereas first-order Tikhonov regularization cannot exploit smoothness beyond qualification one. The plate isolates this bias mechanism; variance and parameter choice must be added before it becomes a risk comparison.

<figure class="viz" data-figure="spectral-qualification-saturation" data-alt="Two log-log panels compare worst-case bias scales for three source smoothness exponents. Tikhonov curves for exponents one and two coincide, while spectral cutoff retains distinct slopes."><figcaption>Saturation means that a smoother source stops improving the regularization bias. Tikhonov's qualification-one curves merge for \(\nu=1\) and \(\nu=2\); spectral cutoff has no finite qualification in this stylized source calculation. The figure compares bias orders only, not total statistical risk.</figcaption></figure>

A source condition types the target relative to the operator:

$$
f_\star=T^r w,\qquad \lVert w\rVert_{\mathcal H}\leq R.
$$

It is not a generic smoothness statement. Changing \(P_X\), the kernel, or the domain changes \(T\) and therefore changes the class. Under the eigendecomposition \(Te_j=\mu_j e_j\),

$$
\sum_j\frac{|\langle f_\star,e_j\rangle|^2}{\mu_j^{2r}}\leq R^2.
$$

::: {.definition #def-inverse-qualification}
[Definition (qualification)]{.box-title}

A filter has qualification at least \(q\) if, for every \(0\leq\nu\leq q\),

$$
\sup_{0\lt\mu\leq\kappa^2}\mu^\nu |r_\lambda(\mu)|
\leq C_\nu\lambda^\nu.
$$

Qualification is a property of the filter and the chosen scale convention. It is not a property of the target.
:::

:::: {.theorem #thm-ridge-source-bias}
[Theorem (Tikhonov source bias and saturation)]{.box-title}

**Proof status.** Complete below.

Let \(T\) be bounded, positive, and self-adjoint. Suppose \(f_\star=T^r w\) with
\(\lVert w\rVert\leq R\), and let \(f_\lambda=(T+\lambda I)^{-1}Tf_\star\).
For \(s\geq0\) with \(0\leq r+s\leq1\),

$$
\lVert T^s(f_\lambda-f_\star)\rVert
\leq R\lambda^{r+s}.
$$

If \(r+s\gt1\), the same scalar argument gives only

$$
\lVert T^s(f_\lambda-f_\star)\rVert
\leq R\lambda\lVert T\rVert^{r+s-1}.
$$

Thus ordinary Tikhonov has qualification one in this convention.

**Assumptions.** The source condition is measured in the same Hilbert space on which \(T\) acts. The claim is population bias only.
::::

:::: {.proof}
Write \(a=r+s\). Functional calculus and \(r_\lambda(\mu)=\lambda/(\mu+\lambda)\) give

$$
T^s(f_\lambda-f_\star)
=-T^{r+s}r_\lambda(T)w.
$$

For \(0\leq a\leq1\), substitute \(\mu=\lambda u\):

$$
\mu^a\frac{\lambda}{\mu+\lambda}
=\lambda^a\frac{u^a}{1+u}
\leq\lambda^a,
$$

because \(u^a\leq1+u\). Taking the spectral supremum proves the first claim. If \(a\gt1\),

$$
\mu^a\frac{\lambda}{\mu+\lambda}
\leq\lambda\mu^{a-1}
\leq\lambda\lVert T\rVert^{a-1},
$$

which proves the second. Extra source smoothness beyond \(a=1\) therefore cannot improve this bias power without changing the filter. \(\square\)
::::

The failure witness is concrete. Place \(w\) in the top eigendirection. For \(r+s\gt1\), the bias is asymptotically proportional to \(\lambda\mu_1^{r+s-1}\), not \(\lambda^{r+s}\). Saturation is not a defect in the proof; it is achieved by an allowed target.

## Paper module: source, capacity, and minimax rates {#inverse-paper-rates}

Caponnetto and De Vito asked when regularized least squares attains a rate that no learning algorithm can uniformly improve over the same model class [@caponnetto2007]. Their key advance was to couple a source condition with eigenvalue decay instead of describing complexity only by a global RKHS radius.

Their exact scalar specialization assumes a separable RKHS with uniformly bounded evaluation, i.i.d. observations, a Bernstein-type moment condition on \(Y-f_{\mathcal H}(X)\), and existence of a minimum-norm risk minimizer. Let

$$
T=\sum_{j\geq1}\mu_j e_j\otimes e_j.
$$

For fixed constants \(M,\Sigma,R,\alpha,\beta\), their class \(\mathcal P(b,c)\) uses

$$
f_{\mathcal H}=T^{(c-1)/2}g,\qquad \lVert g\rVert_{\mathcal H}^2\leq R,
$$

with \(1\leq c\leq2\), and for \(1\lt b\lt\infty\),

$$
\alpha\leq j^b\mu_j\leq\beta
\quad\text{for every }j.
$$

The finite-rank case is denoted \(b=\infty\). These are joint assumptions: \(c\) controls target alignment, \(b\) controls capacity, and the moment condition controls stochastic fluctuations.

:::: {.theorem #thm-caponnetto-rate}
[Theorem (source-capacity rate, after Caponnetto and De Vito)]{.box-title}

**Assumptions.** The source, capacity, bounded-kernel, and noise assumptions stated below are all in force.

**Proof status.** Reconstructed proof skeleton below; constants remain with the cited primary theorem.

For \(1\lt b\lt\infty\) and \(1\lt c\leq2\), choose

$$
\lambda_n=n^{-b/(bc+1)}.
$$

Under the \(\mathcal P(b,c)\) assumptions above, the excess square-loss risk of kernel ridge regression is uniformly

$$
\mathcal E(\widehat f_{\lambda_n})-\mathcal E(f_{\mathcal H})
=O_{\mathbb P}\!\left(n^{-bc/(bc+1)}\right).
$$

For finite-dimensional outputs, the paper proves a matching minimax lower exponent over the same prior class. At \(c=1\), its upper result carries a logarithmic boundary term.

**Source locator.** Definition 1 and Theorems 1 and 2 of [@caponnetto2007].
::::

The exponent can be reconstructed without reproducing the paper's concentration constants. Since \(\mu_j\asymp j^{-b}\),

$$
\mathcal N(\lambda)
=\operatorname{tr}\{T(T+\lambda I)^{-1}\}
=\sum_j\frac{\mu_j}{\mu_j+\lambda}
\asymp\lambda^{-1/b}.
$$

The source condition corresponds in prediction space to a squared bias of order \(\lambda^c\). The stochastic term is of order \(\mathcal N(\lambda)/n\). Balancing

$$
\lambda^c\asymp\frac{\lambda^{-1/b}}{n}
$$

gives \(\lambda\asymp n^{-b/(bc+1)}\) and risk
\(n^{-bc/(bc+1)}\). This calculation explains the exponent; the theorem additionally requires operator concentration, noise control, and a lower-bound construction.

**Failure boundary.** Universality alone supplies neither \(b\) nor \(c\). An eigenvalue upper bound can support an upper rate, but the matching lower rate needs a lower eigenvalue condition and an explicitly defined distribution class. If the target is outside the source class or the noise lacks the required moments, the theorem does not transfer by analogy. General spectral methods and statistical inverse problems require their own qualification and link assumptions; [@blanchard2018] develops that broader setting.

## Interpolation spaces and misspecification {#inverse-interpolation-spaces}

The power spaces of \(L=SS^\ast\) express target regularity even when the target is not an RKHS element.

::: {.definition #def-inverse-power-space}
[Definition (power space)]{.box-title}

For \(\theta\geq0\), define

$$
[\mathcal H]^\theta=\operatorname{ran}L^{\theta/2}
=\left\{\sum_j a_j\mu_j^{\theta/2}\psi_j:
\sum_j a_j^2\lt\infty\right\},
$$

with the minimum-preimage norm and with the null space of \(L\) factored out.
:::

The endpoint \(\theta=0\) is the closure of the positive eigenspace in \(L^2(P_X)\), while \(\theta=1\) is the embedded RKHS. Under the compact-embedding assumptions made explicit in [@steinwart2012], intermediate \(\theta\) describes interpolation between them. The identification depends on both kernel and measure. It must not be replaced by an unqualified statement such as “a Matérn RKHS is a Sobolev space” on an arbitrary domain.

Misspecified regression often places \(f_\rho\) in \([\mathcal H]^\theta\) with \(\theta\lt1\). Then the target can be approximated in prediction norm even though no finite RKHS norm exists. Bounds in \(\mathcal H\)-norm may be meaningless while \(L^2(P_X)\) bounds remain valid.

## Paper module: early stopping from empirical complexity {#inverse-iterative}

Raskutti, Wainwright, and Yu asked for a stopping rule computed from the training design, without a hold-out set, that achieves the kernel class's statistical scale [@raskutti2014early]. Their result is more specific than the slogan “gradient descent regularizes.”

Fix the design \(x_1,\ldots,x_n\), let \(\widehat K=K/n\), and write its eigenvalues as
\(\widehat\mu_1\geq\cdots\geq\widehat\mu_n\geq0\). Assume

- \(f_\star\) lies in the unit ball of an RKHS whose functions are uniformly bounded;
- \(y_i=f_\star(x_i)+w_i\), where the \(w_i\) are independent, mean-zero, and sub-Gaussian with parameter \(\sigma\);
- the iteration starts at zero;
- step sizes \(\alpha_t\) are nonincreasing, satisfy
  \(0\leq\alpha_t\leq\min\{1,\widehat\mu_1^{-1}\}\), and have divergent cumulative travel
  \(\eta_t=\sum_{\tau=0}^{t-1}\alpha_\tau\).

Define the empirical local complexity

$$
\widehat{\mathcal R}_K(\varepsilon)
=\left\{\frac1n\sum_{j=1}^n
\min(\widehat\mu_j,\varepsilon^2)\right\}^{1/2}
$$

and let \(\widehat\varepsilon_n\) be the smallest positive solution of

$$
\widehat{\mathcal R}_K(\varepsilon)
\leq \frac{\varepsilon^2}{2e\sigma}.
$$

Their stopping rule takes the last \(t\) before
\(\widehat{\mathcal R}_K(\eta_t^{-1/2})\) exceeds
\((2e\sigma\eta_t)^{-1}\).

:::: {.theorem #thm-raskutti-stopping}
[Theorem (fixed-design early stopping, after Raskutti et al.)]{.box-title}

**Assumptions.** The fixed-design regression, sub-Gaussian noise, step-size, and empirical critical-radius assumptions stated below are all in force.

**Proof status.** Reconstructed proof skeleton below; the full concentration argument is in the cited primary source.

Under the assumptions above, universal constants \(c_1,c_2\gt0\) exist such that, with probability at least

$$
1-c_1\exp(-c_2n\widehat\varepsilon_n^2),
$$

the stopped iterate satisfies

$$
\lVert \widehat f_{\widehat T}-f_\star\rVert_n^2
\leq 12\widehat\varepsilon_n^2.
$$

The paper also gives a variance lower bound after the stopping time and a random-design extension under a population critical-radius condition.

**Source locator.** Equations (4) to (6) and Theorem 1 of [@raskutti2014early].
::::

The central derivation is spectral. For constant step \(\eta\), gradient descent on the empirical square loss yields

$$
q_t(\mu)=1-(1-\eta\mu)^t,
\qquad
r_t(\mu)=(1-\eta\mu)^t.
$$

In the eigenbasis of \(\widehat K\), write \(y=s+w\). The fitted vector is
\(\widehat s_t=q_t(\widehat K)y\), so

$$
\widehat s_t-s
=-r_t(\widehat K)s+q_t(\widehat K)w.
$$

The first term is bias and decreases with \(t\); the second is variance and increases as more weak directions pass. For \(0\leq\eta\mu\leq1\),

$$
q_t(\mu)\leq\min\{1,\eta t\mu\},
\qquad
r_t(\mu)\leq e^{-\eta t\mu}.
$$

Consequently, the variance scale contains

$$
\frac{\sigma^2}{n}\sum_j q_t(\widehat\mu_j)^2
\leq \frac{\sigma^2\eta t}{n}
\sum_j\min\{\widehat\mu_j,(\eta t)^{-1}\},
$$

which is the empirical complexity evaluated at \((\eta t)^{-1/2}\). The stopping rule is therefore a computable bias-variance balance, not an analogy to ridge.

**Failure boundary.** The theorem does not license arbitrary adaptive optimizers, arbitrary initialization, unknown-noise substitution, classification losses, or unrestricted multiple passes of stochastic gradient descent. Its empirical-norm theorem is conditional on the design. The random-design conclusion needs an additional comparison between empirical and population complexity.

## A four-direction inverse audit {#inverse-example}

A useful example should report both prediction and instability. Consider a normalized empirical operator with eigenvalues

$$
\widehat\mu=(1,\ 0.25,\ 0.04,\ 0.0025),
$$

signal coordinates \(s=(2,1,0.5,0.25)\), and observed coordinates
\(y=s+(0.05,-0.05,0.05,-0.05)\).

::: {.example #example-inverse-filter}
[Example (prediction can look stable while inverse coefficients explode)]{.box-title}

Apply four fitted-value filters: interpolation, ridge with \(\lambda=0.05\), cutoff at \(0.05\), and ten Landweber steps with \(\eta=0.8\).

| Method | Retained fractions \(q_j\) | Prediction MSE to \(s\) | Inverse-coordinate norm \(\lVert(q_jy_j/\widehat\mu_j)_j\rVert_2\) |
|---|---|---:|---:|
| interpolation | \(1,1,1,1\) | \(0.002500\) | \(81.288\) |
| ridge | \(0.9524,0.8333,0.4444,0.0476\) | \(0.042202\) | \(8.105\) |
| cutoff | \(1,1,0,0\) | \(0.079375\) | \(4.318\) |
| Landweber | \(1,0.8926,0.2776,0.0198\) | \(0.051689\) | \(5.727\) |

Interpolation wins this one low-noise prediction comparison, yet its inverse norm is ten to nineteen times larger. A small change in the fourth observation is divided by \(0.0025\). The example therefore rejects two shortcuts: the smallest training or oracle prediction error need not identify the most stable inverse, and coefficient norm alone does not identify the best predictor.

**Verification artifact.** checks/example-ch-inverse-example-inverse-filter.json records the example source hash and verification scope.
:::

## Parameter choice, Krylov methods, and computation {#inverse-parameter-choice}

The regularization path is incomplete until its selection rule is specified.

- A discrepancy principle uses a known or independently estimated noise scale.
- Cross-validation estimates prediction risk but must respect dependence and preprocessing.
- Generalized cross-validation uses the trace of a linear smoother and can fail under leverage heterogeneity.
- Lepskiĭ-type rules compare scales and require a stochastic envelope.
- Marginal likelihood selects a variance ratio under a probabilistic model, not an arbitrary downstream loss.

Conjugate gradients add another complication. At step \(t\), the estimate lies in

$$
\mathcal K_t(T_n,g_n)
=\operatorname{span}\{g_n,T_ng_n,\ldots,T_n^{t-1}g_n\}.
$$

Its residual polynomial is data dependent. Fast reduction of the linear-system residual does not imply stable reconstruction, because the polynomial can begin resolving noisy weak directions. Preconditioning changes both the Krylov space and the implicit regularizer. It must be reported as part of the estimator.

:::: {.algorithm #algo-inverse-comparison}
[Algorithm (auditable spectral-regularizer comparison)]{.box-title}

1. Fix the normalization of \(K\), the prediction norm, and the source of the noise estimate.
2. Estimate spectral bounds and verify positive self-adjoint matrix products.
3. Run ridge, cutoff, Landweber, and conjugate-gradient paths at matched kernel-product budgets.
4. Record prediction residual, validation loss, solution norm, effective degrees of freedom, and inverse amplification.
5. Apply the stopping or selection rule without test reuse.
6. Perturb observations at the declared noise scale and rerun the selected estimator.
7. Report preconditioner, precision, jitter, convergence threshold, and whether the rule or compute budget stopped the solve.
::::

Stochastic gradients require a separate analysis. Mini-batch size, sampling scheme, averaging, and number of passes jointly determine the estimator. There may be no deterministic scalar filter. Calling every undertrained stochastic optimizer “spectral regularization” hides this distinction.

## What the methods can and cannot claim {#inverse-comparisons}

| Claim | Needed evidence | What does not suffice |
|---|---|---|
| deterministic convergence | filter consistency and perturbation control | decreasing training loss |
| source-dependent rate | source exponent, filter qualification, noise, parameter rule | target described as “smooth” |
| capacity-dependent rate | effective dimension or eigenvalue assumptions | universality |
| minimax optimality | matching lower bound over the same class | one upper bound |
| adaptive rate | data-driven rule independent of unknown exponents | test-set selection |
| stable inverse | perturbation sensitivity in the target norm | accurate fitted values |

The same framework appears in conditional mean operators, instrumental-variable equations, vector-valued regression, deconvolution, and scientific inverse problems. In each case, the operator, its range, the norm of interest, and the source condition must be retyped. A regularized solution outside the identifiable range is not recovered truth.

## Common mistakes and practical implications {#inverse-practice}

- \(S^\ast S\) and \(SS^\ast\) have related spectra but act on different spaces.
- The empirical problem perturbs the operator and the right-hand side.
- Source conditions are relative to \(T\), not universal smoothness labels.
- Qualification limits exploitable source regularity.
- Effective dimension controls a variance scale, not numerical conditioning by itself.
- Early stopping includes initialization, step sizes, and the stopping rule.
- A small residual can coexist with an unstable inverse.
- A minimax exponent is meaningful only with its model class and norm.
- Finite precision adds an implicit cutoff that can dominate the mathematical filter.

## Summary and further reading {#inverse-summary}

Inverse learning begins with a typed operator equation. Spectral filters stabilize it by replacing division by \(\mu\) with a controlled function \(g_\lambda(\mu)\). Rosasco et al. identify sampling as random operator discretization [@rosinverse2004]. Caponnetto and De Vito couple source and capacity assumptions to obtain matching rate exponents [@caponnetto2007]. Raskutti et al. derive a computable early-stopping rule from empirical eigenvalues and local complexity [@raskutti2014early]. The transferable lesson is not that every method is secretly ridge. It is that bias, stochastic variance, qualification, identifiability, and numerical stability must be audited in the same spectral coordinates.

## Exercises {#exercises}

1. [warm-up]{.ex-tag} Let \(T\) be positive and self-adjoint. Derive \(q_\lambda\) and \(r_\lambda\) for Tikhonov regularization, and explain their limits as \(\mu/\lambda\) tends to zero and infinity.
2. [computation]{.ex-tag} Reproduce every retained fraction, prediction MSE, and inverse-coordinate norm in the four-direction worked example.
3. [proof]{.ex-tag} Prove the Tikhonov source-bias theorem, including the saturation bound for \(r+s\gt1\), and give a one-eigendirection target that attains order \(\lambda\).
4. [proof]{.ex-tag} Starting from zero, diagonalize constant-step Landweber iteration and derive \(q_t(\mu)=1-(1-\eta\mu)^t\). Show that stability fails for an eigendirection when \(\eta\mu\gt2\).
5. [proof]{.ex-tag} If \(\mu_j\asymp j^{-b}\) with \(b\gt1\), prove \(\mathcal N(\lambda)\asymp\lambda^{-1/b}\) by splitting the sum at the index where \(\mu_j\) is comparable to \(\lambda\). Balance bias \(\lambda^c\) and variance \(\mathcal N(\lambda)/n\).
6. [synthesis]{.ex-tag} A paper claims the rate \(n^{-bc/(bc+1)}\) because its kernel is universal and its target is smooth. List the missing assumptions required to invoke the source-capacity theorem.
7. [synthesis]{.ex-tag} Compare ridge, spectral cutoff, and early stopping under a fixed budget of twenty kernel matrix-vector products. Specify a fair parameter grid, at least four diagnostics, and a failure perturbation.
8. [challenge]{.ex-tag} For a positive empirical operator \(T_n\), prove the resolvent identity used in the Rosasco module and derive a prediction-norm perturbation bound that improves on the displayed RKHS-norm bound by using \(\lVert T_n^{1/2}(T_n+\lambda I)^{-1}\rVert\leq(2\sqrt\lambda)^{-1}\).
