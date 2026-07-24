---
id: ch-inverse
slug: inverse-learning-and-spectral-regularization
title: Inverse Learning and Spectral Regularization
part: XIV · Advanced Extensions
order: 49
tier: advanced
prerequisites:
  - mercer-and-rates
  - kernel-ridge-and-friends
objectives:
  - >-
    Diagonalize a kernel inverse problem and identify noise amplification in
    weak directions.
  - >-
    Compare Tikhonov, cutoff, Landweber, and conjugate-gradient regularization
    by their filters.
  - Translate source conditions into interpolation-space membership.
  - >-
    State qualification and saturation together with the assumptions needed for
    a rate.
  - >-
    Select a stopping or regularization level with residual, conditioning, and
    validation diagnostics.
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
---
# Inverse Learning and Spectral Regularization

<p class="lead">Why does stopping gradient descent early prevent overfitting, and why does it feel like the same medicine as adding a ridge penalty? The two look nothing alike: one is a term in the objective, the other an interrupted optimizer. The resolution is that kernel regression is an inverse problem. A compact operator has attenuated the target's high-frequency directions, and recovering them means dividing by eigenvalues that approach zero, so small perturbations in the data can dominate the solution. Every cure damps that division, and every cure is a spectral filter: ridge shrinks smoothly, spectral cutoff deletes weak directions outright, and gradient descent stopped at iteration \(t\) is a filter in which the iteration count plays the role of an inverse regularization parameter. Once the filter is the object of study, methods as different as conjugate gradients and stochastic iteration become comparable, and source conditions, qualification, and saturation say precisely where each method's accuracy ceiling sits.</p>

## Learning as an inverse problem {#inverse-formulation}

The claim that learning is an inverse problem should be made literal: which operator stands between us and the target, and why is undoing it dangerous? Let \(S:\mathcal H_k\to L^2(P_X)\) be the canonical inclusion, \((Sf)(x)=f(x)\). Two related operators appear:

$$
T=S^\ast S:\mathcal H_k\to\mathcal H_k,
\qquad
L=SS^\ast:L^2(P_X)\to L^2(P_X).
$$

The first is the RKHS covariance operator,
\(Tf=\int k_x\langle k_x,f\rangle_{\mathcal H_k}\,dP_X(x)\), with \(k_x=k(x,\cdot)\); the second is the usual kernel integral operator. Their nonzero eigenvalues agree, and \(S\) maps corresponding eigendirections between the two spaces. The population normal equation in \(\mathcal H_k\) is \(Tf=g\). When these operators are compact, their eigenvalues \(\mu_j\) approach zero. Direct inversion multiplies perturbations in the \(j\)-th coordinate by \(1/\mu_j\), so empirical noise can dominate the solution [@rosinverse2004].

::: {.definition #def-spectral-filter}
[Definition (spectral regularization family)]{.box-title}

A family \(g_\lambda:[0,\kappa^2]\to\mathbb{R}\) defines the estimator \(f_\lambda=g_\lambda(T)g\). Its residual function is \(r_\lambda(\mu)=1-\mu g_\lambda(\mu)\). A regularization family keeps \(g_\lambda\) bounded for fixed \(\lambda\) and makes \(r_\lambda(\mu)\) tend to zero for every positive \(\mu\) as \(\lambda\) tends to zero.
:::

Tikhonov regularization uses \(g_\lambda(\mu)=1/(\mu+\lambda)\). Spectral cutoff uses \(g_\lambda(\mu)=\mu^{-1}\mathbf{1}\{\mu\ge\lambda\}\). Both approach inversion, but one damps continuously and the other deletes weak directions.

## Iterative regularization and early stopping {#inverse-iterative}

Nothing in gradient descent mentions a penalty, yet stopping it early has long been observed to act like one. The filter formalism turns that observation into an identity. Gradient descent on squared loss, initialized at zero with step \(\eta\), gives after \(t\) iterations

$$
g_t(\mu)=\frac{1-(1-\eta\mu)^t}{\mu},
\qquad r_t(\mu)=(1-\eta\mu)^t.
$$

Thus the iteration count is a regularization parameter. Large eigenvalues are learned first; small ones remain damped until later. This is iterative regularization, not explicit ridge. The two filters can have comparable effective scales, but they are not algebraically identical.

:::: {.proposition #prop-landweber-stability}
[Proposition (Landweber stability)]{.box-title}

If \(0\lt\eta\le 1/\lVert T\rVert\), then \(0\le r_t(\mu)\le 1\) on the spectrum of \(T\), and \(r_t(\mu)\) decreases to zero for every \(\mu\gt 0\).

**Assumptions.** \(T\) is bounded, self-adjoint, and positive. **Proof status.** Proved by scalar functional calculus: \(1-\eta\mu\) lies in \([0,1]\).
::::

Data-dependent stopping can balance bias and noise. A discrepancy rule stops when the residual reaches an estimate of the observation noise. Cross-validation is more robust when the noise model is unknown, but its search range and reuse of validation data must be reported.

## Source conditions, qualification, and saturation {#inverse-source}

How much bias a filter leaves depends on the target: one concentrated on strong eigendirections is recovered easily, one buried in the weak tail is nearly hopeless. Source conditions measure where between those extremes the target sits. A source condition writes the target as

$$
f_\star=T^r w,\qquad \lVert w\rVert\le R.
$$

Larger \(r\) means the target is better aligned with stable, high-eigenvalue directions. The bias is controlled through \(\mu^r r_\lambda(\mu)\). A filter has qualification \(q\) when this quantity is bounded at the expected regularization scale for source exponents up to \(q\). Beyond that range, additional target smoothness may not improve the method's rate; this ceiling is called saturation.

::: {.remark #remark-inverse-rates}
[Proof-status warning]{.box-title}

No standalone convergence rate follows from a source condition. A valid rate also needs assumptions on eigenvalue or effective-dimension decay, noise, sampling, and the rule selecting \(\lambda\) or \(t\). Statements may be upper bounds, lower bounds, or minimax equivalences, and those statuses must remain separate [@caponnetto2007].
:::

## Interpolation spaces and the source ladder {#inverse-interpolation-spaces}

The source condition reads as a statement about a single target, but it secretly defines a family of function spaces, and making that family explicit is what lets the theory speak about targets the RKHS does not contain. If \(f_\star = T^r w\) for some \(w\), then \(f_\star\) belongs to the range of \(T^r\), and as \(r\) varies these ranges form a nested ladder: larger \(r\) gives a smaller, smoother class, smaller \(r\) a larger, rougher one.

::: {.definition #def-inverse-power-space}
[Definition (power spaces of the integral operator)]{.box-title}

For \(\theta \ge 0\), use the integral operator \(L=SS^\ast\) on \(L^2(P_X)\) and define the power space, modulo the null space of \(L\), by

$$
[\mathcal{H}]^{\theta} := \operatorname{ran} L^{\theta/2}
= \Big\{ \textstyle\sum_i a_i\, \mu_i^{\theta/2}\, \psi_i \;:\; \textstyle\sum_i a_i^2 \lt \infty \Big\},
$$

with the quotient norm induced by the minimum-norm preimage under \(L^{\theta/2}\). Here \((\mu_i,\psi_i)\) is the positive eigensystem of \(L\) on \(L^2(P_X)\).
:::

The endpoints anchor the ladder on the closure of the positive eigenspace: \(\theta=0\) gives the relevant \(L^2(P_X)\) space, and \(\theta=1\) gives the image of the RKHS under \(S\). Between them, \(0\lt\theta\lt1\) describes targets rougher than RKHS members yet spectrally controlled; under the standard compact-embedding hypotheses these are interpolation spaces between \(L^2(P_X)\) and the embedded RKHS [@steinwart2012]. Values \(\theta\gt1\) describe extra alignment with stable directions, where filter qualification can become the bottleneck. For Matérn kernels on regular bounded domains with a compatible sampling measure, these power spaces are norm-equivalent to an associated Sobolev smoothness scale; the identification is not automatic on an arbitrary domain or measure.

The payoff is a precise language for misspecification. An RKHS target is the case \(\theta=1\); smaller positive \(\theta\) permits rougher targets, with rates that depend jointly on this source exponent, effective-dimension decay, noise, and parameter choice. The exact relation between a filter's qualification \(q\) and the largest exploitable \(\theta\) depends on the convention used for the source power, so a theorem must state both rather than quote a bare factor of two. Reading the ladder alongside effective dimension gives the modern rate picture: one exponent describes target alignment, one decay law describes capacity, and the loss and noise model complete the bound.

## A finite spectral diagnostic {#inverse-example}

Three eigenvalues are enough to see the whole mechanism in numbers.

::: {.example #example-inverse-filter}
[Example (three spectral directions)]{.box-title}

Suppose the Gram eigenvalues are \(10,1,0.01\), with ridge parameter \(\lambda=0.1\). The fitted-value filter \(\mu/(\mu+\lambda)\) is approximately \(0.990,0.909,0.091\). The weak third direction is strongly suppressed. Direct inversion would instead amplify its coefficient by \(100\). A numerical report should show both the data-fit residual and this spectral amplification profile.

**Verification artifact.** checks/example-ch-inverse-example-inverse-filter.json records the example source hash and verification scope.
:::

:::: {.algorithm #algo-spectral-regularization}
[Algorithm (matrix-free early-stopped kernel regression)]{.box-title}

1. Implement the product \(v\mapsto Kv\) without requiring a stored dense matrix when possible.
2. Estimate an upper spectral bound and choose a stable step size.
3. Iterate the residual update from zero coefficients.
4. At checkpoints, record training residual, validation error, and coefficient norm.
5. Stop by a preregistered discrepancy or validation rule; retain the entire trace for reproducibility.
6. Verify the final linear-system residual and compare against a ridge baseline at matched effective degrees of freedom.
::::

## A catalog of spectral filters {#inverse-filter-catalog}

Once each method is reduced to its filter, the whole catalog fits in a table. Regularization methods can be compared by their fitted-value filter \(q_\lambda(\mu)=\mu g_\lambda(\mu)\) and residual \(r_\lambda=1-q_\lambda\).

| Method | Inverse filter \(g\) | Main behavior |
|---|---|---|
| Tikhonov | \((\mu+\lambda)^{-1}\) | smooth shrinkage |
| Iterated Tikhonov | repeated resolvent product | higher qualification |
| Spectral cutoff | \(\mu^{-1}\mathbf 1\{\mu\ge\lambda\}\) | hard deletion |
| Landweber | \(\{1-(1-\eta\mu)^t\}/\mu\) | early-stopped iteration |
| Gradient flow | \(\{1-\exp(-t\mu)\}/\mu\) | continuous-time shrinkage |

Cutoff has excellent bias behavior on retained directions but is discontinuous in the estimated spectrum. Tikhonov is stable and convenient but saturates for sufficiently smooth source conditions. Iteration can increase qualification, although numerical and sampling errors accumulate.

<figure class="viz" data-figure="spectral-filters" data-alt="A logarithmic eigenvalue axis compares three retained-fraction curves. Ridge rises smoothly, spectral cutoff jumps at the regularization threshold, and early stopping rises gradually with a shape distinct from ridge."><figcaption>Regularizers differ by which eigendirections they allow through: cutoff makes a hard decision, ridge shrinks continuously, and early stopping learns strong directions before weak ones. Matching methods by the label \(\lambda\) or by iteration count is meaningless unless their retained spectra or effective degrees of freedom are also matched.</figcaption></figure>

Accelerated first-order methods produce polynomial filters. Their optimization acceleration does not automatically imply better statistical regularization: oscillating residual polynomials can amplify noisy directions. Every accelerated method needs a spectral stability analysis together with its stopping rule.

## Conjugate gradients as regularization {#inverse-conjugate-gradients}

The solver most often reached for on large symmetric systems is itself a regularizer, and a subtler one than anything in the table. Conjugate gradients applied to the normal equations chooses, at iteration \(t\), the best solution in a data-dependent Krylov space. Its filter is a polynomial whose roots depend on the observed spectrum. Large, well-estimated eigendirections are often resolved early.

Unlike Landweber, conjugate-gradient filters are data dependent and not monotone direction by direction. The method can converge rapidly in linear-system residual while beginning to fit noise in weak directions. Early stopping therefore remains necessary.

Preconditioning changes the spectrum and the Krylov subspace. If stopping time is selected after preconditioning, the preconditioner is part of the statistical estimator. Matching two methods by iteration count is meaningless; match effective degrees of freedom, residual scale, or validation risk.

## Choosing the regularization level {#inverse-parameter-choice}

Every filter leaves one number undetermined, \(\lambda\) or \(t\), and the entire statistical behavior hangs on it. Parameter-choice rules use different information:

- **Discrepancy principle:** stop when the data residual reaches a known noise scale.
- **Hold-out or cross-validation:** select predictive error on untouched observations.
- **Generalized cross-validation:** exploit linear smoother structure and approximate leverage.
- **Balancing or Lepskiĭ rules:** compare estimators across scales and stop when changes become compatible with noise.
- **Marginal likelihood:** choose a probabilistic variance ratio under a GP model.

No rule is universally adaptive. A discrepancy principle is sensitive to noise estimation and forward-model error. Random cross-validation can violate dependence. Marginal likelihood targets its model evidence, not an arbitrary downstream loss.

::: {.proposition #prop-inverse-discrepancy}
[Proposition (monotone discrepancy crossing for Landweber)]{.box-title}

For a positive self-adjoint empirical operator, stable Landweber step size, and zero initialization, the training residual norm is nonincreasing with iteration. Hence the first crossing of a fixed discrepancy threshold is well defined whenever the limiting residual lies below that threshold.

**Assumptions.** Exact arithmetic, a fixed positive self-adjoint operator, step size in the stable interval, and a reachable threshold. **Proof status.** Proved by diagonalizing the residual update and observing that each spectral residual magnitude is multiplied by a factor in the unit interval.
:::

Model discrepancy can prevent the residual from reaching the nominal noise scale. The correct response is not unlimited iteration; it is to revise the noise or forward model.

## Statistical rates and effective dimension {#inverse-statistical-rates}

Bias is only half of the risk; the other half counts how many directions the filter leaves open to noise. A source condition controls bias, while the effective dimension

$$
\mathcal N(\lambda)=\operatorname{tr}\{T(T+\lambda I)^{-1}\}
$$

controls variance for Tikhonov-type methods. Under eigenvalue decay and noise assumptions, risk is bounded by a bias term depending on \(r\) and a variance term involving \(\mathcal N(\lambda)/n\). Balancing them selects the statistical scale [@caponnetto2007].

Upper rates should state whether they are in prediction norm, RKHS norm, or another interpolation norm. A minimax claim additionally needs a matching lower bound over a specified source and capacity class. Adaptivity means achieving the rate without knowing those exponents; selecting the best rate after observing the test set is not adaptivity.

## Stochastic and online iterative regularization {#inverse-stochastic}

Stochastic gradient methods introduce an algorithmic noise source in addition to observation noise. Step size, mini-batch size, averaging, sampling with or without replacement, and number of passes jointly define the filter-like behavior. Early stopping can regularize, but there may be no single deterministic polynomial filter.

Multiple passes reduce optimization bias and can eventually fit observation noise. Tail averaging and decaying step sizes change both variance and implicit weighting of iterations. Compare stochastic and deterministic methods at matched kernel products or wall time, and record the full learning curve.

## Beyond scalar regression {#inverse-beyond-scalar}

Nothing in the filter story used the fact that the unknown was a scalar regression function; wherever a compact operator separates the data from the target, the same analysis applies. The same framework applies to:

- conditional mean embeddings, where a covariance operator is inverted;
- kernel instrumental variables and proximal causal equations;
- vector-valued regression with block covariance operators;
- scientific inverse problems with linearized forward maps;
- deconvolution and inverse source recovery.

In each case, compactness and partial identification determine which directions can be recovered. Regularization selects a stable solution but cannot manufacture information in the operator null space. The scientific treatment in [[ch:scientific-computing-and-operator-learning]] distinguishes operator discretization from statistical noise.

:::: {.algorithm #algo-inverse-comparison}
[Algorithm (spectral regularizer comparison)]{.box-title}

**Input.** A matrix-free empirical operator, observations, a noise estimate or validation set, candidate filters, and a compute budget.

**Output.** A selected estimator with filter and stability diagnostics.

1. Estimate spectral bounds and verify self-adjoint positive products.
2. Run ridge, Landweber, and conjugate-gradient paths at matched operator-product checkpoints.
3. Record residual, validation loss, solution norm, effective degrees of freedom where available, and spectral amplification.
4. Apply a preregistered discrepancy or validation rule without test reuse.
5. Perturb observations at the estimated noise scale and measure solution sensitivity.
6. Report the selected filter, parameter or stopping time, preconditioner, arithmetic precision, and final residual.

The solve stops only when the selection rule is met or the compute budget is exhausted; either outcome is part of the result.
::::

## Common mistakes and practical implications {#inverse-practice}

- Early stopping is a regularizer only together with an initialization, step-size schedule, and stopping rule.
- A small training residual does not certify a stable inverse.
- Source conditions constrain the target relative to \(T\); they are not generic smoothness labels.
- Qualification is filter-specific, and saturation is not universal across algorithms.
- Finite precision creates an additional cutoff near machine accuracy.

Matrix-free iteration can reduce memory and exploit fast kernel products. Preconditioning changes convergence across spectral directions, so its interaction with early stopping must be treated as part of the estimator rather than an invisible implementation detail.

## Summary and further reading {#inverse-summary}

Compact covariance operators turn learning into an ill-posed inverse problem. Spectral filters stabilize the inverse by damping weak directions. Ridge, cutoff, and iterative methods use different filters; source conditions and qualification explain their bias, while noise and effective dimension govern variance. See [@rosinverse2004] for the learning-to-inverse-problem bridge, [@caponnetto2007] for statistical rates, and [@raskutti2014early] for early stopping.

## Exercises {#exercises}

1. [warm-up]{.ex-tag} Plot the ridge residual \(r_\lambda(\mu)\) for three values of \(\lambda\) and explain which spectral directions remain biased.
2. [computation]{.ex-tag} Compute the first five Landweber fitted-value filters for \(\mu\in\{1,0.1,0.01\}\) with \(\eta=1/2\).
3. [proof]{.ex-tag} Prove the Landweber stability proposition and show what can fail when \(\eta\gt 2/\lVert T\rVert\).
4. [challenge]{.ex-tag} For a source condition \(f_\star=T^rw\), derive a uniform ridge-bias bound from \(\sup_\mu \mu^r\lambda/(\mu+\lambda)\). State the range of \(r\) for which your bound has the asserted power of \(\lambda\).
