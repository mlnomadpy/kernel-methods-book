---
id: ch-dkl
slug: deep-kernel-learning
title: Deep Kernel Learning
part: XIV · Advanced Extensions
order: 50
tier: advanced
prerequisites:
  - gaussian-processes-and-rvm
  - kernels-and-deep-learning
  - large-scale-kernels
objectives:
  - >-
    Construct a positive-definite kernel by composing a trainable feature map
    with a base kernel.
  - >-
    Differentiate the marginal likelihood and interpret its fit and
    covariance-volume terms.
  - 'Distinguish DKL from fixed NTK, deep-GP, and frozen-feature baselines.'
  - >-
    Choose among exact, inducing-point, and structured-interpolation
    computation.
  - >-
    Detect representation collapse, scale non-identifiability, and unsupported
    OOD confidence.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-dkl.yml
verification_date: null
bibliography:
  - wilson2016dkl
  - wilson2015kissgp
  - ober2021dkl
---
# Deep Kernel Learning

<p class="lead">A neural network learns the representation a task demands but returns a bare number; a Gaussian process returns honest uncertainty but only over the geometry it is handed. Deep kernel learning promises both at once: push the inputs through a trainable feature map, place a positive definite kernel on the learned coordinates, and train the whole stack by marginal likelihood, so predictions carry task-specific similarity and a predictive covariance. The promise is real, but naive joint optimization can quietly break both halves. The feature map can collapse distinct inputs to a point while the likelihood keeps improving through noise and mean terms, scale symmetries leave the objective flat and the parameters uninterpretable, and the error bars can end up most confident exactly where the data give the least support. This chapter builds the model, derives the marginal-likelihood gradient that trains it, scales it with inducing points and structured interpolation, and assembles the diagnostics that separate a working deep kernel from a confident failure: calibration, geometry, and honest ablations.</p>

## The model and its distinction from NTK {#dkl-model}

The construction takes one line; its consequences take the rest of the chapter. Let \(h_\theta:\mathcal{X}\to\mathbb{R}^p\) be a neural feature map and let \(k_\phi\) be a positive definite base kernel. Define

$$
k_{\theta,\phi}(x,z)=k_\phi(h_\theta(x),h_\theta(z)).
$$

For every fixed \((\theta,\phi)\), this is positive definite by composition. Training changes \(\theta\), and therefore changes the geometry itself.

::: {.definition #def-dkl}
[Definition (deep kernel learning)]{.box-title}

Deep kernel learning jointly estimates a parametric feature map and the hyperparameters of a kernel or Gaussian-process readout, usually by optimizing marginal likelihood or a predictive objective [@wilson2016dkl].
:::

This is not the neural tangent kernel limit. In standard NTK analysis, the tangent kernel becomes effectively fixed and training is described by kernel gradient flow. In DKL, movement of \(h_\theta\) is the point of the method. Nor is DKL the same as a deep Gaussian process, which places stochastic processes at multiple layers and integrates over intermediate functions.

## Marginal-likelihood training {#dkl-marginal-likelihood}

With network weights now inside the kernel, something must train them, and cross-validating millions of parameters is not an option. The natural objective is the same evidence that tuned length scales for an ordinary GP, now asked to shape an entire representation. For Gaussian regression, let \(C=K_{\theta,\phi}+\sigma^2I\). The negative log marginal likelihood, up to an additive constant, is

$$
\mathcal{L}(\theta,\phi,\sigma)=\frac12y^\top C^{-1}y+\frac12\log\det C.
$$

The first term rewards data fit; the log determinant penalizes covariance volume. Their competition is sometimes called an automatic Occam effect, but it does not guarantee calibrated predictions under misspecification or approximate inference.

::: {.proposition #prop-dkl-gradient}
[Proposition (covariance-parameter gradient)]{.box-title}

For any scalar parameter \(\psi\) entering \(C\),

$$
\frac{\partial\mathcal{L}}{\partial\psi}=\frac12\operatorname{tr}\!\left[(C^{-1}-\alpha\alpha^\top)\frac{\partial C}{\partial\psi}\right],
\qquad \alpha=C^{-1}y.
$$

**Assumptions.** \(C\) is differentiable in \(\psi\) and strictly positive definite. **Proof status.** Proved using the derivative of a quadratic inverse and \(\partial\log\det C=\operatorname{tr}(C^{-1}\partial C)\).
:::

The representation can shrink pairwise distances while the base length scale expands, creating non-identifiability. Constrain or regularize feature scales and log both feature norms and kernel hyperparameters.

## Exact and approximate computation {#dkl-computation}

Every gradient step above touches the full Gram matrix, so the first practical question is what a step costs. Exact training requires an \(n\times n\) factorization, \(O(n^3)\) time and \(O(n^2)\) memory. Inducing-point methods replace the full process by \(m\) representative locations, often costing \(O(nm^2+m^3)\). Structured kernel interpolation places inducing points on a grid and interpolates feature-space covariances, enabling fast matrix-vector products when the learned representation and grid retain the required structure [@wilson2015kissgp].

:::: {.algorithm #algo-dkl-training}
[Algorithm (auditable DKL training)]{.box-title}

1. Split data before fitting preprocessing, features, inducing locations, or kernel hyperparameters.
2. Initialize a feature map and a strictly positive observation-noise parameter.
3. Optimize marginal likelihood with jitter, gradient clipping, and logged condition estimates.
4. Evaluate predictive log density, calibration, and task loss on untouched data.
5. Compare against a fixed base-kernel GP and a neural point predictor of similar capacity.
6. Repeat over seeds and report approximation settings, wall time, peak memory, and failures.
::::

## Calibration and failure modes {#dkl-failures}

The most instructive DKL failures are quiet: training proceeds smoothly and produces confident nonsense. The simplest such failure is worth examining first.

::: {.example #example-dkl-collapse}
[Example (representation collapse)]{.box-title}

If \(h_\theta(x)\) maps many distinct inputs to nearly the same point, an RBF readout assigns them nearly identical covariance. Training likelihood can still improve through noise and mean parameters, while uncertainty away from the training set becomes misleading. Monitor pairwise feature distances and evaluate shifted inputs, not just interpolation error.

**Verification artifact.** checks/example-ch-dkl-example-dkl-collapse.json records the example source hash and verification scope.
:::

<figure class="viz" data-figure="dkl-collapse" data-alt="Two kernel similarity matrices compare the same ordered inputs. Before feature learning the matrix has a narrow diagonal band; after a saturating feature map collapses the inputs, nearly the entire matrix has high similarity."><figcaption>A composed kernel remains positive definite after representation collapse, so validity alone cannot protect the geometry. When distant inputs become nearly identical in feature space, the GP readout loses the distances it would need to express unfamiliarity and can become confident for the wrong reason.</figcaption></figure>

Other failures include vanishing learned noise, unstable Cholesky factors, inducing points that cover only dense modes, and overconfident extrapolation. Approximate objectives can distort posterior variances even when point predictions look strong. Deep-kernel model selection should therefore include proper scoring rules and coverage curves rather than RMSE alone. Empirical work has shown that DKL uncertainty behavior is sensitive to architecture and training choices [@ober2021dkl].

## Non-Gaussian likelihoods {#dkl-nongaussian}

Classification, counts, censored responses, and ordinal outcomes replace the Gaussian likelihood by a nonconjugate one. The posterior over latent values is no longer Gaussian, so Laplace, expectation propagation, variational inference, or sampling is required. Feature learning and posterior approximation then interact: a representation can change to make the chosen approximation look favorable rather than to improve the exact predictive distribution.

For classification, a variational objective has the schematic form

$$
\mathbb E_q\log p(y\mid f)-\operatorname{KL}(q\Vert p_{\theta,\phi}),
$$

where the prior depends on learned features. Mini-batch training is possible when the expected log likelihood decomposes, but inducing and feature parameters receive noisy coupled gradients. Report the likelihood, link, quadrature or Monte Carlo estimator, and variance-reduction method.

Class probabilities should be assessed with log loss, Brier score, reliability diagrams, and decision-specific utility. Accuracy alone cannot diagnose a collapsed posterior or a representation that makes all logits extreme.

## Variational inducing-point DKL {#dkl-variational}

Sparse approximation acquires a new twist when the space the inducing points live in is itself being learned. Let inducing variables \(u=f(Z)\) live at locations \(Z\) in learned feature space. A sparse variational GP optimizes a lower bound over \(q(u)\), kernel parameters, inducing locations, and network parameters. The inducing set is not merely a computational cache: it determines which posterior directions can be represented.

Three design choices must be distinguished:

1. inducing inputs in original input space, transformed by \(h_\theta\);
2. free inducing locations directly in latent feature space;
3. structured grids used for interpolation.

Free latent inducing points may not correspond to any realizable input. That is acceptable for algebra but complicates interpretation and shift diagnostics. A structured grid assumes learned features remain within and sufficiently cover the interpolation domain [@wilson2015kissgp].

The evidence lower bound can improve while predictive variance deteriorates if the variational family is too restrictive. Increase inducing rank until proper scores and coverage stabilize, not only until training loss stops improving.

## Identifiability and geometry control {#dkl-identifiability}

When training logs show feature norms drifting upward while the length scale grows in lockstep, the optimizer is not misbehaving; the model has a flat direction. For an RBF readout, multiplying all features by \(a\) and multiplying the length scale by the same factor leaves pairwise normalized distances unchanged. Neural weights, feature normalization, output scale, kernel amplitude, and observation noise create further symmetries or near-symmetries.

::: {.proposition #prop-dkl-scale-invariance}
[Proposition (feature-length-scale invariance)]{.box-title}

For an RBF base kernel

$$
k_{\ell}\{h(x),h(z)\}
=\sigma_f^2\exp\left\{-\frac{\lVert h(x)-h(z)\rVert^2}{2\ell^2}\right\},
$$

the simultaneous transformation \(h\mapsto ah\), \(\ell\mapsto |a|\ell\) leaves the covariance unchanged for nonzero \(a\).

**Assumptions.** The base kernel is the displayed isotropic RBF and no other model component depends on the absolute feature scale. **Proof status.** Proved by direct substitution.
:::

This flat direction harms optimization diagnostics and makes raw parameter values uninterpretable. Feature normalization, explicit priors, norm penalties, or fixed length scales can break it. The remedy changes the model and should be included in ablations.

## Derivatives, invariances, and structured inputs {#dkl-structured-inputs}

Because the composed kernel differentiates through \(h_\theta\), DKL can incorporate derivative observations when both network and base kernel are smooth enough. The derivative covariance includes Jacobians of the representation, so unstable network derivatives can create ill-conditioned blocks even when feature values look benign.

Domain structure can enter before or inside the learned map: convolution for images, message passing for graphs, equivariant layers for symmetry, and sequence encoders for text or paths. Positive definiteness of the readout remains automatic for fixed features, but data leakage, invariance errors, and representation collapse remain possible.

For multiple outputs, combine a learned input map with the operator-valued kernels in [[ch:vector-and-operator-valued-kernels]]. Learning both input and output geometry creates strong non-identifiability; independent-output and fixed-feature baselines are mandatory.

## OOD uncertainty and conformal calibration {#dkl-ood}

A stationary base kernel in learned feature space measures distance after the network transformation. Inputs far apart in raw space can map nearby, so GP variance need not increase under semantic or covariate shift. Training likelihood rarely constrains behavior in regions without data.

OOD evaluation should include:

- synthetic directions that move away from the training support;
- realistic source or time shifts;
- feature-space nearest-neighbor distances;
- predictive entropy and proper scores;
- interval or set coverage after calibration;
- comparison with a fixed kernel and a deterministic ensemble.

Conformal prediction from [[ch:distribution-shift-robustness-and-conformal-prediction]] can calibrate marginal coverage on representative data. It does not repair unsupported feature mappings, and its exchangeability assumptions can fail under the very shifts being tested.

## Comparing DKL with neighboring models {#dkl-comparisons}

A DKL score in isolation answers nothing; the claim that joint feature learning earns its complexity is inherently comparative. DKL sits among several models:

- **fixed deep features plus GP:** isolates the value of joint feature learning;
- **NTK or NNGP:** uses a fixed infinite-width kernel and convex readout dynamics;
- **deep GP:** composes stochastic functions and integrates intermediate uncertainty;
- **neural ensemble:** learns features without a GP covariance interpretation;
- **neural operator:** maps functions to functions, often across discretizations;
- **ordinary GP:** tests whether learned geometry is necessary at all.

Match training data, validation budget, preprocessing, and compute. If DKL receives a pretrained network, report that data and cost. If baselines do not, the comparison measures pretraining plus DKL rather than the readout alone.

:::: {.algorithm #algo-dkl-stress-test}
[Algorithm (DKL stress-test protocol)]{.box-title}

**Input.** In-distribution data, at least one declared shift, fixed and learned-feature baselines, and a compute budget.

**Output.** Accuracy, calibration, geometry, and failure diagnostics.

1. Fit preprocessing on training data and preserve untouched calibration and shifted test sets.
2. Train exact or variational DKL while logging feature norms, pairwise distances, noise, jitter, and condition estimates.
3. Increase inducing rank or interpolation resolution until predictive scores stabilize.
4. Compare joint training with frozen features, random features, a fixed GP, and a deterministic predictor.
5. Evaluate proper scores, coverage, width, accuracy, OOD ranking, seed variability, wall time, and memory.
6. Inspect collapsed or saturated feature directions and report failed factorizations and recovery actions.

Optimization stops on validation proper score and stable geometry diagnostics, not training marginal likelihood alone.
::::

## Common mistakes and practical implications {#dkl-practice}

- Positive definiteness of the composed kernel does not imply a well-conditioned Gram matrix.
- Marginal likelihood is an objective, not proof that the model is correctly specified.
- Test inputs must not influence feature normalization or inducing-point selection.
- Comparing DKL only with a weak fixed kernel does not isolate the value of uncertainty modeling.
- A feature-learning model should not be described with fixed-NTK conclusions.

Use double precision for kernel linear algebra when conditioning is difficult, constrain noise away from zero, and record the jitter actually added. Evaluate both in-distribution and shifted data. If uncertainty is not a product requirement, compare the extra complexity against a deterministic representation plus calibrated residual model.

## Summary and further reading {#dkl-summary}

DKL learns a representation inside a valid kernel and uses probabilistic or kernel inference on the transformed inputs. Marginal likelihood balances fit and covariance complexity but introduces coupled optimization. Scalable approximations change both computation and uncertainty, so calibration, conditioning, and ablations are first-class checks. The original scalable construction is [@wilson2016dkl], with structured interpolation in [@wilson2015kissgp] and a focused empirical analysis in [@ober2021dkl].

## Exercises {#exercises}

1. [warm-up]{.ex-tag} Prove that \(k(h_\theta(x),h_\theta(z))\) is positive definite for fixed \(\theta\).
2. [computation]{.ex-tag} For a \(2\times2\) covariance matrix, compute both marginal-likelihood terms and their change when observation noise doubles.
3. [proof]{.ex-tag} Derive the covariance-parameter gradient proposition from matrix differential identities.
4. [synthesis]{.ex-tag} Design an evaluation comparing DKL, a fixed-kernel GP, an NTK predictor, and a neural ensemble. Specify accuracy, calibration, shift, compute, and seed reporting criteria.
