---
id: ch-dkl
slug: deep-kernel-learning
title: Deep Kernel Learning
part: XI · Learning the Representation
order: 55
tier: advanced
prerequisites:
  - gaussian-processes-and-rvm
  - kernels-and-deep-learning
  - large-scale-kernels
objectives:
  - Prove validity of a kernel composed with a learned representation.
  - >-
    Derive exact marginal-likelihood gradients and identify their
    stationary-scale consequence.
  - >-
    Reconstruct structured kernel interpolation and quantify approximation
    sensitivity.
  - >-
    Diagnose representation collapse, non-identifiability, and unsupported
    uncertainty.
  - >-
    Design comparisons that isolate feature learning, GP inference, and
    approximation.
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

<p class="lead">A neural network can learn the representation a task demands, while a Gaussian process can return a distribution rather than a point. Deep kernel learning joins them by placing a kernel on learned coordinates. That construction is valid for every fixed representation, but validity is the easy part. Joint training can collapse distant inputs onto the same latent point, trade feature scale against length scale along a flat objective direction, and exploit a flexible covariance matrix until marginal likelihood becomes a training criterion rather than protection from overfitting. Approximation adds another layer: an inducing or interpolation scheme changes the covariance whose determinant, inverse, and uncertainty drive training. This chapter reconstructs the original scalable model, proves the central matrix derivatives and failure mechanism, derives structured kernel interpolation, and builds an audit in which accuracy, calibration, geometry, approximation, and shift behavior must agree.</p>

## The composed kernel and the model being fitted {#dkl-model}

Let \(\mathcal X\) be the input domain, let
\(h_\theta:\mathcal X\to\mathbb R^p\) be a deterministic feature map, and let
\(k_\phi\) be a positive-definite kernel on \(\mathbb R^p\). Define

$$
k_{\theta,\phi}(x,z)=k_\phi\{h_\theta(x),h_\theta(z)\}.
$$

:::: {.proposition #prop-dkl-composition}
[Proposition (validity under composition)]{.box-title}

**Assumptions.** The feature map is deterministic for fixed parameters and the base kernel is positive definite.

**Proof status.** Complete below.

For every fixed \((\theta,\phi)\), \(k_{\theta,\phi}\) is positive definite. It is strictly positive definite on a finite set only if the base kernel is strictly positive definite on the distinct feature images and the map does not identify relevant inputs.
::::

:::: {.proof}
For \(x_1,\ldots,x_n\in\mathcal X\) and \(a\in\mathbb R^n\), set
\(z_i=h_\theta(x_i)\). Then

$$
\sum_{i,j}a_i a_j k_{\theta,\phi}(x_i,x_j)
=\sum_{i,j}a_i a_j k_\phi(z_i,z_j)\geq0.
$$

If \(z_i=z_j\) for two distinct inputs, the corresponding Gram rows coincide. Taking coefficients \(a_i=1\), \(a_j=-1\), and all others zero gives quadratic form zero, so strict positive definiteness fails on that input set. \(\square\)
::::

This proof establishes a covariance function, not useful geometry, identifiability, calibration, or good conditioning. A constant network produces a valid rank-one signal covariance.

For Gaussian regression, assume

$$
y_i=f(x_i)+\varepsilon_i,\qquad
\varepsilon_i\overset{\mathrm{iid}}{\sim}\mathcal N(0,\sigma_n^2),
$$

with GP prior \(f\sim\mathcal{GP}(m,k_{\theta,\phi})\). Write

$$
K_{\theta,\phi}=[k_{\theta,\phi}(x_i,x_j)]_{ij},
\qquad
C=K_{\theta,\phi}+\sigma_n^2 I.
$$

Unless stated otherwise, \(m=0\), \(C\) is strictly positive definite, and all parameters are point estimates. This last clause matters: uncertainty in \(f\) conditional on a point-estimated network is not uncertainty over the network weights.

::: {.definition #def-dkl}
[Definition (deep kernel learning)]{.box-title}

Deep kernel learning jointly estimates a parametric representation and kernel or Gaussian-process parameters, usually through marginal likelihood or a variational surrogate [@wilson2016dkl].
:::

DKL is not a fixed neural tangent kernel, because movement of \(h_\theta\) changes the kernel. It is not a deep Gaussian process, because the intermediate map is deterministic at a point estimate. It is not merely a neural last layer, because a general base kernel can represent infinitely many basis functions in feature space.

## Paper module: Wilson et al. train the geometry by evidence {#dkl-paper-wilson}

The obstacle addressed by [@wilson2016dkl] was a three-way tension. Flexible neural features did not by themselves provide GP inference, expressive GPs were expensive at scale, and fitting a GP only after freezing a network could not adapt the representation to the probabilistic objective. Their construction composes a deep map with an RBF or spectral-mixture kernel, initializes from a trained network, and then jointly optimizes network and kernel parameters through GP marginal likelihood. Structured kernel interpolation supplies scale.

### Exact objective and gradient {#dkl-marginal-likelihood}

The negative log marginal likelihood is

$$
\mathcal L(\psi)
=\frac12y^\top C^{-1}y
+\frac12\log\det C
+\frac n2\log(2\pi),
$$

where \(\psi\) can be a network weight, length scale, signal variance, or noise parameter.

:::: {.theorem #thm-dkl-gradient}
[Theorem (exact covariance-parameter gradient)]{.box-title}

**Assumptions.** The covariance matrix has a positive noise floor and is differentiable in the parameter under consideration.

**Proof status.** Complete derivation below.

If \(C(\psi)\) is differentiable and strictly positive definite, and
\(\alpha=C^{-1}y\), then

$$
\frac{\partial\mathcal L}{\partial\psi}
=\frac12\operatorname{tr}\left[
\left(C^{-1}-\alpha\alpha^\top\right)
\frac{\partial C}{\partial\psi}
\right].
$$

For a network parameter \(\theta_r\),

$$
\frac{\partial K_{ij}}{\partial\theta_r}
=\nabla_1 k_\phi(h_i,h_j)^\top
\frac{\partial h_i}{\partial\theta_r}
+\nabla_2 k_\phi(h_i,h_j)^\top
\frac{\partial h_j}{\partial\theta_r},
$$

provided the base kernel and feature map are differentiable.
::::

:::: {.proof}
The matrix differential identities

$$
dC^{-1}=-C^{-1}(dC)C^{-1},
\qquad
d\log\det C=\operatorname{tr}(C^{-1}dC)
$$

give

$$
d(y^\top C^{-1}y)
=-y^\top C^{-1}(dC)C^{-1}y
=-\operatorname{tr}(\alpha\alpha^\top dC).
$$

Adding the log-determinant differential and multiplying by \(1/2\) yields the trace formula. The second identity follows by the chain rule applied to both arguments of the symmetric kernel. \(\square\)
::::

The first quadratic term rewards covariance aligned with \(y\); the log determinant rewards small covariance volume. Calling the second term a complexity penalty is an interpretation, not a generalization theorem. The objective is exact only for the declared Gaussian model and exact covariance.

Wilson et al. used pretraining followed by joint optimization and compared DKL with the same deep network and with a GP on frozen neural features. That ablation is logically necessary: without it, an improvement cannot be attributed to joint probabilistic feature learning.

**Failure boundary.** The trace formula assumes exact solves and log determinants. Stochastic trace estimators, truncated conjugate gradients, and approximate covariances optimize a different or noisy objective. A point estimate of millions of network weights also falls outside the low-dimensional empirical-Bayes intuition often attached to GP hyperparameter selection.

## Paper module: structured kernel interpolation {#dkl-ski}

Exact DKL requires \(O(n^3)\) factorization time and \(O(n^2)\) memory. Wilson and Nickisch replace cross-covariances to a set of inducing locations \(U=(u_1,\ldots,u_m)\) by local interpolation [@wilson2015kissgp]. Let \(W\in\mathbb R^{n\times m}\) contain interpolation weights, usually with a fixed small number of nonzeros per row. Then

$$
K_{XU}\approx W K_{UU},
\qquad
\widetilde K=WK_{UU}W^\top.
$$

This is structured kernel interpolation, or SKI.

:::: {.proposition #prop-ski-psd}
[Proposition (SKI preserves positive semidefiniteness)]{.box-title}

**Assumptions.** The inducing covariance is positive semidefinite and the interpolation matrix is real and finite.

**Proof status.** Complete below.

If \(K_{UU}\succeq0\), then \(\widetilde K=WK_{UU}W^\top\succeq0\) for every real interpolation matrix \(W\).
::::

:::: {.proof}
For any \(a\in\mathbb R^n\),

$$
a^\top\widetilde K a
=(W^\top a)^\top K_{UU}(W^\top a)\geq0.
$$

No positivity assumption on individual interpolation weights is needed. \(\square\)
::::

The algebra reveals what SKI buys. A matrix-vector product costs one multiplication by \(W^\top\), one by \(K_{UU}\), and one by \(W\). With \(c\) nonzeros per row and Toeplitz structure in one-dimensional \(K_{UU}\), this is \(O(cn+m\log m)\) time and \(O(cn+m)\) storage. Kronecker structure gives a different complexity depending on grid dimension. Conjugate gradients then solve systems using only these products. The 2015 paper derives this construction in its Section 3 and Equation (8), and reports the structured costs in Section 3.2 [@wilson2015kissgp].

### What covariance approximation does to inference {#dkl-approximation}

Matrix approximation, posterior approximation, and objective approximation are different claims. Let

$$
C=K+\sigma_n^2I,\qquad
\widetilde C=\widetilde K+\sigma_n^2I,
$$

with \(K,\widetilde K\succeq0\), and assume
\(\lVert K-\widetilde K\rVert_2\leq\varepsilon\).

:::: {.proposition #prop-dkl-perturbation}
[Proposition (solve and log-determinant sensitivity)]{.box-title}

**Assumptions.** Both covariance matrices share a positive noise floor and the approximation error is measured in spectral norm.

**Proof status.** Complete below.

The inverse perturbation obeys

$$
\lVert C^{-1}-\widetilde C^{-1}\rVert_2
\leq\frac{\varepsilon}{\sigma_n^4}.
$$

If additionally \(\varepsilon\lt\sigma_n^2\), then

$$
|\log\det C-\log\det\widetilde C|
\leq n\{-\log(1-\varepsilon/\sigma_n^2)\}.
$$

**Assumptions.** The same positive noise floor is used in both matrices and the error is measured in spectral norm.
::::

The resolvent identity

$$
C^{-1}-\widetilde C^{-1}
=C^{-1}(\widetilde C-C)\widetilde C^{-1}
$$

and \(\lVert C^{-1}\rVert,\lVert\widetilde C^{-1}\rVert\leq\sigma_n^{-2}\) prove the first bound. For the second, write

$$
\widetilde C=C^{1/2}(I+E)C^{1/2},
\qquad
E=C^{-1/2}(\widetilde C-C)C^{-1/2}.
$$

Then \(\lVert E\rVert\leq\varepsilon/\sigma_n^2\). Every eigenvalue of \(I+E\) lies in
\([1-\delta,1+\delta]\) with \(\delta=\varepsilon/\sigma_n^2\), so the absolute sum of their log values is at most \(n\{-\log(1-\delta)\}\).

The noise floor appears to the fourth power in the solve bound. An approximation harmless at \(\sigma_n^2=0.1\) can be destructive after training drives the noise toward \(10^{-5}\).

**Failure boundary.** SKI assumes the learned features remain in a region covered by the inducing grid and are smooth enough for the interpolation order. Joint feature learning can move the data while the grid stays fixed. PSD preservation does not bound \(\varepsilon\), and a good training Gram approximation does not guarantee accurate cross-covariances for shifted test features.

## Paper module: when marginal likelihood overfits {#dkl-paper-ober}

Ober, Rasmussen, and van der Wilk challenged the claim that a GP marginal likelihood automatically protects an overparameterized deep kernel from overfitting [@ober2021dkl]. Their question was empirical and mechanistic: why can exact or variational DKL achieve excellent training evidence while generalization deteriorates?

Their setting uses deterministic neural features whose weights are point-estimated with GP hyperparameters. They study exact Gaussian-regression DKL on a one-dimensional toy problem, stochastic variational DKL on regression and image tasks, batch-size effects, and Bayesian inference over network weights. Their central observation is that a highly flexible kernel can overcorrelate training observations and exploit the determinant term.

The paper's Proposition 1 supplies a precise warning about the usual “data fit plus complexity” story.

:::: {.theorem #thm-dkl-scale-stationarity}
[Theorem (stationary signal scale fixes the quadratic term)]{.box-title}

**Assumptions.** Signal and noise scales are jointly optimized in the Gaussian regression model stated above.

**Proof status.** Complete below.

Write the covariance as

$$
C=sA,
\qquad
s=\sigma_f^2\gt0,
$$

where \(A=\widehat K+\tau^2I\) contains kernel shape and the noise-to-signal ratio
\(\tau^2=\sigma_n^2/\sigma_f^2\). If the log marginal likelihood has an interior stationary point in \(s\), with \(A\) fixed at that derivative, then

$$
y^\top C^{-1}y=n.
$$

Hence the log marginal likelihood's quadratic data-fit contribution is \(-n/2\) at that stationary scale.

**Source locator.** Proposition 1 and Appendix A of [@ober2021dkl].
::::

:::: {.proof}
Ignoring the constant \(-n\log(2\pi)/2\), the log marginal likelihood is

$$
\ell(s)
=-\frac{1}{2s}y^\top A^{-1}y
-\frac n2\log s
-\frac12\log\det A.
$$

Differentiate:

$$
\ell'(s)
=\frac{1}{2s^2}y^\top A^{-1}y-\frac{n}{2s}.
$$

At an interior stationary point,
\(y^\top A^{-1}y=ns\). Since \(C^{-1}=s^{-1}A^{-1}\), it follows that
\(y^\top C^{-1}y=n\). \(\square\)
::::

The new contribution is not that GP hyperparameters can overfit. It is the demonstration that adding a highly parameterized feature map can make this failure severe, the analysis showing why the quadratic term ceases to distinguish fitted kernels after scale optimization, and experiments connecting overcorrelation, batch size, and Bayesian treatment of the network.

**Failure boundary of the theorem.** A prior or penalty on \(s\), a boundary optimum, a fixed signal scale, heteroscedastic covariance not sharing the global scale, or integration over \(s\) changes the stationarity equation. The proposition does not prove that every DKL model overfits. It removes one common informal argument that marginal likelihood must prevent it.

The paper's later finding is a response to this limitation: integrating over network uncertainty with Hamiltonian Monte Carlo on the toy problem, and approximate stochastic-gradient sampling on larger tasks, can improve behavior. That is evidence for a fully Bayesian treatment in their tested regimes, not a distribution-free calibration theorem.

## A worked collapse calculation {#dkl-failures}

Representation collapse can be made numerical without training a network. Suppose three inputs are mapped so that their unit-variance signal Gram matrix is

$$
K_\rho=(1-\rho)I+\rho\mathbf1\mathbf1^\top,
\qquad 0\leq\rho\lt1,
$$

and let the noise variance be \(\tau^2=0.01\). The two contrast eigenvalues of
\(C_\rho=K_\rho+\tau^2I\) are

$$
a=1-\rho+\tau^2,
$$

while the constant eigenvalue is

$$
b=1+2\rho+\tau^2.
$$

::: {.example #example-dkl-collapse}
[Example (label-aligned collapse can win the training evidence)]{.box-title}

Take \(y=(1,1,1)^\top\). Since \(y\) lies in the constant eigenspace,

$$
\mathcal L_\rho-\frac32\log(2\pi)
=\frac{3}{2b}+\frac12\{2\log a+\log b\}.
$$

At \(\rho=0\), \(a=b=1.01\) and the displayed quantity is approximately \(1.5000\). At
\(\rho=0.99\), \(a=0.02\), \(b=2.99\), and it is approximately \(-2.8628\). The training evidence strongly prefers the nearly collapsed covariance because the labels are constant and the two unused contrast directions contribute tiny determinant eigenvalues.

Now map a genuinely unfamiliar test input to the same collapsed feature region, with prior covariance \(\rho\) to each training point. Its latent posterior variance is

$$
v_\star
=1-\rho^2\mathbf1^\top C_\rho^{-1}\mathbf1
=1-\frac{3\rho^2}{b}.
$$

At \(\rho=0.99\), \(v_\star\approx0.0166\). The model is highly confident because the learned representation declares the test point familiar. Positive definiteness and high training evidence do not reveal that the raw input was remote.

For contrast labels such as \(y=(1,-2,1)\), the quadratic term is \(6/a\) and collapse is heavily penalized. Collapse is therefore label aligned, not inevitable.
:::

<figure class="viz" data-figure="dkl-collapse" data-alt="Two kernel similarity matrices compare separated and collapsed feature geometries."><figcaption>A composed kernel remains positive definite after representation collapse. When distant raw inputs become nearly identical in feature space, posterior uncertainty loses the geometry needed to express unfamiliarity.</figcaption></figure>

## Identifiability and geometry control {#dkl-identifiability}

For an isotropic RBF readout,

$$
k_\ell\{h(x),h(z)\}
=\sigma_f^2\exp\left\{
-\frac{\lVert h(x)-h(z)\rVert^2}{2\ell^2}
\right\}.
$$

:::: {.proposition #prop-dkl-scale-invariance}
[Proposition (feature-length-scale non-identifiability)]{.box-title}

**Assumptions.** The base kernel and network contain the reciprocal scale transformation stated below.

**Proof status.** Complete below.

For every nonzero scalar \(c\), the simultaneous transformation

$$
h\mapsto ch,\qquad \ell\mapsto |c|\ell
$$

leaves the covariance unchanged, provided no other component depends on absolute feature scale.
::::

This is exact non-identifiability, not merely poor conditioning. Raw feature norms and raw length scales have no separate interpretation along this orbit. Batch normalization, feature normalization, weight decay, priors, or a fixed length scale can break the symmetry, but each remedy changes the model.

Geometry diagnostics should include:

- pairwise feature-distance quantiles, separated by class or response difference;
- singular values and effective rank of the centered feature matrix;
- Gram eigenvalues before and after noise;
- nearest-neighbor distances for in-distribution and shifted inputs;
- sensitivity across seeds and checkpoints;
- the fraction of test features outside an inducing grid or training convex hull.

A single two-dimensional feature plot is not enough. It may conceal collapsed directions, and its projection can create or remove apparent separation.

<figure class="viz" data-figure="dkl-feature-movement-uncertainty" data-alt="Two panels follow deep-kernel optimization. The learned feature scale contracts while the training marginal-likelihood objective improves; at the same time, posterior variance at a raw out-of-distribution input falls."><figcaption>Feature learning can improve the training evidence by moving the geometry itself. In this label-aligned calculation, compression makes a remote raw input look nearby, so its posterior variance falls with the feature scale. Monitoring the objective without monitoring feature movement misses the failure.</figcaption></figure>

## Inducing variables and non-Gaussian likelihoods {#dkl-variational}

For \(m\) inducing variables \(u=f(Z)\), a variational GP chooses
\(q(u)\) and uses

$$
q(f)=\int p(f\mid u)q(u)\,du.
$$

The evidence lower bound is

$$
\mathcal J
=\sum_{i=1}^n\mathbb E_{q(f_i)}\log p(y_i\mid f_i)
-\operatorname{KL}\{q(u)\Vert p(u)\}.
$$

In DKL, \(p(u)\), cross-covariances, and sometimes the meaning of \(Z\) all depend on the representation. Three cases are distinct:

1. original-space inducing inputs transformed by \(h_\theta\);
2. free inducing locations in latent space;
3. a structured latent grid with interpolation.

Free latent locations need not equal \(h_\theta(x)\) for any realizable input. A grid can be left behind as the feature map moves. Original-space inducing inputs are interpretable but can be expensive or awkward for structured objects.

For classification and other non-Gaussian likelihoods, the expected log likelihood needs quadrature or Monte Carlo. Feature gradients then inherit both minibatch and posterior-sampling noise. Report the estimator, number of samples, control variates, and whether inducing rank was increased until proper scores stabilized.

An improved ELBO is not automatically improved posterior variance. The objective controls an average divergence inside the assumed model and variational family. It does not certify shifted-input calibration.

## OOD uncertainty is representation-relative {#dkl-ood}

The GP readout only sees \(h_\theta(x)\). This yields an exact indistinguishability result.

:::: {.proposition #prop-dkl-indistinguishable}
[Proposition (feature collisions have identical predictions)]{.box-title}

**Assumptions.** The two feature maps agree on every training input and all other GP hyperparameters are fixed.

**Proof status.** Complete below.

If \(h_\theta(x)=h_\theta(z)\), then under a zero-mean DKL GP with fixed parameters, \(x\) and \(z\) have identical posterior means, posterior variances, and covariances with every other test point.
::::

The proof is immediate because their kernel rows and diagonal entries coincide. The consequence is not: no readout diagnostic can recover a distinction erased by \(h_\theta\). Raw-space distance, semantic novelty, or acquisition risk may therefore disagree with GP variance.

OOD evaluation needs at least one shift specified before training. Useful tests include support expansion, time shift, sensor corruption, class-conditional shift, and adversarial movement along directions to which the feature map is insensitive. Report proper scores, interval or set coverage, width, OOD ranking, and feature-space distance. Conformal calibration can repair marginal coverage only under its exchangeability or shift assumptions. It cannot make a collapsed representation informative.

## Comparisons that identify the source of an improvement {#dkl-comparisons}

| Model | Features | Bayesian or kernel uncertainty | What the comparison isolates |
|---|---|---|---|
| ordinary GP | fixed raw-input kernel | exact or approximate GP | need for learned geometry |
| frozen features plus GP | pretrained, then fixed | GP readout | value of joint feature training |
| joint DKL | point-estimated learned features | GP conditional on point estimate | full proposed system |
| deterministic network | learned | none unless separately calibrated | value of GP readout |
| neural ensemble | learned per member | empirical ensemble | alternative uncertainty cost |
| NNGP or NTK | fixed limit kernel | GP or kernel dynamics | feature learning versus fixed limit |
| fully Bayesian DKL | integrated feature weights | posterior over both stages | cost and value of weight uncertainty |

Training data, preprocessing, validation budget, architecture, pretraining data, and compute must be matched or reported. If DKL receives a pretrained network unavailable to the baselines, the experiment measures pretraining plus DKL. If only DKL receives extensive hyperparameter search, the experiment measures search budget.

:::: {.algorithm #algo-dkl-stress-test}
[Algorithm (auditable DKL stress test)]{.box-title}

1. Freeze train, validation, calibration, in-distribution test, and shifted test partitions before preprocessing.
2. Train ordinary GP, deterministic network, frozen-feature GP, and joint DKL baselines with matched search budgets.
3. Log exact or approximate objective, feature norms, distance quantiles, centered-feature rank, noise, jitter, condition estimates, and inducing coverage.
4. Increase inducing rank or interpolation resolution until prediction and uncertainty metrics stabilize.
5. Evaluate RMSE or accuracy, negative log predictive density, calibration, coverage and width, OOD ranking, seed variability, time, and peak memory.
6. Repeat with the feature map frozen after pretraining and with a geometry-control intervention.
7. Perturb shifted inputs along a feature-insensitive direction and inspect variance.
8. Report failed factorizations, boundary noise values, approximate-solve tolerances, and every recovery action.
::::

Stopping on training marginal likelihood alone violates the purpose of this audit. The selected checkpoint should use untouched validation proper score together with preregistered geometry and conditioning thresholds.

## Common mistakes and practical implications {#dkl-practice}

- Kernel validity does not imply strict positive definiteness or useful geometry.
- Point-estimated network weights are not integrated Bayesian uncertainty.
- The marginal likelihood is an objective under a model, not a generalization certificate.
- The quadratic “data-fit” term becomes fixed at an interior global-scale stationary point.
- Signal scale, noise ratio, feature scale, and length scale have exact or near symmetries.
- SKI preserves PSD but can still have large covariance and posterior error.
- Tiny learned noise magnifies covariance-approximation and solve errors.
- Inducing locations in latent space may not correspond to realizable inputs.
- OOD variance is only as meaningful as the learned representation.
- An NTK theorem does not describe a feature-learning DKL model.

Use stable solves rather than explicit inverses, double precision when conditioning demands it, constrained positive noise, and recorded jitter. Approximation residuals should be evaluated at the selected parameters, not only at initialization.

<figure class="viz" data-figure="learned-kernel-lifecycle" data-alt="A five-stage learned-kernel lifecycle from raw object to shift audit."><figcaption>A learned kernel is a pipeline, not merely a covariance formula. Representation assumptions enter before training, labels reshape the geometry, approximation acts on the trained kernel, and deployment shift tests whether the resulting similarity remains trustworthy.</figcaption></figure>

## Summary and further reading {#dkl-summary}

Deep kernel learning composes a learned map with a valid base kernel and trains the resulting covariance. Wilson et al. made the construction scalable and jointly optimized it through GP marginal likelihood [@wilson2016dkl]; structured kernel interpolation supplies fast products through \(WK_{UU}W^\top\) [@wilson2015kissgp]. Ober et al. showed why a highly parameterized covariance can overfit despite the familiar evidence decomposition, and why integrating network uncertainty can help in their tested regimes [@ober2021dkl]. A defensible DKL result must therefore identify which gain comes from features, which from GP inference, which from approximation, and which survives shift.

## Exercises {#exercises}

1. [warm-up]{.ex-tag} Prove the composition proposition and construct a two-input example in which a strictly positive-definite RBF base kernel becomes only semidefinite after a feature collision.
2. [proof]{.ex-tag} Derive the exact covariance-parameter gradient and then specialize it to an RBF length scale by computing \(\partial K_{ij}/\partial\ell\).
3. [proof]{.ex-tag} Prove the stationary signal-scale theorem. Identify four modifications of the model or objective for which its conclusion need not hold.
4. [computation]{.ex-tag} Reproduce the compound-symmetry collapse example at \(\rho=0\) and \(\rho=0.99\), including the negative log marginal likelihood without its constant and the shifted-point posterior variance.
5. [proof]{.ex-tag} Prove that SKI preserves PSD and derive the \(O(cn+m\log m)\) matrix-vector cost when \(W\) has at most \(c\) nonzeros per row and \(K_{UU}\) admits a Toeplitz FFT product.
6. [proof]{.ex-tag} Prove both covariance-perturbation bounds. Explain why the conditions become fragile as the learned noise variance approaches zero.
7. [synthesis]{.ex-tag} A DKL model improves RMSE over an ordinary RBF GP but uses a pretrained network, a larger tuning budget, and an SKI approximation with no error report. Design the minimum ablation set needed to attribute the gain.
8. [synthesis]{.ex-tag} Design an experiment that can distinguish representation collapse, inducing-grid misspecification, and ordinary GP hyperparameter overfitting. Specify the data, shift, logged quantities, baselines, stopping rule, and falsifying outcomes.
