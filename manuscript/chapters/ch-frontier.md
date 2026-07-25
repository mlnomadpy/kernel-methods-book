---
id: ch-frontier
slug: the-frontier
title: 'The Frontier: Feature Learning and Beyond'
part: XI · Learning the Representation
order: 57
tier: advanced
prerequisites:
  - kernels-now
objectives:
  - >-
    Distinguish lazy, mean-field, and maximal-update limits by parameter
    movement and representation change.
  - >-
    Explain the sample-complexity barrier of a fixed rotation-invariant kernel
    on single-index targets.
  - >-
    Identify where kernels enter attention, deep architectures, manifold
    methods, and foundation-model features.
  - >-
    Decide whether a claimed learned kernel is fixed, label-adapted,
    task-adapted, or merely a similarity heuristic.
  - >-
    Audit a learned-kernel pipeline for approximation, shift, privacy, fairness,
    and update-sensitive claims.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-frontier.yml
verification_date: null
bibliography:
  - jacot2018
  - arora2019cntk
  - chizat2019
  - chizat2018meanfield
  - mei2018meanfield
  - yang2021tp
  - ghorbani2020
  - abbe2022staircase
  - abbe2023leap
  - bach2017barron
  - neal1996
  - lee2018nngp
  - matthews2018
  - cho2009
  - mairal2016
  - tsai2019transformer
  - katharopoulos2020
  - choromanski2021performer
  - borovitskiy2020matern
  - borovitskiy2021graph
  - havlicek2019quantum
  - chaudhuri2011private
  - vonoswald2023icl
---
# The Frontier: Feature Learning and Beyond

<p class="lead">A kernel is a commitment: once its feature map is fixed, learning may choose coefficients but cannot invent a new geometry. Modern systems blur that boundary. A wide network can either remain inside its initialization kernel or move its features; a quantum circuit can define an enormous feature space while leaving its kernel expensive and noisy to evaluate; a frozen foundation model can become a useful kernel while quietly leaking test information through preprocessing or model selection. This chapter does not catalogue every fashionable hybrid. It develops three modules deeply enough to audit: feature learning beyond the lazy limit, quantum feature kernels, and frozen foundation-model kernels. A final evidence ledger records what is an identity, what is a regime-specific theorem, what is empirical, and what remains conjectural.</p>

## Module I: fixed tangent features or learned features? {#two-limits}

Width alone does not determine whether a network learns features. The architecture, parameterization, output scale, initialization, learning-rate scaling, loss, time horizon, and order of limits all matter. We therefore compare two explicit models rather than using “infinite width” as a synonym for “kernel”.

### Exact tangent dynamics {#lazy-regime}

Let \(f_\theta:\mathcal X\to\mathbb R\) be differentiable, let \(X=(x_1,\ldots,x_n)\), and define \(u_i(t)=f_{\theta_t}(x_i)\). For the unnormalized squared loss

$$L(\theta)=\frac12\sum_{i=1}^n(f_\theta(x_i)-y_i)^2,$$

the empirical neural tangent kernel is

$$\Theta_\theta(x,x')=\langle\nabla_\theta f_\theta(x),\nabla_\theta f_\theta(x')\rangle.$$

It is PSD for every fixed \(\theta\). This certificate says nothing yet about whether it remains fixed during training.

:::: {.proposition #prop-43-1}
[Proposition (exact output dynamics and the frozen-kernel solution)]{.box-title}

Assume \(t\mapsto\theta_t\) solves gradient flow \(\dot\theta_t=-\nabla L(\theta_t)\), and \(f_\theta(x)\) is continuously differentiable along the trajectory. Then

$$\dot u(t)=-\Theta_t(X,X)(u(t)-y).$$

If \(\Theta_t(X,X)=\Theta_0\) for all \(t\), \(\Theta_0\succ0\), and the cross-kernel \(\Theta_t(x,X)=\Theta_0(x,X)\) is also fixed, then

$$u(t)-y=e^{-t\Theta_0}(u(0)-y)$$

and

$$f_\infty(x)=f_0(x)+\Theta_0(x,X)\Theta_0^{-1}(y-u(0)).$$

**Proof status.** Proved below. This finite-dimensional identity does not prove that a nonlinear network has a constant NTK.
::::

::: {.proof}
[Proof]{.box-title}

The chain rule gives

$$\dot f_{\theta_t}(x)=-\sum_i(f_{\theta_t}(x_i)-y_i)\Theta_t(x,x_i).$$

Restricting to \(X\) gives the first display. Under the constant-kernel assumption, the residual \(r=u-y\) solves \(\dot r=-\Theta_0r\), hence \(r(t)=e^{-t\Theta_0}r(0)\). Positive definiteness makes the exponential decay. Integrating the cross-kernel equation and using \(\int_0^\infty e^{-t\Theta_0}\,dt=\Theta_0^{-1}\) yields the predictor. [\(\square\)]{.qed}
:::

Jacot, Gabriel, and Hongler establish deterministic infinite-width NTK recursions and training dynamics under their sequential-width, fully connected, Lipschitz-activation setting [@jacot2018, Theorems 1 and 2]. The statement is not a theorem about arbitrary finite networks. Convolutional extensions require their own architectural construction [@arora2019cntk, Sections 3 and 4].

### The lazy regime is a scaling limit {#lazy-scale}

Consider a scaled differentiable model \(F_\alpha(w)=\alpha h(w)\). Holding the target fixed while increasing \(\alpha\) can make an \(O(\alpha^{-1})\) parameter displacement produce an \(O(1)\) output change. If the derivative of \(h\) is locally Lipschitz and the linearized tangent operator is sufficiently nondegenerate, the nonlinear and linearized trajectories remain close on a fixed time interval. Chizat, Oyallon, and Bach give the finite-horizon statement as Theorem 2.2 and separate assumptions for infinite-horizon convergence [@chizat2019, Theorems 2.2 and 2.3].

The quantifiers matter:

1. the data set and time horizon are fixed before \(\alpha\to\infty\);
2. the model is differentiable with controlled local curvature;
3. the initialization and output normalization are part of the regime;
4. closeness of trajectories is not equality at finite width;
5. zero training loss is not a generalization theorem.

A practical diagnostic is to track both relative parameter movement and tangent-kernel drift,

$$D_\theta(t)=\frac{\|\theta_t-\theta_0\|}{1+\|\theta_0\|},\qquad
D_K(t)=\frac{\|\Theta_t-\Theta_0\|_{\mathrm F}}{\|\Theta_0\|_{\mathrm F}}.$$

Small \(D_\theta\) without small \(D_K\) is not evidence for lazy training, because a small move through a high-curvature parameterization can rotate features substantially.

<figure class="viz" data-widget="frontier-regime-map"><table><thead><tr><th>Representation</th><th>What moves?</th><th>Correct analysis</th></tr></thead><tbody><tr><td>Frozen NTK or encoder</td><td>Coefficients</td><td>Fixed-kernel estimation</td></tr><tr><td>Mean-field network</td><td>Neuron measure</td><td>Nonlinear measure flow</td></tr><tr><td>Tuned circuit, prompt, or adapter</td><td>Feature geometry</td><td>Data-dependent kernel selection</td></tr></tbody></table><figcaption>The decisive axis is representation movement, not model size. Frozen encoders and tangent kernels delegate learning to coefficients; transported particles and tuned feature maps require a different analysis.</figcaption></figure>

### Mean-field feature learning {#parametrization}

For a two-layer network, write

$$f_m(x)=\frac1m\sum_{j=1}^m a_j\sigma(w_j^\top x)
       =\int a\,\sigma(w^\top x)\,d\rho_m(a,w),$$

where \(\rho_m=m^{-1}\sum_j\delta_{(a_j,w_j)}\). In the mean-field parameterization, particle motion remains \(O(1)\) on the chosen macroscopic time scale, and the empirical measure may converge to a time-dependent probability measure \(\rho_t\). The limiting predictor

$$f_{\rho_t}(x)=\int a\,\sigma(w^\top x)\,d\rho_t(a,w)$$

therefore changes its hidden-feature distribution. Chizat and Bach formulate the measure optimization and Wasserstein-gradient-flow perspective in Sections 2–3 [@chizat2018meanfield, Sections 2 and 3]; Mei, Montanari, and Nguyen derive a distributional dynamics for two-layer networks under explicit smoothness and scaling conditions [@mei2018meanfield, Theorems 1 and 2].

This is not merely “an NTK that changes”. The state variable is now a measure over neurons, and its evolution is nonlinear in \(\rho_t\). A fixed RKHS describes optimization over coefficients in a predetermined feature family. Mean-field training transports the feature family itself.

:::: {.definition #def-43-2}
[Definition (regime certificate)]{.box-title}

A feature-learning claim is accompanied by a *regime certificate* if it records:

- architecture and width limit;
- forward normalization and initialization law;
- learning-rate or time rescaling;
- which layers train;
- order of width, sample-size, and time limits;
- the quantity proved to converge;
- the topology or norm of convergence;
- whether the guarantee concerns optimization, approximation, or risk.

Without this certificate, “the network behaves like a kernel” is not a mathematical claim.
::::

### What the separation results do and do not say {#curse}

Rotation-invariant kernels cannot automatically exploit an unknown low-dimensional direction. On spherical single-index targets, spectral mass is organized by polynomial degree; learning a degree-\(k\) component can require a sample scale polynomial in \(d^k\). The precise statements depend on the input distribution, target decomposition, loss, regularization, and algorithm. Ghorbani et al. analyze the separation between kernel methods and feature learning for polynomial scaling regimes [@ghorbani2020, Theorems 1 and 2]. Abbe et al. identify staircase-like and leap-complexity structures for particular neural training models [@abbe2022staircase, Sections 2–3; @abbe2023leap, Definition 1 and main theorem].

These are lower bounds for specified kernel classes and asymptotic models, not a declaration that every kernel needs \(d^k\) samples or every trained network needs \(d\). The transferable lesson is narrower: a fixed symmetry can hide a target direction, while learned features may break that symmetry.

## Module II: quantum feature kernels under finite resources {#quantum-kernels}

A quantum circuit \(U_\phi(x)\) prepares \(|\phi(x)\rangle=U_\phi(x)|0\rangle\). The fidelity kernel

$$k_\phi(x,x')=|\langle\phi(x)\mid\phi(x')\rangle|^2$$

is PSD because it is the Hilbert–Schmidt inner product of density matrices
\(\rho_x=|\phi(x)\rangle\langle\phi(x)|\). Havlíček et al. use this construction for a quantum kernel estimator and a variational classifier [@havlicek2019quantum, main text, “Quantum feature maps” and “Quantum kernel estimator”].

The certificate is exact only for the ideal kernel. A useful quantum-kernel claim must answer three separate questions.

### Evaluation and trainability {#quantum-evaluation}

**Evaluation.** If a circuit estimates each overlap from a finite number \(S\) of shots, the reported matrix \(\widehat K\) is random. Independent entrywise estimates need not form a PSD matrix, even though \(K\) is PSD. Symmetrization repairs asymmetry, not indefiniteness. Projection onto the PSD cone creates a different estimator and must be reported.

**Trainability.** A fixed circuit needs no kernel-parameter training, but its usefulness depends on choosing a data encoding. If circuit parameters are tuned using labels, the method becomes a bilevel learned-kernel problem. Gradients can be noisy or flat, and validation reuse can overfit the kernel family. The 2019 experiment demonstrates feasibility for a particular feature map and small hardware instance; it is not a general trainability theorem [@havlicek2019quantum, Methods, “Quantum kernel estimation”].

**Evaluation complexity.** A hard-to-simulate state preparation does not by itself imply a learning advantage. One must state the classical comparison class, the cost of loading data, shots per entry, number of Gram entries, error tolerance, and downstream solver cost.

### Generalization is controlled by the learned Gram geometry {#quantum-generalization}

Once \(\widehat K\) is supplied to an SVM or ridge estimator, ordinary kernel capacity and margin arguments apply conditionally on that matrix. A huge Hilbert space is not automatically a useful inductive bias. Near-identity Gram matrices can interpolate yet generalize poorly; nearly constant Gram matrices cannot separate labels. The relevant diagnostics are alignment computed on training data only, effective dimension, eigengaps, margin, and stability under shot noise.

::::: {.example #example-frontier-shot-noise}
[Example (entrywise shot noise destroys a valid Gram matrix)]{.box-title}

Start from three identical normalized quantum states. Their exact fidelity Gram is

$$K=\begin{pmatrix}1&1&1\\1&1&1\\1&1&1\end{pmatrix}\succeq0.$$

Suppose independent finite-shot estimates return \(0.9,0.9,0.1\) for the three off-diagonal overlaps:

$$\widehat K=\begin{pmatrix}1&0.9&0.9\\0.9&1&0.1\\0.9&0.1&1\end{pmatrix}.$$

Its eigenvalues are approximately \((-0.2238,0.9000,2.3238)\). The exact feature map is valid, but the estimated matrix is indefinite. Clipping the negative eigenvalue gives a PSD matrix at Frobenius distance \(0.2238\), and changes the entrywise estimates. The right report contains the shot count, confidence intervals, minimum eigenvalue before repair, repair rule, and sensitivity of the final predictor.

**Failure demonstrated.** Kernel validity does not survive arbitrary independent entrywise estimation.

**Verification artifact.** checks/example-ch-frontier-example-frontier-shot-noise.json records the example source hash and verification scope.
:::::

## Module III: foundation models as frozen feature maps {#foundation-model-kernels}

Let \(E_\eta:\mathcal X\to\mathbb R^p\) be a pretrained encoder with frozen release identifier \(\eta\). Any PSD base kernel \(k_0\) induces

$$k_\eta(x,x')=k_0(E_\eta(x),E_\eta(x')).$$

For the linear choice, \(k_\eta(x,x')=E_\eta(x)^\top E_\eta(x')\). This is an ordinary fixed kernel only after the encoder version, prompt/template, layer, pooling rule, normalization, and numerical precision are frozen.

### A leakage-safe frozen-feature protocol {#foundation-protocol}

1. **Freeze before labels.** Record the encoder checksum and preprocessing before inspecting test labels.
2. **Split raw entities first.** Deduplicate people, documents, molecules, sites, or time windows before embedding. Otherwise near-duplicate leakage survives any downstream split.
3. **Fit transforms on training data.** Centering, whitening, PCA, prototype construction, bandwidth selection, and supervised prompt selection use training folds only.
4. **Nest model selection.** Select layer, prompt, kernel, and regularization inside the training partition; evaluate once on the untouched test partition.
5. **Audit pretraining overlap.** Exact membership may be unknowable, so report release dates, known corpora, contamination tests, and the residual uncertainty.
6. **Separate adaptation regimes.** Frozen features, prompt tuning, adapters, and full fine-tuning are different estimators. Only the first is a fixed-kernel pipeline.
7. **Stress shift.** Recompute performance by time, source, subgroup, and semantic distance. A stable Gram matrix does not imply a stable data distribution.

The construction in von Oswald et al. shows that a particular linear self-attention layer can implement a gradient-descent update for linear regression [@vonoswald2023icl, Proposition 1]. It does not prove that arbitrary pretrained transformers implement gradient descent, nor that every embedding kernel inherits that mechanism. The distinction exemplifies the protocol: an architectural possibility is not an empirical guarantee for an unspecified model.

### Leakage changes the object being evaluated {#foundation-leakage}

Suppose a prompt, layer, or projection \(T_y\) is chosen using all labels, including test labels, and the reported kernel is

$$k_y(x,x')=\langle T_yE_\eta(x),T_yE_\eta(x')\rangle.$$

This remains PSD for the realized data, but it is no longer a kernel fixed independently of the held-out outcomes. Standard test-error interpretation fails because the evaluation labels helped choose the geometry. PSD validity and evaluation validity are separate links in the argument.

## Evidence and maturity ledger {#frontier-update-policy}

The ledger is maintained as part of the chapter, not as an optimistic closing paragraph.

| Claim family | Mathematical object | Current evidence | Main boundary | Maturity |
|---|---|---|---|---|
| Frozen NTK dynamics | fixed Jacobian Gram | exact finite identity; infinite-width theorems under regime assumptions [@jacot2018, Theorems 1–2] | finite width, long time, changing NTK | established but scoped |
| Lazy approximation | scaled differentiable model | trajectory bounds [@chizat2019, Theorems 2.2–2.3] | curvature, scale, horizon | established but scoped |
| Mean-field learning | evolving neuron measure | distributional limits [@mei2018meanfield, Theorems 1–2] | two-layer models and specified scaling | active theory |
| Feature-learning separation | target and algorithm class | asymptotic lower/separation results [@ghorbani2020, Theorems 1–2] | distribution and algorithm specificity | active theory |
| Ideal quantum kernel | fidelity Gram | exact PSD certificate; hardware demonstration [@havlicek2019quantum, main text] | usefulness and evaluation cost | construction established |
| Noisy quantum Gram | random matrix estimator | concentration can be analyzed; PSD may fail entrywise | shots, correlated estimation, repair bias | engineering-sensitive |
| Frozen foundation kernel | composed PSD feature map | exact PSD certificate | contamination, split design, shift | method established |
| Transformer-as-optimizer interpretation | architecture-specific construction | linear-attention construction [@vonoswald2023icl, Proposition 1] | not universal mechanism evidence | exploratory |

An annual update should change a row only when it can name the object, theorem or experiment, assumptions, comparator, and failure boundary. “Promising” is not a maturity level.

## A decision procedure for frontier claims {#synthesis}

Ask the questions in order:

1. **What is fixed?** Kernel, random initialization, pretrained encoder, circuit, or data-dependent representation?
2. **What moves?** Coefficients, particles, circuit parameters, prompts, adapters, or all weights?
3. **What limit is taken?** Width, sample size, shots, depth, time, or feature dimension, and in which order?
4. **What is proved?** PSD validity, optimization convergence, approximation, risk, computational advantage, or calibration?
5. **What data entered the geometry?** Training inputs, training labels, validation labels, test inputs, or test labels?
6. **What is the failure witness?** Kernel drift, rank collapse, shot-noise indefiniteness, contamination, shift, or an unfair baseline?

If any answer is missing, the claim belongs in the ledger as unresolved rather than in a theorem box.

## Summary and further reading {#summary-and-further-reading}

The frontier is not one method. It is a set of boundary disputes about what remains fixed while learning occurs. NTK theory is strongest when its parameterization and limit are explicit. Mean-field theory supplies a different state space in which features genuinely move. Quantum kernels inherit exact PSD structure but not automatic trainability, generalization, or computational advantage. Foundation-model kernels are mathematically ordinary frozen-feature kernels whose scientific validity depends on unusually careful split and contamination protocols. The durable habit is to attach every claim to a regime certificate and every empirical advantage to a leakage-safe comparison.

For the tangent-kernel limit, see [@jacot2018, Theorems 1–2]. For lazy scaling, see [@chizat2019, Theorems 2.2–2.3]. For mean-field limits, see [@chizat2018meanfield, Sections 2–3] and [@mei2018meanfield, Theorems 1–2]. For separation results, see [@ghorbani2020, Theorems 1–2]. For the original quantum feature-map experiment, see [@havlicek2019quantum, main text and Methods]. For the scoped in-context gradient construction, see [@vonoswald2023icl, Proposition 1].

## Exercises {#exercises}

1. [warm-up]{.ex-tag} Prove that the empirical NTK is PSD. Give a condition under which its training Gram is singular.
2. [proof]{.ex-tag} Derive the frozen-kernel gradient-flow solution in Proposition 43.1 when \(\Theta_0\succeq0\) is singular. State the limiting fitted values using the Moore–Penrose pseudoinverse.
3. [synthesis]{.ex-tag} Compare the state variables and orders of limits in the NTK and mean-field regimes. Explain why neither theorem automatically describes a finite network trained for an increasing time horizon.
4. [computation]{.ex-tag} Reproduce the quantum shot-noise example. Verify the minimum eigenvalue and the Frobenius distance to the PSD projection.
5. [proof]{.ex-tag} Prove that the fidelity kernel is PSD by embedding pure-state density matrices in the Hilbert–Schmidt space.
6. [synthesis]{.ex-tag} A quantum-kernel paper reports higher accuracy than an RBF SVM but omits shot counts, circuit-evaluation cost, kernel-selection folds, and pre-repair eigenvalues. Classify which claims remain supported.
7. [synthesis]{.ex-tag} Design a nested frozen-feature evaluation protocol when choosing among four encoder layers, three pooling rules, and five ridge parameters.
8. [challenge]{.ex-tag} A team selects a projection using all labels and then freezes it before fitting ridge regression. Prove that the resulting Gram is PSD, then explain why its held-out risk estimate is still invalid.
