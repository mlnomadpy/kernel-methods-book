---
id: ch-manifold
slug: semi-supervised-and-manifold-regularization
title: Semi-Supervised and Manifold Regularization
part: V · Spectral Geometry and Unlabeled Structure
order: 27
tier: advanced
prerequisites:
  - kernel-clustering
  - kernels-and-rkhs
objectives:
  - >-
    State the compatibility assumption that makes unlabeled inputs informative
    about labels.
  - >-
    Derive graph energy, Laplacian eigenmaps, and harmonic extension from finite
    objectives.
  - >-
    Prove the manifold-regularization representer theorem and derive LapRLS
    exactly.
  - >-
    Explain graph-Laplacian scaling, density bias, normalization, and boundary
    effects.
  - >-
    Separate transductive propagation from inductive prediction and
    out-of-sample extension.
  - >-
    Audit semi-supervised gains with supervised and graph-misspecification
    controls.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-manifold.yml
verification_date: null
bibliography:
  - belkin2003
  - belkin2006manifold
  - vonluxburg2007
---
# Semi-Supervised and Manifold Regularization

<p class="lead">A hospital may hold a million scans and labels for only a thousand. The unlabeled scans reveal which anatomies are common, which variations connect smoothly, and where the data form separate populations. They do not reveal which diagnosis belongs to any of those regions. That logical gap is the whole subject: unlabeled inputs identify the marginal distribution \(P_X\), while prediction depends on \(P_{Y\mid X}\). Manifold methods become useful only after making a compatibility bet that connects them. This chapter reconstructs two foundational papers rather than treating that bet as a slogan. Laplacian eigenmaps turn a neighborhood graph into a discrete geometry, while manifold regularization combines that geometry with an ambient RKHS to obtain an inductive predictor. We derive both finite problems, prove the representer and harmonic-extension results, expose the graph-to-manifold limit, and build failure tests for the cases in which more unlabeled data makes the answer worse.</p>

## The identifiability barrier {#manifold-identifiability}

Let \(l\) observations carry labels and let \(u\) additional observations be unlabeled. More unlabeled data estimates \(P_X\) more accurately. It provides no direct information about \(P_{Y\mid X}\). This is not pessimism about a particular algorithm; it is a nonidentifiability statement.

::: {.proposition #prop-manifold-no-free-unlabeled}
[Proposition (unlabeled data alone cannot identify the labeling rule)]{.box-title}

Fix any marginal distribution \(P_X\) on \(\mathcal X\). There exist two joint distributions \(P^{(0)}\) and \(P^{(1)}\) with the same marginal \(P_X\) and incompatible Bayes classifiers. Consequently, no amount of unlabeled data can distinguish those two labeling rules.

**Assumptions.** Binary labels and a marginal assigning positive probability to a measurable set \(A\) and its complement. **Proof status.** Proved by construction below.
:::

**Proof.** Under \(P^{(0)}\), set \(Y=1\) on \(A\) and \(Y=-1\) on \(A^c\). Under \(P^{(1)}\), reverse the two labels. Both models generate exactly the same unlabeled samples because both have marginal \(P_X\), but their Bayes classifiers disagree everywhere. No statistic of unlabeled inputs can tell which conditional law generated the labels. \(\square\)

A semi-supervised guarantee therefore needs a restricted family of conditional laws. Common restrictions include:

- the regression function varies slowly along high-density directions;
- class boundaries avoid high-density regions;
- labels are approximately constant on well-connected graph components;
- the target has small Dirichlet energy on a latent manifold.

These assumptions are related but not equivalent. A smooth regression function can cross zero in a dense region. A low-density separator can coexist with rapid within-class variation. A graph component can merge two classes through one narrow bridge.

::: {.definition #def-manifold-assumption}
[Definition (geometric compatibility assumption)]{.box-title}

A joint distribution is geometrically compatible with a chosen intrinsic regularizer \(\mathcal E\) when its target function \(f_\rho\), such as the conditional mean or a calibrated score, belongs to the domain of \(\mathcal E\) and has small intrinsic energy \(\mathcal E(f_\rho)\) relative to competing labelings.

The regularizer and its scale are part of the assumption. Saying only that data lie on a manifold is insufficient.
:::

The supervised model \(\gamma_I=0\) is therefore not an optional baseline. It tests whether the compatibility assumption helped at the chosen sample size and graph scale.

## Paper module I: Laplacian eigenmaps as discrete geometry {#manifold-paper-eigenmaps}

Belkin and Niyogi's Laplacian eigenmaps paper begins with a representation problem. Euclidean distance in the ambient space can put two points close even when every path along the data manifold between them is long. The proposed move is local: construct a graph that trusts only nearby relationships, then find coordinates that vary as little as possible across strong edges [@belkin2003].

**Exact finite setting.** Let \(x_1,\ldots,x_n\) be observations. Choose a symmetric nonnegative weight matrix \(W\), with \(W_{ii}=0\), by a documented neighborhood rule. Let

$$
D_{ii}=\sum_jW_{ij},
\qquad
L=D-W.
$$

No manifold is required to define this finite problem. The manifold enters only when interpreting the graph as an estimator of continuum geometry.

::: {.proposition #prop-graph-energy}
[Proposition (graph Dirichlet energy)]{.box-title}

For every \(f\in\mathbb R^n\),

$$
f^\top Lf
=
\frac12\sum_{i,j=1}^nW_{ij}(f_i-f_j)^2.
$$

Hence \(L\succeq0\), and its null space consists exactly of vectors constant on each connected component.

**Assumptions.** The graph is finite, undirected, and has nonnegative edge weights. **Proof status.** Proved by expanding the quadratic form and using \(D_{ii}=\sum_jW_{ij}\).
:::

The embedding cannot minimize energy without a scale constraint because the zero vector would win. It also must exclude the constant null mode. A one-dimensional Laplacian eigenmap solves

$$
\min_{f\in\mathbb R^n} f^\top Lf
\quad\text{subject to}\quad
f^\top Df=1,\qquad f^\top D\mathbf1=0.
$$

The \(D\)-weighted normalization prevents high-degree vertices from dominating merely through scale.

::: {.theorem #thm-manifold-eigenmap}
[Theorem (finite Laplacian eigenmap)]{.box-title}

Suppose every vertex has positive degree and the graph is connected. A minimizer of the constrained problem is a generalized eigenvector associated with the smallest positive generalized eigenvalue:

$$
Lf=\lambda Df.
$$

For a \(p\)-dimensional embedding, take \(p\) \(D\)-orthonormal generalized eigenvectors after the constant mode.

**Assumptions.** \(W\) is symmetric and nonnegative, \(D\) is positive diagonal, and the graph is connected. **Proof status.** Derived below from the Rayleigh quotient. This is the finite optimization at the center of [@belkin2003].
:::

**Derivation.** Form the Lagrangian

$$
\mathcal L(f,\lambda,\mu)
=f^\top Lf-\lambda(f^\top Df-1)-\mu f^\top D\mathbf1.
$$

Stationarity gives

$$
2Lf-2\lambda Df-\mu D\mathbf1=0.
$$

Left-multiply by \(\mathbf1^\top\). Since \(\mathbf1^\top L=0\), \(f^\top D\mathbf1=0\), and \(\mathbf1^\top D\mathbf1\gt0\), we obtain \(\mu=0\). Thus \(Lf=\lambda Df\). The Courant-Fischer principle for the symmetric matrix \(D^{-1/2}LD^{-1/2}\) identifies the constrained minimum with its smallest positive eigenvalue. \(\square\)

**Executable object.** The algorithm is graph construction followed by a sparse generalized eigensolve. The output is a coordinate for each current vertex, not a prediction rule on \(\mathcal X\).

**Failure boundary.** A disconnected graph has one zero mode per component, so the phrase "the first nonconstant eigenvector" becomes ambiguous. Isolated vertices make \(D^{-1/2}\) undefined. A shortcut edge can collapse distant manifold regions. A graph built at one bandwidth can encode a different geometry from the graph built at another. The paper introduces a locality-preserving representation; it does not make every neighborhood graph a consistent manifold estimator.

**Comparison and afterlife.** Kernel PCA preserves variance in an ambient-kernel feature space; Laplacian eigenmaps preserve local graph relations through low Dirichlet energy. Spectral clustering uses closely related eigenvectors but discretizes them into groups. Later convergence work asks when these finite eigenvectors approach eigenfunctions of a continuum operator, while out-of-sample methods address the fact that the original finite embedding has no automatic value at a new point.

## Harmonic extension as a discrete boundary-value problem {#manifold-harmonic}

The same energy becomes a semi-supervised method when labels provide boundary values. Partition the vertices into labeled and unlabeled sets and write

$$
L=
\begin{pmatrix}
L_{\ell\ell}&L_{\ell u}\\
L_{u\ell}&L_{uu}
\end{pmatrix},
\qquad
f=
\begin{pmatrix}
y\\f_u
\end{pmatrix}.
$$

Minimizing \(f^\top Lf\) over \(f_u\) gives

$$
L_{uu}f_u=-L_{u\ell}y.
$$

::: {.theorem #thm-manifold-harmonic}
[Theorem (existence and uniqueness of harmonic extension)]{.box-title}

The matrix \(L_{uu}\) is positive definite exactly when every connected component containing an unlabeled vertex also contains a labeled vertex. Under this condition the harmonic extension is unique. At every unlabeled vertex \(i\),

$$
f_i=\frac{\sum_jW_{ij}f_j}{D_{ii}}.
$$

**Assumptions.** A finite undirected graph with nonnegative weights and fixed labels. **Proof status.** Proved below from the graph-energy identity.
:::

**Proof.** For \(v\in\mathbb R^u\), pad it with zeros on labeled vertices to form \(\widetilde v\). Then

$$
v^\top L_{uu}v
=
\widetilde v^\top L\widetilde v
=
\frac12\sum_{i,j}W_{ij}(\widetilde v_i-\widetilde v_j)^2.
$$

This is zero exactly when \(\widetilde v\) is constant on each connected component. Any component meeting a labeled vertex has constant zero because \(\widetilde v\) is zero there. Thus the only null vector is zero precisely under the stated component condition. The first-order condition on an unlabeled coordinate is \((Lf)_i=0\), which rearranges to the weighted-average identity. \(\square\)

The averaging equation yields a random-walk interpretation. With labeled vertices absorbing, \(f_i\) is a weighted average of boundary labels using absorption probabilities. It also exposes a limitation: a disconnected unlabeled component is unidentified, not merely assigned a large standard error.

## Worked example: one bridge controls the answer {#manifold-worked-bridge}

::: {.example #example-manifold-bridge}
[Example (two regions joined by one weighted bridge)]{.box-title}

Take the four-vertex path \(1-2-3-4\). Give edges \((1,2)\) and \((3,4)\) weight one, and the bridge \((2,3)\) weight \(\varepsilon\gt0\). Fix \(f_1=1\) and \(f_4=-1\). Harmonicity at vertices 2 and 3 gives

$$
(1+\varepsilon)f_2-\varepsilon f_3=1,
\qquad
-\varepsilon f_2+(1+\varepsilon)f_3=-1.
$$

Symmetry implies \(f_3=-f_2\), hence

$$
f_2=\frac{1}{1+2\varepsilon},
\qquad
f_3=-\frac{1}{1+2\varepsilon}.
$$

When \(\varepsilon=1/10\), the scores are \(5/6\) and \(-5/6\): the weak bridge preserves the two label regions. When \(\varepsilon=10\), they are \(1/21\) and \(-1/21\): the strong bridge nearly erases both labels. If the bridge is an artifact of the ambient metric, adding more unlabeled points around it can reinforce the wrong answer.

**Verification.** Substitution verifies both linear equations. This is a sensitivity calculation on one graph, not evidence that a particular graph rule is statistically consistent. The existing deterministic figure artifact records the chapter's graph-energy visualization.
:::

<figure class="viz" data-figure="manifold-graph-energy" data-alt="Two moon-shaped point clouds are connected by a sparse neighborhood graph. Four square vertices carry labels in the first panel; the second panel shows harmonic scores propagated along each moon, with blue positive values on the upper arc and red negative values on the lower arc."><figcaption>Graph smoothness propagates labels along connected paths, not across ambient empty space. The mechanism helps only when labels agree with the graph. A shortcut edge or a labeling rule that cuts across each moon turns the same regularizer into bias.</figcaption></figure>

The bridge example supplies a deployment diagnostic: perturb the neighbor rule or bandwidth and track how much the predictions move. A method whose claimed gain disappears under a small graph perturbation has learned graph-construction details, not a stable intrinsic pattern.

## Paper module II: manifold regularization {#manifold-paper-regularization}

Pure graph propagation is transductive. It returns values on the graph vertices and has no canonical prediction at a new \(x\). Belkin, Niyogi, and Sindhwani's central contribution is to combine an intrinsic graph penalty with an ambient RKHS norm, obtaining a function on the original input space [@belkin2006manifold].

**Exact empirical setting.** Let \(x_1,\ldots,x_n\) contain \(l\) labeled and \(u=n-l\) unlabeled inputs. Let \(k\) be a scalar positive definite kernel with RKHS \(\mathcal H_k\). Let \(L\succeq0\) be a graph Laplacian built from all \(n\) inputs. For a loss \(V\), define

$$
\mathcal J(f)
=
\frac1l\sum_{i=1}^lV\{y_i,f(x_i)\}
+\gamma_A\|f\|_{\mathcal H_k}^2
+\frac{\gamma_I}{n^2}f_X^\top Lf_X,
$$

where \(f_X=(f(x_1),\ldots,f(x_n))^\top\), \(\gamma_A\gt0\), and \(\gamma_I\ge0\). The \(n^{-2}\) scaling is a convention tied to the empirical intrinsic norm. Changing it changes how \(\gamma_I\) must scale with sample size.

:::: {.theorem #thm-manifold-representer}
[Theorem (empirical manifold representer theorem)]{.box-title}

If a minimizer of \(\mathcal J\) exists, every minimizer has the form

$$
f^*(\cdot)=\sum_{i=1}^{n}\alpha_i k(x_i,\cdot).
$$

**Assumptions.** Point evaluation is bounded in \(\mathcal H_k\), the loss depends only on labeled evaluations, \(L\succeq0\), \(\gamma_A\gt0\), and a minimizer exists. Strict convexity of the loss is not needed for the representation. **Proof status.** Proved below. This is Theorem 2 and Section 3.4 of [@belkin2006manifold].
::::

**Proof.** Let

$$
\mathcal S=\operatorname{span}\{k(x_i,\cdot):1\le i\le n\}
$$

and decompose \(f=f_\parallel+f_\perp\) with \(f_\perp\perp\mathcal S\). Reproduction gives

$$
f_\perp(x_i)
=
\langle f_\perp,k(x_i,\cdot)\rangle_{\mathcal H_k}
=0
$$

for every labeled and unlabeled input. Therefore the loss and graph penalty are identical for \(f\) and \(f_\parallel\). Pythagoras gives

$$
\gamma_A\|f\|_{\mathcal H_k}^2
=
\gamma_A\|f_\parallel\|_{\mathcal H_k}^2
+\gamma_A\|f_\perp\|_{\mathcal H_k}^2.
$$

Since \(\gamma_A\gt0\), a minimizer cannot have \(f_\perp\ne0\). \(\square\)

**Contribution relative to inherited machinery.** The orthogonal proof is inherited from the standard representer theorem. The new move is recognizing that the empirical intrinsic penalty still depends only on the \(n\) sample evaluations, so unlabeled points enter the representer span even though they carry no loss term. The ambient norm turns that finite geometric regularizer into an inductive function.

**Failure boundary.** If \(\gamma_A=0\), many functions agree on all graph vertices and differ elsewhere, so the graph objective does not determine an inductive extension. If deployment points lie away from the sampled graph, the intrinsic term contributes no local evidence there; extrapolation comes entirely from the ambient kernel.

## Deriving Laplacian regularized least squares {#manifold-laprls}

For squared loss, let \(J\in\mathbb R^{l\times n}\) select the labeled coordinates and let \(K\) be the \(n\times n\) Gram matrix. Substitution of \(f_X=K\alpha\) gives

$$
\mathcal J(\alpha)
=
\frac1l\|y-JK\alpha\|_2^2
+\gamma_A\alpha^\top K\alpha
+\frac{\gamma_I}{n^2}\alpha^\top KLK\alpha.
$$

Differentiating yields

$$
K\left[
\frac1lJ^\top(JK\alpha-y)
+\gamma_A\alpha
+\frac{\gamma_I}{n^2}LK\alpha
\right]=0.
$$

The bracketed equation is a valid coefficient system:

$$
\left\{
\frac1lJ^\top JK+\gamma_A I
+\frac{\gamma_I}{n^2}LK
\right\}\alpha
=
\frac1lJ^\top y.
$$

It is not symmetric in this coordinate form. A symmetric positive definite formulation uses \(\beta=K^{1/2}\alpha\):

$$
\left\{
\frac1lK^{1/2}J^\top JK^{1/2}
+\gamma_A I
+\frac{\gamma_I}{n^2}K^{1/2}LK^{1/2}
\right\}\beta
=
\frac1lK^{1/2}J^\top y.
$$

Every term on the left is positive semidefinite and \(\gamma_A I\) supplies strict positivity. This is the stable system to use when a square root or equivalent factorization is available. Applying conjugate gradients to a nonsymmetric rearrangement merely because it looks smaller violates the method's assumptions.

For a new input \(x\),

$$
f(x)=k_X(x)^\top\alpha.
$$

That formula is the inductive distinction. Harmonic propagation would instead require attaching \(x\) to the graph and solving a new boundary-value problem.

:::: {.algorithm #algo-manifold-regularization}
[Algorithm (auditable LapRLS)]{.box-title}

**Input.** Labeled and unlabeled inputs, a graph rule, an ambient kernel, candidate \(\gamma_A,\gamma_I\), and labeled validation data.

**Output.** An inductive predictor and a graph-sensitivity report.

1. Standardize features from training data only.
2. Build a symmetric graph and record metric, neighbor rule, bandwidth, connected components, and degree distribution.
3. Form sparse \(L\) and the ambient Gram matrix or a positive semidefinite low-rank factor.
4. Solve the symmetric regularized system and monitor its original residual.
5. Tune graph scale and both regularizers using labeled validation data only.
6. Predict new inputs through the ambient kernel expansion.
7. Compare with \(\gamma_I=0\), pure harmonic propagation, and at least two perturbed graph constructions.
::::

**Comparison.** Supervised KRR uses labeled values and ambient smoothness. Harmonic propagation uses all inputs but only predicts graph vertices. LapRLS uses all inputs to regularize an ambient function. The methods coincide only in special limits; replacing one by another without changing the estimand is not valid.

## From graph sums to a differential operator {#manifold-graph-limit}

The finite derivations above are exact. Calling \(L\) a manifold Laplacian requires an asymptotic argument. Consider a compact \(m\)-dimensional smooth Riemannian manifold \(\mathcal M\) without boundary, sampled from a density \(p\) that is positive and twice continuously differentiable. Let \(f\) be three times continuously differentiable and let \(\eta\) be a bounded radial kernel with finite second moment. Define the population nonlocal operator

$$
\mathcal L_\varepsilon f(x)
=
\frac{1}{\varepsilon^{m+2}}
\int_{\mathcal M}
\eta\left(\frac{\|x-z\|^2}{\varepsilon^2}\right)
\{f(x)-f(z)\}p(z)\,dV(z).
$$

In normal coordinates \(z=\exp_x(\varepsilon u)\), Taylor expansion gives

$$
f(z)=f(x)+\varepsilon\langle\nabla f,u\rangle
+\frac{\varepsilon^2}{2}u^\top\nabla^2f\,u+O(\varepsilon^3),
$$

and an analogous expansion for \(p(z)\). Odd terms vanish under a radial kernel. If

$$
c_\eta=\frac1m\int_{\mathbb R^m}\eta(\|u\|^2)\|u\|^2\,du,
$$

then, away from boundary effects,

$$
\mathcal L_\varepsilon f(x)
=
-\frac{c_\eta}{2}
\left\{
p(x)\Delta_{\mathcal M}f(x)
+2\langle\nabla_{\mathcal M}p(x),\nabla_{\mathcal M}f(x)\rangle
\right\}
+O(\varepsilon).
$$

The sign follows the positive graph-Laplacian convention \(f(x)-f(z)\). The formula exhibits density bias directly. The unnormalized graph does not generally converge to a constant multiple of \(-\Delta_{\mathcal M}\); it carries \(p\) and \(\nabla p\).

This is a population bias expansion, not a complete empirical spectral-convergence theorem. Replacing the integral by a random graph sum introduces stochastic error. One must couple \(n\to\infty\) and \(\varepsilon\to0\) so neighborhoods shrink while still containing enough points. Pointwise operator convergence is weaker than eigenvector, resolvent, or learned-predictor convergence. The review [@vonluxburg2007] explains why normalization choices and convergence targets cannot be interchanged.

**Boundary and singularity failure.** Near a boundary, radial neighborhoods are truncated and the odd terms no longer cancel in the same way. The leading behavior and effective boundary condition can change. Intersections, varying dimension, and sampling holes violate the smooth-manifold expansion. A graph may still be computationally useful in those settings, but the Laplace-Beltrami interpretation has left its proved domain.

## Normalization changes the population geometry {#manifold-normalization}

Three common graph operators are

$$
L=D-W,\qquad
L_{\mathrm{rw}}=I-D^{-1}W,\qquad
L_{\mathrm{sym}}=I-D^{-1/2}WD^{-1/2}.
$$

The first is symmetric in the ordinary Euclidean inner product. The random-walk operator is generally nonsymmetric there but self-adjoint under a degree-weighted inner product. The symmetric normalized operator is similar to \(L_{\mathrm{rw}}\) and shares its eigenvalues when every degree is positive.

These are not cosmetic rescalings:

- unnormalized energy weights dense regions heavily;
- random-walk normalization describes one-step averaging relative to local degree;
- symmetric normalization changes both eigenvectors and the norm in which they are orthogonal.

The chosen continuum limit also depends on whether the weights are explicitly density corrected. A theorem about one normalization cannot justify software using another.

::: {.proposition #prop-manifold-normalized-similarity}
[Proposition (normalized Laplacians share a spectrum)]{.box-title}

If every degree is positive, then

$$
L_{\mathrm{sym}}
=
D^{1/2}L_{\mathrm{rw}}D^{-1/2}.
$$

Thus \(L_{\mathrm{sym}}\) and \(L_{\mathrm{rw}}\) are similar and have the same eigenvalues. Their right eigenvectors are related by multiplication by \(D^{-1/2}\).

**Assumptions.** A finite graph with strictly positive degrees. **Proof status.** Verified by direct multiplication. This relation does not make their Euclidean eigenvectors identical.
:::

## When unlabeled data help and when they hurt {#manifold-help-hurt}

The two paper modules now share one currency: a graph estimates geometry, and the learning objective penalizes variation in that estimated geometry. Improvement requires four events:

1. the graph preserves the target-relevant local relationships;
2. the target has low energy in that geometry;
3. enough labels anchor the relevant components;
4. \(\gamma_I\) reduces variance more than it adds bias.

Remove any one and unlabeled data can hurt. Concrete witnesses include:

<figure class="viz" data-figure="manifold-bandwidth-oversmoothing" data-alt="Label accuracy and between-manifold score contrast are plotted against graph bandwidth, showing disconnection at very small scales and oversmoothing at large scales."><figcaption>Graph bandwidth has a narrow useful regime. A graph that is too local leaves labels unable to propagate; a graph that is too global creates shortcuts and drives the two manifold scores together. The peak is data-dependent, so bandwidth must be selected without test-label leakage and stress-tested under graph perturbations.</figcaption></figure>

- **alternating labels:** labels oscillate along a densely sampled curve, so smoothing erases signal;
- **shortcut bridge:** a few spurious edges join two label regions, as in the worked example;
- **density confounding:** acquisition density varies by hospital or device rather than outcome;
- **component without labels:** harmonic values are unidentified;
- **off-manifold deployment:** the ambient kernel extrapolates where graph regularization supplied no evidence;
- **class imbalance:** normalized propagation can alter class mass even when local edges are sensible.

A claim that "unlabeled data helped" must compare against the same ambient model with \(\gamma_I=0\), use a fixed labeled validation protocol, and include graph perturbations. Choosing the best graph after inspecting test labels converts graph selection into leakage.

## Out-of-sample prediction is a modeling choice {#manifold-out-of-sample}

Several mechanisms are called out-of-sample extension, but they define different estimators:

1. **Ambient RKHS extension:** evaluate the LapRLS expansion \(k_X(x)^\top\alpha\).
2. **Nyström eigenfunction extension:** connect \(x\) to the training graph and extend a spectral coordinate.
3. **Graph rebuild:** insert a deployment batch and recompute the graph solution.
4. **Distillation:** fit a parametric map to graph coordinates or propagated labels.

The ambient extension is stable with respect to keeping old vertices fixed, but behavior far from training support is inherited from \(k\), not learned from the manifold. Nyström extension assumes the new point has meaningful neighbors. Rebuilding can change old predictions, which complicates reproducibility and monitoring. Distillation adds approximation and optimization error.

State which estimator is deployed. Transductive test performance on vertices present during graph construction does not establish inductive performance on future samples.

## Scaling without changing the estimator silently {#manifold-scaling}

Sparse neighbor graphs make \(Lv\) cost \(O(|E|)\). The ambient kernel can remain the bottleneck. With random features \(K\approx ZZ^\top\), writing \(f_X=Zw\) gives

$$
\frac1l\|y-JZw\|^2
+\gamma_A\|w\|^2
+\frac{\gamma_I}{n^2}w^\top Z^\top LZw.
$$

The graph-feature matrix \(Z^\top LZ\) can be accumulated edge by edge:

$$
Z^\top LZ
=
\frac12\sum_{i,j}W_{ij}(z_i-z_j)(z_i-z_j)^\top.
$$

This avoids dense \(L\) and reveals a positive semidefinite factorization. Nyström features give an analogous reduced system.

Approximate nearest-neighbor search changes \(W\), hence the statistical estimator. Report recall alone only if the missed edges are exchangeable with retained ones, which they rarely are. Better diagnostics compare component counts, degree distributions, graph energies of candidate predictions, and final held-out risk under exact and approximate graphs on a tractable subset.

:::: {.algorithm #algo-manifold-large-scale}
[Algorithm (large-scale graph and kernel audit)]{.box-title}

1. Build a sparse mutual-neighbor graph and record approximate-neighbor settings.
2. Measure isolated vertices, connected components, degree quantiles, and cross-label edges on labeled data.
3. Construct positive semidefinite ambient features and form \(Z^\top LZ\) by sparse products.
4. Solve the symmetric primal system with a deterministic tolerance.
5. Repeat with at least one neighbor count and one bandwidth perturbation.
6. Compare supervised, harmonic, and manifold-regularized predictions at matched feature rank.
7. Report runtime, memory, residual, and task error separately.
::::

## Common mistakes and practical implications {#manifold-practice}

- Unlabeled samples identify \(P_X\), not \(P_{Y\mid X}\).
- A graph Laplacian is not automatically the Laplace-Beltrami operator.
- Fixed bandwidth estimates a nonlocal operator rather than a differential limit.
- Shrinking bandwidth too quickly disconnects the graph and increases stochastic error.
- Unnormalized and normalized Laplacians have different population biases.
- A transductive score is not an out-of-sample prediction guarantee.
- Approximate neighbor search changes the estimator, not only its runtime.
- Test-label selection of graph scale is leakage.

The minimum credible report includes the supervised ablation, graph construction, connectivity, degree distribution, bandwidth, normalization, labeled-validation protocol, and sensitivity of each task or class to graph perturbation.

## Summary and further reading {#manifold-summary}

Laplacian eigenmaps turn local graph relationships into coordinates by minimizing a finite Dirichlet energy under a degree-weighted normalization [@belkin2003]. Manifold regularization uses the same geometry as an intrinsic penalty and combines it with an ambient RKHS norm, yielding the representer expansion over both labeled and unlabeled inputs and a genuine out-of-sample function [@belkin2006manifold]. The two contributions solve different problems: one estimates coordinates on a point cloud; the other regularizes a predictor.

Neither paper removes the compatibility assumption. Graph construction, normalization, density, boundary, and deployment support define the geometry actually used. The spectral tutorial [@vonluxburg2007] is valuable precisely because it separates these graph choices and their consequences. The practical standard is therefore comparative and falsifiable: unlabeled data count as useful only when the joint method beats its supervised counterpart under a label-only validation protocol and remains stable across plausible graphs.

## Exercises {#exercises}

1. [warm-up]{.ex-tag} Prove that the multiplicity of eigenvalue zero of an unnormalized graph Laplacian equals the number of connected components.
2. [proof]{.ex-tag} Derive the generalized eigenproblem \(Lf=\lambda Df\) from the Laplacian-eigenmap constrained optimization, including the elimination of the multiplier for \(f^\top D\mathbf1=0\).
3. [computation]{.ex-tag} For the four-vertex bridge example, compute \(f_2\) and \(f_3\) at \(\varepsilon=1/4\) and \(\varepsilon=4\), and compare both graph energies.
4. [proof]{.ex-tag} Prove the harmonic-extension uniqueness theorem by padding an unlabeled vector with zeros on labeled vertices.
5. [proof]{.ex-tag} Prove the empirical manifold representer theorem and explain precisely why unlabeled kernel sections enter the span.
6. [computation]{.ex-tag} Starting from the LapRLS coefficient objective, derive both the nonsymmetric coefficient equation and the symmetric equation in \(\beta=K^{1/2}\alpha\). Identify the term that makes the latter strictly positive definite.
7. [synthesis]{.ex-tag} Compare supervised KRR, harmonic propagation, Laplacian eigenmaps, and LapRLS by estimand, output domain, out-of-sample mechanism, and principal failure boundary.
8. [exploration]{.ex-tag} Design a two-moons study with one geometry-aligned labeling and one labeling that cuts across both moons. Specify graph perturbations, label budgets, validation rules, baselines, and the evidence required to conclude that unlabeled data helped.
