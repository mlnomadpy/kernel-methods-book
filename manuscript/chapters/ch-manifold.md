---
id: ch-manifold
slug: semi-supervised-and-manifold-regularization
title: Semi-Supervised and Manifold Regularization
part: XIV · Advanced Extensions
order: 48
tier: advanced
prerequisites:
  - kernel-clustering
  - kernels-and-rkhs
objectives:
  - Explain the manifold assumption and when unlabeled data can help.
  - Derive graph-Laplacian and ambient RKHS penalties.
  - >-
    Distinguish transductive label propagation from inductive manifold
    regularization.
  - 'Assess consistency, scaling, and out-of-sample limitations.'
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-manifold.yml
verification_date: null
bibliography:
  - belkin2006manifold
  - vonluxburg2007
---
# Semi-Supervised and Manifold Regularization

<p class="lead">Labels are expensive and unlabeled data is nearly free: a few thousand scans a specialist can annotate sit beside millions no one has time to touch, and a handful of labeled documents float in an ocean of unlabeled text. A classifier trained only on the labels we can afford throws that ocean away. The trouble is that unlabeled points reveal only where the data lies, the marginal \(P_X\), and \(P_X\) alone cannot say how the labels fall; without some assumption tying the two together, more unlabeled data buys nothing. Manifold regularization supplies that link as a precise conditional bet: the target varies smoothly along the high-density shape the data traces out. A graph Laplacian estimates that shape from all the points at once, and an ambient RKHS norm turns the estimate into a function that also predicts at points never seen, separating transductive label propagation from inductive prediction and marking exactly when the unlabeled data helps and when it hurts.</p>

## What unlabeled data can and cannot identify {#manifold-assumption}

Let \(l\) examples carry labels and let \(u\) additional examples be unlabeled. The marginal distribution \(P_X\) is better observed than the conditional \(P_{Y\mid X}\), but \(P_X\) alone cannot determine the labels. Semi-supervised learning therefore requires a compatibility assumption.

::: {.definition #def-manifold-assumption}
[Definition (manifold or cluster assumption)]{.box-title}

The prediction function is assumed to vary slowly along high-density directions of \(P_X\), and decision boundaries are assumed to avoid those regions. This is a modeling assumption linking \(P_X\) to \(P_{Y\mid X}\), not a consequence of having more data.
:::

If two classes overlap throughout the same connected high-density region, unlabeled data can reinforce the wrong smoothness bias. A supervised baseline and a sensitivity analysis over graph construction are therefore essential.

## Graph energy and label propagation {#graph-energy}

To act on the manifold assumption we need a computable stand-in for the shape of the data. A neighborhood graph is the standard one: join nearby points, let the graph's connectivity trace the high-density regions the assumption cares about, and measure how much a labeling disagrees across neighbors by a single quadratic form. Build a weighted graph on all \(n=l+u\) observations, with symmetric weights \(W_{ij}\ge 0\), degree matrix \(D_{ii}=\sum_jW_{ij}\), and Laplacian \(L=D-W\). For the vector \(f=(f(x_1),\ldots,f(x_n))^\top\),

$$
f^\top Lf=\frac12\sum_{i,j}W_{ij}(f_i-f_j)^2.
$$

::: {.proposition #prop-graph-energy}
[Proposition (Laplacian energy)]{.box-title}

For symmetric nonnegative \(W\), \(L\) is positive semidefinite and \(f^\top Lf=0\) exactly when \(f\) is constant on every connected component.

**Assumptions.** The graph is finite and undirected. **Proof status.** Proved by expanding the displayed sum of squares; the null-space claim follows edge by edge.
:::

Label propagation minimizes this energy while fixing or penalizing the labeled values. Its output is naturally transductive: it labels the vertices already in the graph. Adding a new point requires rebuilding or extending the graph.

## Manifold regularization {#manifold-objective}

The transductive limit of the previous section is exactly what we now remove: to score a point that was never in the graph, the model must be a function defined everywhere, not a list of vertex values. Manifold regularization adds both an ambient and an intrinsic penalty:

$$
\min_{f\in\mathcal{H}_k}\frac1l\sum_{i=1}^l\ell(y_i,f(x_i))
+\gamma_A\lVert f\rVert_{\mathcal{H}_k}^2
+\frac{\gamma_I}{n^2}f_X^\top Lf_X.
$$

The ambient norm controls behavior away from the observed graph and makes evaluation at a new point well defined. The intrinsic term encourages smoothness over the sample geometry [@belkin2006manifold].

:::: {.theorem #thm-manifold-representer}
[Theorem (manifold representer form)]{.box-title}

When \(\gamma_A\gt 0\), every minimum-norm solution has the expansion

$$
f(\cdot)=\sum_{i=1}^{l+u}\alpha_i k(x_i,\cdot).
$$

**Assumptions.** \(k\) is positive definite, the loss depends on labeled evaluations, and a minimizer exists. **Proof status.** The standard representer proof applies because both penalties depend only on the sample evaluations and the RKHS norm [@belkin2006manifold].
::::

For squared loss, substituting the expansion yields a convex linear system. Solve it with a symmetric method and matrix-vector products when possible; forming dense \(K\) and \(L\) costs \(O(n^2)\) memory. Sparse \(k\)-nearest-neighbor graphs keep the Laplacian product cheap, while Nyström or random features approximate the ambient kernel.

:::: {.algorithm #algo-manifold-regularization}
[Algorithm (inductive manifold-regularized regression)]{.box-title}

1. Standardize features using training data only.
2. Construct a symmetric neighbor graph and record its metric, neighbor count, and weight bandwidth.
3. Form \(L\), choose an ambient kernel, and tune \((\gamma_A,\gamma_I)\) using labeled validation data.
4. Solve the coefficient system with residual and condition-number diagnostics.
5. Predict a new \(x\) by \(f(x)=\sum_i\alpha_i k(x_i,x)\).
6. Compare with the supervised model \(\gamma_I=0\) and report whether unlabeled data helped.
::::

## Consistency and graph choices {#manifold-consistency}

The graph Laplacian is not a fixed population operator. Its limit depends on neighbor radius, bandwidth, normalization, sampling density, and manifold regularity. Unnormalized, random-walk, and symmetric normalized Laplacians encode different geometries; they should not be exchanged without changing the theorem being invoked [@vonluxburg2007]. Consistency statements require coupled limits in which sample size grows and the graph scale shrinks at an admissible rate.

## Harmonic extension and label propagation {#manifold-harmonic}

Return to pure label propagation and make its solution explicit. We hold the labeled values fixed and minimize the graph energy over the unlabeled ones, and that minimizer has a closed form. Partition graph vertices into labeled and unlabeled sets and write the Laplacian in blocks:

$$
L=
\begin{pmatrix}
L_{\ell\ell}&L_{\ell u}\\
L_{u\ell}&L_{uu}
\end{pmatrix}.
$$

Fixing labeled values \(f_\ell=y\) and minimizing \(f^\top Lf\) over \(f_u\) gives

$$
L_{uu}f_u=-L_{u\ell}y.
$$

::: {.proposition #prop-harmonic-extension}
[Proposition (graph harmonic extension)]{.box-title}

If every connected component containing an unlabeled vertex also contains a labeled vertex, then \(L_{uu}\) is positive definite and the harmonic extension is unique. At each unlabeled vertex, its value is a weighted average of its neighbors.

**Assumptions.** A finite undirected graph with nonnegative weights and at least one labeled vertex in every relevant component. **Proof status.** Proved by the graph-energy identity and the null-space characterization of the Laplacian.
:::

The averaging identity follows from \((Lf)_i=0\). It gives a random-walk interpretation: under suitable absorbing-boundary conventions, the prediction is an average of labels weighted by hitting probabilities. A disconnected unlabeled component makes the solution unidentified, not merely uncertain.

Soft label propagation penalizes disagreement with labels rather than fixing them. Normalized variants change the averaging geometry and class-mass behavior. Their names are similar, but their fixed points and population limits differ.

## Laplacian regularized least squares and SVMs {#manifold-laprls-lapsvm}

Substitute \(f_X=K\alpha\) into the manifold objective. With squared loss on labeled vertices and selector matrix \(J\), one obtains an objective proportional to

$$
\lVert y-JK\alpha\rVert^2
+\gamma_A\alpha^\top K\alpha
+\gamma_I\alpha^\top K L K\alpha.
$$

Its normal equations contain the ambient Gram geometry and the graph geometry in different positions. Symmetrizing and preconditioning the system matters because \(KLK\) can be badly scaled relative to \(K\).

For a hinge loss, the same intrinsic penalty augments an SVM and produces a Laplacian SVM. The dual is no longer the ordinary SVM dual with a new kernel substituted casually; the regularizer changes the effective inverse geometry. A robust implementation derives the finite problem from the stated primal and verifies the KKT residual.

## When unlabeled data help or hurt {#manifold-help-hurt}

Unlabeled inputs estimate \(P_X\), while prediction needs \(P_{Y|X}\). Improvement is possible only through an assumption connecting them. Three common connections are:

- decision boundaries pass through low-density regions;
- labels are nearly constant on graph communities;
- the target is smooth in the intrinsic manifold metric.

Counterexamples are easy. If labels alternate rapidly along a well-sampled manifold, intrinsic smoothing erases signal. If a narrow bridge connects classes, graph construction can propagate labels across it. If sampling density varies for reasons unrelated to the outcome, an unnormalized Laplacian can impose density-dependent bias.

The correct baseline is \(\gamma_I=0\) with the same labeled data and ambient kernel. Report the distribution of improvements across label budgets and graph perturbations, not only the best graph.

## Graph limits and density bias {#manifold-graph-limits}

For data sampled on a smooth manifold, a properly scaled graph Laplacian can converge to a differential operator. The limiting operator depends on the kernel bandwidth, graph normalization, sampling density, boundary, and manifold dimension. Shrinking bandwidth reduces geometric bias but increases variance and can disconnect the graph.

The order of limits matters. Keeping a fixed bandwidth as sample size grows estimates a nonlocal integral operator, not the Laplace-Beltrami operator. Shrinking it too quickly produces a noisy graph. Estimating intrinsic dimension and density adds further uncertainty.

Consistency of spectral clustering, label propagation, or manifold regularization needs more than pointwise convergence of graph weights. One must control eigenvectors or resolvents and then propagate that error through the learning algorithm [@vonluxburg2007].

## Out-of-sample extensions {#manifold-out-of-sample}

Pure harmonic extension is transductive. Several inductive strategies are available:

1. Manifold regularization learns an ambient RKHS function over all labeled and unlabeled training points.
2. A Nyström extension connects a new point to the existing graph and extends graph eigenfunctions.
3. A parametric or neural map learns from graph coordinates to predictions.
4. A new graph is constructed for each deployment batch.

Each strategy defines a different estimator. Nyström extension assumes the new point lies within the sampled geometry. Rebuilding a graph changes old predictions and complicates reproducibility. An ambient RKHS extrapolates off the graph according to its own kernel, where the intrinsic penalty supplies no direct information.

## Scaling sparse graphs {#manifold-scaling}

Sparse neighbor graphs allow \(Lf\) in time proportional to the number of edges. The ambient kernel may still be dense. Random features yield a primal problem with graph penalty \(Z^\top LZ\), while Nyström features yield a smaller landmark system. Approximate nearest-neighbor search changes the graph and therefore the estimator, not only runtime.

:::: {.algorithm #algo-manifold-large-scale}
[Algorithm (large-scale manifold regularization)]{.box-title}

**Input.** Labeled and unlabeled inputs, graph and ambient-kernel budgets, validation labels, and approximation tolerances.

**Output.** An inductive predictor and graph-sensitivity report.

1. Build a sparse mutual-neighbor graph and record connectivity and approximate-neighbor recall.
2. Normalize features using training data and construct low-rank ambient features.
3. Form graph-feature products without materializing dense \(L\) or \(K\).
4. Solve the convex system by preconditioned iteration and monitor the original residual.
5. Tune graph scale, ambient regularization, and intrinsic regularization on labeled validation data only.
6. Compare with supervised, transductive, and graph-free low-rank baselines.

Complexity is governed by graph edges, feature rank, and solver iterations. Stop when the system residual and held-out labeled error are stable across consecutive checkpoints.
::::

## Common mistakes and practical implications {#manifold-practice}

- Selecting graph hyperparameters using all labels leaks validation information.
- A disconnected graph propagates no information between components.
- Very wide graphs collapse geometry toward a global smoothness penalty; very narrow graphs are unstable.
- Transductive performance does not establish accurate out-of-sample prediction.
- More unlabeled data can hurt when the manifold assumption is false.

Report graph connectivity, degree distribution, bandwidth, scaling method, and the supervised ablation. For large data, use sparse graphs and matrix-free solvers; do not materialize a dense Laplacian merely because the ambient kernel is dense.

## Summary and further reading {#manifold-summary}

Graph energy translates a geometry estimate into a smoothness penalty. Pure graph methods are transductive; manifold regularization combines that penalty with an RKHS norm to obtain an inductive function. Its gains are conditional on a relationship between marginal geometry and labels. The primary framework is [@belkin2006manifold], and [@vonluxburg2007] explains the spectral graph choices behind it.

## Exercises {#exercises}

1. [warm-up]{.ex-tag} Show that the multiplicity of eigenvalue zero of \(L\) equals the number of connected components.
2. [computation]{.ex-tag} For a path of three vertices with endpoint labels \(+1\) and \(-1\), minimize Laplacian energy over the middle value and interpret the result.
3. [proof]{.ex-tag} Derive the representer expansion by decomposing \(f\) into the span of \(k(x_i,\cdot)\) and its orthogonal complement.
4. [exploration]{.ex-tag} Construct a two-moons experiment in which unlabeled data helps, then rotate the labeling rule so the same geometry becomes misleading. State a rubric based on supervised ablation, graph sensitivity, and held-out error.
