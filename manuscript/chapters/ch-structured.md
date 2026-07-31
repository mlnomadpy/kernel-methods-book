---
id: ch-structured
slug: structured-prediction-with-kernels
title: Structured Prediction with Kernels
part: II · Learning with a Fixed Kernel
order: 9
tier: advanced
prerequisites:
  - kernel-tricks
  - support-vector-machines
  - ranking-and-ordinal-regression
objectives:
  - >-
    Formulate structured prediction through joint feature maps and
    operator-valued score functions.
  - >-
    Distinguish task loss, surrogate loss, ordinary decoding, and loss-augmented
    inference.
  - >-
    Derive the structured hinge, its subgradients, and its finite representer
    reduction.
  - >-
    Prove correctness and finite termination of an exact cutting-plane solver
    under explicit oracle assumptions.
  - >-
    State calibration and Fisher-consistency claims only within their valid
    output, loss, and decoding regimes.
  - >-
    Diagnose the computational-statistical decoding gap with exact, approximate,
    and restricted inference.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-structured.yml
verification_date: null
example_code_policy: visible-for-executable
narrative_link_policy: exact
bibliography:
  - bach2024learning
  - taskar2003m3n
  - tsochantaridis2005structured
  - joachims2009struct
  - finley2008approx
  - osokin2017structured
  - cortes1995
  - crammer2001
  - joachims1999
  - scholkopf2002
  - shawe2004
  - steinwart2008
  - bartlett2006
  - zhang2004
  - boyd2004
  - micchelli2005vv
---
# Structured Prediction with Kernels

<p class="lead">A classifier chooses one label. A sequence tagger must choose a whole path of labels whose neighboring decisions agree; a matching system must choose a globally feasible set of pairs; a parser must choose a tree. Scoring each component independently can produce an object that is locally plausible and globally impossible. Structured prediction begins exactly where [[ch:support-vector-machines|the SVM chapter]] and [[ch:ranking-and-ordinal-regression|the ranking chapter]] stop: the margin now compares one feasible object with every competing object, and finding the worst competitor is itself an algorithm. Kernels still provide geometry, but geometry alone is no longer the bottleneck. Training repeatedly calls a decoder, the surrogate may disagree with the task loss, and an approximate oracle can change the optimization problem being solved. We will keep one three-position sequence in view while following that chain from a failed componentwise predictor to joint features, a structured hinge, finite reduction, cutting planes, calibration boundaries, and visible executable code.</p>

## The prediction object and its two geometries {#structured-setting}

Maximum-margin Markov networks provide the graphical-model route into this setting
[@taskar2003m3n, Sections 2--3], while the structural-SVM formulation makes the joint-feature
and loss-augmented-oracle pattern explicit [@tsochantaridis2005structured, Sections 2--3].

Let \(\mathcal X\) be an input space. Each input \(x\) has a nonempty feasible output set \(\mathcal Y(x)\), possibly exponential in the size of \(x\). A predictor returns

$$
\widehat y_f(x)\in\arg\max_{y\in\mathcal Y(x)} f(x,y),
$$

where \(f\) is a real-valued compatibility score. The task loss

$$
\Delta_x:\mathcal Y(x)\times\mathcal Y(x)\to[0,\infty)
$$

measures the cost of predicting its second argument when the first is correct. It need not be a metric, symmetric, decomposable, or bounded independently of structure size. Those properties must be stated when an algorithm or theorem uses them.

There are two distinct geometries:

1. the **score geometry**, induced by a kernel or feature map on input-output pairs;
2. the **decision geometry**, induced by feasibility and \(\Delta_x\).

A rich score class cannot rescue a decoder that searches the wrong feasible set. Conversely, exact decoding cannot rescue a surrogate whose population minimizer does not induce a Bayes decision for the task loss.

::: {.definition #def-structured-joint-map}
[Definition (joint feature score)]{.box-title}

Let \(\mathcal H\) be a real Hilbert space and let
\(\Psi:\{(x,y):x\in\mathcal X,\ y\in\mathcal Y(x)\}\to\mathcal H\).
For \(w\in\mathcal H\), define

$$
f_w(x,y)=\langle w,\Psi(x,y)\rangle_{\mathcal H}.
$$

The induced joint kernel is

$$
k_J\{(x,y),(x',y')\}
=
\langle\Psi(x,y),\Psi(x',y')\rangle_{\mathcal H}.
$$
:::

The kernel is positive semidefinite by construction. In a chain model,

$$
\Psi(x,y)
=
\sum_{t=1}^{T}\psi_{\mathrm{emit}}(x,t,y_t)
+
\sum_{t=2}^{T}\psi_{\mathrm{trans}}(y_{t-1},y_t),
$$

so the score decomposes over nodes and edges. Dynamic programming is possible because of this factorization, not because the joint kernel is positive semidefinite.

The distinction becomes concrete in the running example. There are three positions and
two labels, \(0\) and \(1\). Emission scores prefer the componentwise sequence \(010\),
but a transition reward of \(0.6\) favors adjacent equal labels. Choosing the largest
emission independently gives a score of \(2.4\); the feasible *joint* maximizer is
\(000\), with score \(2.8\). Nothing is wrong with any local decision. The failure is
that local maximization has discarded the transition features.

This is the first pressure inherited from the ordinary kernel machine. The representer
theorem in [[ch:kernel-tricks|the kernel-trick chapter]] can reduce a variational problem
to coefficients, and the hinge construction in [[ch:support-vector-machines|the SVM
chapter]] can compare two labels. Neither result tells us how to search an exponential
output set. The new object is therefore not merely a more elaborate loss: it is a
**learning-and-inference contract**. We must specify both the score geometry and the
algorithm that maximizes it.

For this tiny chain, the joint feature map can be written without abstraction. Give
each position-label pair one coordinate and each transition type one coordinate:

$$
\Psi(y)
=
\left(
\mathbf 1\{y_t=c\}_{t\in\{1,2,3\},\,c\in\{0,1\}},
\;
\sum_{t=2}^3\mathbf 1\{(y_{t-1},y_t)=(a,b)\}_{a,b\in\{0,1\}}
\right).
$$

The first six coordinates record which label occurs at each position. The final four
count \(00,01,10,11\) transitions. Put the six emission scores and four transition
scores into \(w\); then \(\langle w,\Psi(y)\rangle\) is exactly the chain score used
throughout the chapter. Learning \(w\) means learning how much each local event should
contribute, while Viterbi exploits the fact that those contributions factor along the
chain.

The associated joint kernel has a direct interpretation:

$$
k_J(y,y')
=
\sum_{t=1}^3\mathbf 1\{y_t=y'_t\}
+
\sum_{a,b}
N_{ab}(y)N_{ab}(y'),
$$

where \(N_{ab}(y)\) counts \(a\!\to b\) transitions. The first term counts positional
label agreement; the second compares transition profiles. This kernel is PSD because
it is the inner product of the displayed feature vectors. It is also cheap to
evaluate. Yet neither fact gives the maximizing sequence: tractable decoding follows
from the additive score factorization. This one calculation separates the chapter's
three claims: valid geometry, learnable weights, and efficient inference. That distinction matters before larger
output spaces make them easy to blur.

## The operator-valued view {#structured-operator-view}

When the output set is finite and common to every input, write

$$
F_w(x)
=
\bigl(f_w(x,y)\bigr)_{y\in\mathcal Y}
\in\mathbb R^{|\mathcal Y|}.
$$

Define \(\Phi(x):\mathbb R^{|\mathcal Y|}\to\mathcal H\) by

$$
\Phi(x)a=\sum_{y\in\mathcal Y}a_y\Psi(x,y).
$$

Then

$$
K(x,x')=\Phi(x)^*\Phi(x')
$$

is an operator-valued kernel, and its entries are

$$
K(x,x')_{y,y'}
=
k_J\{(x,y),(x',y')\}.
$$

This factorization certifies block positivity:

$$
\sum_{i,j}\langle a_i,K(x_i,x_j)a_j\rangle
=
\left\|\sum_i\Phi(x_i)a_i\right\|_{\mathcal H}^{2}
\ge0.
$$

The vector-valued RKHS viewpoint from [@micchelli2005vv] is therefore not a competing model. It is another coordinate system for the same family of scores. The joint-map view is convenient for combinatorial factorization; the operator-valued view exposes coupling among output coordinates.

::: {.proposition #prop-structured-equivalence}
[Proposition (joint and operator-valued representations agree)]{.box-title}

For finite \(\mathcal Y\), the score vectors \(F_w(x)=\Phi(x)^*w\) generated by a joint feature map belong to the vector-valued RKHS induced by \(K(x,x')=\Phi(x)^*\Phi(x')\). Conversely, every finite linear combination

$$
F(\cdot)=\sum_{i=1}^nK(\cdot,x_i)a_i
$$

has joint-space parameter \(w=\sum_i\Phi(x_i)a_i\).

**Assumptions.** Common finite output set, Hilbert-valued joint features, and the stated bounded feature operators. **Proof status.** Complete.
:::

::: {.proof}
[Proof]{.box-title}

The first claim follows from the feature-operator construction of a vector-valued RKHS. For the converse,

$$
F(x)
=
\sum_i\Phi(x)^*\Phi(x_i)a_i
=
\Phi(x)^*
\left\{\sum_i\Phi(x_i)a_i\right\}
=F_w(x).
$$

Thus both forms generate the same finite span. Completion handles norm limits. [\(\square\)]{.qed}
:::

For input-dependent or enormous output spaces, explicitly materializing \(F_w(x)\) is impossible. The joint representation remains useful because a factorized decoder can maximize the score without enumerating all coordinates.

## Loss, decoding, and loss-augmented decoding {#structured-loss-decoding}

Three operations must not be conflated:

$$
\begin{aligned}
\text{prediction:}\quad&
\widehat y_w(x)
\in\arg\max_y f_w(x,y),\\
\text{evaluation:}\quad&
\Delta_x\{y^\star,\widehat y_w(x)\},\\
\text{loss-augmented inference:}\quad&
\widetilde y_i(w)
\in\arg\max_{y\in\mathcal Y(x_i)}
\left[
\Delta_{x_i}(y_i,y)
+f_w(x_i,y)-f_w(x_i,y_i)
\right].
\end{aligned}
$$

Prediction uses only score. Loss augmentation adds the training loss. If \(\Delta\) decomposes over the same factors as the score, the same dynamic program often works after modifying local potentials. If it does not, exact loss augmentation may be harder than ordinary prediction.

For Hamming loss on a sequence,

$$
\Delta(y,y')=\sum_{t=1}^{T}\mathbf 1\{y_t\ne y'_t\},
$$

loss augmentation adds one local cost to every wrong label. For exact-match loss
\(\mathbf 1\{y\ne y'\}\), one global bit records whether any discrepancy has occurred. For F-scores, intersection-over-union, or matching-specific losses, the required state may expand dramatically or destroy the original factorization.

Why can we not train by repeatedly calling the ordinary decoder? Because the most
dangerous competitor is not necessarily the currently highest-scoring one. A wrong
structure with slightly lower model score but very large task loss can violate the
desired margin more severely. The training oracle must maximize *loss plus score* so
that the required score gap grows with the cost of the mistake. In the running example,
ordinary decoding prefers \(000\), but loss augmentation prefers \(111\). Reusing the
ordinary decoder would never expose the constraint associated with \(111\), even
though it defines the hinge value.

That distinction also prevents a common implementation bug. A decoder API that accepts
only model potentials is not yet a loss-augmented oracle. For Hamming loss, the repair
is local: add a unit cost to each incorrect label before running Viterbi. For exact
match, an additional state must remember whether the path has departed from the truth.
For a nondecomposable F-score, the dynamic program may need to track a sufficient
count such as the number of predicted positives and true positives. The state space,
and therefore the computational cost, is determined jointly by the score
factorization and the loss.

The contract should be tested, not inferred from a function name. On a small structure,
enumerate every feasible output and compare:

1. the score-only maximizer with the ordinary decoder;
2. the loss-plus-score maximizer with the loss-augmented decoder;
3. the reported maximum values, not only the returned structures;
4. deterministic behavior under ties.

Agreement on the returned label alone is insufficient: two implementations can select
the same maximizer while disagreeing on its value because a constant truth score was
subtracted in one place but not another. That offset does not affect the argmax, but it
does affect the hinge, slack, stopping test, and dual objective.

<figure class="viz" data-widget="structured-decoding-gap"><figcaption>Three operations that are easy to conflate: ordinary prediction ranks candidates by model score, loss augmentation changes those scores during training, and structured decoding enforces global feasibility. Local winners need not assemble into the maximizing valid structure.</figcaption></figure>

## The structured hinge {#structured-hinge}

Given training pairs \((x_i,y_i)_{i=1}^n\), define

$$
\delta\Psi_i(y)
=
\Psi(x_i,y_i)-\Psi(x_i,y).
$$

The margin-rescaled structured hinge is

$$
H_i(w)
=
\max_{y\in\mathcal Y(x_i)}
\left[
\Delta_i(y)-\langle w,\delta\Psi_i(y)\rangle
\right],
\qquad
\Delta_i(y)=\Delta_{x_i}(y_i,y).
$$

The term with \(y=y_i\) is zero when \(\Delta_i(y_i)=0\), so \(H_i(w)\ge0\).
It extends the binary maximum-margin construction [@cortes1995] and its multiclass
score-vector generalization [@crammer2001] by replacing the finite label comparison
with a loss-weighted comparison over feasible structures.

For the running sequence, take the truth to be \(010\) and use Hamming loss. Ordinary
decoding returns \(000\), which makes one error. Loss augmentation instead asks for the
sequence maximizing

$$
f(y)+\Delta(010,y).
$$

It returns \(111\) with augmented value \(4.6\). Subtracting the truth score \(2.4\)
gives hinge \(2.2\). The same model has therefore produced three different objects:
the truth \(010\), the deployed prediction \(000\), and the training adversary \(111\).
Keeping those objects separate is the central discipline of the chapter.

::: {.proposition #prop-structured-upper-bound}
[Proposition (structured hinge upper-bounds task loss)]{.box-title}

Let \(\widehat y_i\in\arg\max_y f_w(x_i,y)\), and assume
\(\Delta_i(y_i)=0\) and \(\Delta_i(y)\ge0\). Then

$$
\Delta_i(\widehat y_i)\le H_i(w).
$$

**Assumptions.** Exact score decoding over the same feasible set used by the hinge and nonnegative loss vanishing at the truth. **Proof status.** Complete.
:::

::: {.proof}
[Proof]{.box-title}

Since \(\widehat y_i\) maximizes the score,

$$
\langle w,\Psi(x_i,\widehat y_i)-\Psi(x_i,y_i)\rangle\ge0.
$$

Evaluating the maximum defining \(H_i\) at \(\widehat y_i\) gives

$$
H_i(w)
\ge
\Delta_i(\widehat y_i)
+
\langle w,\Psi(x_i,\widehat y_i)-\Psi(x_i,y_i)\rangle
\ge
\Delta_i(\widehat y_i).
$$

[\(\square\)]{.qed}
:::

The regularized estimator is

$$
\min_{w\in\mathcal H}
J(w)
=
\frac{\lambda}{2}\|w\|_{\mathcal H}^{2}
+
\frac1n\sum_{i=1}^{n}H_i(w),
\qquad \lambda\gt0.
$$
{#eq-structured-normalized-objective}

Each \(H_i\) is a pointwise maximum of affine functions, hence convex. If
\(\widetilde y_i(w)\) is any exact maximizer, then

$$
-\delta\Psi_i\{\widetilde y_i(w)\}
\in\partial H_i(w),
$$

and therefore

$$
\lambda w-\frac1n\sum_i
\delta\Psi_i\{\widetilde y_i(w)\}
\in\partial J(w).
$$

Ties produce a set of subgradients: every convex combination of active feature differences is valid.

## Representer reduction and finite coefficients {#structured-representer}

The structured hinge depends on \(w\) through inner products with every
\(\delta\Psi_i(y)\). If every training output set is finite, their total span is finite even when \(\mathcal H\) is not.

::: {.theorem #thm-structured-representer}
[Theorem (structured representer reduction)]{.box-title}

Assume every \(\mathcal Y(x_i)\) is finite and \(\lambda\gt0\). The objective \(J\) has a unique minimizer \(w^\star\), and

$$
w^\star
\in
\mathcal S
=
\operatorname{span}
\left\{
\delta\Psi_i(y):
1\le i\le n,\ y\in\mathcal Y(x_i)
\right\}.
$$

Equivalently,

$$
w^\star
=
\sum_{i=1}^{n}
\sum_{y\in\mathcal Y(x_i)}
\alpha_{iy}\delta\Psi_i(y)
$$

for finite coefficients \(\alpha_{iy}\).

**Assumptions.** Finite training output sets, finite-valued feature vectors and losses, and positive quadratic regularization. **Proof status.** Complete.
:::

::: {.proof}
[Proof]{.box-title}

The maximum of finitely many continuous affine functions is continuous and convex. Adding the coercive term \(\lambda\|w\|^2/2\) gives existence; strong convexity gives uniqueness. Decompose \(w=w_{\parallel}+w_{\perp}\) with \(w_{\parallel}\in\mathcal S\) and \(w_{\perp}\perp\mathcal S\). Every hinge term is unchanged because

$$
\langle w,\delta\Psi_i(y)\rangle
=
\langle w_{\parallel},\delta\Psi_i(y)\rangle.
$$

If \(w_{\perp}\ne0\), the regularizer is strictly larger by
\(\lambda\|w_\perp\|^2/2\). Hence the unique minimizer has \(w_\perp=0\). [\(\square\)]{.qed}
:::

All finite optimization can be expressed through the joint kernel because

$$
\langle\delta\Psi_i(y),\delta\Psi_j(y')\rangle
$$

expands into four evaluations of \(k_J\). The reduction is finite but may still be exponentially large. The cutting-plane method constructs only the constraints that matter.

The contrast with the ordinary representer theorem is worth making explicit. In binary
classification, one training example contributes one evaluation functional and hence
one kernel section. Here, example \(i\) contributes a feature *difference* for every
competitor \(y\). The theorem removes the infinite dimension of \(\mathcal H\), but it
does not remove the combinatorics of \(\mathcal Y(x_i)\). That second reduction must
come from factorized inference, constraint generation, or approximation. “Kernelized”
and “computationally tractable” remain different claims.

## The margin program and its restricted form {#structured-margin-program}

Introduce one slack \(\xi_i\) per example:

$$
\begin{aligned}
\min_{w,\xi}\quad&
\frac{\lambda}{2}\|w\|^2+\frac1n\sum_i\xi_i\\
\text{subject to}\quad&
\langle w,\delta\Psi_i(y)\rangle
\ge \Delta_i(y)-\xi_i,
\qquad
y\in\mathcal Y(x_i),\\
&\xi_i\ge0.
\end{aligned}
$$

At fixed \(w\), the smallest feasible slack is exactly \(H_i(w)\). Thus this program and the unconstrained structured-hinge objective are equivalent.

Check the equivalence with two candidate outputs whose violations are \(0.3\) and
\(-0.2\). Feasibility requires \(\xi_i\geq0.3\), so minimization sets
\(\xi_i=0.3=H_i(w)\). If every violation is negative, nonnegativity sets \(\xi_i=0\).
Dropping \(\xi_i\geq0\) would reward an already-correct example with negative slack and
change the loss. This calculation is the scalar oracle for the restricted master.

The dual reveals what the cutting plane is actually constructing. Attach a
nonnegative multiplier \(\alpha_{iy}\) to each margin constraint, while keeping the
nonnegative slack domain explicit. The Lagrangian is

$$
\mathcal L(w,\xi,\alpha)
=
\frac{\lambda}{2}\|w\|^2
+
\frac1n\sum_i\xi_i
+
\sum_{i,y}\alpha_{iy}
\left[
\Delta_i(y)-\xi_i-\langle w,\delta\Psi_i(y)\rangle
\right].
$$

Minimizing over \(w\) gives

$$
w
=
\frac1\lambda
\sum_{i,y}\alpha_{iy}\delta\Psi_i(y).
$$

This is more informative than merely restating the representer theorem: the coefficient
of a feature difference is a dual weight on a violated structured comparison.
Minimizing over the restricted domain \(\xi_i\ge0\) is finite exactly when

$$
\sum_y\alpha_{iy}\le\frac1n.
$$

The inequality, rather than equality, is the dual trace of the explicit
\(\xi_i\ge0\) constraint. Substitution yields

$$
\begin{aligned}
\max_{\alpha\ge0}\quad&
\sum_{i,y}\alpha_{iy}\Delta_i(y)
-
\frac{1}{2\lambda}
\left\|
\sum_{i,y}\alpha_{iy}\delta\Psi_i(y)
\right\|_{\mathcal H}^2\\
\text{subject to}\quad&
\sum_y\alpha_{iy}\le\frac1n
\qquad\text{for every }i.
\end{aligned}
$$

The quadratic term is computable from four joint-kernel evaluations for each pair of
feature differences. A working-set method begins with almost all
\(\alpha_{iy}=0\), then gives a new coordinate permission to become nonzero whenever
the separation oracle finds a violated comparison. In this sense, constraint
generation in the primal and column generation in the dual are the same act viewed
from opposite sides.

The dual also exposes two distinct notions of sparsity. Only some training examples
may carry nonzero total dual mass, echoing ordinary support vectors. Within an active
example, only some competing structures may receive nonzero weights. Neither kind of
sparsity is guaranteed to make prediction cheap: decoding a new input still maximizes
over its full feasible output set. Training sparsity and inference complexity answer
different questions.

For working sets \(W_i\subseteq\mathcal Y(x_i)\), the restricted problem keeps only those constraints. Let its exact minimizer be \((w_W,\xi_W)\), and define the most violated value

The numerical diagnostic is therefore not only the restricted optimizer residual. After
each solve, loss-augmented inference over the full output space reports the largest omitted
violation. A small restricted residual is not a global certificate when that violation
exceeds tolerance. This failure forces the cutting-plane loop developed next.

$$
v_i(w_W)
=
\max_{y\in\mathcal Y(x_i)}
\left[\Delta_i(y)-\langle w_W,\delta\Psi_i(y)\rangle\right].
$$

The violation is \(v_i(w_W)-\xi_{W,i}\).

## Exact cutting planes: correctness and termination {#structured-cutting-plane}

The one-slack cutting-plane contract and its iteration analysis follow the structural-SVM
development of Joachims, Finley, and Yu [@joachims2009struct, Sections 3--4]. The theorem below
states the oracle and tolerance conventions locally because changing either changes the claim.
The working-set logic inherits the decomposition principle used for large ordinary
SVMs [@joachims1999], but the structured separation oracle now generates a whole
output constraint rather than selecting a scalar training point.

That difference changes the cost model. In an ordinary decomposition method, checking
one candidate point requires a kernel score and a KKT test. In a structured method,
checking one training case can require a complete dynamic program, matching solve, or
integer optimization. Report wall time per oracle call, the number of calls, and the
restricted-QP time separately. Otherwise an apparently slow optimizer may really be a
fast optimizer wrapped around an expensive decoder, and changing QP software will not
address the bottleneck.

:::: {.algorithm #algo-structured-cutting-plane}
[Algorithm (exact \(n\)-slack cutting plane)]{.box-title}

**Input.** Training pairs, \(\lambda\gt0\), tolerance \(\varepsilon\gt0\), finite feasible sets, and an exact loss-augmented oracle.

**Output.** A feasible point whose objective is at most \(\varepsilon\) above the full optimum.

1. Initialize \(W_i=\{y_i\}\) for all \(i\).
2. Solve the restricted convex quadratic program exactly.
3. For every \(i\), call the oracle for \(\widetilde y_i\in\arg\max_y\{\Delta_i(y)-\langle w,\delta\Psi_i(y)\rangle\}\).
4. If \(v_i(w)-\xi_i\gt\varepsilon\), add \(\widetilde y_i\) to \(W_i\).
5. If no constraint is added, return \((w,\xi+\varepsilon\mathbf1)\); otherwise repeat.

**Numerical contract.** The restricted solver's primal feasibility and duality gap must be below a declared inner tolerance. Oracle ties may be broken deterministically. Cached scores may be reused only if they refer to the current \(w\).
::::

::: {.theorem #thm-structured-cutting-correct}
[Theorem (finite termination and additive certificate)]{.box-title}

Under the algorithm's assumptions, no output constraint is added twice, so the method terminates after at most

$$
\sum_{i=1}^{n}\{|\mathcal Y(x_i)|-1\}
$$

additions. At termination, \((w,\xi+\varepsilon\mathbf1)\) is feasible for the full program, and

$$
J(w)\le J^\star+\varepsilon,
$$

where \(J^\star\) is the full structured-hinge optimum.

**Assumptions.** Finite output sets, exact restricted solves, exact loss-augmented maximization, common additive violation tolerance, and the normalized objective in [[eq:eq-structured-normalized-objective]]. **Proof status.** Complete.
:::

::: {.proof}
[Proof]{.box-title}

If an oracle returns a constraint already in \(W_i\), restricted feasibility gives
\(v_i(w)-\xi_i\le0\), so it cannot pass the addition test. Every addition is therefore new, proving the finite bound.

At termination,

$$
\Delta_i(y)-\langle w,\delta\Psi_i(y)\rangle
\le \xi_i+\varepsilon
$$

for every \(i,y\). Hence \((w,\xi+\varepsilon\mathbf1)\) is fully feasible. Let
\(P_W\) be the restricted optimum and \(P^\star\) the full optimum. Removing constraints can only decrease an optimum, so \(P_W\le P^\star\). The returned feasible point has objective

$$
P_W+\frac1n\sum_i\varepsilon=P_W+\varepsilon\le P^\star+\varepsilon.
$$

Since eliminating slacks recovers \(J\), the same bound holds for \(J(w)\). [\(\square\)]{.qed}
:::

The iteration bound is deliberately crude: it certifies finite termination, not practical speed. Stronger polynomial bounds require bounded feature differences, a particular cutting-plane variant, and quantitative improvement arguments. Those claims need the missing primary cutting-plane citation listed in the chapter handoff.

## Approximate inference changes the contract {#structured-approximate-inference}

Undergenerating search and overgenerating relaxations fail in different directions; Finley
and Joachims analyze precisely why only some guarantees survive approximate inference
[@finley2008approx, Sections 3--4].

Suppose an oracle returns \(\bar y\) whose augmented score is within \(\eta\) of the maximum:

$$
\Delta_i(\bar y)-\langle w,\delta\Psi_i(\bar y)\rangle
\ge
v_i(w)-\eta.
$$

If the algorithm stops because the returned violation is at most \(\varepsilon\), then

$$
v_i(w)-\xi_i\le\varepsilon+\eta.
$$

The preceding proof therefore yields an \(\varepsilon+\eta\) objective certificate when \(\eta\) is a valid uniform additive oracle bound.

Heuristic beam search usually provides no such \(\eta\). In that case, “no violated constraint found” is not a certificate. The algorithm has optimized a dynamically restricted surrogate whose relation to the intended structured hinge is unknown.

Two approximations that are often grouped together have opposite logical behavior.

| Oracle behavior | Search domain | What a returned value proves | Main failure |
|---|---|---|---|
| undergenerating search | \(\mathcal C(x)\subset\mathcal Y(x)\) | a lower bound on the true maximum violation | stopping can miss an omitted constraint |
| overgenerating relaxation | \(\mathcal Y(x)\subset\overline{\mathcal Y}(x)\) | an upper bound when the relaxation objective dominates the integral problem | the returned object may be fractional or infeasible |
| certified additive approximation | full problem with error \(\eta\) | maximum is within declared score units | certificate may scale badly with structure size |
| heuristic local search | implicit neighborhood | only that no improving move was found | neither separation nor stopping is certified |

Suppose beam search examines four of eight sequences in the running example and omits
\(111\). It may report \(000\) as the largest augmented competitor with value \(3.8\),
while exact enumeration finds value \(4.6\). The resulting hinge is understated by
\(0.8\). If the cutting-plane tolerance is \(0.1\), a restricted solver can appear
fully converged even though the missing oracle error is eight times the stopping
tolerance. Increasing QP accuracy cannot repair the absent structure.

An overgenerating relaxation can be useful in the opposite way. If a relaxed maximizer
has augmented value no larger than the current slack plus tolerance, then no integral
structure can violate the constraint more: the upper bound certifies separation.
But when the relaxed optimum is fractional and violated, rounding it may reduce the
value, so the rounded structure need not provide the violated integral constraint that
the optimizer expects. A correct implementation records both values and labels them:
the relaxed upper bound for certification and the feasible rounded value for adding a
constraint.

::: {.proposition #prop-structured-oracle-gap}
[Proposition (additive oracle error propagates additively)]{.box-title}

If every oracle call has certified additive error at most \(\eta_i\), and termination uses returned violation tolerance \(\varepsilon_i\), then increasing slack \(i\) by \(\varepsilon_i+\eta_i\) gives full feasibility. The returned objective is at most

$$
P^\star+\frac1n\sum_i(\varepsilon_i+\eta_i).
$$

**Assumptions.** Exact restricted solve and valid per-example additive oracle certificates over the full feasible sets. **Proof status.** Complete by the same feasibility argument as the cutting-plane theorem.
:::

Approximation ratios are not interchangeable with additive score errors. Negative augmented scores, arbitrary offsets, and structure-size scaling can make a multiplicative ratio meaningless. State the oracle guarantee in the same units as the stopping test.

## Calibration and Fisher-consistency boundaries {#structured-calibration}

For general finite structured losses, calibration must be proved for the chosen surrogate,
feasible score set, and decoder. Osokin, Bach, and Lacoste-Julien provide the calibration
function framework and show how output cardinality enters the statistical and optimization
cost [@osokin2017structured, Sections 3--4].

Surrogate optimization and statistical decision quality are separate claims. Let
\(P(\cdot\mid x)\) be the conditional law of \(Y\). The Bayes action for task loss is

$$
y^\star_P(x)
\in
\arg\min_{a\in\mathcal Y(x)}
\mathbb E[\Delta_x(Y,a)\mid X=x].
$$

A surrogate \(L(s,Y)\), where \(s\) is a score vector or structured score function, is Fisher consistent for \(\Delta\) under decoder \(d\) if every conditional surrogate-risk minimizer \(s^\star\) satisfies

$$
d(s^\star)\in
\arg\min_a\mathbb E[\Delta_x(Y,a)\mid X=x].
$$

Binary hinge calibration does not automatically lift to arbitrary structures. Results for ordinary multiclass zero-one loss do not automatically cover Hamming, F-score, matching, or ranking losses. Convex classification calibration is developed for specific loss and score constructions in [@bartlett2006; @zhang2004]; applying those results requires matching their setting.

::: {.proposition #prop-structured-hamming-calibration}
[Proposition (a positive boundary: separable proper scores for Hamming loss)]{.box-title}

Let \(\mathcal Y=\mathcal A^T\), let
\(\Delta(y,a)=\sum_t\mathbf1\{y_t\ne a_t\}\), and suppose a surrogate estimates each conditional marginal \(p_t(c\mid x)\) consistently with a strictly proper multiclass loss. Coordinatewise decoding

$$
\widehat y_t(x)\in\arg\max_{c\in\mathcal A}p_t(c\mid x)
$$

is Fisher consistent for Hamming loss.

**Assumptions.** Unconstrained product output space, additive Hamming loss, correct conditional-risk minimization for every marginal, and coordinatewise Bayes decoding. **Proof status.** Complete.
:::

::: {.proof}
[Proof]{.box-title}

Conditional Hamming risk separates:

$$
\mathbb E[\Delta(Y,a)\mid x]
=
\sum_t\{1-P(Y_t=a_t\mid x)\}.
$$

Each summand is minimized by a marginal mode. Strict propriety recovers the required marginal probabilities at the population optimum. [\(\square\)]{.qed}
:::

This proposition fails as stated when outputs obey a global constraint. Coordinatewise marginal modes can be infeasible. It also does not establish consistency for exact-match loss, whose Bayes action is the joint mode, not the vector of marginal modes.

## From an empirical margin to population task risk {#structured-generalization}

The upper-bound proposition was pointwise on the training sample. Calibration was a
population statement about conditional risk minimizers. Between them lies the finite
sample question: if we minimize the regularized empirical hinge, how far can its
population hinge risk move when one training structure changes?

This question belongs to the same approximation–estimation–optimization ledger
developed in [[ch:learning-theory|the learning-theory chapter]]. Structured prediction
adds one important scaling issue. A feature difference may grow with the size of the
output. For a length-\(T\) chain built by summing bounded local features,
\(\|\delta\Psi\|\) can grow like \(T\) unless the score is normalized or a sharper
orthogonality argument is available. A statement that suppresses this dependence can
look sample-efficient while becoming vacuous on long sequences.

::: {.theorem #thm-structured-stability}
[Theorem (expected stability bound for the regularized structured hinge)]{.box-title}

Let \(S=(Z_1,\ldots,Z_n)\) be IID, and let

$$
\widehat w_S
\in
\arg\min_w
\left\{
\frac{\lambda}{2}\|w\|_{\mathcal H}^2
+
\frac1n\sum_{i=1}^n H_{Z_i}(w)
\right\}.
$$

Assume

$$
\sup_{z}\sup_{y\in\mathcal Y(x)}
\|\Psi(x,y_z)-\Psi(x,y)\|_{\mathcal H}
\le R,
$$

where \(y_z\) is the observed output in \(z=(x,y_z)\), and assume the
relevant hinge risks are integrable. Then changing one observation changes the loss
on any test structure by at most

$$
\beta=\frac{2R^2}{\lambda n}.
$$

Consequently, with
\(\mathcal H_{\mathrm{pop}}(w)=\mathbb E_Z H_Z(w)\),

$$
\mathbb E_S\!\left[
\mathcal H_{\mathrm{pop}}(\widehat w_S)
-
\frac1n\sum_{i=1}^nH_{Z_i}(\widehat w_S)
\right]
\le
\frac{2R^2}{\lambda n}.
$$

If deployment uses exact score decoding over the same feasible set and the task loss
is nonnegative and vanishes at the truth, then

$$
\mathbb E_S R_\Delta(\widehat w_S)
\le
\mathbb E_S\!\left[\frac1n\sum_iH_{Z_i}(\widehat w_S)\right]
+
\frac{2R^2}{\lambda n}.
$$

**Assumptions.** IID sampling, positive quadratic regularization, uniformly bounded
feature differences, integrable losses, and exact deployment decoding for the last
inequality. **Proof status.** Complete. This is an expectation bound, not a
high-probability calibration theorem.
:::

::: {.proof}
[Proof]{.box-title}

Every hinge \(H_z\) is \(R\)-Lipschitz because it is a maximum of affine functions
whose slopes are \(-\delta\Psi_z(y)\). Let \(S\) and \(S'\) differ in one observation,
and write \(w=\widehat w_S\), \(w'=\widehat w_{S'}\). Strong convexity and optimality
give

$$
F_S(w')-F_S(w)\ge\frac{\lambda}{2}\|w'-w\|^2,
\qquad
F_{S'}(w)-F_{S'}(w')\ge\frac{\lambda}{2}\|w'-w\|^2.
$$

Adding cancels the \(n-1\) common losses. The two remaining loss differences are each
at most \(R\|w-w'\|\), so

$$
\lambda\|w-w'\|^2
\le
\frac{2R}{n}\|w-w'\|.
$$

Thus \(\|w-w'\|\le 2R/(\lambda n)\), and for any test structure \(z\),

$$
|H_z(w)-H_z(w')|
\le
R\|w-w'\|
\le
\frac{2R^2}{\lambda n}.
$$

The replace-one identity for IID samples turns this uniform stability inequality into
the expected generalization-gap bound. Finally, the pointwise structured-hinge
upper bound gives \(R_\Delta(w)\le\mathcal H_{\mathrm{pop}}(w)\) under exact decoding.
Combining the two inequalities proves the last display. [\(\square\)]{.qed}
:::

The theorem shows what a finite-sample guarantee does and does not repair. Increasing
\(\lambda\) improves the stability term but increases regularization bias. Normalizing
the sequence score can control \(R\), but it also changes the meaning of the margin.
Most importantly, the last inequality disappears when deployment uses an uncertified
decoder: the proof of the task-loss upper bound evaluated the hinge at the *actual
score maximizer*. Bach's structured-prediction development makes the same
surrogate–generalization–computation separation explicit [@bach2024learning,
Chapter 13].

## A failure witness: marginally best, jointly impossible {#structured-failure-witness}

Let the feasible set be

$$
\mathcal Y=\{000,110,101,011\},
$$

the binary strings of even parity. Consider

$$
P(110)=0.30,\quad
P(101)=0.30,\quad
P(011)=0.30,\quad
P(000)=0.10.
$$

Each coordinate has marginal probability \(0.60\) of being one, so independent marginal modes produce \(111\), which is infeasible. Under Hamming loss, the best feasible decisions are \(110,101,011\), each with conditional risk \(1.4\); \(000\) has risk \(1.8\). Projecting \(111\) onto the feasible set needs a tie-breaking rule and happens to recover a Bayes action here, but no general theorem licenses arbitrary projection after independent training.

Under exact-match loss, every one of the three probability-\(0.30\) structures is Bayes optimal. Marginals alone cannot identify which joint structures carry the mass. The failure is informational, not numerical.

## Worked sequence-labeling example {#structured-sequence-example}

Consider a length-three binary chain. Let the score be

$$
f(y)
=
\sum_{t=1}^{3}e_t(y_t)
+
\sum_{t=2}^{3}q(y_{t-1},y_t),
$$

with emission scores

| position \(t\) | \(e_t(0)\) | \(e_t(1)\) |
|---:|---:|---:|
| 1 | \(0.8\) | \(0.2\) |
| 2 | \(0.1\) | \(0.9\) |
| 3 | \(0.7\) | \(0.3\) |

and transition score \(q(a,b)=0.6\) if \(a=b\), zero otherwise. Independent emissions choose \(010\), whose score is \(2.4\). Global decoding gives \(000\), whose score is

$$
0.8+0.1+0.7+0.6+0.6=2.8.
$$

Let the true sequence be \(010\) and use Hamming loss. Loss augmentation adds one to the emission of every wrong label:

| position \(t\) | augmented score for \(0\) | augmented score for \(1\) |
|---:|---:|---:|
| 1 | \(0.8\) | \(1.2\) |
| 2 | \(1.1\) | \(0.9\) |
| 3 | \(0.7\) | \(1.3\) |

The loss-augmented maximizer is \(111\), with value

$$
1.2+0.9+1.3+0.6+0.6=4.6.
$$

Because the true-sequence score is \(2.4\), the structured hinge is \(4.6-2.4=2.2\). The decoded task loss is \(\Delta(010,000)=1\), confirming the upper bound.

::: {.example #example-structured-viterbi}
[Example (prediction and loss augmentation call the same dynamic program)]{.box-title}

The Viterbi recurrence is

$$
V_t(b)
=
e_t(b)+
\max_{a\in\{0,1\}}
\{V_{t-1}(a)+q(a,b)\}.
$$

Replacing \(e_t(b)\) by
\(e_t(b)+\mathbf1\{b\ne y_t^\star\}\) performs Hamming loss augmentation without changing the state graph. The deterministic check enumerates all eight sequences and verifies the ordinary optimum \(000\), augmented optimum \(111\), hinge \(2.2\), and upper-bound inequality.

The complete check is shown below. The enumerated baseline is deliberate: it tests the
dynamic program against the definition rather than merely running the same recurrence
twice.

```python
from itertools import product

emission = (
    {0: 0.8, 1: 0.2},
    {0: 0.1, 1: 0.9},
    {0: 0.7, 1: 0.3},
)
truth = (0, 1, 0)

def transition(previous, current):
    return 0.6 if previous == current else 0.0

def score(sequence):
    node_score = sum(emission[t][label] for t, label in enumerate(sequence))
    edge_score = sum(
        transition(sequence[t - 1], sequence[t])
        for t in range(1, len(sequence))
    )
    return node_score + edge_score

def hamming(sequence):
    return sum(label != target for label, target in zip(sequence, truth))

def viterbi(loss_augmented=False):
    values = {
        label: emission[0][label] + (label != truth[0] if loss_augmented else 0)
        for label in (0, 1)
    }
    paths = {label: (label,) for label in (0, 1)}
    for t in range(1, len(truth)):
        candidates = {}
        for label in (0, 1):
            candidates[label] = max(
                (
                    values[previous]
                    + transition(previous, label)
                    + emission[t][label]
                    + (label != truth[t] if loss_augmented else 0),
                    paths[previous] + (label,),
                )
                for previous in (0, 1)
            )
        values = {label: value for label, (value, _) in candidates.items()}
        paths = {label: path for label, (_, path) in candidates.items()}
    best_label = max(values, key=lambda label: (values[label], paths[label]))
    return paths[best_label], values[best_label]

sequences = list(product((0, 1), repeat=3))
ordinary = max((score(y), y) for y in sequences)
augmented = max((score(y) + hamming(y), y) for y in sequences)

assert viterbi() == ordinary[::-1] == ((0, 0, 0), 2.8)
assert viterbi(True) == augmented[::-1] == ((1, 1, 1), 4.6)

hinge = augmented[0] - score(truth)
decoded_loss = hamming(ordinary[1])
assert abs(hinge - 2.2) < 1e-12
assert decoded_loss == 1
assert decoded_loss <= hinge
```
:::

The example is tiny for a reason: every claim is hand-checkable. On a chain with \(T\) positions and \(L\) labels, the same recurrence costs \(O(TL^2)\) time and \(O(TL)\) memory with backpointers, or \(O(L)\) score memory if only the optimum value is needed.

## The computational-statistical decoding gap {#structured-decoding-gap}

There are at least four predictors hiding behind the phrase “trained structured model”:

1. the exact minimizer of the population surrogate;
2. the exact empirical minimizer with exact loss-augmented inference;
3. the output of a finite optimization run with certified residual and oracle gaps;
4. the deployed predictor using its actual approximate decoder.

Their risks can differ for different reasons. A useful decomposition is conceptual:

$$
\begin{aligned}
R_\Delta(\widehat y_{\mathrm{deploy}})-R_\Delta^\star
=&
\underbrace{\text{surrogate calibration gap}}_{\text{population mismatch}}
+
\underbrace{\text{estimation gap}}_{\text{finite sample}}
\\
&+
\underbrace{\text{optimization gap}}_{\text{finite solve}}
+
\underbrace{\text{training-oracle gap}}_{\text{approximate separation}}
+
\underbrace{\text{deployment-decoding gap}}_{\text{approximate prediction}}.
\end{aligned}
$$

This is a bookkeeping identity only after each term is defined through intermediate predictors. It is not a theorem that every term is nonnegative or separately observable.

The intermediate predictors turn vague debugging into controlled comparisons. Hold
the learned score fixed and replace only the deployment decoder: the change in task
risk measures a deployment-decoding effect, not estimation. Hold the working set fixed
and tighten the QP tolerance: the change isolates numerical optimization inside the
restricted problem. Replace beam separation with exact enumeration on short
structures: the difference measures training-oracle error on a regime where ground
truth is available. Finally, increase the sample size while keeping the model,
regularization convention, and exact oracle fixed: only then does an empirical
learning curve speak primarily to estimation.

| Question | Controlled comparison | Required diagnostic |
|---|---|---|
| Did the restricted QP converge? | same working set, tighter inner tolerance | primal feasibility and duality gap |
| Did separation succeed? | approximate versus exact oracle on small cases | maximum augmented-score gap |
| Did the candidate set erase the answer? | full versus restricted feasible set | gold and high-loss-candidate recall |
| Did approximate deployment change decisions? | exact versus deployed decoder at fixed \(w\) | task loss and score gap |
| Did more data help? | nested samples with the same full pipeline | held-out task risk with uncertainty |

Reporting only final task accuracy collapses all five questions. A worse score may come
from insufficient data, a poorly calibrated surrogate, an unfinished QP, a missed
training constraint, or a beam that cannot represent the correct output. Those causes
suggest different repairs; the experiment must keep them distinguishable.

If deployment uses a restricted candidate set \(\mathcal C(x)\subsetneq\mathcal Y(x)\), then even the best learned score cannot return a Bayes action outside \(\mathcal C(x)\). Candidate recall is therefore part of the statistical system. Measure:

- oracle objective gaps on instances where exact decoding is possible;
- candidate-set recall of the gold output and high-loss competitors;
- task risk as beam width or relaxation tolerance changes;
- train-time and deploy-time decoder agreement;
- feasibility and deterministic tie handling.

## Matching as a second structured domain {#structured-matching}

For bipartite matching, an output is a binary matrix \(Y\) satisfying row and column constraints. A linear assignment score has the form

$$
f_w(x,Y)=\sum_{a,b}Y_{ab}\langle w,\psi(x,a,b)\rangle.
$$

Ordinary decoding is a maximum-weight matching. If the loss decomposes over selected edges, loss augmentation changes edge costs and preserves polynomial-time assignment. A nondecomposable graph-level loss may not.

The continuous relaxation is exact only when the feasible polytope is integral for the stated constraints. Adding side constraints can destroy integrality. Solving the relaxed linear program then returns a fractional object that needs rounding, creating a separate deployment-decoding gap.

## Practical workflow and common mistakes {#structured-practice}

An end-to-end structured-kernel experiment should record:

1. the feasible set and whether it is enumerated, dynamically factorized, or relaxed;
2. the task loss and its decomposition;
3. the joint feature map or joint kernel certificate;
4. the exact prediction and loss-augmented oracle contracts;
5. restricted-solver primal feasibility, duality gap, and stopping tolerance;
6. oracle approximation evidence, if inference is not exact;
7. task risk under the deployed decoder, not an unavailable exact decoder;
8. a componentwise baseline and a candidate-set or beam-width sensitivity curve.

Common mistakes are predictable:

- treating a positive semidefinite joint kernel as a certificate of tractable decoding;
- reporting training with beam search as optimization of the exact structured hinge;
- using the ordinary decoder where loss-augmented inference is required;
- quoting binary or multiclass calibration for a different structured loss;
- normalizing Hamming loss at evaluation but not in margin constraints;
- hiding deterministic tie-breaking, which can change reproducibility;
- evaluating infeasible coordinatewise outputs after training a structured model;
- confusing a small QP gap with a small separation-oracle gap.

## Summary and further reading {#structured-summary}

Structured prediction adds a combinatorial decision layer to kernel learning. A joint feature map produces a valid joint kernel and, for finite output sets, an equivalent operator-valued score RKHS. The structured hinge is convex, upper-bounds task loss under exact decoding, and admits a representer reduction to training feature differences. Strong convexity and bounded feature differences also give an expected stability bound of order \(R^2/(\lambda n)\), but the feature radius can grow with structure size and the task-risk step still requires exact deployment decoding. Its exponentially constrained empirical program can be solved by cutting planes: with finite output sets, exact restricted solves, and exact separation, the algorithm terminates and carries an additive objective certificate. Approximate inference preserves that certificate only when its additive error is itself certified.

The statistical boundary is just as important. Fisher consistency is a statement about a specific surrogate, task loss, score space, and decoder. Results for binary hinge or ordinary multiclass classification [@bartlett2006; @zhang2004] do not automatically establish consistency for arbitrary structured losses. The primary structured-prediction line runs from maximum-margin Markov networks [@taskar2003m3n] and structural SVMs [@tsochantaridis2005structured] through cutting-plane training [@joachims2009struct], approximate inference [@finley2008approx], and calibrated structured surrogates [@osokin2017structured]. Bach connects multicategory surrogates, structured losses, generalization, and computation in one learning-theory treatment [@bach2024learning, Chapter 13]. General RKHS and SVM foundations are available in [@scholkopf2002; @shawe2004; @steinwart2008], while the operator-valued interpretation follows [@micchelli2005vv]. Convex-programming details and duality conventions follow [@boyd2004].

The unresolved pressure is computational. A representer theorem gave finite
coefficients, but the constraint set can still be exponential; a cutting plane exposed
only the constraints it needed, but each separation step still called an inference
algorithm. [[ch:solving-the-svm|The solver chapter]] now studies working sets, KKT
residuals, and decomposition methods directly. [[ch:online-kernel-learning|The online
chapter]] will ask what remains when examples arrive sequentially and the expansion
itself must be budgeted.

## Exercises {#exercises}

1. [warm-up]{.ex-tag} For a finite common output set, prove directly that \(K(x,x')_{y,y'}=k_J\{(x,y),(x',y')\}\) is operator-valued positive semidefinite. Explain why this fact says nothing about the cost of decoding.
2. [computation]{.ex-tag} Enumerate all eight sequences in the worked binary-chain example. Compute the ordinary score, Hamming-augmented score for truth \(010\), ordinary maximizer, augmented maximizer, structured hinge, and decoded task loss.
3. [proof]{.ex-tag} Prove the structured representer theorem, including existence, uniqueness, and the orthogonal-projection argument. State what changes if an output set is infinite.
4. [proof]{.ex-tag} Starting from the constrained margin program, eliminate the slacks and recover the structured-hinge objective. Then derive a valid subgradient when the loss-augmented maximizer is not unique.
5. [computation]{.ex-tag} Suppose a restricted cutting-plane solution has slacks \((0.4,0.7)\), returned oracle values \((0.45,0.72)\), certified additive oracle errors \((0.03,0.05)\), and \(n=2\). If the algorithm stops at returned violation tolerance \(0.05\), give valid full-program slacks and the resulting additive objective certificate.
6. [synthesis]{.ex-tag} Compare joint-feature and operator-valued views of structured scores. Give the map between them, one setting where each notation is preferable, and one assumption needed before an infinite output space can be treated similarly.
7. [proof]{.ex-tag} Prove the Hamming-loss consistency proposition for an unconstrained product output space. Then use the even-parity distribution in the failure witness to show exactly which step fails under a global feasibility constraint.
8. [proof]{.ex-tag} Reproduce the stability theorem. If a chain feature map is the sum of \(T\) local feature vectors of norm at most \(r\), derive one worst-case bound on \(R\). Repeat when the local feature differences are mutually orthogonal, and explain why loss normalization alone does not normalize the score geometry.
9. [challenge]{.ex-tag} Design a cutting-plane experiment with approximate loss-augmented inference. Specify an oracle diagnostic, an optimization certificate, a deployment-decoder sensitivity analysis, and a condition under which you would refuse to claim that the exact structured hinge was optimized.
