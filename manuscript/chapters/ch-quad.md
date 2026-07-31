---
example_code_policy: visible-for-executable
id: ch-quad
slug: kernel-quadrature-and-herding
title: Kernel Quadrature and Herding
part: VII · Distributions as Objects
order: 44
tier: advanced
prerequisites:
  - optimal-transport-and-kernels
objectives:
  - >-
    Prove that RKHS worst-case integration error is an MMD to a weighted node
    measure.
  - >-
    Derive optimal quadrature weights and connect their residual to orthogonal
    projection.
  - >-
    Recover the same weights and error from Gaussian-process conditioning while
    separating posterior uncertainty from frequentist calibration.
  - >-
    Run kernel herding as a Frank-Wolfe method and state exactly when its
    \(O(1/n)\) error rate applies.
  - >-
    Compare greedy, leverage-score, and determinantal node selection under a
    fixed evaluation budget.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-quad.yml
verification_date: null
bibliography:
  - ohagan1991
  - rasmussen2003bmc
  - welling2009herding
  - chen2010herding
  - bach2012herding
  - huszar2012
  - bach2017quadrature
  - briol2019
  - muandet2017
narrative_link_policy: exact
---
# Kernel Quadrature and Herding

<p class="lead">Almost every quantity we care about in probability is an integral: a mean, a variance, a marginal likelihood, a posterior prediction, an expected loss. When the integral has no closed form we fall back on a weighted sum \(\sum_i w_i f(x_i)\), and the only question is where to place the nodes \(x_i\) and how to weight them. Plain Monte Carlo answers by sampling the nodes at random, and pays for that convenience with a slow \(1/\sqrt{n}\) error. This chapter asks whether the RKHS geometry of [[ch:kernel-mean-embeddings|the previous chapters]] can do better. The answer is clean and complete: the worst-case error of a quadrature rule over the RKHS unit ball is exactly the distance between the mean embedding of the target measure and the embedding of the weighted node set, a maximum mean discrepancy. Minimizing that error over the weights recovers Bayesian quadrature and its posterior variance; minimizing it greedily over the nodes is kernel herding, a way to manufacture deterministic super-samples. The same identity ties both to leverage scores, determinantal sampling, and coresets, so that numerical integration becomes one more thing the kernel does for free.</p>

## The integration problem, and why randomness is wasteful {#integration-problem}

Before improving on randomness we should state exactly what randomness is estimating, since every rule in this chapter will be scored against the same target. Fix a probability measure \(P\) on a set \(\mathcal X\) and a function \(f:\mathcal X\to\mathbb R\). We want the integral

$$ P[f] \;=\; \mathbb E_{X\sim P}[f(X)] \;=\; \int_{\mathcal X} f\,dP, $$

and we are allowed to evaluate \(f\) at finitely many nodes \(x_1,\dots,x_n\). A *quadrature rule* is a choice of those nodes together with weights \(w_1,\dots,w_n\), and it estimates the integral by

$$ \widehat{P[f]} \;=\; \sum_{i=1}^n w_i\, f(x_i). $$

The classical answer is Monte Carlo: draw the nodes i.i.d. from \(P\) and weight them uniformly, \(w_i=1/n\). It is unbiased and needs nothing but a sampler, but its root-mean-square error decays like \(1/\sqrt n\), so cutting the error in half costs four times the evaluations. That rate is the price of ignoring \(f\): the nodes are scattered blindly, some landing almost on top of each other and wasting evaluations, others leaving whole regions unprobed. If we knew that \(f\) were smooth, we could place nodes deliberately, spread them out, and reweight them to cancel the redundancy. To make \"smooth\" precise and the optimization tractable, we assume \(f\) lives in the RKHS \(\mathcal H\) of a positive definite kernel \(k\), and we measure a rule by how badly it can do against any such \(f\).

:::: {.definition #def-32-1}
[Definition (quadrature rule and its error)]{.box-title}

A quadrature rule for \(P\) is a finite set of nodes \(x_1,\dots,x_n\in\mathcal X\) with weights \(w=(w_1,\dots,w_n)\in\mathbb R^n\). Its *error on \(f\)* is

$$ \mathrm{err}(f;w) \;=\; P[f] - \sum_{i=1}^n w_i f(x_i). $$

The weights are not required to be nonnegative, nor to sum to one.
::::

Two remarks fix expectations. The weights are free real numbers: releasing them from the probability simplex is exactly what will let an optimal rule outperform any averaging scheme. And a single \(f\) tells us nothing, since a rule can be exact on one function by luck; the honest figure of merit is the largest error the rule can make over a whole class of integrands, which we take to be the RKHS unit ball.

## Worst-case error is a maximum mean discrepancy {#worst-case-error}

The right way to grade a rule is adversarial: how large can \(\mathrm{err}(f;w)\) be as \(f\) ranges over all integrands of unit RKHS norm? This worst case turns out to have a closed form, and it is precisely a distance between mean embeddings, which is why the machinery of [[ch:kernel-mean-embeddings|mean embeddings and MMD]] transfers wholesale to integration.

:::: {.definition #def-32-2}
[Definition (worst-case error)]{.box-title}

The *worst-case error* of the rule \((x_i,w_i)\) over the unit ball of \(\mathcal H\) is

$$ e(w) \;=\; \sup_{\substack{f\in\mathcal H\\ \|f\|_{\mathcal H}\le 1}}\ \Big|\, P[f] - \sum_{i=1}^n w_i f(x_i)\,\Big|. $$
::::

Recall the two facts an RKHS gives us. Evaluation is an inner product with the canonical feature, \(f(x)=\langle f,k(x,\cdot)\rangle_{\mathcal H}\), and integration against \(P\) is an inner product with the mean embedding \(\mu_P=\mathbb E_{X\sim P}[k(X,\cdot)]\), the generalized kernel trick \(P[f]=\langle f,\mu_P\rangle_{\mathcal H}\). Both the integral and the finite sum are therefore linear functionals of \(f\), and their difference is a single inner product. That collapses the supremum by Cauchy-Schwarz.

:::: {.theorem #thm-32-3}
[Theorem (worst-case error is an embedding distance)]{.box-title}

Let \(\mu_P\in\mathcal H\) exist. For any nodes \(x_1,\dots,x_n\) and weights \(w\), the worst-case error equals the RKHS distance between the target embedding and the weighted node embedding,

$$ e(w) \;=\; \Big\| \mu_P - \sum_{i=1}^n w_i\, k(x_i,\cdot) \Big\|_{\mathcal H} \;=\; \mathrm{MMD}\big(P,\, Q_w\big), \qquad Q_w := \sum_{i=1}^n w_i\,\delta_{x_i}, $$

the maximum mean discrepancy between \(P\) and the (signed) weighted node measure \(Q_w\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

Write the two functionals through their representers. For any \(f\in\mathcal H\), the generalized kernel trick gives \(P[f]=\langle f,\mu_P\rangle_{\mathcal H}\), and the reproducing property gives \(f(x_i)=\langle f,k(x_i,\cdot)\rangle_{\mathcal H}\). Hence

$$ P[f]-\sum_i w_i f(x_i) \;=\; \Big\langle f,\ \mu_P-\sum_i w_i\,k(x_i,\cdot)\Big\rangle_{\mathcal H}. $$

Taking the supremum of the absolute value over \(\|f\|_{\mathcal H}\le 1\), Cauchy-Schwarz bounds it by \(\|\mu_P-\sum_i w_i k(x_i,\cdot)\|_{\mathcal H}\), and the bound is attained at the unit vector aligned with \(\mu_P-\sum_i w_i k(x_i,\cdot)\). The right-hand side is \(\|\mu_P-\mu_{Q_w}\|_{\mathcal H}\) because the embedding of \(Q_w=\sum_i w_i\delta_{x_i}\) is \(\mu_{Q_w}=\sum_i w_i k(x_i,\cdot)\), which is the definition of \(\mathrm{MMD}(P,Q_w)\). [\(\square\)]{.qed}
::::

The identity is the hinge of the chapter. Designing a quadrature rule is the same problem as approximating one point of \(\mathcal H\), the target embedding \(\mu_P\), by a weighted combination of the feature vectors \(k(x_i,\cdot)\) sitting at the nodes. Everything reduces to making that approximation good. Expanding the squared norm turns it into ordinary linear algebra in the Gram matrix, which is all any numerical routine ever touches.

:::: {.proposition #prop-32-4}
[Proposition (worst-case error in coordinates)]{.box-title}

Let \(K\in\mathbb R^{n\times n}\) be the Gram matrix \(K_{ij}=k(x_i,x_j)\), let \(z\in\mathbb R^n\) be the *kernel mean vector* \(z_i=\mu_P(x_i)=\mathbb E_{X\sim P}[k(x_i,X)]\), and let \(C=\mathbb E_{X,X'\sim P}[k(X,X')]=\|\mu_P\|_{\mathcal H}^2\). Then

$$ e(w)^2 \;=\; C \;-\; 2\,w^\top z \;+\; w^\top K w. $$

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

Expand the squared norm by bilinearity:

$$ \Big\|\mu_P-\sum_i w_i k(x_i,\cdot)\Big\|_{\mathcal H}^2 = \langle\mu_P,\mu_P\rangle_{\mathcal H} - 2\sum_i w_i\langle\mu_P,k(x_i,\cdot)\rangle_{\mathcal H} + \sum_{i,j} w_i w_j\langle k(x_i,\cdot),k(x_j,\cdot)\rangle_{\mathcal H}. $$

The first term is \(\|\mu_P\|_{\mathcal H}^2=\mathbb E_{X,X'\sim P}[k(X,X')]=C\). In the second, \(\langle\mu_P,k(x_i,\cdot)\rangle_{\mathcal H}=\mu_P(x_i)=z_i\) by the reproducing property. In the third, \(\langle k(x_i,\cdot),k(x_j,\cdot)\rangle_{\mathcal H}=k(x_i,x_j)=K_{ij}\). Collecting the three pieces gives \(C-2w^\top z+w^\top K w\). [\(\square\)]{.qed}
::::

The three ingredients have plain meanings. The constant \(C\) is the self-similarity of \(P\), fixed once the target and kernel are chosen. The vector \(z\) records how well each node \"sees\" the target, since \(z_i=\mu_P(x_i)\) is the target embedding read off at \(x_i\). The Gram matrix \(K\) records how redundant the nodes are with each other. A good rule wants nodes with large \(z\) (near the mass of \(P\)) but small off-diagonal \(K\) (not clustered), and the weights trade these off. Before optimizing, it is worth pinning down exactly what Monte Carlo achieves in this language, so we know the bar to beat.

:::: {.proposition #prop-32-5}
[Proposition (Monte Carlo error)]{.box-title}

If the nodes are drawn i.i.d. from \(P\) and weighted uniformly, \(w_i=1/n\), then

$$ \mathbb E\big[e(w)^2\big] \;=\; \frac{1}{n}\Big(\mathbb E_{X\sim P}[k(X,X)] - \mathbb E_{X,X'\sim P}[k(X,X')]\Big). $$

In particular \(e(w)=O_P(1/\sqrt n)\), and for a normalized kernel with \(k(x,x)=1\) the bracket is \(1-C\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

With uniform weights, \(e(w)^2=C-\frac{2}{n}\sum_i\mu_P(x_i)+\frac{1}{n^2}\sum_{i,j}k(x_i,x_j)\). Take the expectation over the i.i.d. draw. Each \(\mathbb E[\mu_P(x_i)]=\mathbb E_{X,X'\sim P}[k(X,X')]=C\). In the double sum, the \(n\) diagonal terms average to \(\mathbb E[k(X,X)]\) and the \(n(n-1)\) off-diagonal terms, having independent arguments, average to \(C\). Thus

$$ \mathbb E[e(w)^2] = C - 2C + \tfrac1n\mathbb E[k(X,X)] + \tfrac{n-1}{n}C = \tfrac1n\big(\mathbb E[k(X,X)] - C\big). $$

[\(\square\)]{.qed}
::::

So Monte Carlo's expected squared worst-case error falls like \(1/n\), and its worst-case error like \(1/\sqrt n\). Two levers remain untouched in that bound: the weights are frozen at \(1/n\), and the nodes are placed at random. The rest of the chapter pulls each lever. Optimizing the weights on fixed nodes gives Bayesian quadrature; optimizing the nodes greedily gives herding; optimizing both, or choosing the sampling law of the nodes cleverly, connects to leverage scores.

## Optimally weighted quadrature {#optimal-weights}

Hold the nodes fixed and minimize the worst-case error over the weights. Because \(e(w)^2=C-2w^\top z+w^\top Kw\) is a convex quadratic in \(w\) with Hessian \(2K\succeq 0\), the minimizer is found by setting the gradient to zero.

::::: {.proposition #prop-32-6}
[Proposition (optimal weights)]{.box-title}

For fixed nodes with positive definite Gram matrix \(K\), the worst-case error \(e(w)^2\) is minimized by

$$ w^\star \;=\; K^{-1} z, $$

and the minimal value is

$$ e(w^\star)^2 \;=\; C - z^\top K^{-1} z. $$

The optimal weights need not be nonnegative and need not sum to one.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::::

::: {.proof}
[Proof]{.box-title}

Differentiate \(e(w)^2=C-2w^\top z+w^\top K w\) in \(w\): the gradient is \(-2z+2Kw\), which vanishes at \(w^\star=K^{-1}z\). Since the Hessian \(2K\) is positive definite, \(w^\star\) is the unique global minimum. Substituting, \(e(w^\star)^2=C-2(K^{-1}z)^\top z+(K^{-1}z)^\top K(K^{-1}z)=C-2z^\top K^{-1}z+z^\top K^{-1}z=C-z^\top K^{-1}z\). Nothing constrains the sign or the sum of the entries of \(K^{-1}z\). [\(\square\)]{.qed}
:::

The quantity \(z^\top K^{-1}z\) is the squared norm of the projection of \(\mu_P\) onto the span of the node features, so \(e(w^\star)^2=C-z^\top K^{-1}z\) is the squared distance from \(\mu_P\) to that span: the optimal rule is the orthogonal projection of the target embedding onto the nodes, and no reweighting can beat a projection. A small numeric instance shows the projection at work and how much it saves over uniform averaging.

::::: {.example #example-32-1}
[Example (optimal weights beat uniform)]{.box-title}

:::: wex
::: wex-setup
Target \(P=\mathcal N(0,1)\). Kernel \(k(x,x')=\exp\!\big(-(x-x')^2/2\big)\), the Gaussian with lengthscale \(1\). Nodes \(x=(-1,0,1)\). For this Gaussian-kernel, Gaussian-measure pair the two integrals are closed form: the kernel mean is \(\mu_P(x)=\tfrac{1}{\sqrt2}e^{-x^2/4}\) and the self-similarity is \(C=\mathbb E_{X,X'}[k(X,X')]=1/\sqrt3=0.577350\).
:::

1.  [Assemble the Gram matrix.]{.wex-op} With \(k(x_i,x_j)=e^{-(x_i-x_j)^2/2}\),

$$ K=\begin{pmatrix} 1 & 0.606531 & 0.135335\\ 0.606531 & 1 & 0.606531\\ 0.135335 & 0.606531 & 1\end{pmatrix}. $$
2.  [Read off the kernel mean vector.]{.wex-op} Evaluate \(z_i=\mu_P(x_i)=\tfrac{1}{\sqrt2}e^{-x_i^2/4}\) at the three nodes: \(z=(0.550695,\,0.707107,\,0.550695)\). The middle node, sitting on the mode of \(P\), sees the most mass.
3.  [Score the uniform rule.]{.wex-op} With \(w=(\tfrac13,\tfrac13,\tfrac13)\), \(e(w)^2=C-2w^\top z+w^\top Kw=0.004662\), so \(e(w)=0.068281\).
4.  [Solve for the optimal weights.]{.wex-op} \(w^\star=K^{-1}z=(0.304856,\,0.337297,\,0.304856)\). These sum to \(0.947010\), not \(1\): the optimal rule deliberately undershoots the total mass to cancel the overlap between neighbouring nodes.
5.  [Score the optimal rule.]{.wex-op} \(e(w^\star)^2=C-z^\top K^{-1}z=0.003079\), so \(e(w^\star)=0.055490\).

**Reading.** Reweighting the very same three nodes drops the squared worst-case error from \(0.004662\) to \(0.003079\), a fall of \(0.001583\), about \(34\%\), at zero extra evaluations of \(f\). The gain is pure geometry: the uniform rule places \(\mu_{Q_w}\) somewhere in the node span, while \(w^\star\) places it at the foot of the perpendicular from \(\mu_P\).

[[lst:lst-quadrature-optimal-weights]] is short enough to inspect in full. It uses a
linear solve, evaluates both rules from the same quadratic form, and asserts
the claimed improvement rather than relying on rounded output.

```python
import numpy as np

x = np.array([-1.0, 0.0, 1.0])
K = np.exp(-(x[:, None] - x[None, :]) ** 2 / 2)
z = np.exp(-x**2 / 4) / np.sqrt(2)
C = 1 / np.sqrt(3)

uniform = np.full(3, 1 / 3)
optimal = np.linalg.solve(K, z)

def squared_error(w):
    return C - 2 * w @ z + w @ K @ w

e2_uniform = squared_error(uniform)
e2_optimal = squared_error(optimal)
assert np.allclose(optimal, [0.304856, 0.337297, 0.304856], atol=1e-6)
assert np.isclose(e2_uniform, 0.004662, atol=1e-6)
assert np.isclose(e2_optimal, 0.003079, atol=1e-6)
assert e2_optimal < e2_uniform
print(optimal, e2_uniform, e2_optimal)
```
{#lst-quadrature-optimal-weights caption="Verify optimal quadrature weights and their worst-case error"}
::::
:::::

## Bayesian quadrature {#bayesian-quadrature}

The optimal-weight rule can be derived a second way that supplies something the worst-case view does not: a posterior distribution for the integral under an explicit model of \(f\). Instead of treating \(f\) as an unknown element of a norm ball, treat it as a random function with a Gaussian process prior whose covariance is the kernel \(k\). Then the integral \(P[f]\) is a random scalar, and conditioning on the observed evaluations gives a full posterior over it. This is Bayesian quadrature, introduced by O'Hagan (1991) as Bayes-Hermite quadrature and revived for machine learning as Bayesian Monte Carlo by Rasmussen and Ghahramani (2003). The construction belongs to the same circle of ideas as [[ch:gaussian-processes-and-rvm|Gaussian process regression]]: a GP prior, linear observations, a Gaussian posterior.

Put \(f\sim\mathcal{GP}(0,k)\). Integration against \(P\) is a linear functional, so the pair \(\big(f(x_1),\dots,f(x_n),\,P[f]\big)\) is jointly Gaussian. The node evaluations have covariance \(K_{ij}=k(x_i,x_j)\); the cross-covariance between \(P[f]\) and \(f(x_i)\) is \(\mathbb E[P[f]\,f(x_i)]=\int k(x,x_i)\,dP(x)=\mu_P(x_i)=z_i\); and the prior variance of \(P[f]\) is \(\mathbb E[P[f]^2]=\iint k(x,x')\,dP(x)\,dP(x')=C\). Conditioning the Gaussian on the observed values \(f_i=f(x_i)\) yields a Gaussian posterior for the integral with the standard formulas.

:::: {.definition #def-32-7}
[Definition (Bayesian quadrature)]{.box-title}

Under a \(\mathcal{GP}(0,k)\) prior on \(f\), given evaluations \(f=(f_1,\dots,f_n)^\top\) at the nodes, the posterior over \(P[f]\) is Gaussian with mean and variance

$$ \mathbb E\big[P[f]\mid f\big] = z^\top K^{-1} f, \qquad \mathrm{Var}\big[P[f]\mid f\big] = C - z^\top K^{-1} z. $$
::::

Two things deserve to be spelled out. First, the posterior mean is itself a quadrature rule: \(z^\top K^{-1}f=\sum_i w_i^\star f_i\) with weights \(w^\star=K^{-1}z\), the very weights of the optimal rule above. Bayesian quadrature and worst-case-optimal quadrature are the same estimator, reached from a probabilistic and a minimax door. Second, the posterior variance \(C-z^\top K^{-1}z\) is precisely the minimal worst-case squared error \(e(w^\star)^2\). This is an exact algebraic identity, but it is not automatic frequentist calibration: the posterior credible interval has its advertised probability only under the GP model, while the RKHS statement is a deterministic unit-ball bound. Example (optimal weights beat uniform) is simultaneously a Bayesian quadrature calculation: the number \(0.003079\) is at once the squared worst-case error and the model-based posterior variance for those three nodes.

:::: {.algorithm #algo-32-1}
[Algorithm (Bayesian quadrature weights)]{.box-title}

::: algo-io
[Input]{.algo-lab} Kernel \(k\); target \(P\) through its kernel mean \(z_i=\mu_P(x_i)\) and self-similarity \(C=\mathbb E_{P\otimes P}[k]\); nodes \(x_1,\dots,x_n\); evaluations \(f_i=f(x_i)\).

[Output]{.algo-lab} Estimate \(\widehat Z\) of \(P[f]\) and its posterior variance \(V\) (\(=\) squared worst-case error).
:::

1.  Form the Gram matrix \(K_{ij}=k(x_i,x_j)\) and the kernel mean vector \(z=(\mu_P(x_i))_i\).
2.  Solve the linear system \(K\,w=z\) for the weights \(w=K^{-1}z\).
3.  Return the estimate \(\widehat Z=w^\top f=\sum_i w_i f_i\).
4.  Return the variance \(V=C-z^\top w=C-z^\top K^{-1}z\).
::::

Implement step 2 with a Cholesky solve rather than an explicit inverse. If the nodes nearly coincide, \(K\) becomes ill-conditioned and the weights can become large with alternating signs; report \(\operatorname{cond}(K)\), add a declared nugget when needed, and check that the computed variance is nonnegative up to rounding. The dense setup costs \(O(n^3)\), but the same factorization serves both the weights and the variance.

The bottleneck is the kernel mean vector \(z\): it needs the integrals \(\mu_P(x_i)=\int k(x_i,x)\,dP(x)\) in closed form or by a cheap sub-routine. These are tabulated for the pairs that matter in practice, Gaussian kernel against a Gaussian or mixture-of-Gaussians measure, and Bayesian quadrature is used exactly where evaluations of \(f\) are expensive enough that solving an \(n\times n\) system to squeeze each one dry is worthwhile. Briol et al. (2019) survey this probabilistic view of integration, its convergence theory, and the calibration of the posterior variance.

## Kernel herding {#kernel-herding}

Bayesian quadrature optimizes the weights but leaves the nodes to us. The complementary move fixes the weights at uniform, \(1/n\), so that the output is a plain sample we can hand to any downstream averaging code, and instead chooses the nodes to drive down the embedding error. Kernel herding, introduced by Welling (2009) and analyzed as a source of \"super-samples\" by Chen, Welling, and Smola (2010), does this greedily, one node at a time.

The objective after choosing \(t\) nodes is the squared worst-case error of the equally weighted rule,

$$ E_t^2 \;=\; \Big\| \mu_P - \frac1t\sum_{s=1}^t k(x_s,\cdot)\Big\|_{\mathcal H}^2, $$

the squared MMD between \(P\) and the uniform empirical measure on the nodes. Herding adds the node that, given those already chosen, most reduces this quantity. To see which node that is, view the procedure through Frank-Wolfe, the observation of Bach, Lacoste-Julien, and Obozinski (2012). Minimizing \(J(g)=\tfrac12\|g-\mu_P\|_{\mathcal H}^2\) over the marginal polytope \(\mathcal M=\overline{\mathrm{conv}}\{k(x,\cdot):x\in\mathcal X\}\), the conditional-gradient step linearizes \(J\) at the current iterate \(g_{t-1}\) and picks the vertex minimizing \(\langle\nabla J(g_{t-1}),\,k(x,\cdot)\rangle=\langle g_{t-1}-\mu_P,\,k(x,\cdot)\rangle\). By the reproducing property this inner product is \(g_{t-1}(x)-\mu_P(x)\), so the step chooses

$$ x_t \;=\; \arg\max_{x\in\mathcal X}\ \Big[\, \mu_P(x) - g_{t-1}(x)\,\Big], \qquad g_{t-1}(x)=\frac{1}{t-1}\sum_{s=1}^{t-1} k(x_s,x), $$

with \(x_1=\arg\max_x\mu_P(x)\), and the step size \(1/t\) makes the running iterate the uniform average \(g_t=\frac1t\sum_{s\le t}k(x_s,\cdot)\). The rule is intuitive on its own terms. The *acquisition* \(\mu_P(x)-g_{t-1}(x)\) is the residual between the target smoothed density and the density already assembled from earlier nodes. Herding pushes the next node wherever \(P\) still has unrepresented mass, which forces the nodes apart: once a region is covered, its residual drops and the search moves on. This self-avoidance is the source of the improvement over random sampling, which happily revisits covered regions.

:::: {.algorithm #algo-32-2}
[Algorithm (kernel herding, greedy selection)]{.box-title}

::: algo-io
[Input]{.algo-lab} Kernel \(k\); target \(P\) through its kernel mean \(x\mapsto\mu_P(x)=\mathbb E_{X\sim P}[k(x,X)]\); candidate set \(\mathcal X\); budget \(n\).

[Output]{.algo-lab} Nodes \(x_1,\dots,x_n\) carrying equal weights \(1/n\).
:::

1.  Select the mode of the embedding: \(x_1\leftarrow\arg\max_{x\in\mathcal X}\mu_P(x)\).
2.  For \(t=2,\dots,n\): form the running model \(g_{t-1}(x)=\frac{1}{t-1}\sum_{s\lt t}k(x_s,x)\) and select \(x_t\leftarrow\arg\max_{x\in\mathcal X}\big[\mu_P(x)-g_{t-1}(x)\big]\).
3.  Return \(\{x_1,\dots,x_n\}\) with uniform weights \(1/n\).
::::

<figure class="viz" data-widget="herding-greedy">

<figcaption>Each step places the exact argmax of the herding criterion for the standard normal target and unit-bandwidth Gaussian kernel, using the closed-form embedding \(\mu_P(x)=e^{-x^2/4}/\sqrt{2}\). The lower panel tracks the true worst-case integration error: uniform weights against the optimally reweighted same nodes and the exact Monte Carlo expectation, the comparison worked in the examples above.</figcaption>
</figure>

Read the lower panel as a comparison at a fixed node budget, not as a universal rate claim. The greedy nodes reduce uncovered embedding residual; optimal reweighting then projects onto their span. Whether the resulting slope beats Monte Carlo asymptotically depends on the marginal-polytope condition stated next.

The greedy criterion is cheap: each step is one pass over the candidates evaluating a scalar acquisition, with no linear solve. We trace three steps on the same target and kernel as before, so the two examples can be compared directly.

::::: {.example #example-32-2}
[Example (three greedy herding steps)]{.box-title}

:::: wex
::: wex-setup
Target \(P=\mathcal N(0,1)\), kernel \(k(x,x')=e^{-(x-x')^2/2}\), so \(\mu_P(x)=\tfrac{1}{\sqrt2}e^{-x^2/4}\) and \(C=1/\sqrt3\), as in the previous example. Candidate grid \(\mathcal X=\{-1,-0.5,0,0.5,1\}\). Ties in the \(\arg\max\) are broken toward the smaller (leftmost) grid point.
:::

1.  [Pick the first node at the mode.]{.wex-op} The acquisition is \(\mu_P\) itself: over the grid it is \((0.550695,\,0.664265,\,0.707107,\,0.664265,\,0.550695)\), maximal at \(x_1=0\). The one-node error is \(E_1^2=C-2\mu_P(0)+1=0.163137\), so \(E_1=0.403902\).
2.  [Subtract the first node and pick again.]{.wex-op} With \(g_1(x)=k(0,x)\), the acquisition \(\mu_P(x)-k(0,x)\) over the grid is \((-0.055835,\,-0.218232,\,-0.292893,\,-0.218232,\,-0.055835)\). It is most negative at the centre, where the first node already over-covers, and least negative, hence maximal, at the symmetric pair \(\pm1\); the tie breaks to \(x_2=-1\). The two-node error falls to \(E_2^2=0.122814\), \(E_2=0.350448\).
3.  [Subtract both and pick the mirror point.]{.wex-op} With \(g_2(x)=\tfrac12\big(k(0,x)+k(-1,x)\big)\), the acquisition \(\mu_P(x)-g_2(x)\) is \((-0.25257,\,-0.218232,\,-0.096159,\,0.060691,\,0.179762)\), maximal at \(x_3=1\): having placed mass on the left, the residual now peaks on the right. The error drops sharply to \(E_3^2=0.004662\), \(E_3=0.068281\).

**Reading.** The errors decrease monotonically, \(0.163137\to0.122814\to0.004662\), and the three greedy nodes are exactly \(\{0,-1,1\}=\{-1,0,1\}\), the node set of Example (optimal weights beat uniform). Herding chose that set from scratch with equal weights and worst-case squared error \(0.004662\); optimally reweighting it, that is, running Bayesian quadrature on the same nodes, presses the error further to \(0.003079\). For contrast, Proposition (Monte Carlo error) says a random \(3\)-node rule has expected squared error \((1-C)/3=0.140883\), some thirty times larger than herding's deterministic \(0.004662\). Deliberate placement, not luck, is what buys the accuracy.
::::

**Reproduce the calculation.**

```python
import numpy as np

def k(a, b):
    return np.exp(-(a - b) ** 2 / 2.0)

def m(x):                                     # kernel mean of N(0,1)
    return (1.0 / np.sqrt(2.0)) * np.exp(-np.asarray(x, float) ** 2 / 4.0)

C = 1.0 / np.sqrt(3.0)
grid = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
print("candidate grid =", list(grid))

def wce2(nodes):
    nodes = np.asarray(nodes, float)
    t = len(nodes)
    G = k(nodes[:, None], nodes[None, :])
    return C - (2.0 / t) * m(nodes).sum() + G.sum() / t ** 2

chosen = []
for step in range(1, 4):
    if not chosen:
        acq = m(grid)
    else:
        gprev = np.array([np.mean([k(xi, c) for c in chosen]) for xi in grid])
        acq = m(grid) - gprev
    j = int(np.argmax(acq))                   # ties -> smallest index (leftmost)
    print(f"step {step}: acquisition a(x) over grid =", np.round(acq, 6))
    print(f"        argmax index {j} -> x_{step} = {grid[j]}")
    chosen.append(float(grid[j]))
    e2 = wce2(chosen)
    print(f"        nodes so far = {chosen}")
    print(f"        E_{step}^2 = {round(float(e2),6)},  E_{step} = {round(float(np.sqrt(e2)),6)}")

print("final nodes =", chosen)
print("(E_1^2, E_2^2, E_3^2) =",
      tuple(round(float(wce2(chosen[:i])), 6) for i in (1, 2, 3)))

# optimally reweighting the herded nodes = Bayesian quadrature on them (example 1):
nodes = np.asarray(chosen, float)
Kf = k(nodes[:, None], nodes[None, :])
zf = m(nodes)
es2 = C - zf @ np.linalg.solve(Kf, zf)
print(f"optimal-reweight E^2 on herded nodes = {float(es2):.6f}")

# Monte Carlo baseline: expected squared worst-case error of n i.i.d. draws with
# uniform weights is (E_P[k(X,X)] - E_{PxP}[k]) / n = (1 - C)/n here (k(x,x)=1).
for nn in (3,):
    print(f"MC expected E^2 at n={nn} nodes = {round(float((1.0 - C) / nn), 6)}")
```
:::::

### Super-samples and the rate, stated carefully {#herding-convergence}

The name \"super-samples\" comes from a convergence claim that must be stated with its hypotheses attached, because the literature is easy to over-read. Chen, Welling, and Smola (2010) prove that when \(\mu_P\) lies in the *interior* of the marginal polytope \(\mathcal M\), with a ball of radius \(r\gt0\) around it inside \(\mathcal M\), the herding nodes satisfy

$$ E_n \;=\; \Big\|\mu_P-\frac1n\sum_{s=1}^n k(x_s,\cdot)\Big\|_{\mathcal H} \;=\; O\!\Big(\frac1n\Big), $$

a quadratic improvement on Monte Carlo's \(O(1/\sqrt n)\). Since \(n\) herding nodes then match the integration error of \(n^2\) i.i.d. draws, they earn the name. The interior condition holds automatically when \(\mathcal H\) is finite-dimensional, for instance for a polynomial kernel of bounded degree, and there the fast rate is real. Bach, Lacoste-Julien, and Obozinski (2012), by identifying herding with Frank-Wolfe, supply both the guarantee and its ceiling. The generic Frank-Wolfe analysis gives \(E_n^2=O(1/n)\), that is \(E_n=O(1/\sqrt n)\), with no assumption at all, matching Monte Carlo. The faster \(O(1/n)\) on \(E_n\) needs the positive-radius interior condition, and that condition typically *fails* in an infinite-dimensional RKHS such as the Gaussian's, where \(\mu_P\) sits on the boundary of \(\mathcal M\) and no interior ball exists. So the precise statement is: herding never does asymptotically worse than Monte Carlo, is provably a square faster when the embedding is interior (in particular in finite dimension), and in the infinite-dimensional characteristic-kernel case the proven worst-case guarantee is the same \(O(1/\sqrt n)\), though the error is deterministic and empirically often better. Bach, Lacoste-Julien, and Obozinski (2012) also show that fully-corrective and line-search Frank-Wolfe variants, which re-optimize the weights at every step, recover faster rates, which is the bridge to the next point.

### Optimally weighted herding is Bayesian quadrature {#optimally-weighted-herding}

Herding fixes the weights at \(1/n\); Bayesian quadrature optimizes them. Composing the two is irresistible: pick the nodes greedily as herding does, then reweight them optimally as Bayesian quadrature does. Huszár and Duvenaud (2012) observed that this composition is not a heuristic patch but exactly Bayesian quadrature restricted to the herded nodes, and that reweighting can only reduce the worst-case error, since the uniform weights are a feasible point of the very minimization \(w^\star\) solves. The improvement in the worked examples is the general phenomenon in miniature: the herded nodes \(\{-1,0,1\}\) carry squared error \(0.004662\) at uniform weights and \(0.003079\) at the optimal weights \(w^\star=K^{-1}z\). Uniform weights make herding's output a genuine sample, usable wherever an unweighted sample is expected; optimal weights make it a quadrature rule with the smallest error those nodes admit. The choice is exactly the trade between wanting super-samples and wanting a minimal-error estimate of one integral.

## Optimal nodes, leverage scores, and determinantal sampling {#leverage-dpp}

Herding and Bayesian quadrature both take the node budget as given and work hard per node. A different question asks, before any greedy search, from what distribution the nodes should be *drawn* so that even random placement is near-optimal. Bach (2017) answers it by showing that kernel quadrature rules and random-feature expansions are two views of one approximation, and that the right sampling law is dictated by the spectrum of the kernel integral operator rather than by \(P\) alone. Sampling nodes proportionally to the *leverage function*, the analogue for integration of the ridge leverage scores that govern column subsampling in [[ch:large-scale-kernels|large-scale kernel machines]], concentrates evaluations where the operator has energy that the current budget cannot yet resolve.

The governing quantity is the effective dimension, or number of degrees of freedom,

$$ d(\lambda) \;=\; \operatorname{tr}\!\big(\Sigma(\Sigma+\lambda I)^{-1}\big) \;=\; \sum_{j}\frac{\sigma_j}{\sigma_j+\lambda}, $$

where \(\sigma_j\) are the eigenvalues of the integral operator of \(k\) under \(P\), the same spectrum that sets learning rates in [[ch:mercer-and-rates|Mercer's theorem and rates]]. Bach (2017) shows that \(O\big(d(\lambda)\log d(\lambda)\big)\) nodes drawn from the leverage distribution suffice to integrate every unit-norm RKHS function to accuracy \(\sqrt\lambda\), so the node count tracks the intrinsic complexity of the problem, not the ambient dimension. Independent leverage sampling still wastes budget on near-duplicate nodes, and the cure is to make the draw repulsive. A *determinantal point process* assigns a set of nodes probability proportional to \(\det K_S\), the Gram determinant of the selected subset, which is the squared volume they span in feature space and is small whenever two nodes are similar. Determinantal sampling therefore builds the self-avoidance of herding into a single random draw, and it delivers quadrature rules whose error matches the optimal-weight bound in expectation, tying the random and greedy routes back together.

## A note on coresets {#coresets}

Stepping back, every construction in this chapter produces a *coreset*: a small, possibly weighted set of points whose empirical measure stands in for \(P\) on a whole class of queries. Here the query class is integration of RKHS functions, and the approximation guarantee is exactly the worst-case error \(\|\mu_P-\mu_{Q_w}\|_{\mathcal H}=\mathrm{MMD}(P,Q_w)\): a set with small MMD to \(P\) answers every unit-norm RKHS integral to within that MMD, uniformly. Chen, Welling, and Smola's super-samples are an MMD coreset built greedily; Bayesian quadrature reweights a coreset to minimize its MMD; leverage and determinantal sampling draw one at random with the right repulsion. This reading also connects the material to [[ch:optimal-transport-and-kernels|optimal transport]], since both the MMD and the Wasserstein distance are ways to measure how well a discrete measure represents a continuous one, and to the broader compression literature, where MMD-coresets summarize massive datasets for downstream averaging. The review of Muandet, Fukumizu, Sriperumbudur, and Schölkopf (2017) collects these threads, from herding and Bayesian quadrature to the sampling schemes, under the single banner of approximating a mean embedding.

## Summary {#summary}

Numerical integration in an RKHS is embedding approximation. The worst-case error of a quadrature rule over the unit ball equals \(\|\mu_P-\sum_i w_i k(x_i,\cdot)\|_{\mathcal H}\), the MMD between the target and the weighted node measure, which in coordinates is \(C-2w^\top z+w^\top Kw\). Minimizing over the weights gives \(w^\star=K^{-1}z\) with error \(C-z^\top K^{-1}z\); the same estimator is the posterior mean of Bayesian quadrature under a Gaussian process prior, and its irreducible error is the posterior variance, so the RKHS worst case and the Bayesian error bar coincide. Minimizing over the nodes greedily, with weights frozen at \(1/n\), is kernel herding, whose acquisition \(\mu_P(x)-g_{t-1}(x)\) drives nodes into unrepresented mass and yields deterministic super-samples: provably a square faster than Monte Carlo when the embedding is interior to the marginal polytope, and never slower, with the fast rate contingent on that condition. Optimally reweighting herded nodes is Bayesian quadrature again and can only help. Choosing the nodes' sampling law by leverage scores or by a determinantal process makes even random placement track the effective dimension \(d(\lambda)\), and every one of these rules is a coreset for integration, certified by its MMD to \(P\). The next parts put the same embedding machinery to work on [[ch:optimal-transport-and-kernels|transport]] and on [[ch:gaussian-processes-and-rvm|probabilistic prediction]].

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

For **Kernel Quadrature and Herding**, verify that the kernel mean \(z_i=\int k(x_i,x)\,dP(x)\) and the self-similarity \(C\) are known or estimated independently of the expensive integrand. Solve \(Kw=z\) stably and inspect large signed weights, because a small formal worst-case error can coexist with catastrophic numerical sensitivity. Treat the Bayesian variance as model-based uncertainty unless calibration has been checked. Finally, do not quote the \(O(1/n)\) herding error without the interior condition; the generic infinite-dimensional guarantee is weaker.

## Summary and further reading {#summary-and-further-reading}

O'Hagan [@ohagan1991] and Rasmussen and Ghahramani [@rasmussen2003bmc] develop the probabilistic route to quadrature; Welling [@welling2009herding] and Chen et al. [@chen2010herding] supply the deterministic greedy route. Frank-Wolfe analysis explains the rate boundary [@bach2012herding], while optimal reweighting [@huszar2012], leverage sampling [@bach2017quadrature], and probabilistic-numerics calibration [@briol2019] expose different consequences of approximating the same residual mean embedding. The mean-embedding review [@muandet2017] places these routes in a common framework. In practice, choose the route by the desired output: signed optimal weights and a model-based variance for one integral, or uniformly weighted representative points for reuse across many downstream queries. The next decision problem is no longer where to integrate but where to observe; [[ch:bayesian-optimization-and-bandits|Bayesian optimization]] reuses posterior variance to allocate that sequential budget.

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} Show that a quadrature rule is exact on the constant function \(f\equiv 1\) if and only if its weights sum to one. Then, using \(e(w)^2=C-2w^\top z+w^\top Kw\), explain in one sentence why the optimal weights \(w^\star=K^{-1}z\) of Proposition (optimal weights) are generally *not* forced to sum to one, and what property of the kernel would be needed for the constant to lie in the RKHS.
2.  [computation]{.ex-tag} Take \(P=\tfrac12\delta_{a}+\tfrac12\delta_{b}\) with \(a\ne b\), a two-point target, and any normalized kernel with \(k(x,x)=1\). Compute \(C=\mathbb E_{X,X'\sim P}[k(X,X')]\) and the kernel mean \(\mu_P(x)=\tfrac12 k(a,x)+\tfrac12 k(b,x)\). Using the single node \(x_1=a\) with the optimal weight from \(w^\star=K^{-1}z\) (here a scalar), find the weight and the resulting worst-case squared error \(C-z^\top K^{-1}z\). For \(k(a,b)=\rho\), express both in terms of \(\rho\).
3.  [proof]{.ex-tag} Prove Proposition (Monte Carlo error) in the biased-estimator form: for i.i.d. nodes and uniform weights, show directly that \(\mathbb E[e(w)^2]=\tfrac1n(\mathbb E[k(X,X)]-\mathbb E[k(X,X')])\), being careful to separate the \(n\) diagonal and \(n(n-1)\) off-diagonal terms of \(\frac{1}{n^2}\sum_{i,j}k(x_i,x_j)\). Conclude that the worst-case error is \(O_P(1/\sqrt n)\) and that it does not depend on the placement, only on the kernel and \(P\).
4.  [computation]{.ex-tag} Reproduce the first herding step of Example (three greedy herding steps) with a coarser grid \(\mathcal X=\{-2,-1,0,1,2\}\) and the same \(P=\mathcal N(0,1)\), \(k(x,x')=e^{-(x-x')^2/2}\). Compute the step-2 acquisition \(\mu_P(x)-k(0,x)\) on this grid, show that it is maximized at \(x=\pm2\), and verify numerically that adding \(x_2=-2\) with equal weights *raises* the error above \(E_1^2\). Explain why this does not contradict the convergence claim of Section (super-samples and the rate).
    Hint

    ::: hint-body
    Use \(\mu_P(x)=\tfrac{1}{\sqrt2}e^{-x^2/4}\), \(C=1/\sqrt3\). The uniform-weight, fixed step \(1/t\) update is Frank-Wolfe without line search, whose objective can rise on a single step even though the bound \(E_n^2=O(1/n)\) still holds; a line search, that is optimal reweighting, would forbid the increase.
    :::
5.  [proof]{.ex-tag} Show that Bayesian quadrature reproduces optimal weighting. Starting from the jointly Gaussian vector \(\big(f(x_1),\dots,f(x_n),P[f]\big)\) with the covariances \(K_{ij}=k(x_i,x_j)\), \(\operatorname{Cov}(P[f],f(x_i))=z_i\), and \(\operatorname{Var}(P[f])=C\), apply the Gaussian conditioning formula to derive the posterior mean \(z^\top K^{-1}f\) and variance \(C-z^\top K^{-1}z\) of Definition (Bayesian quadrature). Identify the posterior-mean weights with \(w^\star=K^{-1}z\) and the posterior variance with \(e(w^\star)^2\).
    Hint

    ::: hint-body
    For a jointly Gaussian \((u,v)\) the conditional mean is \(\mathbb E[v\mid u]=\Sigma_{vu}\Sigma_{uu}^{-1}u\) and variance \(\Sigma_{vv}-\Sigma_{vu}\Sigma_{uu}^{-1}\Sigma_{uv}\). Here \(u=f\), \(v=P[f]\), \(\Sigma_{uu}=K\), \(\Sigma_{vu}=z^\top\), \(\Sigma_{vv}=C\).
    :::
6.  [proof]{.ex-tag} Derive the herding acquisition from the Frank-Wolfe step. For \(J(g)=\tfrac12\|g-\mu_P\|_{\mathcal H}^2\) on \(\mathcal M=\overline{\mathrm{conv}}\{k(x,\cdot)\}\), compute the gradient \(\nabla J(g_{t-1})=g_{t-1}-\mu_P\), and show that the conditional-gradient vertex \(\arg\min_{x}\langle\nabla J(g_{t-1}),k(x,\cdot)\rangle\) equals \(\arg\max_x[\mu_P(x)-g_{t-1}(x)]\). Then verify that the step \(g_t=(1-\tfrac1t)g_{t-1}+\tfrac1t k(x_t,\cdot)\) unrolls to the uniform average \(g_t=\frac1t\sum_{s\le t}k(x_s,\cdot)\).
    Hint

    ::: hint-body
    Use the reproducing property \(\langle h,k(x,\cdot)\rangle_{\mathcal H}=h(x)\) to turn the inner product into a pointwise value, and note that minimizing \(g_{t-1}(x)-\mu_P(x)\) is maximizing \(\mu_P(x)-g_{t-1}(x)\). Induct on \(t\) for the unrolling, with base case \(g_1=k(x_1,\cdot)\).
    :::
7.  [challenge]{.ex-tag} A determinantal point process on a finite candidate set \(\{x_1,\dots,x_m\}\) with kernel matrix \(K\) draws a subset \(S\) with probability proportional to \(\det K_S\), the principal minor on \(S\). For a size-\(2\) draw, show that \(\Pr(\{i,j\})\propto k(x_i,x_i)k(x_j,x_j)-k(x_i,x_j)^2\), and deduce that for a normalized kernel the pair is chosen with probability proportional to \(1-k(x_i,x_j)^2\). Explain in one or two sentences how this realizes the same self-avoidance that herding produces greedily, and why it therefore tends to place quadrature nodes far apart in feature space.
    Hint

    ::: hint-body
    The \(2\times2\) minor is \(\det\begin{pmatrix}k_{ii}&k_{ij}\\k_{ij}&k_{jj}\end{pmatrix}=k_{ii}k_{jj}-k_{ij}^2\). With \(k_{ii}=k_{jj}=1\) this is \(1-k_{ij}^2\), which vanishes as the two points coincide (\(k_{ij}\to1\)) and is largest when they are dissimilar (\(k_{ij}\to0\)).
    :::
:::
