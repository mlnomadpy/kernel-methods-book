---
narrative_link_policy: exact
example_code_policy: visible-for-executable
id: ch-ksd
slug: kernel-stein-discrepancy
title: Kernel Stein Discrepancy and Stein Methods
part: 'VIII · Conditional, Stein, and Causal Inference'
order: 46
tier: advanced
prerequisites:
  - conditional-mean-embeddings
objectives:
  - >-
    Prove Stein's identity and identify the boundary condition that makes
    integration by parts valid.
  - >-
    Derive the four-term Stein kernel and compute empirical KSD from samples and
    an unnormalized target score.
  - >-
    Separate zero-discrepancy identification from the stronger claim that KSD
    controls weak convergence.
  - Calibrate a KSD goodness-of-fit test with a null-valid wild bootstrap.
  - >-
    Decompose an SVGD step into score attraction and kernel repulsion and
    diagnose bandwidth or step-size failure.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-ksd.yml
verification_date: null
bibliography:
  - stein1972
  - gorham2015
  - liu2016ksd
  - chwialkowski2016ksd
  - gorham2017
  - liu2016svgd
  - oates2017
  - hyvarinen2005
  - kernelbook-code-ch-ksd-ex1
  - kernelbook-code-ch-ksd-ex2
  - kernelbook-code-ch-ksd-ex3
  - muandet2017
---
# Kernel Stein Discrepancy and Stein Methods

<p class="lead">A Bayesian posterior is known through its numerator: likelihood times prior sits in closed form, while the normalizing constant is exactly the high-dimensional integral that makes Bayesian computation hard. An energy-based model hides its normalizer inside a partition function. We still need to ask of such models the most basic question: is this sample consistent with this distribution? Every comparison built so far needs samples from both sides; the maximum mean discrepancy of [[ch:kernel-mean-embeddings|the mean-embedding chapter]] cannot help, and neither can the operator extension of [[ch:conditional-mean-embeddings|the conditional embedding]], because both need draws from \(p\) and we have none. The escape is a classical trick of Charles Stein: build an operator \(\mathcal{A}_p\) that annihilates \(p\) in expectation and depends on \(p\) only through its score \(\nabla \log p\), which is blind to the normalizer. Pairing that operator with an RKHS turns it into the kernel Stein discrepancy, a computable distance between a sample and an unnormalized model. The same operator, run in reverse, becomes Stein variational gradient descent, a deterministic particle sampler that transports points toward \(p\). Both need only the score of \(p\), never its normalizing constant.</p>

## Goodness of fit for a model you can only score {#unnormalized}

Fix a probability density \(p\) on \(\mathbb{R}^d\) and a sample \(x_1,\dots,x_n\) drawn from some unknown \(q\). The goodness-of-fit question is whether the sample could have come from \(p\): is \(q = p\)? When we can sample from \(p\) as well, this is just the two-sample problem, and the MMD two-sample test of [[ch:kernel-hypothesis-testing|the hypothesis-testing chapter]] settles it. The difficulty that defines this chapter is that in the most important cases we cannot sample from \(p\), and we do not even know \(p\) exactly. We know it only in the form

$$ p(x) = \frac{1}{Z}\,\tilde p(x), \qquad Z = \int_{\mathbb{R}^d} \tilde p(x)\, dx, $$

where the unnormalized density \(\tilde p\) is available in closed form but the constant \(Z\) is a high-dimensional integral we cannot evaluate. This is the rule, not the exception. A Bayesian posterior \(p(\theta \mid \mathcal D) \propto p(\mathcal D \mid \theta)\,p(\theta)\) is known through its numerator, while the marginal likelihood \(Z = p(\mathcal D)\) is exactly the intractable integral that makes Bayesian computation hard, a point we return to when SVGD reappears as posterior sampling and which the [[ch:gaussian-processes-and-rvm|Gaussian-process chapter]] meets from the marginal-likelihood side. An energy-based model \(p(x) \propto e^{-E(x)}\) hides its normalizer inside the partition function. In every such case the density is a moving target only up to \(Z\), and any discrepancy that requires \(Z\), or requires sampling from \(p\), is useless.

The one object that survives is the gradient of the log density. Because the logarithm turns the product \(p = \tilde p / Z\) into a difference and \(Z\) is constant in \(x\),

$$ \nabla_x \log p(x) = \nabla_x \log \tilde p(x) - \nabla_x \log Z = \nabla_x \log \tilde p(x). $$

The normalizer contributes nothing to the gradient. So if we can build a discrepancy that reads \(p\) only through \(\nabla \log p\), it will cost us nothing that \(Z\) is unknown. That is precisely what the Stein operator delivers.

## Stein's identity and the Stein operator {#stein-operator}

The gradient of the log density has a name.

:::: {.definition #def-34-1}
[Definition (score function)]{.box-title}

The *score* of a differentiable density \(p\) on \(\mathbb{R}^d\) is the vector field

$$ s_p(x) := \nabla_x \log p(x) = \frac{\nabla_x p(x)}{p(x)} \in \mathbb{R}^d. $$

It is invariant to normalization: if \(p \propto \tilde p\), then \(s_p = \nabla_x \log \tilde p\), so the score can be computed from the unnormalized model alone.
::::

Stein's method, introduced by Stein (1972) for bounding errors in the normal approximation, rests on a single observation: from the score one can manufacture a family of functions whose expectation under \(p\) is exactly zero. Averaging one of them against a sample therefore probes whether the sample really follows \(p\). The construction is an operator.

:::: {.definition #def-34-2}
[Definition (Langevin Stein operator)]{.box-title}

For a density \(p\) with score \(s_p\), the *Stein operator* acts on a smooth vector field \(f = (f_1,\dots,f_d) : \mathbb{R}^d \to \mathbb{R}^d\) by

$$ \mathcal{A}_p f(x) := s_p(x)^\top f(x) + \nabla \cdot f(x) = \sum_{i=1}^d \Big( s_{p,i}(x)\, f_i(x) + \partial_{x_i} f_i(x) \Big), $$

where \(\nabla \cdot f = \sum_i \partial_{x_i} f_i\) is the divergence. In one dimension this is simply \(\mathcal{A}_p f(x) = s_p(x) f(x) + f'(x)\).
::::

The operator earns its keep through the following identity, whose proof is nothing but integration by parts arranged so the score appears.

:::: {.theorem #thm-34-3}
[Theorem (Stein's identity)]{.box-title}

Let \(p\) be a strictly positive \(C^1\) density on \(\mathbb{R}^d\). Let \(f:\mathbb{R}^d\to\mathbb{R}^d\) be \(C^1\), assume \(p\,\nabla\cdot f\) and \(\nabla p^\top f\) are integrable, and require the boundary flux

$$
\lim_{R\to\infty}\int_{\lVert x\rVert=R}p(x)f(x)^\top n_R(x)\,dS(x)=0.
$$

Then \(f\) belongs to the Langevin Stein class of \(p\) and

$$ \mathbb{E}_{X\sim p}\big[\mathcal{A}_p f(X)\big] = 0. $$

On a proper open domain \(\Omega\), replace the limit by zero normal flux on \(\partial\Omega\). Merely assuming differentiability is insufficient: a nonzero boundary term invalidates the identity.

**Assumptions.** Strict positivity and \(C^1\) regularity of \(p\); \(C^1\) regularity of \(f\); integrability of both product-rule terms; and vanishing boundary flux.
**Source.** The integration-by-parts identity is the Langevin specialization used in Chwialkowski, Strathmann, and Gretton (2016), Appendix Lemma 5.1; see also Gorham and Mackey (2017), Proposition 1, for an RKHS Stein-class condition.
**Proof status.** Proved immediately below.
::::

::::: {.proof}
[Proof]{.box-title}

Write the expectation as an integral and use \(s_p(x)\,p(x) = \nabla p(x)\), the defining property of the score:

$$ \mathbb{E}_{X\sim p}\big[\mathcal{A}_p f(X)\big] = \int_{\mathbb{R}^d} \big( s_p(x)^\top f(x) + \nabla \cdot f(x)\big) p(x)\, dx = \int_{\mathbb{R}^d} \big( \nabla p(x)^\top f(x) + p(x)\, \nabla \cdot f(x)\big)\, dx. $$

The integrand is exactly the divergence of the product field \(p f\), since \(\nabla \cdot (p f) = (\nabla p)^\top f + p\, \nabla \cdot f\) by the product rule. By the divergence theorem the integral equals the flux of \(p f\) through the sphere at infinity, which vanishes for \(f\) in the Stein class:

$$ \int_{\mathbb{R}^d} \nabla \cdot \big(p(x) f(x)\big)\, dx = \lim_{R\to\infty} \oint_{\lVert x\rVert = R} p(x) f(x)^\top n(x)\, dS = 0. $$

In one dimension this is the fundamental theorem of calculus applied to \((p f)' = p' f + p f'\), giving \(\int_{\mathbb{R}} (p f)'\, dx = [p f]_{-\infty}^{\infty} = 0\). [\(\square\)]{.qed}
:::::

Read the identity as a test. If our sample truly came from \(p\), then the sample average \(\frac1n \sum_i \mathcal{A}_p f(x_i)\) should hover near zero for every admissible \(f\), because its population value is zero. If the sample came from a different \(q\), there is no reason for the average to vanish, and the amount by which it fails to vanish measures the mismatch. To see exactly what it measures, subtract the identity for \(q\) from the expectation under \(q\). Since \(\mathbb{E}_{X\sim q}[\mathcal{A}_q f(X)] = 0\) by the same theorem applied to \(q\),

$$ \mathbb{E}_{X\sim q}\big[\mathcal{A}_p f(X)\big] = \mathbb{E}_{X\sim q}\big[\mathcal{A}_p f(X) - \mathcal{A}_q f(X)\big] = \mathbb{E}_{X\sim q}\Big[\big(s_p(X) - s_q(X)\big)^\top f(X)\Big], $$

because the divergence terms cancel and only the two scores differ. The Stein operator of \(p\), evaluated under \(q\), sees precisely the gap between the two score fields, weighted by the test field \(f\). It is zero for all \(f\) exactly when \(s_p = s_q\) holds \(q\)-almost everywhere, which for regular densities forces \(p = q\). The score difference \(s_p - s_q\) is the raw material of every discrepancy in this chapter.

## From the identity to a discrepancy {#stein-discrepancy}

A single test field \(f\) can be fooled: a badly chosen \(f\) might average to zero under \(q\) even when \(q \neq p\). The cure is the same one the MMD used for the mean embedding, namely to search over a whole class of test fields and report the worst case. This defines the Stein discrepancy.

:::: {.definition #def-34-4}
[Definition (Stein discrepancy)]{.box-title}

For a class \(\mathcal F\) of vector fields in the Stein class of \(p\), the *Stein discrepancy* between \(q\) and \(p\) is

$$ \mathbb{S}(q,p) := \sup_{f \in \mathcal F}\ \mathbb{E}_{X\sim q}\big[\mathcal{A}_p f(X)\big] = \sup_{f \in \mathcal F}\ \mathbb{E}_{X\sim q}\Big[\big(s_p(X) - s_q(X)\big)^\top f(X)\Big]. $$
::::

Everything now hinges on the class \(\mathcal F\). Too small and the supremum misses real differences; too large and it is neither finite nor computable. The choice made by Gorham and Mackey (2015) that opened the computable theory was a smoothness ball; the choice that closes it in one line, made independently by Liu, Lee, and Jordan (2016) and by Chwialkowski, Strathmann, and Gretton (2016), is the unit ball of a reproducing kernel Hilbert space. As with the MMD, the RKHS ball is the class rich enough to separate distributions yet structured enough that the supremum has a closed form.

## The kernel Stein discrepancy {#ksd}

The RKHS ball earns its place here the same way it did for the MMD: it is rich enough to separate distributions, yet the supremum over it collapses into a norm we can compute. Let \(k\) be a positive definite kernel on \(\mathbb{R}^d\) with RKHS \(\mathcal H\), and let \(\mathcal H^d = \mathcal H \times \cdots \times \mathcal H\) be the vector-valued RKHS of fields \(f = (f_1,\dots,f_d)\) with \(f_i \in \mathcal H\) and squared norm \(\lVert f\rVert_{\mathcal H^d}^2 = \sum_i \lVert f_i\rVert_{\mathcal H}^2\). Taking \(\mathcal F\) to be its unit ball gives the kernel Stein discrepancy.

:::: {.definition #def-34-5}
[Definition (kernel Stein discrepancy)]{.box-title}

The *kernel Stein discrepancy* (KSD) between \(q\) and the target \(p\) is the Stein discrepancy over the unit ball of \(\mathcal H^d\),

$$ \mathrm{KSD}(q,p) := \sup_{\lVert f\rVert_{\mathcal H^d} \le 1}\ \mathbb{E}_{X\sim q}\big[\mathcal{A}_p f(X)\big]. $$
::::

The supremum collapses because, just as expectations of RKHS functions are inner products with the mean embedding, the whole functional \(f \mapsto \mathbb{E}_q[\mathcal{A}_p f]\) is an inner product with a single element of \(\mathcal H^d\). Recall from [[ch:kernels-and-rkhs|the RKHS chapter]] the two reproducing identities \(f_i(x) = \langle f_i, k(x,\cdot)\rangle_{\mathcal H}\) and, for a differentiable kernel, \(\partial_{x_i} f_i(x) = \langle f_i, \partial_{x_i} k(x,\cdot)\rangle_{\mathcal H}\). Substituting them into \(\mathcal{A}_p f\) and pulling the expectation inside the inner product,

$$ \mathbb{E}_{X\sim q}\big[\mathcal{A}_p f(X)\big] = \sum_{i=1}^d \Big\langle f_i,\ \mathbb{E}_{X\sim q}\big[ s_{p,i}(X)\, k(X,\cdot) + \partial_{x_i} k(X,\cdot)\big]\Big\rangle_{\mathcal H} = \big\langle f,\ \xi_{q,p}\big\rangle_{\mathcal H^d}, $$

where the *Stein witness* \(\xi_{q,p} \in \mathcal H^d\) has coordinates \(\xi_{q,p,i} = \mathbb{E}_{X\sim q}[ s_{p,i}(X) k(X,\cdot) + \partial_{x_i} k(X,\cdot)]\). This is the Stein-transformed analogue of the mean embedding: where \(\mu_q = \mathbb{E}_q[k(X,\cdot)]\) averaged the plain feature, \(\xi_{q,p}\) averages the feature after the Stein operator has acted on it. By Cauchy-Schwarz the supremum of \(\langle f, \xi_{q,p}\rangle\) over the unit ball is \(\lVert \xi_{q,p}\rVert_{\mathcal H^d}\), attained at \(f^\star = \xi_{q,p} / \lVert \xi_{q,p}\rVert_{\mathcal H^d}\). So the KSD is the RKHS norm of the Stein witness, exactly as the MMD was the RKHS norm of the difference of embeddings.

### The Stein kernel and its closed form {#stein-kernel}

To turn that norm into something computable, expand \(\lVert \xi_{q,p}\rVert^2 = \langle \xi_{q,p}, \xi_{q,p}\rangle\) as a double expectation. Write the per-point Stein feature as \(\beta_p(x,\cdot) = s_p(x) k(x,\cdot) + \nabla_x k(x,\cdot) \in \mathcal H^d\), so \(\xi_{q,p} = \mathbb{E}_{X\sim q}[\beta_p(X,\cdot)]\). Then \(\lVert \xi_{q,p}\rVert^2 = \mathbb{E}_{X,X'\sim q}\langle \beta_p(X,\cdot), \beta_p(X',\cdot)\rangle_{\mathcal H^d}\), and the inner product of two Stein features is a kernel in its own right.

:::: {.definition #def-34-6}
[Definition (Stein kernel)]{.box-title}

The *Stein kernel* associated to the target \(p\) and base kernel \(k\) is

$$ u_p(x,x') := \big\langle \beta_p(x,\cdot),\ \beta_p(x',\cdot)\big\rangle_{\mathcal H^d} = s_p(x)^\top s_p(x')\, k(x,x') + s_p(x)^\top \nabla_{x'} k(x,x') + \nabla_x k(x,x')^\top s_p(x') + \sum_{i=1}^d \partial_{x_i}\partial_{x'_i} k(x,x'). $$

Equivalently \(u_p(x,x') = \mathcal{A}_p^{x}\,\mathcal{A}_p^{x'}\, k(x,x')\), the Stein operator applied in the \(x\) slot and again in the \(x'\) slot of the kernel.
::::

The four terms come from expanding the inner product coordinatewise with the four reproducing identities: \(\langle k(x,\cdot), k(x',\cdot)\rangle = k(x,x')\) pairs the two score parts, the two cross identities \(\langle k(x,\cdot), \partial_{x'_i} k(x',\cdot)\rangle = \partial_{x'_i} k(x,x')\) pair a score against a derivative, and \(\langle \partial_{x_i} k(x,\cdot), \partial_{x'_i} k(x',\cdot)\rangle = \partial_{x_i}\partial_{x'_i} k(x,x')\) pairs the two derivatives. The decisive feature is written into the formula: \(u_p\) depends on \(p\) only through the score \(s_p\), and on \(k\) through values and derivatives we can differentiate in closed form. Nowhere does the normalizer \(Z\) appear. This is what the whole construction was for.

:::: {.theorem #thm-34-7}
[Theorem (KSD as a double expectation, Liu, Lee, and Jordan, 2016)]{.box-title}

Let \(p\) be as in Theorem 34.3. Let \(k\) be a real \(C^{(1,1)}\) positive-definite kernel whose value and first-derivative sections belong to \(\mathcal H\). Assume the vector Stein feature

$$
\beta_p(X,\cdot)=s_p(X)k(X,\cdot)+\nabla_Xk(X,\cdot)
$$

is Bochner integrable under \(q\), and assume every field in the unit ball of \(\mathcal H^d\) satisfies the boundary and integrability conditions of Theorem 34.3. Then, with \(u_p\) the Stein kernel,

$$ \mathrm{KSD}^2(q,p) = \mathbb{E}_{X,X'\sim q}\big[u_p(X,X')\big], $$

where \(X, X'\) are independent draws from \(q\). The quantity is non-negative.

**Assumptions.** The stated \(C^{(1,1)}\), derivative-representer, Bochner-integrability, and RKHS Stein-class conditions.
**Source.** Liu, Lee, and Jordan (2016), Theorem 3.6; Chwialkowski, Strathmann, and Gretton (2016), Theorem 2.1.
**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

The witness expansion above gives \(\mathrm{KSD}(q,p) = \lVert \xi_{q,p}\rVert_{\mathcal H^d}\), so \(\mathrm{KSD}^2 = \langle \xi_{q,p}, \xi_{q,p}\rangle_{\mathcal H^d}\). Writing each copy of \(\xi_{q,p}\) as an expectation of \(\beta_p\) and using bilinearity and continuity of the inner product to exchange it with the two independent expectations,

$$ \mathrm{KSD}^2(q,p) = \big\langle \mathbb{E}_{X}[\beta_p(X,\cdot)],\ \mathbb{E}_{X'}[\beta_p(X',\cdot)]\big\rangle_{\mathcal H^d} = \mathbb{E}_{X,X'\sim q}\big\langle \beta_p(X,\cdot), \beta_p(X',\cdot)\big\rangle_{\mathcal H^d} = \mathbb{E}_{X,X'\sim q}\big[u_p(X,X')\big]. $$

Non-negativity is immediate, since it is a squared norm. [\(\square\)]{.qed}
::::

The norm identity alone does not make KSD a metric. The implication \(\mathrm{KSD}(q,p)=0\Rightarrow q=p\) additionally needs the Stein identity to hold for the chosen RKHS, enough smoothness and positivity of the densities, integrability of the Stein feature, and a base kernel rich enough to identify the score difference; Chwialkowski, Strathmann, and Gretton (2016) give one such set of conditions. Even identification at zero is weaker than convergence control: a sequence can have KSD tending to zero while its mass escapes to infinity, the failure addressed by the inverse-multiquadric result later in the chapter.

::: {.corollary #cor-34-8}
[Corollary (the Stein kernel is positive definite)]{.box-title}

For fixed \(p\) and a differentiable base kernel \(k\) whose derivative representers belong to \(\mathcal H\), the Stein kernel \(u_p\) is a positive definite kernel on \(\mathbb{R}^d\), because \(u_p(x,x') = \langle \beta_p(x,\cdot), \beta_p(x',\cdot)\rangle_{\mathcal H^d}\) is a Gram inner product of the feature map \(x \mapsto \beta_p(x,\cdot)\). Consequently the empirical V-statistic \(\frac1{n^2}\sum_{i,j}u_p(x_i,x_j)\) is a genuine squared feature norm and is non-negative. The diagonal-free U-statistic is unbiased for the population square but can be negative.

**Assumptions.** The score is finite at the evaluated points, and the kernel value and first-derivative sections used to define \(\beta_p\) belong to \(\mathcal H\).
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
:::

The next result is deliberately only an identification statement. It says what equality to zero means for one fixed pair \((q,p)\); it says nothing about whether small values force a sequence \(q_m\) to approach \(p\).

:::: {.theorem #thm-34-8a}
[Theorem (zero KSD identifies the target under score-separation assumptions)]{.box-title}

Let \(p\) and \(q\) be strictly positive \(C^1\) densities on the same connected open subset of \(\mathbb{R}^d\). Assume the RKHS unit ball is in the Stein classes of both densities, \(\mathbb E_q[u_p(X,X)]\lt\infty\), and

$$
\mathbb E_q\left[\left\lVert\nabla\log\frac{p(X)}{q(X)}\right\rVert^2\right]\lt\infty.
$$

If \(k\) is \(cc\)-universal on the domain, then

$$
\mathrm{KSD}(q,p)=0\quad\Longleftrightarrow\quad q=p.
$$

**Scope.** This is population identification. It does not assert that \(\mathrm{KSD}(q_m,p)\to0\) implies \(q_m\Rightarrow p\), and it does not validate a finite-sample test.

**Assumptions.** Common connected domain; strictly positive \(C^1\) densities; Stein-class boundary conditions; finite Stein-kernel diagonal moment; square-integrable score ratio; and \(cc\)-universality.
**Source.** Chwialkowski, Strathmann, and Gretton (2016), Theorem 2.2 and its proof in Section 3.1.
**Proof status.** The reverse implication follows from Stein's identity. The forward implication is source-backed; the key step uses injectivity of the \(cc\)-universal kernel embedding on the integrable score-ratio field.
::::

::: {.remark}
[Remark (only the score, and a mean-embedding reading)]{.box-title}

The construction parallels the mean embedding term for term, with one change: the feature \(k(x,\cdot)\) is replaced by the Stein feature \(\beta_p(x,\cdot)\), which folds the target into the map before averaging. The KSD is then the MMD-style RKHS norm of a single embedded object \(\xi_{q,p}\), and its double-expectation formula uses \(u_p\) exactly where the MMD used \(k\). But whereas the MMD between \(q\) and \(p\) needed samples from both, the KSD needs samples from \(q\) alone and the score of \(p\) alone. This asymmetry is the entire point: it is what lets us test against a model we can only evaluate up to its normalizer.
:::

### Worked closed form: RBF kernel against a Gaussian target {#rbf-gaussian}

Nothing here is abstract once the base kernel is fixed. Take the RBF kernel \(k(x,x') = \exp\!\big(-\lVert x - x'\rVert^2 / (2h^2)\big)\) in one dimension, whose derivatives are \(\partial_x k = -\frac{x-x'}{h^2} k\), \(\partial_{x'} k = \frac{x-x'}{h^2} k\), and \(\partial_x \partial_{x'} k = \big(\frac{1}{h^2} - \frac{(x-x')^2}{h^4}\big) k\). Take the target to be the Gaussian \(p = \mathcal N(\mu, \sigma^2)\), whose score is the linear field \(s_p(x) = -(x-\mu)/\sigma^2\), a formula that plainly never saw the normalizer \(1/\sqrt{2\pi\sigma^2}\). Substituting into the four-term Stein kernel and collecting, with \(g = x - x'\),

$$ u_p(x,x') = \Big[ s_p(x) s_p(x') + \frac{g}{h^2}\big(s_p(x) - s_p(x')\big) + \frac{1}{h^2} - \frac{g^2}{h^4}\Big]\, \exp\!\Big(-\frac{g^2}{2h^2}\Big). $$

Specializing further to the standard Gaussian \(p = \mathcal N(0,1)\) with \(s_p(x) = -x\) and bandwidth \(h = 1\), the middle terms telescope, \(-x g + x' g = -g^2\), leaving the compact form

$$ u_p(x,x') = \big[\, x x' + 1 - 2(x-x')^2\,\big]\, \exp\!\Big(-\tfrac12 (x-x')^2\Big). $$

This is the concrete object we now compute with. It is symmetric, and on the diagonal it reduces to \(u_p(x,x) = x^2 + 1\), which grows as a point sits farther from the mean \(0\), already hinting that badly placed points inflate the discrepancy.

::::: {.example #example-34-1}
[Example (empirical KSD, a fitting and a misfitting sample)]{.box-title}

:::: wex
::: wex-setup
Target \(p = \mathcal N(0,1)\), score \(s_p(x) = -x\), RBF kernel with \(h = 1\), so \(u_p(x,x') = [x x' + 1 - 2(x-x')^2]\,e^{-(x-x')^2/2}\). Two samples of size \(n = 3\): sample A \(= (-1, 0, 1)\), symmetric about the mean and consistent with \(p\); sample B \(= (2, 3, 4)\), shifted well to the right. The values are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-ksd-ex1].
:::

1.  [Build the Stein kernel matrix for sample A.]{.wex-op} Evaluating \(u_p(x_i,x_j)\) on \((-1,0,1)\),

$$ U^A = \begin{pmatrix} 2.0000 & -0.6065 & -1.0827 \\ -0.6065 & 1.0000 & -0.6065 \\ -1.0827 & -0.6065 & 2.0000 \end{pmatrix}, $$

    with diagonal \((2, 1, 2)\) equal to \(x_i^2 + 1\) as predicted.
2.  [Average it two ways.]{.wex-op} The V-statistic averages all \(n^2 = 9\) entries; the U-statistic averages only the \(6\) off-diagonal ones:

$$ \widehat{\mathrm{KSD}}^2_V(A) = \frac{1}{9}\sum_{i,j} U^A_{ij} = 0.0454, \qquad \widehat{\mathrm{KSD}}^2_U(A) = \frac{1}{6}\sum_{i\neq j} U^A_{ij} = -0.7652. $$
3.  [Build the Stein kernel matrix for sample B.]{.wex-op} On \((2,3,4)\) the score magnitudes \(\lvert s_p\rvert = 2,3,4\) are large, so the entries swell,

$$ U^B = \begin{pmatrix} 5.0000 & 3.0327 & 0.1353 \\ 3.0327 & 10.0000 & 6.6718 \\ 0.1353 & 6.6718 & 17.0000 \end{pmatrix}, $$

    with diagonal \((5, 10, 17)\).
4.  [Average it the same two ways.]{.wex-op}

$$ \widehat{\mathrm{KSD}}^2_V(B) = \frac{51.6797}{9} = 5.7422, \qquad \widehat{\mathrm{KSD}}^2_U(B) = \frac{19.6797}{6} = 3.2799. $$

**Reading.** The manifestly non-negative V-statistic reports \(0.045\) for the fitting sample against \(5.74\) for the shifted one, a hundredfold gap that flags B as inconsistent with \(p\). The unbiased U-statistic sharpens the contrast, dipping slightly negative for A (its target is \(0\), so finite-sample noise puts it just below, exactly as the unbiased MMD does under its null) while staying firmly positive at \(3.28\) for B. Every entry was computed from the score \(s_p(x) = -x\) alone, never from the constant \(1/\sqrt{2\pi}\).
::::

**Reproduce the calculation.**

```python
import numpy as np

h = 1.0

def score(x):                       # s_p for N(0,1)
    return -x

def stein_kernel_matrix(xs):
    """u_p(x_i, x_j) built from RBF derivatives; returns the n x n matrix."""
    xs = np.asarray(xs, dtype=float)
    X = xs[:, None]
    Y = xs[None, :]
    g = X - Y                                   # x - x'
    k = np.exp(-(g ** 2) / (2 * h ** 2))        # k(x,x')
    dk_x = -(g / h ** 2) * k                    # d_x  k
    dk_y = (g / h ** 2) * k                     # d_x' k
    dk_xy = (1.0 / h ** 2 - g ** 2 / h ** 4) * k   # d_x d_x' k
    sx = score(X)
    sy = score(Y)
    u = sx * sy * k + sx * dk_y + sy * dk_x + dk_xy
    # cross-check against the hand-derived closed form for N(0,1), h=1
    closed = (X * Y + 1.0 - 2.0 * g ** 2) * np.exp(-(g ** 2) / 2.0)
    assert np.allclose(u, closed), "four-term u_p disagrees with closed form"
    return u

def ksd_stats(xs):
    U = stein_kernel_matrix(xs)
    n = len(xs)
    V = U.sum() / n ** 2
    off = U.sum() - np.trace(U)
    Ust = off / (n * (n - 1))
    return U, V, Ust

for name, xs in [("A (fits N(0,1))", [-1.0, 0.0, 1.0]),
                 ("B (shifted to +3)", [2.0, 3.0, 4.0])]:
    U, V, Ust = ksd_stats(xs)
    print(f"--- sample {name}: x = {xs} ---")
    print("Stein kernel matrix u_p(x_i,x_j) =")
    print(np.round(U, 4))
    print("diagonal u_p(x_i,x_i) =", np.round(np.diag(U), 4))
    print(f"V-statistic  KSD^2_V = {V:.4f}")
    print(f"U-statistic  KSD^2_U = {Ust:.4f}")
    print()
```
:::::

## Empirical KSD: U-statistics and V-statistics {#empirical-ksd}

The worked example already used the two estimators; here they are in general. Given a sample \(x_1,\dots,x_n \sim q\), replace the double expectation \(\mathbb{E}_{X,X'\sim q}[u_p(X,X')]\) by an average over pairs. Keeping all pairs, including \(i = j\), gives the *V-statistic*

$$ \widehat{\mathrm{KSD}}^2_V = \frac{1}{n^2}\sum_{i=1}^n \sum_{j=1}^n u_p(x_i, x_j), $$

which is biased upward because the diagonal terms \(u_p(x_i,x_i)\) pair a point with itself, but is guaranteed non-negative since \(u_p\) is positive definite. Dropping the diagonal gives the *U-statistic*

$$ \widehat{\mathrm{KSD}}^2_U = \frac{1}{n(n-1)}\sum_{i \neq j} u_p(x_i, x_j), $$

an unbiased estimator of \(\mathrm{KSD}^2(q,p)\), which is the usual choice for testing. Both are simple double sums over the \(n \times n\) Stein kernel matrix, costing \(O(n^2)\) evaluations, each of which needs one score \(s_p(x_i)\) and the kernel derivatives. As with the MMD, the price of unbiasedness is that \(\widehat{\mathrm{KSD}}^2_U\) can fall below zero when \(q = p\).

That sign behavior is the shadow of a deeper fact about the null distribution, and it is the same degeneracy the MMD showed. Under the alternative \(q \neq p\) the statistic concentrates around the positive number \(\mathrm{KSD}^2(q,p)\) and is asymptotically normal. Under the null \(q = p\) the population value is zero, the first-order part of the U-statistic vanishes, and the rescaled statistic has a non-Gaussian limit,

$$ n\,\widehat{\mathrm{KSD}}^2_U \ \xrightarrow{\ d\ }\ \sum_{j=1}^\infty c_j\,(Z_j^2 - 1), \qquad Z_j \overset{\text{i.i.d.}}{\sim} \mathcal N(0,1), $$

an infinite weighted sum of centered \(\chi^2_1\) variables whose weights \(c_j\) are the eigenvalues of the Stein kernel \(u_p\) under \(p\) (Liu, Lee, and Jordan, 2016; Chwialkowski, Strathmann, and Gretton, 2016). A quadratic form in Gaussian noise is not Gaussian, which is why the null cannot be read off a normal table and must be calibrated by resampling.

## A goodness-of-fit test for unnormalized models {#gof-test}

We now turn the estimator into a test of \(H_0 : q = p\) against \(H_1 : q \neq p\). The unknown eigenvalues \(c_j\) put the null quantile out of closed-form reach, so we approximate the null by a *wild bootstrap*, following Chwialkowski, Strathmann, and Gretton (2016). The idea is to perturb the double sum by independent sign flips: multiplying the \((i,j)\) term by \(W_i W_j\) with \(W_i \in \{-1, +1\}\) leaves the diagonal structure intact but randomizes the cross terms so that the resulting statistic mimics a draw from the degenerate null law, because \(\mathbb{E}[W_i W_j] = 0\) for \(i \neq j\). Repeating this many times traces out the null distribution without ever knowing the eigenvalues, precisely the role the permutation test played for the MMD in [[ch:kernel-hypothesis-testing|the testing chapter]].

:::: {.algorithm #algo-34-1}
[Algorithm (kernel Stein goodness-of-fit test)]{.box-title}

::: algo-io
[Input]{.algo-lab} Sample \(x_1,\dots,x_n \sim q\); score function \(s_p = \nabla \log \tilde p\) of the target, computed from the unnormalized model; base kernel \(k\) and its derivatives; bootstrap count \(B\); level \(\alpha\).

[Output]{.algo-lab} Reject or fail to reject \(H_0 : q = p\).
:::

1.  Form the Stein kernel matrix \(U_{ij} = u_p(x_i, x_j)\) from the four-term formula, using \(s_p(x_i)\), \(k(x_i,x_j)\), \(\nabla k\), and \(\nabla\nabla' k\). No normalizer is required.
2.  Compute the source-backed V-statistic \(T = n^{-2}\sum_{i,j} U_{ij}\).
3.  For i.i.d. data, draw independent Rademacher signs \(W_1^{(b)},\dots,W_n^{(b)}\) and set \(T_b=n^{-2}\sum_{i,j}W_i^{(b)}W_j^{(b)}U_{ij}\). For dependent data, use the dependence-matched multiplier process in the theorem below.
4.  Repeat step 3 for \(b = 1,\dots,B\) to build the null sample \(\{T_1,\dots,T_B\}\).
5.  Return the finite-\(B\) corrected value \(\hat p = \dfrac{1+\big\lvert\{\, b : T_b \ge T \,\}\big\rvert}{B+1}\); reject \(H_0\) if \(\hat p \le \alpha\).
::::

:::: {.theorem #thm-34-bootstrap}
[Theorem (scope of wild-bootstrap calibration)]{.box-title}

Let \(X_1,X_2,\ldots\) be a stationary sequence with marginal \(p\). Assume the Stein V-kernel \(u_p\) is Lipschitz, \(\mathbb E[u_p(X_1,X_1)^2]\lt\infty\), and the sequence is \(\tau\)-mixing with

$$
\sum_{t=1}^{\infty}t^2\sqrt{\tau(t)}\lt\infty.
$$

Use the dependent Rademacher multiplier process of Chwialkowski, Strathmann, and Gretton, with its switching probability tending to zero at the prescribed rate. Then the conditional law of the rescaled bootstrap V-statistic consistently approximates the null law of the rescaled KSD V-statistic. Under a fixed alternative with positive population KSD, the data statistic converges to a positive constant while the bootstrap statistic collapses to zero.

For i.i.d. observations, independent Rademacher multipliers are the special case. For dependent observations, replacing the dependent multiplier process by independent signs is not covered.

**Scope.** The result is asymptotic. It is neither an exact finite-sample level guarantee nor a statement about power against every alternative.

**Assumptions.** Stationarity, the displayed \(\tau\)-mixing summability, Lipschitz Stein V-kernel, finite squared diagonal moment, and dependence-matched multipliers.
**Source.** Chwialkowski, Strathmann, and Gretton (2016), Section 3.2, especially Propositions 3.1 and 3.2.
**Proof status.** Source-backed; not reproved here.
::::

The test reuses the Stein kernel matrix \(U\) built once in step 1; every bootstrap replicate is a reweighted quadratic form. A direct implementation costs \(O(n^2)\) to form \(U\) and \(O(Bn^2)\) for \(B\) dense replicates, although batching matrix products improves constants without changing that order. The displayed theorem validates the V-statistic construction. A diagonal-free U-statistic bootstrap is a related but distinct procedure and needs its own degeneracy and resampling argument. Because the test reads \(p\) only through the score, it applies to energy-based models and unnormalized posteriors. Its calibration, identification, and power are separate questions: Theorem 34.8a handles population identification, the theorem above handles null calibration, and neither prevents low power when the observed sample misses the region where the model is wrong.

## Stein variational gradient descent {#svgd}

The Stein operator was built to detect a mismatch between \(q\) and \(p\). Run it in the other direction and it becomes a rule for removing the mismatch: instead of measuring how a sample fails to follow \(p\), move the sample so that it does. This is Stein variational gradient descent (SVGD) of Liu and Wang (2016), a deterministic sampler that carries a set of particles toward \(p\) using nothing but the score.

Represent the current distribution by particles \(x_1,\dots,x_n\), and consider nudging every particle by a small smooth map \(T(x) = x + \epsilon\, \phi(x)\) with \(\phi \in \mathcal H^d\). The pushforward \(q_{[T]}\) of the particle distribution changes its KL divergence to \(p\), and the first-order effect is governed exactly by the Stein operator.

:::: {.proposition #prop-34-9}
[Proposition (steepest descent of the KL is the Stein witness, Liu and Wang, 2016)]{.box-title}

Let \(p\) and \(q\) be strictly positive \(C^1\) densities, and let \(\phi\in\mathcal H^d\) be \(C^1\). Assume \(s_p^\top\phi\) and \(\nabla\cdot\phi\) are \(q\)-integrable, differentiation may pass through the \(q\)-expectation, and \(T_\epsilon(x)=x+\epsilon\phi(x)\) is a \(C^1\) diffeomorphism for all sufficiently small \(\lvert\epsilon\rvert\). Let \(q_{[T_\epsilon]}\) be the pushforward of \(q\). Then

$$ \frac{d}{d\epsilon}\,\mathrm{KL}\big(q_{[T_\epsilon]} \,\|\, p\big)\Big|_{\epsilon = 0} = -\,\mathbb{E}_{X\sim q}\big[\mathcal{A}_p \phi(X)\big]. $$

Maximizing the rate of decrease over the unit ball \(\lVert \phi\rVert_{\mathcal H^d} \le 1\) therefore yields the maximal value \(\mathrm{KSD}(q,p)\), attained at the normalized Stein witness \(\phi^\star = \xi_{q,p} / \lVert \xi_{q,p}\rVert_{\mathcal H^d}\).

This is a derivative at \(\epsilon=0\), not a finite-step descent theorem.

**Assumptions.** Positive \(C^1\) densities, a \(C^1\) perturbation, local diffeomorphism, integrability, and a valid differentiation-under-the-integral interchange.
**Source.** Liu and Wang (2016), Theorem 3.1 and Lemma 3.2.
**Proof status.** Proved immediately below.
::::

::: {.proof}
[Proof (sketch; full derivation in Liu and Wang, 2016)]{.box-title}

By the change-of-variables formula the pushforward density is \(q_{[T]}(y) = q(T^{-1}(y))\,\lvert \det \nabla T^{-1}(y)\rvert\). Differentiating \(\mathrm{KL}(q_{[T]}\|p) = \mathbb{E}_{q_{[T]}}[\log q_{[T]} - \log p]\) at \(\epsilon = 0\), where \(T\) is the identity, the entropy term \(\mathbb{E}[\log q_{[T]}]\) contributes \(-\mathbb{E}_q[\nabla \cdot \phi]\) through the log-determinant, and the cross term contributes \(-\mathbb{E}_q[s_p^\top \phi]\) through the shift of \(\log p\). Adding them gives \(-\mathbb{E}_q[s_p^\top \phi + \nabla \cdot \phi] = -\mathbb{E}_q[\mathcal{A}_p \phi]\). The supremum of this linear functional over the RKHS unit ball is, by the same Cauchy-Schwarz argument that defined the KSD, equal to \(\lVert \xi_{q,p}\rVert_{\mathcal H^d} = \mathrm{KSD}(q,p)\), attained at the aligned unit field. [\(\square\)]{.qed}
:::

The consequence is infinitesimal: at the population distribution \(q\), the steepest first-order direction is the Stein witness. Turning that derivative into an algorithm introduces two new approximations, a finite empirical particle average and a nonzero step size.

:::: {.algorithm #algo-34-2}
[Algorithm (SVGD update, one step)]{.box-title}

::: algo-io
[Input]{.algo-lab} Particles \(x_1,\dots,x_n\); score \(s_p = \nabla\log\tilde p\) of the target; base kernel \(k\); step size \(\epsilon\).

[Output]{.algo-lab} Updated particles \(x_1,\dots,x_n\) that have moved toward \(p\).
:::

1.  For each particle \(x_i\), form the empirical perturbation

$$ \hat\phi(x_i) = \frac{1}{n}\sum_{j=1}^n \Big[\, \underbrace{k(x_j, x_i)\, s_p(x_j)}_{\text{driving force}} + \underbrace{\nabla_{x_j} k(x_j, x_i)}_{\text{repulsion}} \,\Big]. $$
2.  Update every particle in parallel, \(x_i \leftarrow x_i + \epsilon\,\hat\phi(x_i)\).
3.  Repeat from step 1 until a declared iteration budget, a small-update criterion, or a validation diagnostic is reached. Monitor empirical KSD, but do not assume it decreases monotonically for a finite step size.
::::

<figure class="viz" data-widget="svgd-flow">

<figcaption>Eighty particles start in a clump at \(x=-5\) and follow the SVGD update above with the analytic score of the two-mode target \(p=\tfrac12\,\mathcal N(-2,0.6^2)+\tfrac12\,\mathcal N(2,0.8^2)\). Attraction moves particles toward high density while the kernel derivative spreads them; the KSD readout is a diagnostic, not a guaranteed Lyapunov curve for a finite step. Changing bandwidth shows the central trade-off: local repulsion may strand particles in one mode, while global repulsion can move the cloud coherently but blur local structure.</figcaption>
</figure>

The two terms in the update have a clean mechanical reading. The driving force \(k(x_j, x_i) s_p(x_j)\) averages the target score, while \(\nabla_{x_j} k(x_j,x_i)\) separates nearby particles. Both use only \(s_p\). What follows from Proposition 34.9 is only population, infinitesimal KL descent. A finite particle system replaces the expectation by a dependent empirical average; a finite step adds truncation error; repeated steps add optimization and discretization error. The cited 2016 derivation does not prove that a fixed finite particle cloud converges to \(p\), that every finite step decreases KL or empirical KSD, or that the resulting points have Monte Carlo error bars. Those require additional mean-field, propagation-of-chaos, smoothness, and step-schedule analyses not supplied by the algorithm box.

::::: {.example #example-34-2}
[Example (one SVGD step toward a standard Gaussian)]{.box-title}

:::: wex
::: wex-setup
Target \(p = \mathcal N(0,1)\), score \(s_p(x) = -x\), RBF kernel with \(h = 1\) so \(\nabla_{x_j} k(x_j,x_i) = (x_i - x_j)\,k(x_j,x_i)\), step size \(\epsilon = 0.1\). Three particles start at \(x = (1, 2, 3)\), all to the right of the mean. The values are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-ksd-ex2].
:::

1.  [Form the kernel matrix.]{.wex-op} With \(K_{ji} = k(x_j, x_i) = e^{-(x_j - x_i)^2/2}\),

$$ K = \begin{pmatrix} 1.0000 & 0.6065 & 0.1353 \\ 0.6065 & 1.0000 & 0.6065 \\ 0.1353 & 0.6065 & 1.0000 \end{pmatrix}. $$
2.  [Average the driving force.]{.wex-op} The score-weighted term \(D_i = \tfrac1n\sum_j K_{ji}\,s_p(x_j)\) is negative for every particle, pulling all three toward the mean \(0\):

$$ D = (-0.8730,\ -1.4754,\ -1.4495). $$
3.  [Average the repulsion.]{.wex-op} The term \(R_i = \tfrac1n\sum_j (x_i - x_j) K_{ji}\) is antisymmetric: it pushes the left particle further left, the right particle further right, and leaves the middle one alone,

$$ R = (-0.2924,\ 0.0000,\ +0.2924). $$
4.  [Combine into the step and move.]{.wex-op} With \(\hat\phi(x_i) = D_i + R_i\),

$$ \hat\phi = (-1.1654,\ -1.4754,\ -1.1571), \qquad x^{\text{new}} = x + 0.1\,\hat\phi = (0.8835,\ 1.8525,\ 2.8843). $$

**Reading.** Every particle moved left toward the target's center: the cloud mean drops from \(2.0000\) to \(1.8734\) in a single step. The driving term did the transport while the repulsion split the outer two apart by equal and opposite amounts. This one-step calculation does not prove that repeated finite-particle updates converge to \(\mathcal N(0,1)\). As with the test, the update touched \(p\) only through the score \(s_p(x) = -x\).
::::

**Reproduce the calculation.**

```python
import numpy as np

h = 1.0
eps = 0.1
x = np.array([1.0, 2.0, 3.0])       # initial particles
n = len(x)

def score(v):
    return -v

# kernel matrix K[j,i] = k(x_j, x_i) and gradient of first argument
diff = x[:, None] - x[None, :]                  # diff[j,i] = x_j - x_i
K = np.exp(-(diff ** 2) / (2 * h ** 2))         # K[j,i] = k(x_j, x_i)
gradK = (x[None, :] - x[:, None]) / h ** 2 * K  # (x_i - x_j) k, indexed [j,i]

# driving and repulsion parts of phi(x_i), averaging over j
drive = (K * score(x)[:, None]).sum(axis=0) / n     # D_i
repel = gradK.sum(axis=0) / n                        # R_i
phi = drive + repel                                  # phi(x_i)
x_new = x + eps * phi

print("kernel matrix K[j,i] = k(x_j, x_i) =")
print(np.round(K, 4))
print("driving   D_i =", np.round(drive, 4))
print("repulsion R_i =", np.round(repel, 4))
print("phi(x_i)      =", np.round(phi, 4))
print("x (old)       =", np.round(x, 4))
print("x (new)       =", np.round(x_new, 4))
print(f"mean before   = {x.mean():.4f}")
print(f"mean after    = {x_new.mean():.4f}")
```
:::::

## Choosing the kernel, and a bridge to quadrature {#kernel-choice}

Identification of one distribution at KSD zero does not control a sequence. The missing property is tightness: if probability mass can escape every compact set while the discrepancy shrinks, there is no weak convergence to the target.

The first failure is already visible at finite sample size. A rapidly decaying kernel can make widely separated tail particles nearly invisible to one another, so dropping the diagonal leaves almost no cross interaction even as every point moves farther from the target. This estimator witness is distinct from the stronger population non-tightness result stated below.

<figure class="viz" data-figure="ksd-escape-failure" data-alt="For five increasingly separated tail points, the absolute Gaussian-Stein U-statistic falls toward zero while the V-statistic grows and the nearest support point moves away from the target."><figcaption>A diagonal-free Gaussian-Stein statistic can miss isolated tail points: off-diagonal interactions vanish exponentially, so the U-statistic approaches zero, while the diagonal-retaining V-statistic grows with the score magnitude. This is a finite-sample estimator witness. The theorem below addresses the stronger question of population KSD convergence and tightness.</figcaption></figure>

:::: {.theorem #thm-34-convergence}
[Theorem (IMQ KSD determines weak convergence in the distantly dissipative regime)]{.box-title}

Let \(P\) have a strictly positive \(C^1\) density on \(\mathbb R^d\) with globally Lipschitz score \(b=\nabla\log p\). Assume \(P\) is *distantly dissipative*: with

$$
\kappa(r)=\inf_{\lVert x-y\rVert=r}
\left\{-\,2\frac{\langle b(x)-b(y),x-y\rangle}{\lVert x-y\rVert^2}\right\},
$$

we have \(\liminf_{r\to\infty}\kappa(r)\gt0\). Let

$$
k(x,y)=\big(c^2+\lVert x-y\rVert^2\big)^\beta,
\qquad c\gt0,\quad \beta\in(-1,0).
$$

For any sequence of probability measures \((Q_m)\), if the associated Langevin KSD satisfies \(\mathrm{KSD}(Q_m,P)\to0\), then \(Q_m\Rightarrow P\).

**Scope.** The target assumptions and the slowly decaying IMQ tail do the work. The result does not cover arbitrary targets, arbitrary kernels, or a finite empirical U-statistic.

**Assumptions.** Strictly positive \(C^1\) target density, globally Lipschitz score, distant dissipativity, \(c\gt0\), and IMQ exponent \(\beta\in(-1,0)\).
**Source.** Gorham and Mackey (2017), Definition 4 and Theorem 8; proof in their Supplement, Section H.
**Proof status.** Source-backed; not reproved here.
::::

The contrast is sharp. Gorham and Mackey (2017), Theorem 6, construct non-tight sequences in dimension \(d\ge3\) whose KSD tends to zero for kernels with sufficiently light tails, including the Gaussian kernel, even when \(P=\mathcal N(0,I_d)\). The theorem is existential and asymptotic. The deterministic calculation below isolates the numerical mechanism without pretending to reproduce that full sequence.

::::: {.example #example-34-3}
[Example (a Gaussian-KSD tail failure witness)]{.box-title}

:::: wex
::: wex-setup
Take \(P=\mathcal N(0,1)\), the unit-bandwidth Gaussian base kernel, and the five-point sample \(x=(10,20,30,40,50)\). All points are deep in the target's tails and separated by ten standard deviations. The values are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-ksd-ex3].
:::

1.  [Inspect off-diagonal visibility.]{.wex-op} The largest absolute off-diagonal Stein-kernel entry is below \(4.0\times10^{-19}\). Gaussian kernel decay has effectively disconnected every pair.
2.  [Compute the diagonal-free statistic.]{.wex-op} The U-statistic has absolute value below \(7.0\times10^{-20}\), numerically indistinguishable from zero despite the catastrophic sample.
3.  [Restore the diagonal.]{.wex-op} The V-statistic is \(220.2\), because \(u_p(x_i,x_i)=x_i^2+1\). The full empirical KSD has not failed; the diagonal-free finite-sample surrogate has.

**Reading.** This witness separates two hazards. At finite \(n\), dropping the diagonal can hide isolated tail points when a light-tailed kernel kills every cross interaction. Asymptotically in \(d\ge3\), Gorham and Mackey's Theorem 6 shows a stronger population failure: even the KSD itself can miss a carefully constructed non-tight sequence. Neither issue is repaired by zero-identification alone.
::::

**Reproduce the calculation.**

```python
import math


def stein_kernel(x: float, y: float) -> float:
    gap = x - y
    return (x * y + 1.0 - 2.0 * gap * gap) * math.exp(-0.5 * gap * gap)


points = (10.0, 20.0, 30.0, 40.0, 50.0)
n = len(points)
matrix = [[stein_kernel(x, y) for y in points] for x in points]

max_off_diagonal = max(
    abs(matrix[i][j])
    for i in range(n)
    for j in range(n)
    if i != j
)
u_statistic = sum(
    matrix[i][j]
    for i in range(n)
    for j in range(n)
    if i != j
) / (n * (n - 1))
v_statistic = sum(sum(row) for row in matrix) / (n * n)
expected_diagonal_v = sum(x * x + 1.0 for x in points) / (n * n)

assert max_off_diagonal < 4.0e-19
assert abs(u_statistic) < 7.0e-20
assert math.isclose(v_statistic, 220.2, rel_tol=0.0, abs_tol=1.0e-12)
assert math.isclose(v_statistic, expected_diagonal_v, rel_tol=0.0, abs_tol=1.0e-12)
assert all(math.isfinite(value) for row in matrix for value in row)

print(f"max |off diagonal| = {max_off_diagonal:.6e}")
print(f"U-statistic = {u_statistic:.6e}")
print(f"V-statistic = {v_statistic:.12f}")
```
:::::

The same Stein-kernel machinery has a second life beyond testing and sampling. If \(u_p\) is a kernel whose functions integrate to a known value against \(p\), one can build control variates that cancel the variance of a Monte Carlo estimator: the control functionals of Oates, Girolami, and Chopin (2017) fit an RKHS function in the range of the Stein operator to an integrand, subtract it, and are left with an estimator of far lower variance, sometimes converging faster than the Monte Carlo rate. It is the same object read a third way. The Stein operator that annihilates \(p\) in expectation gives, at once, a discrepancy for testing, a transport direction for sampling, and a zero-mean correction for integration.

::: {.remark}
[Connections and further reading]{.box-title}

The score that drives the KSD is the same quantity estimated by score matching (Hyvärinen, 2005), and the population KSD with a translation-invariant kernel is a smoothed version of the Fisher divergence \(\mathbb{E}_q\lVert s_q - s_p\rVert^2\); the Stein operator is what lets us evaluate it without ever forming \(s_q\). Like the mean-embedding chapter, this material postdates the standard textbooks, so the primary sources are articles: the kernel Stein discrepancy and its goodness-of-fit test are due independently to Liu, Lee, and Jordan (2016) and Chwialkowski, Strathmann, and Gretton (2016), building on the computable Stein discrepancies of Gorham and Mackey (2015); the convergence-control theory and the case for the inverse multiquadric kernel are from Gorham and Mackey (2017); Stein variational gradient descent is from Liu and Wang (2016); and the control-functional variance reduction is from Oates, Girolami, and Chopin (2017). The survey of Muandet, Fukumizu, Sriperumbudur, and Schölkopf (2017) places all of this within the wider theory of kernel mean embeddings.
:::

Stein witnesses can reject an unnormalized model and SVGD can move particles toward it, but neither operation identifies how an intervention would change a system: observational fit is not causal identification. [[ch:causal-inference-with-kernels|Causal Inference with Kernels]] inherits the operator and conditional-embedding machinery while adding the assumptions, such as independence structure, instruments, or structural equations, needed to turn distributional evidence into a causal claim.

## Summary {#summary}

When a target density \(p\) is known only up to its normalizer, the score \(s_p = \nabla \log p\) is the one handle that survives, because the constant \(Z\) disappears under the log-gradient. Stein's identity turns the score into an operator \(\mathcal{A}_p f = s_p^\top f + \nabla \cdot f\) whose expectation under \(p\) is zero, so its expectation under any other \(q\) measures the score gap \(s_p - s_q\). Maximizing that expectation over the unit ball of a vector RKHS gives the kernel Stein discrepancy, the RKHS norm of a Stein-transformed witness, with the closed-form double expectation \(\mathrm{KSD}^2(q,p) = \mathbb{E}_{X,X'\sim q}[u_p(X,X')]\) of the Stein kernel \(u_p\), itself positive definite and built from four terms in \(s_p\) and the derivatives of the base kernel. Identification at zero and control of weak convergence require additional boundary, tail, target, and kernel assumptions. Estimated by its U-statistic or V-statistic over a sample, the KSD drives a goodness-of-fit test for unnormalized models, calibrated by a wild bootstrap because the null is a degenerate \(\chi^2\) mixture. The same witness is the infinitesimal steepest-descent direction of the KL divergence, so SVGD uses a score-driven attraction and kernel repulsion to construct a deterministic particle approximation. Testing, approximate inference, and control-variate integration all read the target only through its score, but each requires its own validity diagnostics.

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

For **Kernel Stein Discrepancy and Stein Methods**, verify the integration-by-parts boundary condition before invoking Stein's identity and check that score and kernel derivatives are integrable under the sample law. Distinguish three claims: the KSD is computable, zero KSD identifies the target under additional assumptions, and vanishing KSD controls convergence only for suitable target-kernel pairs. For testing, use a bootstrap justified by the dependence structure. For SVGD, report kernel bandwidth, step schedule, particle count, stopping rule, and mode coverage; a falling training KSD alone is not evidence of calibrated posterior samples.

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} Let \(p(x) = \tilde p(x)/Z\) be a density on \(\mathbb{R}^d\) with unknown normalizer \(Z\). Show that the score \(s_p = \nabla_x \log p\) equals \(\nabla_x \log \tilde p\), so that it can be computed from the unnormalized model alone. Then explain in one sentence why the maximum mean discrepancy of the mean-embedding chapter cannot be used to test \(q = p\) in this setting, while the kernel Stein discrepancy can.
2.  [computation]{.ex-tag} Work on \(\mathbb{R}\) with target \(p = \mathcal N(0,1)\), score \(s_p(x) = -x\), and RBF kernel \(k(x,x') = e^{-(x-x')^2/2}\), so the Stein kernel is \(u_p(x,x') = [x x' + 1 - 2(x-x')^2]\,e^{-(x-x')^2/2}\). For the two-point sample \(x_1 = 0,\ x_2 = 1\), write out the \(2\times 2\) Stein kernel matrix entry by entry, then evaluate the V-statistic \(\tfrac{1}{4}\sum_{i,j} u_p(x_i,x_j)\) and the U-statistic \(\tfrac{1}{2}\sum_{i\neq j} u_p(x_i,x_j)\).
    Hint

    ::: hint-body
    The diagonal entries are \(u_p(0,0) = 0^2 + 1 = 1\) and \(u_p(1,1) = 1^2 + 1 = 2\). The off-diagonal entry is \(u_p(0,1) = [0 + 1 - 2]\,e^{-1/2} = -e^{-1/2} = -0.6065\), which is both cross terms. The V-statistic is \((1 + 2 - 2\cdot 0.6065)/4\); the U-statistic averages only the two off-diagonal entries.
    :::
3.  [proof]{.ex-tag} Prove Stein's identity in one dimension: for \(p\) differentiable with score \(s_p = (\log p)' = p'/p\) and \(f\) in the Stein class of \(p\), show \(\mathbb{E}_{X\sim p}[s_p(X) f(X) + f'(X)] = 0\). State exactly where the boundary condition \(\lim_{\lvert x\rvert \to \infty} p(x) f(x) = 0\) enters.
    Hint

    ::: hint-body
    Multiply through by \(p\) inside the integral: \(s_p f\, p = p' f\), so the integrand becomes \(p' f + p f' = (p f)'\). The integral of a derivative is the boundary difference \([p f]_{-\infty}^{\infty}\), which the Stein-class condition sends to zero.
    :::
4.  [proof]{.ex-tag} Starting from \(\mathrm{KSD}(q,p) = \sup_{\lVert f\rVert_{\mathcal H^d}\le 1} \mathbb{E}_{X\sim q}[\mathcal{A}_p f(X)]\), use the reproducing identities \(f_i(x) = \langle f_i, k(x,\cdot)\rangle\) and \(\partial_{x_i} f_i(x) = \langle f_i, \partial_{x_i} k(x,\cdot)\rangle\) to write the functional as \(\langle f, \xi_{q,p}\rangle_{\mathcal H^d}\), identify the witness \(\xi_{q,p}\), and conclude by Cauchy-Schwarz that \(\mathrm{KSD}(q,p) = \lVert \xi_{q,p}\rVert_{\mathcal H^d}\). Name the two properties of the RKHS you use.
    Hint

    ::: hint-body
    You need the reproducing property (to convert function and derivative evaluations into inner products with \(k(x,\cdot)\) and \(\partial_{x_i} k(x,\cdot)\)) and continuity of the inner product (to pull \(\mathbb{E}_q\) inside it). Cauchy-Schwarz on \(\langle f, \xi_{q,p}\rangle\) over \(\lVert f\rVert \le 1\) gives the supremum \(\lVert \xi_{q,p}\rVert\), attained at \(f = \xi_{q,p}/\lVert \xi_{q,p}\rVert\).
    :::
5.  [computation]{.ex-tag} Derive the four-term Stein kernel for a general one-dimensional target and RBF kernel with bandwidth \(h\), obtaining

$$ u_p(x,x') = \Big[ s_p(x) s_p(x') + \tfrac{g}{h^2}\big(s_p(x) - s_p(x')\big) + \tfrac{1}{h^2} - \tfrac{g^2}{h^4}\Big] e^{-g^2/(2h^2)}, \quad g = x - x'. $$

    Then set \(p = \mathcal N(0,1)\) and \(h = 1\) and confirm it collapses to \([x x' + 1 - 2(x-x')^2]\,e^{-(x-x')^2/2}\).
    Hint

    ::: hint-body
    Use \(\partial_x k = -\tfrac{g}{h^2} k\), \(\partial_{x'} k = \tfrac{g}{h^2} k\), and \(\partial_x\partial_{x'} k = (\tfrac{1}{h^2} - \tfrac{g^2}{h^4})k\). For the collapse, \(s_p(x) = -x\) makes the middle term \(\tfrac{g}{1}(-x + x') = -g^2\), which combines with \(-g^2\) from the last term to give \(-2g^2\).
    :::
6.  [proof]{.ex-tag} Show that the Stein kernel \(u_p\) is positive definite for any base kernel \(k\) and target \(p\), so that both the V-statistic and any single Gram matrix \([u_p(x_i,x_j)]\) are non-negative in the quadratic-form sense. Deduce that \(\widehat{\mathrm{KSD}}^2_V \ge 0\) always, and explain why \(\widehat{\mathrm{KSD}}^2_U\) can nonetheless be negative.
    Hint

    ::: hint-body
    Write \(u_p(x,x') = \langle \beta_p(x,\cdot), \beta_p(x',\cdot)\rangle_{\mathcal H^d}\) with the Stein feature \(\beta_p(x,\cdot) = s_p(x) k(x,\cdot) + \nabla_x k(x,\cdot)\); any Gram matrix of a feature map is positive semidefinite, so \(\sum_{ij} c_i c_j u_p(x_i,x_j) = \lVert \sum_i c_i \beta_p(x_i,\cdot)\rVert^2 \ge 0\). The V-statistic takes \(c_i = 1/n\); the U-statistic removes the diagonal, which is not a quadratic form and can go negative.
    :::
7.  [exploration]{.ex-tag} In the SVGD update \(\hat\phi(x_i) = \tfrac1n\sum_j [k(x_j,x_i) s_p(x_j) + \nabla_{x_j} k(x_j,x_i)]\), consider the two limits of the base kernel bandwidth \(h\). Argue what happens to the particle cloud as \(h \to \infty\) (the kernel is nearly constant) and as \(h \to 0\) (the kernel is nearly a spike), and say which term, driving or repulsion, dominates in each limit and why a moderate \(h\) is needed for the particles to approximate \(p\).
    Hint

    ::: hint-body
    As \(h \to \infty\), \(k \to 1\) and \(\nabla k \to 0\): the repulsion vanishes and every particle follows the same averaged score, so the cloud collapses toward a mode. As \(h \to 0\), \(k(x_j,x_i) \to 0\) for \(i \neq j\): each particle sees only itself, follows its own score to the nearest mode, and the particles no longer coordinate to fill out \(p\).
    :::
8.  [challenge]{.ex-tag} Two parts on the goodness-of-fit test. (a) Explain why, under \(H_0 : q = p\), the rescaled statistic \(n\,\widehat{\mathrm{KSD}}^2_U\) has the non-Gaussian limit \(\sum_j c_j (Z_j^2 - 1)\) rather than a normal law, and connect this to the degeneracy that makes the first-order term of the U-statistic vanish. (b) Suppose the wild bootstrap of the test is run with \(B = 500\) sign-flip replicates and \(12\) of the bootstrap statistics \(T_b\) exceed the observed \(T\). Compute the finite-\(B\) corrected \(p\)-value and state the decision at level \(\alpha = 0.05\). Then argue that a target whose score is wrong only on a region the sample never visits could still yield a small statistic, and say what that means for the power of the test.
    Hint

    ::: hint-body
    For (a), under the null the leading linear part of the U-statistic has mean zero and vanishes, leaving a quadratic form in the Gaussian fluctuations of the sample, whose law is a weighted sum of centered \(\chi^2_1\) variables with the eigenvalues of \(u_p\) under \(p\) as weights. For (b), \(\hat p=(12+1)/(500+1)=13/501\approx0.02595\le0.05\), so reject \(H_0\); the test only senses the score difference \(s_p-s_q\) where \(q\) places mass, so a mismatch confined to an unvisited region contributes little and the test loses power there.
    :::
9.  [synthesis]{.ex-tag} Separate the logical content of the following four statements: (i) \(\mathrm{KSD}(q,p)=0\Rightarrow q=p\); (ii) \(\mathrm{KSD}(q_m,p)\to0\Rightarrow q_m\Rightarrow p\); (iii) a wild-bootstrap KSD test has asymptotically valid null level; and (iv) one empirical SVGD update decreases the KL divergence. For each, name the additional assumptions required or explain why the chapter does not claim it.
:::
