---
id: ch-distreg
slug: distribution-regression
title: Distribution Regression and Functional Data
part: 'VIII · Conditional, Stein, and Causal Inference'
order: 48
tier: advanced
prerequisites:
  - causal-inference-with-kernels
objectives:
  - >-
    Formalize the two-stage sampling model and keep bag-count error separate
    from within-bag error.
  - >-
    Compute empirical mean embeddings and Gaussian-on-MMD similarities directly
    from bag samples.
  - >-
    Prove positive definiteness of linear and Gaussian kernels on embedded
    distributions.
  - >-
    Implement two-stage kernel ridge regression with stable solves and an
    explicit \(O(\ell^2N^2)\) Gram-build budget.
  - >-
    State the source, capacity, and bag-growth assumptions behind consistency
    rather than treating characteristicness as a rate theorem.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-distreg.yml
verification_date: null
bibliography:
  - szabo2016dr
  - muandet2012smm
  - christmann2010
  - ramsay2005fda
  - poczos2013
  - muandet2017
  - haussler1999
---
# Distribution Regression and Functional Data

<p class="lead">A pathologist looks at a slide holding thousands of cells, and the diagnosis belongs to the slide, not to any one cell. An astronomer wants the redshift of a source seen only as a cloud of photons; a social scientist wants a county's election result from a survey of its households. In each case the label is attached to a whole bag of samples, so the training example is really a probability distribution, observed only through a finite draw. Every chapter so far has fed the kernel a single point, and none of those methods accepts a distribution as an argument. The remedy is two stages: summarize each bag by its [[ch:kernel-mean-embeddings|kernel mean embedding]], one point in an RKHS, then run [[ch:kernel-ridge-and-friends|kernel ridge regression]] on those points through a kernel built on embeddings, a kernel of kernels. Two questions make the construction delicate. Is the kernel of kernels positive definite? We prove it is. How does the error split between having few bags and having small bags? The consistency theorem of Szabó, Sriperumbudur, Póczos, and Gretton (2016) settles the account.</p>

## When the input is a distribution {#input-is-a-distribution}

Ordinary supervised learning pairs a feature vector with a label. But a great deal of data does not arrive as one vector per example. A pathologist sees not a single cell but a slide of thousands, and the diagnosis labels the slide. An astronomer measures a cloud of photons and wants the redshift of the source that emitted them. A social scientist has a survey of many households in a county and wants to predict the county's median income or election result. In each case the natural input is a bag of points, and the label is attached to the whole bag, not to its members. Treating the bag as an unordered set, the object it represents is a probability distribution: the empirical distribution of its samples, standing in for the population the samples came from.

So we want to learn a function \(f\) whose argument is a probability distribution \(P\) and whose value \(f(P)\) is a real label. The obstacle is that a distribution is an infinite-dimensional object and we never observe it exactly. We observe a finite bag \(\{x^1,\dots,x^N\}\) of samples from \(P\), and even the number of bags we are given is finite. Two approximations are therefore baked in from the start, and keeping them separate is the whole art of the subject. The first idea, and the one that organizes the chapter, is to compress each distribution into a single point of an RKHS using its mean embedding, turning the exotic input space of distributions into an ordinary Hilbert space on which we already know how to regress.

### The two-stage sampled model {#two-stage-model}

We make the sampling structure precise, because the guarantees depend on it. There is a meta-distribution, a distribution over distributions, from which the inputs are drawn.

:::: {.definition #def-36-1}
[Definition (distribution regression problem)]{.box-title}

Let \(\mathcal X\) be a space with a positive definite base kernel \(k\), and let \(\mathcal P\) be a set of Borel probability measures on \(\mathcal X\). A *meta-distribution* \(\mathfrak M\) generates labelled distributions \((P,y)\) with \(P\in\mathcal P\) and \(y\in\mathbb R\). We do not observe the pairs \((P_i,y_i)\); instead we observe, for \(i=1,\dots,\ell\), a bag of i.i.d. samples

$$ X_i=\{x_i^1,\dots,x_i^{N_i}\},\qquad x_i^a\overset{\text{i.i.d.}}{\sim}P_i, $$

together with the label \(y_i\). The goal is to learn a predictor \(f:\mathcal P\to\mathbb R\) minimizing the risk \(\mathcal R(f)=\mathbb E_{(P,y)\sim\mathfrak M}\big[(f(P)-y)^2\big]\).
::::

The two integers \(\ell\) and \(N_i\) name the two error sources. The number of bags \(\ell\) is the ordinary sample size of a regression: with few bags we cannot pin down \(f\), and this error vanishes only as \(\ell\to\infty\). The bag size \(N_i\) is new: even with infinitely many bags, if each bag holds a handful of points we see each input distribution only blurrily, and this error vanishes only as \(N_i\to\infty\). A one-stage regression has only the first source. Distribution regression has both, and its theory is the accounting that keeps them apart. From here on we write \(N\) for a common bag size to keep the notation light.

## Representing each input by its mean embedding {#embedding-the-inputs}

The first stage borrows the central object of [[ch:kernel-mean-embeddings|the mean-embedding chapter]] wholesale. Recall that a base kernel \(k\) on \(\mathcal X\), with RKHS \(\mathcal H\) and feature map \(x\mapsto K_x=k(x,\cdot)\), embeds a distribution \(P\) as the average feature

$$ \mu_P=\mathbb E_{X\sim P}[K_X]\in\mathcal H,\qquad \mu_P(\cdot)=\mathbb E_{X\sim P}[k(X,\cdot)]. $$

This single vector stores the expectation under \(P\) of every RKHS function, through the generalized kernel trick \(\mathbb E_{X\sim P}[g(X)]=\langle g,\mu_P\rangle_{\mathcal H}\), and when \(k\) is characteristic the map \(P\mapsto\mu_P\) is injective, so no information about \(P\) is lost. The embedding is exactly the linearizing device we need: it replaces each unwieldy input distribution \(P_i\) by a point \(\mu_{P_i}\) in the fixed Hilbert space \(\mathcal H\).

We do not have \(P_i\), only its bag, so we use the empirical embedding, the mean embedding of the empirical distribution \(\widehat P_i=\frac1N\sum_a\delta_{x_i^a}\),

$$ \widehat\mu_i:=\mu_{\widehat P_i}=\frac1N\sum_{a=1}^N K_{x_i^a}=\frac1N\sum_{a=1}^N k(x_i^a,\cdot)\in\mathcal H. $$

This is nothing but the average of one feature per sample, the same kernel-smoothed fingerprint of the bag met before. Stage one, then, turns each bag into a point \(\widehat\mu_i\in\mathcal H\), and the gap \(\|\widehat\mu_i-\mu_{P_i}\|_{\mathcal H}\) between this point and the true one is precisely the finite-bag error, which we will bound in a moment.

<figure class="viz" data-figure="bags-to-embeddings" data-alt="Three bags contain different one-dimensional sample clouds. Each bag is transformed into a smooth kernel mean curve, and distances between those curves become a small second-stage Gram matrix."><figcaption>Distribution regression has two geometries: points within a bag are averaged into one mean embedding, then whole bags are compared through distances between those embeddings. Larger bags stabilize the curves; more labelled bags stabilize the regression across curves.</figcaption></figure>

## A kernel on distributions: the kernel of kernels {#kernel-on-distributions}

Now that every input is a point \(\mu_{P}\) in \(\mathcal H\), stage two is a kernel method on those points, and it needs a kernel between them. This is a kernel whose two arguments are themselves distributions, evaluated through their embeddings, a kernel of kernels. Two choices span the range.

::::: {.definition #def-36-2}
[Definition (embedding kernels on distributions)]{.box-title}

Let \(\mu_P\in\mathcal H\) be the mean embedding of \(P\) under the base kernel \(k\). The *linear embedding kernel* is the RKHS inner product of embeddings,

$$ \mathcal K_{\mathrm{lin}}(P,Q)=\langle\mu_P,\mu_Q\rangle_{\mathcal H}=\mathbb E_{X\sim P,\,Y\sim Q}[k(X,Y)]. $$

For a scale \(\gamma\gt 0\), the *Gaussian-on-MMD kernel* is a Gaussian in the RKHS distance between embeddings,

$$ \mathcal K_\gamma(P,Q)=\exp\!\Big(-\frac{\|\mu_P-\mu_Q\|_{\mathcal H}^2}{2\gamma^2}\Big)=\exp\!\Big(-\frac{\mathrm{MMD}^2(P,Q)}{2\gamma^2}\Big), $$

since \(\|\mu_P-\mu_Q\|_{\mathcal H}\) is exactly the maximum mean discrepancy.
:::::

The linear kernel measures raw overlap of the smoothed densities and is a perfectly good similarity, but it can only ever see \(P\) through the single vector \(\mu_P\), so as a second-stage kernel it fits functions that are linear in the embedding. The Gaussian-on-MMD kernel restores nonlinearity: it turns the MMD, our metric on distributions, into a bump-shaped similarity, large when two distributions nearly coincide and decaying as they separate. Both are genuine kernels, which needs an argument, because positive definiteness on the strange domain of distributions is not automatic.

::: {.proposition #prop-36-3}
[Proposition (the embedding kernels are positive definite)]{.box-title}

For any base kernel \(k\) and any \(\gamma\gt 0\), both \(\mathcal K_{\mathrm{lin}}\) and \(\mathcal K_\gamma\) are positive definite kernels on \(\mathcal P\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::

::::: {.proof}
[Proof]{.box-title}

The map \(\Phi:P\mapsto\mu_P\) is a feature map from \(\mathcal P\) into the Hilbert space \(\mathcal H\). For \(\mathcal K_{\mathrm{lin}}\), given distributions \(P_1,\dots,P_m\) and scalars \(c_1,\dots,c_m\),

$$ \sum_{a,b}c_a c_b\,\langle\mu_{P_a},\mu_{P_b}\rangle_{\mathcal H}=\Big\|\sum_a c_a\,\mu_{P_a}\Big\|_{\mathcal H}^2\ \ge\ 0, $$

so \(\mathcal K_{\mathrm{lin}}\) is positive definite, being an inner product of feature vectors. For \(\mathcal K_\gamma\), expand the squared distance as \(\|\mu_P-\mu_Q\|^2=\|\mu_P\|^2+\|\mu_Q\|^2-2\langle\mu_P,\mu_Q\rangle\) and factor the exponential,

$$ \mathcal K_\gamma(P,Q)=\underbrace{e^{-\|\mu_P\|^2/2\gamma^2}\,e^{-\|\mu_Q\|^2/2\gamma^2}}_{g(P)\,g(Q)}\ \cdot\ \exp\!\Big(\tfrac{1}{\gamma^2}\langle\mu_P,\mu_Q\rangle\Big). $$

The first factor is \(g(P)g(Q)\) with \(g(P)=e^{-\|\mu_P\|^2/2\gamma^2}\), a rank-one and hence positive definite kernel. The second is the exponential of \(\tfrac1{\gamma^2}\mathcal K_{\mathrm{lin}}\), which is positive definite because \(\exp(t\,\mathcal K_{\mathrm{lin}})=\sum_{n\ge 0}\tfrac{t^n}{n!}\mathcal K_{\mathrm{lin}}^{\,n}\) is a limit of nonnegative combinations of Schur powers of the positive definite kernel \(\mathcal K_{\mathrm{lin}}\), each power positive definite by the Schur product theorem. A pointwise product of positive definite kernels is positive definite, so the product of the two factors is \(\mathcal K_\gamma\). Christmann and Steinwart (2010) and Muandet, Fukumizu, Sriperumbudur, and Schölkopf (2017) develop these embedding kernels in full. [\(\square\)]{.qed}
:::::

The only property of \(\mathcal H\) the proof used is that it is an inner product space, so the argument is the exact replay of \"the Gaussian kernel on \(\mathbb R^d\) is positive definite,\" now run in the embedding space. In practice we never form \(\mu_P\) explicitly; every entry of the second-stage Gram matrix is computed from bag samples through the base kernel, which is the content of the next procedure.

:::: {.algorithm #algo-36-1}
[Algorithm (kernel-on-embeddings Gram matrix from bags)]{.box-title}

::: algo-io
[Input]{.algo-lab} Bags \(X_1,\dots,X_\ell\) with \(X_i=\{x_i^1,\dots,x_i^{N}\}\); base kernel \(k\); scale \(\gamma\).

[Output]{.algo-lab} Positive semidefinite matrix \(\mathbf K\in\mathbb R^{\ell\times\ell}\), \(\mathbf K_{ij}=\mathcal K_\gamma(P_i,P_j)\).
:::

1.  For each pair \((i,j)\), form the mean cross base-kernel \(B_{ij}=\dfrac{1}{N^2}\displaystyle\sum_{a,b}k(x_i^a,x_j^b)=\langle\widehat\mu_i,\widehat\mu_j\rangle_{\mathcal H}\).
2.  Form the squared empirical MMD \(D_{ij}=B_{ii}+B_{jj}-2B_{ij}=\|\widehat\mu_i-\widehat\mu_j\|_{\mathcal H}^2\).
3.  Set \(\mathbf K_{ij}=\exp\!\big(-D_{ij}/(2\gamma^2)\big)\); for the linear kernel instead set \(\mathbf K_{ij}=B_{ij}\).
::::

The construction is nothing more than: average the base kernel across every pair of bags to get inner products, read off distances, and pass them through a Gaussian. The worked example carries three tiny bags plus a fourth all the way to the matrix.

The exact Gram build costs \(O(\ell^2N^2)\) base-kernel evaluations for \(\ell\) equal-size bags of \(N\) points, before the \(O(\ell^3)\) ridge factorization. Compute each symmetric block once, stream blocks when memory is tight, and reuse the resulting matrix across ridge values. For large bags, explicit finite-dimensional base features can reduce each bag to one averaged feature vector, after which cross-bag inner products cost only the feature dimension; that approximation changes stage one and must be validated against held-out bag pairs.

:::::: {.example #example-36-1}
[Example (empirical embeddings and the Gaussian-on-MMD Gram matrix)]{.box-title}

::::: wex
:::: wex-setup
Four bags of three points each on \(\mathbb R\), with clean generating means \(0,1,2,3\) and varied spreads:

$$ X_1=\{-0.4,0.1,0.3\},\ \ X_2=\{0.7,1.0,1.3\},\ \ X_3=\{1.5,2.1,2.4\},\ \ X_4=\{2.7,3.0,3.3\}. $$

Base kernel Gaussian with bandwidth \(\sigma=1\), that is \(k(a,b)=e^{-(a-b)^2/2}\); embedding scale \(\gamma=1\).
::::

1.  [Average the base kernel over each pair of bags.]{.wex-op} The matrix of embedding inner products \(B_{ij}=\langle\widehat\mu_i,\widehat\mu_j\rangle\), each an average of \(9\) base-kernel values, is

$$ B=\begin{pmatrix}0.9212&0.6065&0.1783&0.0179\\0.6065&0.9438&0.5993&0.1588\\0.1783&0.5993&0.8796&0.6063\\0.0179&0.1588&0.6063&0.9438\end{pmatrix}, $$

    with self-similarities \(\operatorname{diag}(B)=(0.9212,0.9438,0.8796,0.9438)\) measuring how tightly each bag is concentrated.
2.  [Read off the squared MMD distances.]{.wex-op} Using \(D_{ij}=B_{ii}+B_{jj}-2B_{ij}\),

$$ D=\begin{pmatrix}0&0.6520&1.4442&1.8293\\0.6520&0&0.6249&1.5701\\1.4442&0.6249&0&0.6109\\1.8293&1.5701&0.6109&0\end{pmatrix}. $$

    Adjacent bags sit at squared distance about \(0.62\), the far corners at \(1.83\): the MMD grows with the gap between the generating means.
3.  [Pass distances through the Gaussian.]{.wex-op} With \(\mathbf K_{ij}=e^{-D_{ij}/2}\),

$$ \mathbf K=\begin{pmatrix}1&0.7218&0.4857&0.4006\\0.7218&1&0.7317&0.4561\\0.4857&0.7317&1&0.7368\\0.4006&0.4561&0.7368&1\end{pmatrix}. $$
4.  [Confirm it is a valid kernel matrix.]{.wex-op} The eigenvalues of \(\mathbf K\) are \((0.1262,0.3549,0.7411,2.7778)\), all strictly positive, so \(\mathbf K\) is positive definite, exactly as the proposition promised.

**Reading.** Stage one has turned four bags of numbers into four points whose pairwise similarities form the matrix \(\mathbf K\). The kernel behaves geometrically: it is \(1\) on the diagonal, decays monotonically as the generating means pull apart (from \(0.7218\) for neighbours to \(0.4006\) for the extremes), and its positive spectrum certifies that any downstream kernel machine may use it. Every number here came from bag samples through the base kernel alone; the embeddings \(\widehat\mu_i\) were never formed.
:::::
::::::

## The two-stage estimator {#two-stage-estimator}

With a positive definite kernel on distributions in hand, the second stage is ordinary kernel ridge regression, run on the embedded inputs. This is the whole estimator: embed, then ridge.

:::: {.algorithm #algo-36-2}
[Algorithm (two-stage distribution-regression estimator)]{.box-title}

::: algo-io
[Input]{.algo-lab} Labelled bags \((X_i,y_i)_{i=1}^\ell\); base kernel \(k\); embedding kernel \(\mathcal K_\gamma\); ridge \(\lambda\gt 0\).

[Output]{.algo-lab} Predictor \(\widehat f(P_\ast)\) for a new bag \(X_\ast\).
:::

1.  [Stage 1.]{.algo-lab} Represent each bag by its empirical embedding \(\widehat\mu_i=\frac1N\sum_a k(x_i^a,\cdot)\), implicitly through \(k\).
2.  Build the \(\ell\times\ell\) Gram matrix \(\mathbf K_{ij}=\mathcal K_\gamma(P_i,P_j)\) by the previous algorithm.
3.  [Stage 2.]{.algo-lab} Solve the ridge system \(\alpha=(\mathbf K+\ell\lambda I)^{-1}\mathbf y\) for the dual weights.
4.  For a new bag \(X_\ast\), form the vector \((\mathbf k_\ast)_i=\mathcal K_\gamma(P_i,P_\ast)\) and return \(\widehat f(P_\ast)=\mathbf k_\ast^\top\alpha\).
::::

The predictor is the familiar weighted sum of kernel similarities, \(\widehat f(P_\ast)=\sum_i\alpha_i\,\mathcal K_\gamma(P_i,P_\ast)\), only with the similarity now computed between bags. Nothing about the ridge solve knows it is acting on distributions; the distributional character lives entirely in how \(\mathbf K\) and \(\mathbf k_\ast\) were built. We complete the running example by attaching labels and predicting a held-out bag.

::::: {.example #example-36-2}
[Example (predicting a label for a held-out bag)]{.box-title}

:::: wex
::: wex-setup
The four bags of the previous example, with the label of each bag its generating mean. Train on bags \(1,2,4\) with labels \(\mathbf y=(0,1,3)\); hold out bag \(3\), whose true label is \(2.0\). Reuse \(\mathcal K_\gamma\) with \(\gamma=1\), and ridge \(\lambda=0.05\), so \(\ell\lambda=3(0.05)=0.15\).
:::

1.  [Extract the training Gram and the test column.]{.wex-op} From \(\mathbf K\), the training block on \(\{1,2,4\}\) and the similarities of the held-out bag to the training bags are

$$ \mathbf K_{\mathrm{tr}}=\begin{pmatrix}1&0.7218&0.4006\\0.7218&1&0.4561\\0.4006&0.4561&1\end{pmatrix},\qquad \mathbf k_\ast=\begin{pmatrix}0.4857\\0.7317\\0.7368\end{pmatrix}. $$
2.  [Solve the ridge system.]{.wex-op} With \(\ell\lambda=0.15\), \(\alpha=(\mathbf K_{\mathrm{tr}}+0.15\,I)^{-1}\mathbf y=(-1.3679,\ 0.5987,\ 2.8478)\).
3.  [Predict the held-out label.]{.wex-op} \(\widehat f(P_3)=\mathbf k_\ast^\top\alpha=1.8719\), against the true value \(2.0\).
4.  [Compare to the naive baseline.]{.wex-op} Predicting the constant mean of the training labels, \((0+1+3)/3=1.3333\), has absolute error \(0.6667\); the two-stage prediction has absolute error \(0.1281\), five times smaller.

**Reading.** From three labelled bags the estimator predicts the held-out label as \(1.87\), close to the truth \(2.0\). It falls a little short for two reasons that are exactly the two error sources of the model: the ridge penalty shrinks the fit toward zero (a stage-two effect that shrinks with more bags and smaller \(\lambda\)), and each bag has only three points, so the embedding \(\widehat\mu_3\) is a noisy stand-in for \(\mu_{P_3}\) (a stage-one effect that shrinks as the bags grow). The naive baseline, blind to which distribution each bag came from, does far worse.
::::
:::::

## Consistency and the two error sources {#consistency}

There are two sample sizes and therefore two learning curves: the number \(N\) of labelled distributions and the number \(m\) of observations used to represent each distribution. Improving only one eventually hits the floor imposed by the other.

<figure class="viz" data-figure="distribution-regression-two-stage-curve" data-alt="Learning curves separate error from the number of labelled bags and error from samples per bag. A second panel shows the optimal allocation under fixed total sampling budgets."><figcaption>Distribution regression is a two-stage sampling problem. More labelled bags reduce the second-stage regression error but cannot remove a poor empirical embedding; larger bags stabilize each embedding but leave too few labelled tasks under a fixed budget. The right panel shows the resulting interior allocation rather than a universal preference for either axis.</figcaption></figure>

The example's two apologies, ridge bias and tiny bags, are the whole story of the theory. We now make them quantitative. The clean way is to compare the estimator we can compute, which uses the empirical embeddings \(\widehat\mu_i\), against an oracle estimator that uses the true embeddings \(\mu_{P_i}\), and then against the target itself. Two short lemmas supply the pieces.

:::: {.lemma #lem-36-4}
[Lemma (finite-bag embedding error)]{.box-title}

If the base kernel is bounded, \(\sup_x k(x,x)\le\kappa\), then for a bag of \(N\) i.i.d. draws from \(P\),

$$ \mathbb E\,\big\|\widehat\mu_P-\mu_P\big\|_{\mathcal H}^2\ \le\ \frac{\kappa}{N}. $$

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

Write \(\widehat\mu_P-\mu_P=\frac1N\sum_{a=1}^N(K_{x^a}-\mu_P)\), a mean of \(N\) i.i.d. terms with mean zero in \(\mathcal H\). By independence the variance of the mean is the single-term variance over \(N\),

$$ \mathbb E\,\|\widehat\mu_P-\mu_P\|^2=\frac1N\,\mathbb E\,\|K_X-\mu_P\|^2=\frac1N\big(\mathbb E[k(X,X)]-\|\mu_P\|^2\big)\le\frac{\mathbb E[k(X,X)]}{N}\le\frac{\kappa}{N}. $$

Hence \(\|\widehat\mu_P-\mu_P\|_{\mathcal H}=O_P(N^{-1/2})\). [\(\square\)]{.qed}
::::

So the stage-one error per bag decays like \(N^{-1/2}\). It only helps if the second-stage kernel does not amplify it, which is what the next lemma guarantees: the Gaussian-on-MMD feature map is Lipschitz in the embedding, so a small perturbation of \(\mu_P\) moves the second-stage feature by a controlled amount.

:::: {.lemma #lem-36-5}
[Lemma (the distribution kernel is Lipschitz in the embedding)]{.box-title}

Let \(\Psi(P)=\mathcal K_\gamma(P,\cdot)\in\mathcal H_{\mathcal K}\) be the canonical feature of the Gaussian-on-MMD kernel. Then

$$ \big\|\Psi(P)-\Psi(P')\big\|_{\mathcal H_{\mathcal K}}\ \le\ \frac1\gamma\,\big\|\mu_P-\mu_{P'}\big\|_{\mathcal H}. $$

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

::::: {.proof}
[Proof]{.box-title}

Since \(\mathcal K_\gamma(P,P)=1\), the reproducing property gives

$$ \|\Psi(P)-\Psi(P')\|_{\mathcal H_{\mathcal K}}^2=\mathcal K_\gamma(P,P)+\mathcal K_\gamma(P',P')-2\mathcal K_\gamma(P,P')=2\Big(1-e^{-\|\mu_P-\mu_{P'}\|^2/2\gamma^2}\Big). $$

Apply the elementary bound \(1-e^{-u}\le u\) for \(u\ge 0\) with \(u=\|\mu_P-\mu_{P'}\|^2/(2\gamma^2)\):

$$ 2\Big(1-e^{-\|\mu_P-\mu_{P'}\|^2/2\gamma^2}\Big)\le 2\cdot\frac{\|\mu_P-\mu_{P'}\|^2}{2\gamma^2}=\frac{\|\mu_P-\mu_{P'}\|^2}{\gamma^2}. $$

Taking square roots gives the claim. [\(\square\)]{.qed}
:::::

The two lemmas chain: replacing each true embedding by its bag estimate perturbs each second-stage feature by at most \(\frac1\gamma\|\widehat\mu_i-\mu_{P_i}\|=O_P(N^{-1/2})\), and kernel ridge regression is stable to such feature perturbations. This is the mechanism behind the risk decomposition.

:::: {.proposition #prop-36-6}
[Proposition (two-stage risk decomposition)]{.box-title}

Let \(\widehat f\) be the two-stage estimator built from the empirical embeddings, \(\widetilde f\) the oracle ridge estimator built from the same \(\ell\) bags but with the true embeddings \(\mu_{P_i}\), and \(f_\lambda\) the population ridge solution. Then the excess risk splits as

$$ \underbrace{\mathcal R(\widehat f)-\mathcal R(f_\rho)}_{\text{total}}\ \lesssim\ \underbrace{\|\widehat f-\widetilde f\|^2}_{\text{stage 1: finite bags}}\ +\ \underbrace{\|\widetilde f-f_\lambda\|^2}_{\text{stage 2: finite bags count }\ell}\ +\ \underbrace{\|f_\lambda-f_\rho\|^2}_{\text{approximation}}, $$

where all norms are taken in the prediction space induced by the meta-distribution. The first term isolates replacing true embeddings by empirical ones, the second is ordinary finite-\(\ell\) ridge estimation, and the third is regularization bias.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

::: {.proof}
[Proof]{.box-title}

For square loss, excess risk is the squared \(L^2\) distance to the regression function \(f_\rho\). Insert \(\widetilde f\) and \(f_\lambda\), apply the triangle inequality, and then use \((a+b+c)^2\le3(a^2+b^2+c^2)\) to obtain the three terms. The two-stage interpretation is exact. Rates for the individual terms need more assumptions: the finite-bag lemma and the Lipschitz embedding kernel give an \(O_P(N^{-1/2})\) perturbation of each second-stage feature, but the ridge map amplifies that perturbation according to \(\lambda\), the sample geometry, output bounds, and the source/capacity regime. The middle and right terms are the usual estimation and approximation pieces of kernel ridge regression. Szabó et al. (2016) control all three together under the conditions stated in the theorem below. [\(\square\)]{.qed}
:::

The decomposition already tells the qualitative story: drive \(\lambda\to0\) to kill the bias, take \(\ell\to\infty\) to kill the estimation variance, and take \(N\to\infty\) fast enough that \(\lambda^{-2}N^{-1}\to0\) despite the shrinking \(\lambda\). The bag size must grow with the number of bags, but not arbitrarily fast. Szabó, Sriperumbudur, Póczos, and Gretton (2016) turned this heuristic into the first sharp analysis, under the standard smoothness (source) and effective-dimension (capacity) conditions of [[ch:mercer-and-rates|the rates chapter]] on the second-stage regression.

::: {.theorem #thm-36-7}
[Theorem (consistency and optimal rates, Szabó, Sriperumbudur, Póczos, Gretton 2016)]{.box-title}

Under source and capacity conditions on the second-stage kernel ridge problem and mild regularity of the embedding kernel, the two-stage estimator \(\widehat f\) with an appropriately chosen \(\lambda=\lambda(\ell)\to0\) is consistent, \(\mathcal R(\widehat f)-\mathcal R(f_\rho)\to0\) in probability as \(\ell\to\infty\), provided the bag size \(N=N(\ell)\) grows at least polynomially in \(\ell\). Moreover, for a sufficient bag-growth rate the estimator attains the same minimax-optimal rate it would achieve if the true embeddings \(\mu_{P_i}\) were observed: the finite-bag stage costs nothing asymptotically once the bags are large enough.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
:::

The headline is worth stating plainly but not universally. Under the theorem's source, capacity, boundedness, and Hölder-continuity assumptions, the price of seeing only bag samples is asymptotically nil for a sufficient joint schedule of \(N(\ell)\) and \(\lambda(\ell)\). The polynomial exponent is not universal: it changes with the source smoothness, effective dimension, and continuity exponent. Finite bags do not preclude learning, but they constrain how fast \(\lambda\) may shrink and which oracle rate can be matched.

## Set kernels and support measure machines {#set-kernels-smm}

The two-stage recipe did not appear from nothing; its linear case predates the general theory and is worth naming. Take the linear embedding kernel and its empirical form: for two bags \(A\) and \(B\),

$$ \mathcal K_{\mathrm{lin}}(A,B)=\langle\widehat\mu_A,\widehat\mu_B\rangle_{\mathcal H}=\frac{1}{|A|\,|B|}\sum_{a\in A}\sum_{b\in B}k(a,b). $$

This is the *set kernel*, the average base kernel over all cross pairs, a special case of the convolution kernels for structured objects of Haussler (1999). It compares two bags without any distributional language at all, and plugging it into a support vector machine gives a classifier of sets. Muandet, Fukumizu, Dinuzzo, and Schölkopf (2012) generalized this from the linear kernel to the full nonlinear machinery, running kernel machines with the embedding kernels \(\mathcal K_\gamma\) directly on probability measures.

::: {.remark}
[Support measure machines]{.box-title}

A *support measure machine* (Muandet et al. 2012) is a large-margin classifier whose training examples are probability measures \(P_1,\dots,P_\ell\), each with a label, fitted by the SVM dual on the Gram matrix \(\mathbf K_{ij}=\mathcal K_\gamma(P_i,P_j)\). It is exactly the SVM of the earlier chapters with points replaced by embeddings, so the representer theorem, the dual quadratic program, and the kernelization all carry over unchanged. When only samples are available, each \(\mathcal K_\gamma(P_i,P_j)\) is estimated from the two bags as in the Gram-matrix algorithm, making the support measure machine the classification twin of the two-stage regressor.
:::

For any of this to learn an arbitrary continuous target, the embedding kernel must be rich enough on the space of measures, the distributional analogue of a universal kernel. Under the topology and compactness conditions in Christmann and Steinwart (2010), Gaussian-type kernels built on injective embeddings are universal on the relevant measure space, so their RKHS can approximate continuous distribution functionals. This approximation property is one ingredient of consistency, not a finite-sample rate by itself. A complementary line avoids the embedding entirely: Póczos, Rinaldo, Singh, and Wasserman (2013) estimate functionals of the input density directly, by nonparametric density or nearest-neighbour methods, giving a distribution-free route to distribution regression with its own consistency guarantees. The embedding approach and the density approach are the two poles of the field, trading the RKHS's smoothing for direct estimation.

## Functional data: when the input is a function {#functional-data}

A close cousin of a distribution-valued input is a function-valued one. In functional data analysis (Ramsay and Silverman 2005) each example is itself a function: a growth curve \(x_i(t)\) of height against age, a spectrum of absorbance against wavelength, a daily temperature profile. As with bags, we never see the whole function, only its values on a finite grid, so the same finite-resolution error reappears, now as a finite grid rather than a finite bag. The objects live in \(L^2\), the space of square-integrable functions, with the inner product \(\langle x,x'\rangle_{L^2}=\int x(t)x'(t)\,dt\).

:::: {.definition #def-36-8}
[Definition (functional linear model)]{.box-title}

The functional linear regression model predicts a scalar label from a functional input \(x\in L^2\) through

$$ y=\alpha+\int x(t)\,\beta(t)\,dt+\varepsilon=\alpha+\langle x,\beta\rangle_{L^2}+\varepsilon, $$

with intercept \(\alpha\in\mathbb R\) and coefficient function \(\beta\in L^2\) to be estimated.
::::

This is ordinary linear regression carried out in \(L^2\), the coefficient vector replaced by a coefficient function. To move beyond linearity, place a kernel on \(L^2\) directly, for instance the Gaussian \(\mathcal K(x,x')=\exp\!\big(-\|x-x'\|_{L^2}^2/2\gamma^2\big)\), and run kernel ridge regression, exactly stage two of the distribution-regression recipe with \(L^2\) distance in place of the MMD. The two settings are in fact one. A mean embedding \(\mu_P\) is an element of the RKHS \(\mathcal H\), that is, a function, so distribution regression is functional regression on the embedding functions, and the MMD is the \(\mathcal H\)-norm distance between them. Christmann and Steinwart's universality applies here too, so nonparametric functional regression with a Gaussian kernel on \(L^2\) is consistent under the same kind of conditions. The distinctions that remain are practical: functional data are often smooth and are pre-smoothed by splines or a functional principal component basis before the kernel sees them, whereas bags are rough empirical measures whose only smoothing is the base kernel itself.

## Summary {#summary}

When each training example is a probability distribution seen only through a bag of samples, learning proceeds in two stages. Stage one embeds every input distribution as a single RKHS point, its kernel mean embedding, estimated from the bag by the empirical embedding. Stage two runs kernel ridge regression on those points using a kernel on distributions, either the linear embedding kernel \(\langle\mu_P,\mu_Q\rangle\), which is the set kernel and underlies support measure machines, or the Gaussian-on-MMD kernel \(\exp(-\mathrm{MMD}^2/2\gamma^2)\); both are positive definite, the second by a Schur-product argument that reruns \"the Gaussian kernel is a kernel\" inside the embedding space. The estimator carries two error sources, a finite number of bags \(\ell\) and a finite bag size \(N\), and its risk decomposes accordingly: the stage-two term vanishes as \(\ell\to\infty\), and the stage-one term, controlled by the \(N^{-1/2}\) embedding rate and the Lipschitz continuity of the distribution kernel, vanishes as \(N\to\infty\). Szabó, Sriperumbudur, Póczos, and Gretton (2016) proved that with \(\lambda\) shrinking and the bags growing polynomially in \(\ell\) the two-stage estimator is consistent and even minimax-optimal, so finite bags cost nothing asymptotically. The same construction reaches set kernels and support measure machines (Muandet et al. 2012), inherits universality on measures (Christmann and Steinwart 2010), has a density-based rival (Póczos et al. 2013), and merges with functional data analysis (Ramsay and Silverman 2005), where the input is a function in \(L^2\) and the embedding view makes distribution regression a special case. The thread back to [[ch:conditional-mean-embeddings|conditional mean embeddings]] and [[ch:causal-inference-with-kernels|causal inference with kernels]] is the same one throughout: once a distribution is a point in a Hilbert space, every kernel method already built applies to it unchanged.

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

For **Distribution Regression and Functional Data**, split validation by bags, never by points within a bag, or information from one distribution leaks into both train and test sets. Report the number of bags, the distribution of bag sizes, both kernel bandwidths, ridge strength, Gram conditioning, and exact or approximate Gram-build cost. A characteristic base kernel only prevents stage-one collisions; stage two still needs adequate capacity, regularization, and enough labelled bags. Do not quote an oracle rate without the source, capacity, boundedness, Hölder, and \(N(\ell)\) assumptions that make it apply.

## Summary and further reading {#summary-and-further-reading}

Szabó et al. [@szabo2016dr] give the two-stage statistical analysis, Muandet et al. [@muandet2012smm] develop learning directly on measures, and Christmann and Steinwart [@christmann2010] supply universality results under explicit topological conditions. The operational task board is therefore two-dimensional: improve within-bag resolution or its approximation, and independently improve the number and coverage of labelled bags. Neither can substitute for the other.

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} Explain, in terms of the two-stage sampled model, why distribution regression has two sources of error while ordinary regression has one. Name the integer that controls each, say which limit \(\ell\to\infty\) or \(N\to\infty\) removes each error, and give a one-sentence example of a real task whose natural input is a bag of samples rather than a vector.
2.  [computation]{.ex-tag} Take the two bags \(A=\{0,1\}\) and \(B=\{2,3\}\) on \(\mathbb R\) with base kernel \(k(a,b)=e^{-(a-b)^2/2}\). Compute the three empirical inner products \(\langle\widehat\mu_A,\widehat\mu_A\rangle\), \(\langle\widehat\mu_B,\widehat\mu_B\rangle\), and \(\langle\widehat\mu_A,\widehat\mu_B\rangle\) as averages of four base-kernel values each, then form the squared empirical MMD \(D_{AB}=\langle\widehat\mu_A,\widehat\mu_A\rangle+\langle\widehat\mu_B,\widehat\mu_B\rangle-2\langle\widehat\mu_A,\widehat\mu_B\rangle\) and the Gaussian-on-MMD value \(\mathcal K_1(A,B)=e^{-D_{AB}/2}\). Report each number.
3.  [computation]{.ex-tag} Continue Example (predicting a label): keeping \(\gamma=1\) fixed, recompute the held-out prediction \(\widehat f(P_3)=\mathbf k_\ast^\top(\mathbf K_{\mathrm{tr}}+\ell\lambda I)^{-1}\mathbf y\) for the two ridge values \(\lambda=0.2\) and \(\lambda=0.01\). Explain the direction each prediction moves relative to the value \(1.8719\) at \(\lambda=0.05\), and relate it to the bias term of the risk decomposition.
4.  [proof]{.ex-tag} Prove that the linear embedding kernel \(\mathcal K_{\mathrm{lin}}(P,Q)=\langle\mu_P,\mu_Q\rangle_{\mathcal H}\) is positive definite directly from the quadratic form, and identify its feature map. Then show it equals \(\mathbb E_{X\sim P,\,Y\sim Q}[k(X,Y)]\), so its empirical version is the set kernel \(\frac{1}{|A||B|}\sum_{a,b}k(a,b)\).
    Hint

    ::: hint-body
    For positive definiteness, \(\sum_{a,b}c_a c_b\langle\mu_{P_a},\mu_{P_b}\rangle=\|\sum_a c_a\mu_{P_a}\|^2\ge 0\); the feature map is \(\Phi(P)=\mu_P\). For the expectation form, write \(\mu_P=\mathbb E_X[K_X]\), \(\mu_Q=\mathbb E_Y[K_Y]\), and pull both expectations out of the inner product using \(\langle K_X,K_Y\rangle=k(X,Y)\).
    :::
5.  [proof]{.ex-tag} Establish the finite-bag rate directly. For \(\widehat\mu_P=\frac1N\sum_{a=1}^N K_{x^a}\) with \(x^a\overset{\text{i.i.d.}}{\sim}P\) and a bounded base kernel \(k(x,x)\le\kappa\), show \(\mathbb E\|\widehat\mu_P-\mu_P\|_{\mathcal H}^2\le\kappa/N\), hence \(\|\widehat\mu_P-\mu_P\|_{\mathcal H}=O_P(N^{-1/2})\). Then combine this with the Lipschitz lemma to bound \(\|\Psi(\widehat P)-\Psi(P)\|_{\mathcal H_{\mathcal K}}\) in expectation, showing the finite-bag error propagates to the second stage at the same \(N^{-1/2}\) rate.
    Hint

    ::: hint-body
    The estimator is a mean of \(N\) i.i.d. zero-mean terms \(K_{x^a}-\mu_P\), so its squared-norm expectation is \(\frac1N(\mathbb E[k(X,X)]-\|\mu_P\|^2)\le\kappa/N\). Then \(\mathbb E\|\Psi(\widehat P)-\Psi(P)\|^2\le\frac1{\gamma^2}\mathbb E\|\widehat\mu_P-\mu_P\|^2\le\kappa/(\gamma^2 N)\) by the Lipschitz lemma.
    :::
6.  [proof]{.ex-tag} Prove the Lipschitz lemma's key inequality without quoting it: for the Gaussian-on-MMD kernel, show \(\|\Psi(P)-\Psi(P')\|_{\mathcal H_{\mathcal K}}^2=2\big(1-e^{-\|\mu_P-\mu_{P'}\|^2/2\gamma^2}\big)\le\gamma^{-2}\|\mu_P-\mu_{P'}\|^2\). Deduce that as \(\gamma\to\infty\) the second-stage feature map becomes ever flatter in the embedding, and explain what that means for the bias-variance trade-off of the two-stage estimator.
    Hint

    ::: hint-body
    Use \(\mathcal K_\gamma(P,P)=1\) to get \(\|\Psi(P)-\Psi(P')\|^2=2-2\mathcal K_\gamma(P,P')\), then the bound \(1-e^{-u}\le u\) with \(u=\|\mu_P-\mu_{P'}\|^2/2\gamma^2\). Large \(\gamma\) means a nearly constant kernel, low variance but high bias, the usual bandwidth trade-off now on distributions.
    :::
7.  [challenge]{.ex-tag} Consider the linear embedding kernel with the linear base kernel \(k(x,x')=x^\top x'\) on \(\mathbb R^d\). Show that then \(\mu_P=\mathbb E_{X\sim P}[X]=m_P\), the ordinary mean, so \(\mathcal K_{\mathrm{lin}}(P,Q)=m_P^\top m_Q\) and the two-stage regressor can only fit functions of the input mean. Conclude that this choice is blind to any feature of \(P\) beyond its mean, and state which property of the base kernel, in the language of [[ch:kernel-mean-embeddings|the mean-embedding chapter]], must hold for the two-stage estimator to be able to distinguish distributions with equal means. Finally, argue that using a characteristic base kernel is necessary but not sufficient for the two-stage estimator to be consistent, and name the extra ingredient the second stage requires.
    Hint

    ::: hint-body
    With the linear base kernel \(\mu_P(x)=\mathbb E[X]^\top x\), so the embedding is the mean vector and the set kernel is \(m_A^\top m_B\). Distinguishing equal-mean distributions requires a characteristic base kernel (injective embedding). But injectivity of stage one is not enough: the second-stage kernel on embeddings must also be universal on the space of measures, which is the Christmann and Steinwart (2010) condition.
    :::
:::
