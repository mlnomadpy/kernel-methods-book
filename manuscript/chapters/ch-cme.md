---
id: ch-cme
slug: conditional-mean-embeddings
title: Conditional Mean Embeddings and Kernel Bayes' Rule
part: IX · Kernel Probabilistic Inference
order: 35
tier: advanced
prerequisites:
  - kernel-quadrature-and-herding
objectives:
  - >-
    Derive the conditional embedding operator and identify the range assumption
    hidden in its formal inverse.
  - >-
    Compute regularized empirical conditional embeddings and conditional
    expectations from one shared Gram factorization.
  - >-
    Interpret conditional embedding estimation as vector-valued kernel ridge
    regression.
  - >-
    Apply kernel sum, chain, and Bayes rules while distinguishing an RKHS
    embedding from a probability vector.
  - >-
    Diagnose regularization, conditioning, and signed-weight failures in a
    kernel Bayes filter.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-cme.yml
verification_date: null
bibliography:
  - song2009cme
  - baker1973
  - fukumizu2004
  - grunewalder2012
  - muandet2017
  - song2013cme
  - fukumizu2013kbr
---
# Conditional Mean Embeddings and Kernel Bayes' Rule

<p class="lead">Ask what a test result means for a diagnosis, what a sensor reading says about a robot's position, what today's state predicts about tomorrow's: nearly every question probabilistic reasoning answers is a conditional one. Given that we observed \(X=x\), what do we now believe about \(Y\)? The mean embeddings of [[ch:kernel-mean-embeddings|the previous chapter]] turned a whole distribution into a single RKHS vector, but that construction embeds a marginal, an object complete on its own; it offers no way to update the vector once part of the world has been observed. This chapter embeds the conditional distribution \(P(Y\mid X=x)\), not as one vector but as a linear operator that turns the feature of the conditioning value \(x\) into the embedding of the resulting conditional. Once conditioning is an operator, the sum rule, the chain rule, and Bayes' rule become operator algebra on embeddings, and probabilistic inference runs entirely inside the RKHS, with no densities, no parametric model, and no integration, only Gram matrices estimated from a joint sample. The payoff is Kernel Bayes' Rule and, chained step by step, a kernel filter for state-space models.</p>

## From marginal to conditional embeddings {#from-marginal-to-conditional}

The object we are after must do for a conditional law what \(\mu_P\) did for a marginal: hold every conditional expectation inside one element of an RKHS. Because conditioning relates two variables, the setup needs two kernels, one per space. Fix two spaces, an input space \(\mathcal X\) with a positive definite kernel \(k\) and RKHS \(\mathcal H_{\mathcal X}\), and an output space \(\mathcal Y\) with kernel \(\ell\) and RKHS \(\mathcal H_{\mathcal Y}\). Write \(\varphi(x)=k(x,\cdot)\in\mathcal H_{\mathcal X}\) and \(\psi(y)=\ell(y,\cdot)\in\mathcal H_{\mathcal Y}\) for the two feature maps. The marginal embedding of a distribution \(P\) on \(\mathcal Y\) was \(\mu_P=\mathbb E_{Y\sim P}[\psi(Y)]\), the average feature. The natural conditional analogue simply averages the output feature under the conditional law.

:::: {.definition #def-33-1}
[Definition (conditional mean embedding)]{.box-title}

Let \((X,Y)\) be a random pair on \(\mathcal X\times\mathcal Y\). For each \(x\in\mathcal X\) at which \(P(Y\mid X=x)\) is well defined, the *conditional mean embedding* of \(P(Y\mid X=x)\) is

$$\mu_{Y\mid X=x}\ :=\ \mathbb E\big[\psi(Y)\mid X=x\big]\ =\ \int_{\mathcal Y}\ell(y,\cdot)\,dP(y\mid x)\ \in\ \mathcal H_{\mathcal Y}.$$
::::

This is not one embedding but a whole family, one element of \(\mathcal H_{\mathcal Y}\) for every conditioning value \(x\). What makes it useful is the same generalized reproducing property that made the marginal embedding useful, now conditional: averaging the kernel trick \(g(Y)=\langle g,\psi(Y)\rangle_{\mathcal H_{\mathcal Y}}\) under the conditional law gives, for every \(g\in\mathcal H_{\mathcal Y}\),

$$\big\langle g,\ \mu_{Y\mid X=x}\big\rangle_{\mathcal H_{\mathcal Y}}\ =\ \mathbb E\big[g(Y)\mid X=x\big].$$

Read it as the promise of the whole chapter: the single vector \(\mu_{Y\mid X=x}\) stores the conditional expectation of *every* RKHS function \(g\), recovered by one inner product. If we can produce these vectors from data, we can answer any question of the form \"what is the expected value of \(g(Y)\) once we learn \(X=x\)\" without ever writing down the conditional density. The reproducing-property background here is exactly that of [[ch:kernels-and-rkhs|the RKHS chapter]]; the new content is that conditioning turns out to be linear in the feature \(\varphi(x)\).

## The conditional mean embedding operator {#cme-operator}

The move introduced by Song, Huang, Smola, and Fukumizu (2009) is to seek a single linear operator that produces the entire family at once, mapping the input feature to the conditional embedding:

$$\mu_{Y\mid X=x}\ =\ \mathcal C_{Y\mid X}\,\varphi(x),\qquad \mathcal C_{Y\mid X}:\mathcal H_{\mathcal X}\to\mathcal H_{\mathcal Y}.$$

If such an operator exists, conditioning is a matrix-vector product in feature space, and this is what will let the sum and Bayes rules become operator identities. To build it we need the second-order objects that summarize how the two feature maps covary, the covariance operators, studied for RKHS first by Baker (1973) and brought into learning by Fukumizu, Bach, and Jordan (2004).

::::: {.definition #def-33-2}
[Definition (covariance operators)]{.box-title}

Assume \(\mathbb E[k(X,X)]\lt\infty\) and \(\mathbb E[\ell(Y,Y)]\lt\infty\). The *(uncentered) covariance operator* \(\mathcal C_{XX}:\mathcal H_{\mathcal X}\to\mathcal H_{\mathcal X}\) and the *cross-covariance operator* \(\mathcal C_{YX}:\mathcal H_{\mathcal X}\to\mathcal H_{\mathcal Y}\) are

$$\mathcal C_{XX}=\mathbb E\big[\varphi(X)\otimes\varphi(X)\big],\qquad \mathcal C_{YX}=\mathbb E\big[\psi(Y)\otimes\varphi(X)\big],$$

where \((a\otimes b)c=\langle b,c\rangle\,a\). They are characterized by

$$\langle f,\mathcal C_{XX}f'\rangle_{\mathcal H_{\mathcal X}}=\mathbb E\big[f(X)f'(X)\big],\qquad \langle g,\mathcal C_{YX}f\rangle_{\mathcal H_{\mathcal Y}}=\mathbb E\big[f(X)g(Y)\big],$$

for all \(f,f'\in\mathcal H_{\mathcal X}\) and \(g\in\mathcal H_{\mathcal Y}\).
:::::

These are the RKHS versions of a variance and a covariance matrix: \(\mathcal C_{XX}\) records how input features co-vary with themselves, \(\mathcal C_{YX}\) how output features co-vary with input features. With them the operator is forced.

:::: {.proposition #prop-33-3}
[Proposition (the conditional embedding operator, Song et al. 2009)]{.box-title}

Suppose that for every \(g\in\mathcal H_{\mathcal Y}\) the conditional expectation \(x\mapsto\mathbb E[g(Y)\mid X=x]\) belongs to \(\mathcal H_{\mathcal X}\). Then the operator \(\mathcal C_{Y\mid X}\) satisfying \(\mu_{Y\mid X=x}=\mathcal C_{Y\mid X}\varphi(x)\) for all \(x\) obeys

$$\mathcal C_{Y\mid X}\,\mathcal C_{XX}\ =\ \mathcal C_{YX},\qquad\text{so formally}\qquad \mathcal C_{Y\mid X}\ =\ \mathcal C_{YX}\,\mathcal C_{XX}^{-1}.$$

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

Write \(h_g(x):=\mathbb E[g(Y)\mid X=x]\), which lies in \(\mathcal H_{\mathcal X}\) by assumption. The defining reproducing property says \(\langle g,\mathcal C_{Y\mid X}\varphi(x)\rangle=\langle g,\mu_{Y\mid X=x}\rangle=h_g(x)\), while the reproducing property in \(\mathcal H_{\mathcal X}\) gives \(h_g(x)=\langle h_g,\varphi(x)\rangle\). Since both hold for every \(x\), the adjoint acts as \(\mathcal C_{Y\mid X}^{\ast}g=h_g\). Now test \(\mathcal C_{XX}\mathcal C_{Y\mid X}^{\ast}g\) against any \(f\in\mathcal H_{\mathcal X}\), using the tower property \(\mathbb E[f(X)g(Y)]=\mathbb E\big[f(X)\,\mathbb E[g(Y)\mid X]\big]\):

$$\langle f,\mathcal C_{XX}\mathcal C_{Y\mid X}^{\ast}g\rangle=\langle f,\mathcal C_{XX}h_g\rangle=\mathbb E\big[f(X)h_g(X)\big]=\mathbb E\big[f(X)g(Y)\big]=\langle f,\mathcal C_{YX}^{\ast}g\rangle.$$

As \(f\) and \(g\) range over their spaces, \(\mathcal C_{XX}\mathcal C_{Y\mid X}^{\ast}=\mathcal C_{YX}^{\ast}\); taking adjoints and using that \(\mathcal C_{XX}\) is self-adjoint gives \(\mathcal C_{Y\mid X}\mathcal C_{XX}=\mathcal C_{YX}\). Wherever \(\mathcal C_{XX}^{-1}\) is meaningful this reads \(\mathcal C_{Y\mid X}=\mathcal C_{YX}\mathcal C_{XX}^{-1}\). [\(\square\)]{.qed}
::::

The identity is clean, but the words \"formally\" and \"wherever it is meaningful\" are doing real work, and honesty requires unpacking them.

::: {.remark}
[Remark (why the inverse is delicate)]{.box-title}

Two subtleties sit inside \(\mathcal C_{YX}\mathcal C_{XX}^{-1}\). First, for a bounded kernel \(\mathcal C_{XX}\) is a compact (indeed trace-class) operator, so its eigenvalues accumulate at zero and its inverse \(\mathcal C_{XX}^{-1}\) is *unbounded*: the product \(\mathcal C_{YX}\mathcal C_{XX}^{-1}\) need not be a bounded operator on \(\mathcal H_{\mathcal X}\) at all. Second, the assumption that \(\mathbb E[g(Y)\mid X=\cdot]\in\mathcal H_{\mathcal X}\) for every \(g\in\mathcal H_{\mathcal Y}\) is genuinely restrictive: the conditional expectation can be smoother or rougher than the input RKHS contains, and then no exact operator \(\mathcal C_{Y\mid X}\) reproduces the family. Both problems are met the same way, by regularizing: replace \(\mathcal C_{XX}^{-1}\) by \((\mathcal C_{XX}+\lambda I)^{-1}\) for some \(\lambda\gt 0\), which is bounded and always defined. The regression view of the next section removes the range assumption entirely, by never claiming an exact operator, only a best approximation.
:::

## The empirical estimate {#empirical-cme}

In practice we never see \(\mathcal C_{XX}\) or \(\mathcal C_{YX}\); we see a joint sample \(\{(x_i,y_i)\}_{i=1}^n\) drawn from \(P(X,Y)\). Collect the features into the \"feature matrices\" \(\Phi=(\varphi(x_1),\dots,\varphi(x_n))\) and \(\Upsilon=(\psi(y_1),\dots,\psi(y_n))\), whose columns are RKHS elements, and form the two Gram matrices \(K_{ij}=k(x_i,x_j)\) and \(L_{ij}=\ell(y_i,y_j)\). The empirical covariance operators are \(\widehat{\mathcal C}_{XX}=\tfrac1n\Phi\Phi^{\ast}\) and \(\widehat{\mathcal C}_{YX}=\tfrac1n\Upsilon\Phi^{\ast}\), and the regularized conditional embedding operator is

$$\widehat{\mathcal C}_{Y\mid X}=\widehat{\mathcal C}_{YX}\big(\widehat{\mathcal C}_{XX}+\lambda I\big)^{-1}=\Upsilon\,(K+n\lambda I)^{-1}\Phi^{\ast},$$

where the second equality is the push-through identity \(\Phi^{\ast}(\Phi\Phi^{\ast}+cI)^{-1}=(\Phi^{\ast}\Phi+cI)^{-1}\Phi^{\ast}\) with \(c=n\lambda\). Applying it to the feature of a test input \(x\) gives the embedding as a weighted sum of the training output features,

$$\widehat\mu_{Y\mid X=x}=\widehat{\mathcal C}_{Y\mid X}\varphi(x)=\sum_{i=1}^{n}\beta_i(x)\,\psi(y_i),\qquad \boldsymbol\beta(x)=(K+n\lambda I)^{-1}\mathbf k_x,$$

with \(\mathbf k_x=(k(x_1,x),\dots,k(x_n,x))^{\top}\). The weight vector \(\boldsymbol\beta(x)\) is exactly a kernel ridge regression solution, the same object as the Gaussian-process posterior weights of [[ch:gaussian-processes-and-rvm|the GP chapter]], except that here the \"targets\" are the output features \(\psi(y_i)\) rather than scalar labels. Reading off any conditional expectation is then a finite sum, since for \(g\in\mathcal H_{\mathcal Y}\),

$$\widehat{\mathbb E}\big[g(Y)\mid X=x\big]=\big\langle g,\widehat\mu_{Y\mid X=x}\big\rangle=\sum_{i=1}^{n}\beta_i(x)\,g(y_i).$$

:::: {.algorithm #algo-33-1}
[Algorithm (empirical conditional mean embedding)]{.box-title}

::: algo-io
[Input]{.algo-lab} Joint sample \(\{(x_i,y_i)\}_{i=1}^n\); input kernel \(k\), output kernel \(\ell\); regularization \(\lambda\gt 0\); test input \(x\); query function \(g\in\mathcal H_{\mathcal Y}\).

[Output]{.algo-lab} Weights \(\boldsymbol\beta(x)\in\mathbb R^{n}\) with \(\widehat\mu_{Y\mid X=x}=\sum_i\beta_i(x)\psi(y_i)\), and the estimate \(\widehat{\mathbb E}[g(Y)\mid X=x]\).
:::

1.  Form the input Gram matrix \(K\in\mathbb R^{n\times n}\), \(K_{ij}=k(x_i,x_j)\).
2.  Form the test vector \(\mathbf k_x=(k(x_1,x),\dots,k(x_n,x))^{\top}\).
3.  Solve the ridge system \((K+n\lambda I)\,\boldsymbol\beta(x)=\mathbf k_x\) for the weights \(\boldsymbol\beta(x)\).
4.  Return \(\widehat{\mathbb E}[g(Y)\mid X=x]=\sum_{i=1}^n\beta_i(x)\,g(y_i)\), and, if the full embedding is needed, the weighted set \(\{(\beta_i(x),y_i)\}\).
::::

<figure class="viz" data-widget="cme-explore">

<figcaption>Dragging the conditioning point applies the exact estimator \(\beta(x^*)=(K_X+n\lambda I)^{-1}k_X(x^*)\) to sixty fixed samples: the side profile is the embedded conditional \(\sum_i\beta_i\,l(\cdot,y_i)\) and the dot its conditional mean. The Cholesky factor is shared across every \(x^*\), so each drag costs one triangular solve; the sliders expose the bandwidth and ridge trade-off the chapter derives.</figcaption>
</figure>

The visual question is how a point observation becomes a distribution-valued prediction. As \(x^\ast\) moves, the weights change first, then the entire conditional profile moves; bandwidth controls locality and \(\lambda\) controls how violently the weights can respond. A plausible-looking conditional mean is not enough, so inspect the profile, weight norm, and condition number together.

Regularization is not optional decoration here. The matrix \(K\) is often numerically singular (repeated or nearby inputs make its columns almost dependent), and even in the population the operator inverse is unbounded, so the raw \(K^{-1}\mathbf k_x\) would be a wild, high-variance vector. The term \(n\lambda I\) floors the spectrum of \(K\) at \(n\lambda\), trading a little bias for a large reduction in variance, exactly the bias-variance dial of ridge regression.

For many test inputs, factor \(K+n\lambda I\) once by Cholesky at \(O(n^3)\) cost and reuse the factor, reducing each new conditional query to two triangular solves plus kernel evaluation. Never form the inverse explicitly. Report \(\lambda\), \(\operatorname{cond}(K+n\lambda I)\), the norm and sum of \(\boldsymbol\beta(x)\), and a held-out conditional prediction error; these reveal extrapolation and unstable cancellation that the displayed embedding alone can hide.

::::: {.example #example-33-1}
[Example (predicting a conditional expectation)]{.box-title}

:::: wex
::: wex-setup
Four joint pairs \((x_i,y_i)=(0,0.0),(1,1.2),(2,1.8),(3,3.0)\), so \(y\) roughly tracks \(x\). Gaussian kernels \(k(x,x')=e^{-(x-x')^2/2}\) and \(\ell(y,y')=e^{-(y-y')^2/2}\) (both bandwidths \(1\)). Regularization \(\lambda=0.125\), so \(n\lambda=0.5\). Test input \(x_\ast=1.5\). All numbers from `checks/ch-cme-ex1.py`.
:::

1.  [Form the input Gram matrix.]{.wex-op} With \(e^{-1/2}=0.6065\), \(e^{-2}=0.1353\), \(e^{-9/2}=0.0111\),

$$K=\begin{pmatrix}1&0.6065&0.1353&0.0111\\0.6065&1&0.6065&0.1353\\0.1353&0.6065&1&0.6065\\0.0111&0.1353&0.6065&1\end{pmatrix}.$$
2.  [Build the test vector.]{.wex-op} The distances from \(x_\ast=1.5\) to \((0,1,2,3)\) give \(\mathbf k_{x_\ast}=(0.3247,\ 0.8825,\ 0.8825,\ 0.3247)\).
3.  [Solve the ridge system.]{.wex-op} \(\boldsymbol\beta(x_\ast)=(K+0.5\,I)^{-1}\mathbf k_{x_\ast}=(0.0111,\ 0.4150,\ 0.4150,\ 0.0111)\), with \(\sum_i\beta_i=0.8522\).
4.  [Predict the conditional mean of the response.]{.wex-op} Using the plug-in \(g(y)=y\), \(\widehat{\mathbb E}[Y\mid X=1.5]=\sum_i\beta_i(x_\ast)\,y_i=0.4150\,(1.2)+0.4150\,(1.8)+0.0111\,(0.0+3.0)=1.2784\).
5.  [Predict a genuine RKHS query.]{.wex-op} For \(g=\ell(2.0,\cdot)\in\mathcal H_{\mathcal Y}\), evaluate \(g(y_i)=\ell(y_i,2.0)=(0.1353,\ 0.7261,\ 0.9802,\ 0.6065)\), so \(\widehat{\mathbb E}[\ell(2.0,Y)\mid X=1.5]=\sum_i\beta_i(x_\ast)g(y_i)=0.7164\).

**Reading.** The two central training points, at \(x=1\) and \(x=2\), carry almost all the weight and receive equal shares because \(x_\ast=1.5\) sits symmetrically between them; the far points are all but ignored. The response estimate \(1.28\) is close to the sensible \(1.5\) but pulled down, because the weights sum to \(0.85\) rather than \(1\): regularization shrinks the embedding toward the origin, which biases the plug-in mean toward \(0\). Note also that the RKHS query \(0.7164\) is not equal to \(\ell(2.0,\widehat{\mathbb E}[Y\mid X=1.5])=0.7708\): the embedding estimates \(\mathbb E[g(Y)\mid X]\), the average of \(g\) over the conditional, not \(g\) evaluated at the conditional mean, and the two differ whenever \(g\) is nonlinear.
::::

**Verification artifact.** checks/example-ch-cme-example-33-1.json records the example source hash and verification scope.
:::::

The scalar query \(g(y)=y\) in step 4 is deliberately labelled a plug-in: on an unbounded domain the identity function need not belong to the Gaussian RKHS, so the conditional-reproducing guarantee does not automatically cover it. The genuine RKHS query in step 5 is covered. To estimate a raw conditional mean rigorously, use an output kernel whose RKHS contains or uniformly approximates the identity on the working domain, and include the approximation error.

## The regression view {#cme-as-regression}

The estimator \(\Upsilon(K+n\lambda I)^{-1}\Phi^{\ast}\) was derived by regularizing an operator inverse, which left open both the range assumption and the question of what, precisely, it estimates when that assumption fails. Grünewälder, Lever, Baldassarre, Patterson, Gretton, and Pontil (2012) gave the reassuring answer: the empirical conditional mean embedding is the solution of a vector-valued ridge regression, and this holds with no assumption on the range of \(\mathcal C_{XX}\).

The idea is to regress the output feature onto the input feature. Since \(\mathbb E[\psi(Y)\mid X=x]=\mu_{Y\mid X=x}\), the feature \(\psi(Y)\) is an unbiased, noisy observation of the target embedding \(\mu_{Y\mid X}\), exactly the setup of least-squares regression, only with values in the Hilbert space \(\mathcal H_{\mathcal Y}\) instead of \(\mathbb R\). The population target minimizes the surrogate risk \(\mathcal E(C)=\mathbb E\,\|\psi(Y)-C\varphi(X)\|_{\mathcal H_{\mathcal Y}}^2\) over operators \(C\), whose minimizer over all measurable maps is the true conditional embedding; the empirical, regularized version is what we compute.

:::: {.proposition #prop-33-4}
[Proposition (CME as vector-valued regression, Grünewälder et al. 2012)]{.box-title}

Among Hilbert-Schmidt operators \(C:\mathcal H_{\mathcal X}\to\mathcal H_{\mathcal Y}\), the regularized empirical risk

$$\widehat{\mathcal E}_\lambda(C)=\frac1n\sum_{i=1}^n\big\|\psi(y_i)-C\varphi(x_i)\big\|_{\mathcal H_{\mathcal Y}}^2+\lambda\,\|C\|_{\mathrm{HS}}^2$$

is minimized by \(\widehat{\mathcal C}_{Y\mid X}=\Upsilon(K+n\lambda I)^{-1}\Phi^{\ast}\), the empirical conditional embedding operator.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
::::

Three payoffs follow from this identification. It is an honest definition: \(\widehat{\mathcal C}_{Y\mid X}\) is always the minimizer of a well-posed empirical risk, whether or not the population range assumption holds, so computation never requires \(\mathbb E[g(Y)\mid X=\cdot]\in\mathcal H_{\mathcal X}\). It exposes the estimator as ridge regression, with \(\boldsymbol\beta(x)\) the ridge weights, making the vector-valued regression toolbox available. Consistency is then a learning-theory statement rather than a consequence of the formal inverse: it requires a schedule \(\lambda_n\to0\), enough data relative to that schedule, and assumptions on the target and kernel spectrum. Song et al. (2009) and Grünewälder et al. (2012) give such results under their stated source and regularity conditions; [[ch:mercer-and-rates|the rates chapter]] explains why those conditions cannot be omitted.

## Kernel probability rules {#kernel-prob-rules}

With a conditioning operator in hand we can lift the elementary rules of probability to embeddings. Throughout, a distribution is represented by its embedding and a conditional by an operator; the rules say how to combine them. Fix a *prior* \(\pi\) on \(\mathcal X\) with embedding \(\mu^{\pi}_X=\mathbb E_{X\sim\pi}[\varphi(X)]\), and read \(\mathcal C_{Y\mid X}\) as the likelihood model \(P(Y\mid X)\) it was estimated from.

### The kernel sum rule {#kernel-sum-rule}

The sum rule of probability computes a marginal, \(Q(y)=\int P(y\mid x)\,d\pi(x)\). Its embedding form is a single operator application. Draw \(X\sim\pi\) and then \(Y\sim P(Y\mid X)\); the embedding of the resulting \(Y\)-marginal is

$$\mu^{\pi}_Y=\mathbb E_{X\sim\pi}\big[\mathbb E[\psi(Y)\mid X]\big]=\mathbb E_{X\sim\pi}\big[\mathcal C_{Y\mid X}\varphi(X)\big]=\mathcal C_{Y\mid X}\,\mu^{\pi}_X,$$

the law of total expectation with the linear operator pulled outside the expectation. Empirically, if the prior embedding is carried by weights \(\mathbf m\in\mathbb R^n\) on the training inputs, \(\widehat\mu^{\pi}_X=\sum_j m_j\varphi(x_j)=\Phi\mathbf m\), then

$$\widehat\mu^{\pi}_Y=\widehat{\mathcal C}_{Y\mid X}\,\Phi\mathbf m=\Upsilon\,(K+n\lambda I)^{-1}K\mathbf m=\sum_{i=1}^n\rho_i\,\psi(y_i),\qquad \boldsymbol\rho=(K+n\lambda I)^{-1}K\mathbf m.$$

The weights \(\boldsymbol\rho\) are the sum-rule reweighting of the sample induced by the prior.

### The kernel chain rule {#kernel-chain-rule}

The chain rule factors a joint as \(Q(x,y)=P(y\mid x)\pi(x)\). In embedding language the joint is represented by an (uncentered) cross-covariance operator, and the chain rule says it is obtained by reweighting the empirical joint feature outer products with the same sum-rule weights \(\boldsymbol\rho\):

$$\mathcal C^{\pi}_{XY}=\sum_{i=1}^n\rho_i\,\varphi(x_i)\otimes\psi(y_i)=\Phi\,\mathrm{diag}(\boldsymbol\rho)\,\Upsilon^{\ast},\qquad \mathcal C^{\pi}_{YY}=\sum_{i=1}^n\rho_i\,\psi(y_i)\otimes\psi(y_i)=\Upsilon\,\mathrm{diag}(\boldsymbol\rho)\,\Upsilon^{\ast}.$$

These are the prior-weighted second-order embeddings that Bayes' rule will invert. Song, Fukumizu, and Gretton (2013) develop the sum and chain rules in this operator form as the building blocks of kernel probabilistic inference.

## Kernel Bayes' Rule {#kernel-bayes-rule}

Bayes' rule inverts the conditioning: from a prior \(\pi(x)\) and a likelihood \(P(y\mid x)\) it produces the posterior \(Q(x\mid y)\propto P(y\mid x)\pi(x)\). We want its embedding, \(\mu^{\pi}_{X\mid Y=y}\), computed entirely from the joint sample and the prior weights, with no densities anywhere. Fukumizu, Song, and Gretton (2013) obtained it by recognizing the posterior as a conditional embedding of \(X\) given \(Y\) under the prior-weighted model \(Q\), and applying the conditional embedding operator built from the reweighted covariances:

$$\mu^{\pi}_{X\mid Y=y}=\mathcal C^{\pi}_{XY}\big(\mathcal C^{\pi}_{YY}\big)^{-1}\psi(y).$$

The catch is that \(\mathcal C^{\pi}_{YY}=\Upsilon\,\mathrm{diag}(\boldsymbol\rho)\,\Upsilon^{\ast}\) is built from the signed weights \(\boldsymbol\rho\), which need not be nonnegative, so it is generally not a positive operator and a plain Tikhonov inverse \((\mathcal C^{\pi}_{YY}+\delta I)^{-1}\) can behave badly. Fukumizu et al. (2013) therefore invert the square, which is always positive, and use

$$\mu^{\pi}_{X\mid Y=y}=\mathcal C^{\pi}_{XY}\big((\mathcal C^{\pi}_{YY})^2+\delta I\big)^{-1}\mathcal C^{\pi}_{YY}\,\psi(y).$$

Turning the operators into Gram matrices with the push-through identity gives the finite-sample rule below, where \(D=\mathrm{diag}(\boldsymbol\rho)\) and \(\boldsymbol\ell_y=(\ell(y_1,y),\dots,\ell(y_n,y))^{\top}\).

:::: {.algorithm #algo-33-2}
[Algorithm (Kernel Bayes' Rule)]{.box-title}

::: algo-io
[Input]{.algo-lab} Joint sample \(\{(x_i,y_i)\}_{i=1}^n\); kernels \(k,\ell\) with Gram matrices \(K,L\); prior weights \(\mathbf m\in\mathbb R^n\) (embedding \(\sum_i m_i\varphi(x_i)\)); regularizers \(\varepsilon,\delta\gt 0\); observation \(y\); query \(g\in\mathcal H_{\mathcal X}\).

[Output]{.algo-lab} Posterior weights \(\mathbf w(y)\in\mathbb R^n\) with \(\widehat\mu^{\pi}_{X\mid Y=y}=\sum_i w_i(y)\varphi(x_i)\), and \(\widehat{\mathbb E}[g(X)\mid Y=y]\).
:::

1.  Kernel sum rule: solve \((K+n\varepsilon I)\boldsymbol\rho=K\mathbf m\) for the prior-weighted embedding weights \(\boldsymbol\rho\), and set \(D=\mathrm{diag}(\boldsymbol\rho)\).
2.  Form the observation vector \(\boldsymbol\ell_y=(\ell(y_1,y),\dots,\ell(y_n,y))^{\top}\).
3.  Kernel Bayes' rule: compute \(\mathbf w(y)=DL\big((DL)^2+\delta I\big)^{-1}D\,\boldsymbol\ell_y\).
4.  Return \(\widehat{\mathbb E}[g(X)\mid Y=y]=\sum_{i=1}^n w_i(y)\,g(x_i)\). If a literal discrete distribution is required downstream, construct and validate a separate nonnegative approximation; normalizing or projecting \(\mathbf w\) changes the RKHS estimator and must be reported as an additional approximation.
::::

::::: {.example #example-33-2}
[Example (a Bayesian posterior update)]{.box-title}

:::: wex
::: wex-setup
Five joint pairs \((x_i,y_i)=(0,0),(1,1),(2,2),(3,3),(4,4)\), Gaussian kernels of bandwidth \(1\) on both sides. A non-uniform prior favoring large inputs, \(\mathbf m=(0.05,0.05,0.10,0.30,0.50)\), so the prior mean of \(X\) is \(3.15\). Regularizers \(\varepsilon=0.1\) (so \(n\varepsilon=0.5\)) and \(\delta=0.01\). We observe \(y=1.0\). All numbers from `checks/ch-cme-ex2.py`.
:::

1.  [Apply the sum rule.]{.wex-op} \(\boldsymbol\rho=(K+0.5\,I)^{-1}K\mathbf m=(0.0363,\ 0.0436,\ 0.0995,\ 0.2620,\ 0.3488)\), inheriting the prior's tilt toward large \(x\); set \(D=\mathrm{diag}(\boldsymbol\rho)\).
2.  [Encode the observation.]{.wex-op} \(\boldsymbol\ell_y=(\ell(y_i,1.0))=(0.6065,\ 1.0000,\ 0.6065,\ 0.1353,\ 0.0111)\), peaked at the training pair whose \(y_i=1\).
3.  [Apply Bayes' rule.]{.wex-op} \(\mathbf w(y)=DL((DL)^2+0.01\,I)^{-1}D\,\boldsymbol\ell_y=(0.1100,\ 0.1847,\ 0.2679,\ 0.0513,\ -0.0721)\), with \(\sum_i w_i=0.542\).
4.  [Read off the posterior mean.]{.wex-op} Raw, \(\sum_i w_i x_i=0.5862\); after normalizing the weights to sum to one, the posterior mean of \(X\) is \(1.0817\).

**Reading.** The prior placed the mean of \(X\) at \(3.15\), betting on large inputs. Observing \(y=1\), which in this sample is the signature of \(x=1\), pulls the posterior mass down to concentrate near \(x=1\): the normalized posterior mean is \(1.08\), so the likelihood has correctly overwhelmed the prior. Two honesty notes. The posterior weights do not sum to one and one of them is negative (\(-0.072\) at \(x=4\)): the output of Kernel Bayes' Rule is an RKHS embedding, not a probability vector, so to read it as a distribution one normalizes or projects onto the simplex. And the answer depends on the two regularizers \(\varepsilon,\delta\), the price of the unbounded operator inverses that would otherwise be ill-defined.
::::

**Verification artifact.** checks/example-ch-cme-example-33-2.json records the example source hash and verification scope.
:::::

::: {.remark}
[Remark (consistency and its price)]{.box-title}

Fukumizu, Song, and Gretton (2013) prove that the Kernel Bayes' Rule estimator is consistent: with the regularizers decayed at appropriate rates as \(n\to\infty\), the estimated posterior embedding converges to the true one, and posterior expectations of RKHS functions converge to their correct values. The convergence is slower than for a single conditional embedding, because Bayes' rule chains two regularized inversions, the sum rule and the operator-square inverse, and each contributes error. This is the recurring bargain of the chapter: operator inverses that are unbounded in the population must be regularized in the sample, which buys well-posedness and consistency at the cost of a bias controlled by \(\lambda,\varepsilon,\delta\) and of outputs that are embeddings rather than literal probabilities.
:::

## Kernelized inference and the kernel Bayes filter {#kernel-bayes-filter}

The rules assembled above are exactly what a filtering algorithm needs. Consider a state-space model with a hidden state \(s_t\) that evolves through an unknown transition \(s_t\to s_{t+1}\) and emits an observation \(o_t\) through an unknown observation model. Classical filters, the Kalman filter for the linear-Gaussian case and particle filters more generally, maintain the belief state \(P(s_t\mid o_{1:t})\) and update it as each observation arrives. The *kernel Bayes filter* of Fukumizu, Song, and Gretton (2013) carries out the same recursion with the belief represented as a mean embedding \(\mu_{s_t\mid o_{1:t}}\) and both models learned nonparametrically from training trajectories, alternating two steps.

:::: {.algorithm #algo-33-3}
[Algorithm (kernel Bayes filter, one step)]{.box-title}

::: algo-io
[Input]{.algo-lab} Belief weights for the embedding \(\mu_{s_t\mid o_{1:t}}\); a transition conditional embedding \(\mathcal C_{s_{t+1}\mid s_t}\) and an observation model, both estimated from training trajectories; the new observation \(o_{t+1}\).

[Output]{.algo-lab} Updated belief weights for the embedding \(\mu_{s_{t+1}\mid o_{1:t+1}}\).
:::

1.  Prediction: push the current belief through the transition operator with the kernel sum rule, \(\mu_{s_{t+1}\mid o_{1:t}}=\mathcal C_{s_{t+1}\mid s_t}\,\mu_{s_t\mid o_{1:t}}\).
2.  Correction: fold in \(o_{t+1}\) with Kernel Bayes' Rule, using the predicted embedding as the prior, to obtain the filtered belief \(\mu_{s_{t+1}\mid o_{1:t+1}}\).
::::

Nothing in this recursion writes down a density or assumes linearity or Gaussianity; the dynamics and the observation model live entirely in Gram matrices built from data. That is what makes the kernel Bayes filter attractive when the state space is a complex object, when the transition and observation models are unknown but sampled trajectories are available, or when the true dynamics are strongly non-Gaussian. The same operator machinery also underpins kernel treatments of causal effect estimation, where a conditional embedding stands in for an interventional distribution, taken up in [[ch:causal-inference-with-kernels|the causal inference chapter]]. The survey of Muandet et al. (2017) catalogues these inference applications and the assumptions each one needs.

## Summary {#summary}

Conditioning, the workhorse of probabilistic modeling, becomes a linear operator on embeddings. The conditional mean embedding \(\mu_{Y\mid X=x}=\mathbb E[\psi(Y)\mid X=x]\) stores every conditional expectation \(\mathbb E[g(Y)\mid X=x]\) as an inner product, and it is produced by a single operator \(\mathcal C_{Y\mid X}=\mathcal C_{YX}\mathcal C_{XX}^{-1}\) applied to the input feature. That population identity hides two hazards, an unbounded covariance inverse and a restrictive range assumption, both cured by regularization, giving the empirical estimate \(\widehat{\mathcal C}_{Y\mid X}=\Upsilon(K+n\lambda I)^{-1}\Phi^{\ast}\), whose weights \(\boldsymbol\beta(x)=(K+n\lambda I)^{-1}\mathbf k_x\) are a kernel ridge regression. Grünewälder et al. (2012) made that precise: the conditional mean embedding is vector-valued regression of the output feature on the input feature, which frees it from the range assumption and delivers consistency. Lifting the sum rule, chain rule, and Bayes' rule to operators gives kernel probabilistic inference, culminating in Kernel Bayes' Rule, \(\mathbf w(y)=DL((DL)^2+\delta I)^{-1}D\boldsymbol\ell_y\), which updates a prior embedding into a posterior embedding using only a joint sample. Chaining prediction by the sum rule with correction by Bayes' rule yields the kernel Bayes filter, a nonparametric recursion for state-space inference. The honest thread throughout is that these operator inverses are unbounded in the population and must be regularized in the sample, so the outputs are embeddings, consistent as the sample grows but biased at finite \(n\) and not literal probability distributions.

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

For **Conditional Mean Embeddings and Kernel Bayes' Rule**, never read \(\mathcal C_{YX}\mathcal C_{XX}^{-1}\) as an everywhere-defined bounded operator without a range argument. Use the regularized estimator, solve rather than invert, and report conditioning and weight diagnostics. Keep three objects separate: the exact conditional embedding, its regularized population approximation, and its finite-sample ridge estimate. Kernel Bayes weights may be signed and need not sum to one; silently normalizing them creates a different estimator, not a repaired theorem.

## Summary and further reading {#summary-and-further-reading}

Song et al. [@song2009cme] develop empirical conditional embeddings, while Baker [@baker1973] and Fukumizu et al. [@fukumizu2004] supply the operator background. The safest practical route is the regression route: declare the vector-valued prediction target, regularize the Gram solve, validate conditional queries on held-out pairs, and treat the operator calculus as a compact language for composing those fitted maps rather than as permission to invert compact population operators.

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} Starting from the definition \(\mu_{Y\mid X=x}=\mathbb E[\psi(Y)\mid X=x]\) with \(\psi(y)=\ell(y,\cdot)\), use the reproducing property of \(\mathcal H_{\mathcal Y}\) to show the generalized conditional reproducing property \(\langle g,\mu_{Y\mid X=x}\rangle=\mathbb E[g(Y)\mid X=x]\) for every \(g\in\mathcal H_{\mathcal Y}\). Specialize to \(g=\ell(y_0,\cdot)\) and read off what the number \(\mu_{Y\mid X=x}(y_0)\) computes. In one sentence, say how this generalizes the marginal embedding of [[ch:kernel-mean-embeddings|the mean-embedding chapter]].
2.  [computation]{.ex-tag} Use the setup of the worked conditional-embedding example (same four points, kernels, and \(n\lambda=0.5\)), but move the test input to \(x_\ast=2.5\). Form \(\mathbf k_{x_\ast}\), solve \((K+0.5\,I)\boldsymbol\beta=\mathbf k_{x_\ast}\), and report \(\widehat{\mathbb E}[Y\mid X=2.5]=\sum_i\beta_i y_i\) together with \(\sum_i\beta_i\). Explain, from where \(x_\ast\) now sits, which two training points dominate the weights, and why the weight sum is again below one.
3.  [proof]{.ex-tag} Prove the operator identity \(\mathcal C_{Y\mid X}\mathcal C_{XX}=\mathcal C_{YX}\) under the assumption that \(h_g(x):=\mathbb E[g(Y)\mid X=x]\in\mathcal H_{\mathcal X}\) for every \(g\in\mathcal H_{\mathcal Y}\). State clearly where the tower property \(\mathbb E[f(X)g(Y)]=\mathbb E[f(X)h_g(X)]\) enters, and explain in a sentence why dropping the assumption breaks the argument.
    Hint

    ::: hint-body
    The reproducing property forces the adjoint to act as \(\mathcal C_{Y\mid X}^{\ast}g=h_g\). Test \(\mathcal C_{XX}\mathcal C_{Y\mid X}^{\ast}g=\mathcal C_{XX}h_g\) against an arbitrary \(f\in\mathcal H_{\mathcal X}\): \(\langle f,\mathcal C_{XX}h_g\rangle=\mathbb E[f(X)h_g(X)]=\mathbb E[f(X)g(Y)]=\langle f,\mathcal C_{YX}^{\ast}g\rangle\). Take adjoints, using \(\mathcal C_{XX}=\mathcal C_{XX}^{\ast}\). Without \(h_g\in\mathcal H_{\mathcal X}\) the quantity \(\mathcal C_{XX}h_g\) is not even defined, so no exact operator exists.
    :::
4.  [proof]{.ex-tag} Establish the push-through identity used to pass from operators to Gram matrices: for a bounded operator \(\Phi\) and any \(c\gt 0\), \(\Phi^{\ast}(\Phi\Phi^{\ast}+cI)^{-1}=(\Phi^{\ast}\Phi+cI)^{-1}\Phi^{\ast}\). Use it to show \(\widehat{\mathcal C}_{YX}(\widehat{\mathcal C}_{XX}+\lambda I)^{-1}=\Upsilon(K+n\lambda I)^{-1}\Phi^{\ast}\), identifying \(\Phi^{\ast}\Phi=K\).
    Hint

    ::: hint-body
    From the identity \(\Phi^{\ast}(\Phi\Phi^{\ast}+cI)=(\Phi^{\ast}\Phi+cI)\Phi^{\ast}\), left-multiply by \((\Phi^{\ast}\Phi+cI)^{-1}\) and right-multiply by \((\Phi\Phi^{\ast}+cI)^{-1}\). Then \(\widehat{\mathcal C}_{YX}=\tfrac1n\Upsilon\Phi^{\ast}\) and \(\widehat{\mathcal C}_{XX}=\tfrac1n\Phi\Phi^{\ast}\), and the \(\tfrac1n\) factors turn \(\lambda\) into \(n\lambda\); the identity moves \(\Phi^{\ast}\) across so the inverse acts on the \(n\times n\) matrix \(K=\Phi^{\ast}\Phi\).
    :::
5.  [computation]{.ex-tag} Derive the finite-sample kernel sum rule. With a prior embedding \(\widehat\mu^{\pi}_X=\Phi\mathbf m\) carried by weights \(\mathbf m\), show that \(\widehat\mu^{\pi}_Y=\widehat{\mathcal C}_{Y\mid X}\widehat\mu^{\pi}_X=\sum_i\rho_i\psi(y_i)\) with \(\boldsymbol\rho=(K+n\lambda I)^{-1}K\mathbf m\). Then, in the setup of the Kernel Bayes' Rule example, replace the tilted prior by the uniform prior \(m_i=1/5\), recompute \(\boldsymbol\rho\), and say which direction the sum-rule weights shift and why.
6.  [exploration]{.ex-tag} Rerun the Kernel Bayes' Rule example (same sample, kernels, prior, and regularizers) but with the observation changed to \(y=3.0\). Compute the posterior weights \(\mathbf w(y)\) and the normalized posterior mean of \(X\). Compare with the prior mean \(3.15\) and with the \(y=1.0\) result, and explain how prior and likelihood now reinforce rather than oppose each other. Comment on any negative weight that appears.
    Hint

    ::: hint-body
    Now \(\boldsymbol\ell_y\) peaks at the pair with \(y_i=3\), so the likelihood points where the prior already leans; the posterior mean should land near \(3\), close to but not exactly the prior mean, since the likelihood is informative. Edit `yobs` in `checks/ch-cme-ex2.py` to reproduce every number.
    :::
7.  [challenge]{.ex-tag} Derive the finite-sample Kernel Bayes' Rule weights from the operator form \(\mu^{\pi}_{X\mid Y=y}=\mathcal C^{\pi}_{XY}((\mathcal C^{\pi}_{YY})^2+\delta I)^{-1}\mathcal C^{\pi}_{YY}\psi(y)\), where \(\mathcal C^{\pi}_{XY}=\Phi D\Upsilon^{\ast}\) and \(\mathcal C^{\pi}_{YY}=\Upsilon D\Upsilon^{\ast}\) with \(D=\mathrm{diag}(\boldsymbol\rho)\). Show that the posterior embedding is \(\Phi\mathbf w(y)\) with \(\mathbf w(y)=DL((DL)^2+\delta I)^{-1}D\boldsymbol\ell_y\), where \(\boldsymbol\ell_y=\Upsilon^{\ast}\psi(y)\). Then explain why the square \((\mathcal C^{\pi}_{YY})^2\) is inverted rather than \(\mathcal C^{\pi}_{YY}\) itself.
    Hint

    ::: hint-body
    Seek the vector \(z=((\mathcal C^{\pi}_{YY})^2+\delta I)^{-1}\mathcal C^{\pi}_{YY}\psi(y)\) in the range of \(\Upsilon\), writing \(z=\Upsilon\mathbf c\); the component orthogonal to that range is killed by \(\delta I\) and must vanish. Substituting gives \(((DL)^2+\delta I)\mathbf c=D\boldsymbol\ell_y\), and then \(\mathcal C^{\pi}_{XY}z=\Phi DL\mathbf c\). The square is inverted because \(\mathcal C^{\pi}_{YY}\) is built from the signed weights \(\boldsymbol\rho\) and need not be positive, whereas \((\mathcal C^{\pi}_{YY})^2+\delta I\) is always positive definite.
    :::
:::
