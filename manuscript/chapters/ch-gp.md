---
id: ch-gp
slug: gaussian-processes-and-rvm
title: Gaussian Processes and the RVM
part: XI · The Bayesian View
order: 42
tier: core
prerequisites:
  - kernels-and-deep-learning
objectives:
  - >-
    Explain the central definitions and claims in Gaussian Processes and the
    RVM.
  - Apply the chapter's principal methods and interpret their outputs.
  - >-
    State the assumptions behind formal results and connect them to earlier
    chapters.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-gp.yml
verification_date: null
bibliography:
  - rasmussen2006
  - williams1996
  - neal1996
  - tipping2001
  - mackay1992
  - scholkopf2002
  - williams1998classification
  - quinonero2005
  - snelson2006fitc
  - titsias2009svgp
  - hensman2013bigdata
  - wilson2013
  - damianou2013deepgp
---
# Gaussian Processes and the RVM

<p class="lead">A forecast that reads 0.8 could mean 0.8 give or take 0.01, or 0.8 give or take 1.0; the decisions those two numbers justify are entirely different, and nothing built so far can tell them apart. Every method to this point writes down a loss, adds a regularizer, and minimizes, returning a single curve that is silent about its own reliability. The Bayesian route reaches the same algorithms from the opposite side: it places a probability distribution over functions before any data arrive, then updates that belief with Bayes' rule once the data are in. The answer is no longer a single curve but a whole posterior distribution, so every prediction comes with an honest error bar, and the regularizer that seemed like a free modelling choice turns into a prior, with the object doing the work again a kernel, this time read as a covariance. The centerpiece of this chapter is a single identity: the Gaussian process posterior mean is exactly the kernel ridge regression fit of [[ch:kernel-ridge-and-friends|the ridge chapter]], with the noise variance playing the role of the regularization strength. Around that identity we develop marginal-likelihood learning of hyperparameters, Gaussian process classification, and the sparse Bayesian prior behind the Relevance Vector Machine.</p>

## The Bayesian view: a prior over functions {#bayesian-view}

The framework of risk minimization asks which function best fits the data under a complexity penalty. The Bayesian framework asks a different question: given a prior belief about which functions are plausible, and given a model of how observations are generated from a function, how plausible is each function after we see the data? The two ingredients are a likelihood and a prior, and the machinery that combines them is Bayes' rule. This account follows Schölkopf and Smola (2002) and MacKay (1992).

Suppose we observe inputs \(X=(x_1,\dots,x_m)\) and targets \(Y=(y_1,\dots,y_m)\), and we posit a hypothesis \(f\) together with a model \(P(y\mid x,f(x))\) of how each label is produced from the underlying value \(f(x)\). The prototypical case is additive noise, \(y=f(x)+\xi\), so that \(P(y\mid x, f(x))=P(y-f(x))\). Under the usual independence assumption the likelihood factorizes,

$$P(Y\mid X,f)=\prod_{i=1}^m P(y_i\mid x_i,f(x_i)).$$

The likelihood tells us how probable the observed sample is if \(f\) is responsible for it. The prior \(p(f)\) encodes what we believe before seeing data: a preference for smooth functions, for small values, or for a particular correlation structure among the values \(f(x_i)\).

Bayes' rule turns the likelihood around. Since \(p(Y\mid f,X)p(f)=p(f\mid X,Y)\,p(Y)\), and \(p(Y)\) does not depend on \(f\), the posterior over hypotheses is

$$p(f\mid X,Y)\ \propto\ p(Y\mid f,X)\,p(f).$$

To predict at a new location \(x\) we integrate the label model against the posterior,

$$p(y\mid X,Y,x)=\int p(y\mid f,x)\,p(f\mid X,Y)\,df,$$

which yields not a point estimate but a predictive distribution. When this integral is intractable, a common shortcut is the maximum a posteriori (MAP) estimate, the mode of the posterior,

$$f_{\mathrm{MAP}}=\arg\min_f\big[-\ln p(Y\mid f,X)-\ln p(f)\big].$$

Read the two terms: the negative log likelihood is a data-fit term and the negative log prior is a penalty independent of the data. This is precisely the regularized risk functional \(R_{\mathrm{emp}}[f]+\Omega[f]\), with the loss identified as the negative log likelihood and the regularizer as the negative log prior. Squared loss corresponds to Gaussian noise, and a quadratic RKHS penalty corresponds to a Gaussian prior. The correspondence is exact at the level of the mode; the Bayesian side additionally offers the mean and the variance, which risk minimization has no counterpart for.

## The Gaussian process {#gaussian-processes}

To put a prior on a function we cannot write down a density on an infinite-dimensional space directly. The trick, going back to Neal (1996) and Williams and Rasmussen (1996), is to specify the prior only through the joint distribution of the function values at any finite set of points, and to make that joint distribution Gaussian. A collection of function values that is jointly Gaussian for every finite subset of locations is a Gaussian process.

:::: {.definition #def-40-1}
[Definition (Gaussian process)]{.box-title}

A stochastic process \(t(x)\) indexed by \(x\in\mathcal X\) is a *Gaussian process* if, for every finite set \(x_1,\dots,x_m\in\mathcal X\), the vector \((t(x_1),\dots,t(x_m))\) is normally distributed. It is determined by a mean function \(\mu(x)=\mathbb E[t(x)]\) and a covariance function

$$k(x,x')=\operatorname{cov}\big(t(x),t(x')\big),\qquad K_{ij}=k(x_i,x_j).$$

We write \((t(x_1),\dots,t(x_m))\sim\mathcal N(\mu,K)\), and take \(\mu\equiv 0\) unless stated otherwise.
::::

The covariance function is not an arbitrary symmetric function. For any coefficients \(c\in\mathbb R^m\) the variance of the linear combination \(\sum_i c_i t(x_i)\) is nonnegative,

$$0\le\operatorname{Var}\Big(\sum_i c_i t(x_i)\Big)=\sum_{i,j}c_i c_j\operatorname{cov}\big(t(x_i),t(x_j)\big)=c^\top K c,$$

so \(K\) is positive semidefinite for every choice of points. In other words the covariance function of a Gaussian process is exactly a positive definite kernel in the sense of [[ch:kernels-now|the kernel chapter]], and every Mercer kernel is a legitimate covariance. This is the bridge: choosing a covariance is choosing a kernel, and the smoothness properties the kernel enforces in an RKHS reappear as the correlation structure of the prior. A zero-mean prior on the values \(t=(t(x_1),\dots,t(x_m))\) reads

$$p(t)=(2\pi)^{-m/2}(\det K)^{-1/2}\exp\!\Big(-\tfrac12 t^\top K^{-1}t\Big),$$

and \(-\ln p(t)=\tfrac12 t^\top K^{-1}t+\text{const}\) is the same quadratic form that the RKHS regularizer \(\|f\|_{\mathcal H}^2\) produces (Schölkopf and Smola 2002). Kernels favoring slowly varying functions, such as the Gaussian covariance \(k(x,x')=\exp(-\|x-x'\|^2/2\ell^2)\), give priors that assign high probability to smooth realizations, as one sees by expanding \(t\) in the eigenvectors of \(K\): directions with large eigenvalue \(\lambda_i\) are cheap because \(t^\top K^{-1}t\) weights them by \(1/\lambda_i\), so the prior prefers the smooth, large-eigenvalue modes. The link back to [[ch:mercer-and-rates|the Mercer eigen-analysis]] is direct: the spectrum of \(K\) ranks hypotheses by prior plausibility.

## Gaussian process regression {#gp-regression}

In regression we do not observe the latent function directly. We observe it through additive Gaussian noise, \(y_i=t(x_i)+\xi_i\) with \(\xi_i\sim\mathcal N(0,\sigma^2)\) independent. Because a sum of independent Gaussians is Gaussian, the observed vector \(y\) and any future latent value are jointly Gaussian, and everything we need follows from one fact about Gaussians: conditioning a joint Gaussian on part of its coordinates produces another Gaussian, with closed-form mean and covariance.

Write \(k_*=(k(x_1,x_*),\dots,k(x_m,x_*))^\top\) for the vector of covariances between the training points and a test point \(x_*\), and \(k_{**}=k(x_*,x_*)\). Since \(y_i=t(x_i)+\xi_i\), the training observations have covariance \(K+\sigma^2 I\), while the latent test value \(f_*=t(x_*)\) has covariance \(k_*\) with the training values and variance \(k_{**}\) with itself. The joint law is

$$\begin{pmatrix} y\\ f_*\end{pmatrix}\sim\mathcal N\!\left(0,\ \begin{pmatrix} K+\sigma^2 I & k_*\\ k_*^\top & k_{**}\end{pmatrix}\right).$$

The Gaussian conditioning formula, that \((a\mid b)\) has mean \(\Sigma_{ab}\Sigma_{bb}^{-1}b\) and covariance \(\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}\), gives the predictive distribution of \(f_*\) given the data at once.

:::: {.theorem #thm-40-2}
[Theorem (GP predictive equations)]{.box-title}

Under a zero-mean Gaussian process prior with covariance \(k\) and additive Gaussian noise of variance \(\sigma^2\), the posterior over the latent value at a test point \(x_*\) is Gaussian, \(f_*\mid X,Y,x_*\sim\mathcal N(\bar m(x_*),v(x_*))\), with

$$\bar m(x_*)=k_*^\top (K+\sigma^2 I)^{-1}y,\qquad v(x_*)=k_{**}-k_*^\top (K+\sigma^2 I)^{-1}k_*.$$

The predictive distribution of a noisy observation \(y_*\) adds \(\sigma^2\) to \(v(x_*)\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

::: {.proof}
[Proof]{.box-title}

Apply the conditioning identity with \(a=f_*\) and \(b=y\), reading the blocks off the joint covariance: \(\Sigma_{bb}=K+\sigma^2 I\), \(\Sigma_{ab}=k_*^\top\), \(\Sigma_{aa}=k_{**}\). The conditional mean is \(\Sigma_{ab}\Sigma_{bb}^{-1}b=k_*^\top(K+\sigma^2 I)^{-1}y\) and the conditional variance is \(\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}=k_{**}-k_*^\top(K+\sigma^2 I)^{-1}k_*\). For a noisy observation \(y_*=f_*+\xi_*\) with independent \(\xi_*\sim\mathcal N(0,\sigma^2)\), the variance of \(f_*\) and \(\xi_*\) add, giving \(v(x_*)+\sigma^2\). [\(\square\)]{.qed}
:::

Three features of these equations deserve emphasis. The mean is a linear combination of kernel functions centered at the training points, \(\bar m(x_*)=\sum_i \alpha_i k(x_i,x_*)\) with \(\alpha=(K+\sigma^2 I)^{-1}y\), so the posterior mean lives in the span of the training kernels exactly as the representer theorem predicts. The variance does not depend on the observed targets \(y\) at all, only on where the data were taken: it shrinks near training points and grows in the gaps, which is what a principled error bar should do. And the noise variance \(\sigma^2\) is added to the diagonal of \(K\), which is what keeps the inverse well conditioned. Rasmussen and Williams (2006) is the standard reference for this development.

:::: {.algorithm #algo-40-1}
[Algorithm (GP regression)]{.box-title}

::: algo-io
[Input]{.algo-lab} Training data \(X,y\); covariance function \(k\) with hyperparameters \(\theta\); noise variance \(\sigma^2\); test point \(x_*\).

[Output]{.algo-lab} Predictive mean \(\bar m(x_*)\), variance \(v(x_*)\), and (optionally) a hyperparameter gradient step.
:::

1.  Form the Gram matrix \(K\) with \(K_{ij}=k(x_i,x_j)\) and set \(A=K+\sigma^2 I\).
2.  Compute the Cholesky factor \(A=LL^\top\) (numerically stabler than inverting \(A\)).
3.  Solve \(A\alpha=y\) by two triangular solves against \(L\).
4.  Form \(k_*=(k(x_1,x_*),\dots,k(x_m,x_*))^\top\); return \(\bar m(x_*)=k_*^\top\alpha\).
5.  Solve \(Lw=k_*\); return \(v(x_*)=k(x_*,x_*)-w^\top w\).
6.  To learn hyperparameters, compute the log marginal likelihood \(\mathcal L(\theta,\sigma^2)\) and its gradient \(\partial\mathcal L/\partial\theta_j=\tfrac12\,y^\top A^{-1}(\partial A/\partial\theta_j)A^{-1}y-\tfrac12\operatorname{tr}\!\big(A^{-1}\partial A/\partial\theta_j\big)\), take a gradient-ascent step, and repeat from step 1.
::::

::::: {.example #example-40-1}
[Example (GP posterior on three points)]{.box-title}

:::: wex
::: wex-setup
Training inputs \(x=(0,1,2)\), targets \(y=(1,\ 0.5,\ -0.5)\), Gaussian kernel \(k(x,x')=e^{-(x-x')^2/2}\) (length scale \(\ell=1\)), noise variance \(\sigma^2=0.1\). Predict at \(x_*=0.5\). All numbers from `checks/ch-gp-ex1.py`.
:::

1.  [Form the Gram matrix.]{.wex-op} With \(e^{-1/2}=0.6065\) and \(e^{-2}=0.1353\),

$$K=\begin{pmatrix}1&0.6065&0.1353\\0.6065&1&0.6065\\0.1353&0.6065&1\end{pmatrix},\qquad A=K+0.1\,I.$$
2.  [Solve for the coefficients.]{.wex-op} \(\alpha=A^{-1}y=(0.7321,\ 0.5046,\ -0.8228)\).
3.  [Evaluate the test covariances.]{.wex-op} \(k_*=(e^{-1/8},e^{-1/8},e^{-9/8})=(0.8825,\ 0.8825,\ 0.3247)\).
4.  [Read off the mean.]{.wex-op} \(\bar m(x_*)=k_*^\top\alpha=0.8242\).
5.  [Read off the variance.]{.wex-op} \(v(x_*)=k_{**}-k_*^\top A^{-1}k_*=1-0.9176=0.0824\), so the posterior standard deviation is \(0.287\).
6.  [Score the fit.]{.wex-op} The log marginal likelihood is \(-\tfrac12 y^\top A^{-1}y-\tfrac12\ln\det A-\tfrac{3}{2}\ln 2\pi=-3.2002\), using \(y^\top A^{-1}y=1.3958\) and \(\ln\det A=-0.509\).

**Reading.** The prediction at \(x_*=0.5\) is \(0.82\pm0.29\). The error bar is tight here because \(x_*\) sits close to two training points; it would widen in a region with no data, since the variance ignores \(y\) and tracks only the geometry of the inputs.
::::

**Verification artifact.** checks/example-ch-gp-example-40-1.json records the example source hash and verification scope.
:::::

## Kernel ridge regression is the GP posterior mean {#krr-gp}

The coefficient vector \(\alpha=(K+\sigma^2 I)^{-1}y\) should look familiar. In [[ch:kernel-ridge-and-friends|the ridge chapter]] the kernel ridge solution was \(\alpha=(K+\lambda n I)^{-1}y\), where \(n\) is the number of training points and \(\lambda\) the regularization strength, and the fitted function was \(\hat f(x_*)=\sum_i\alpha_i k(x_i,x_*)=k_*^\top\alpha\). The GP posterior mean is the identical expression with \(\sigma^2\) in place of \(\lambda n\). This is not a coincidence; it is the MAP-equals-mean phenomenon made concrete.

:::: {.proposition #prop-40-3}
[Proposition (KRR / GP correspondence)]{.box-title}

Kernel ridge regression with regularization parameter \(\lambda\) on \(n\) points produces the same predictor as the Gaussian process posterior mean with noise variance

$$\sigma^2=\lambda n.$$

Concretely, \(\hat f_{\mathrm{KRR}}(x_*)=k_*^\top(K+\lambda n I)^{-1}y=k_*^\top(K+\sigma^2 I)^{-1}y=\bar m(x_*)\) for every \(x_*\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

With additive Gaussian noise the negative log posterior in coefficient space is, up to constants,

$$-\ln p(t\mid y)=\frac{1}{2\sigma^2}\|y-K\alpha\|^2+\frac12\alpha^\top K\alpha,$$

writing \(t=K\alpha\). This is the regularized least-squares objective whose stationarity condition is \(K[(K+\sigma^2 I)\alpha-y]=0\), solved by \(\alpha=(K+\sigma^2 I)^{-1}y\). The ridge objective \(\tfrac1n\|K\alpha-y\|^2+\lambda\,\alpha^\top K\alpha\) has stationarity condition \(K[(K+\lambda n I)\alpha-y]=0\), solved by \(\alpha=(K+\lambda n I)^{-1}y\). The two coincide iff \(\sigma^2=\lambda n\), and since the predictor is \(k_*^\top\alpha\) in both cases, the functions agree everywhere. Because the noise is Gaussian the posterior is Gaussian, so its mode and mean coincide, which is why the MAP fit (ridge) equals the posterior mean (GP). [\(\square\)]{.qed}
::::

::::: {.example #example-40-2}
[Example (the two fits agree)]{.box-title}

:::: wex
::: wex-setup
Same kernel \(k(x,x')=e^{-(x-x')^2/2}\) on \(x=(0,1,2)\), now with \(y=(1,0,-1)\). Ridge parameter \(\lambda=0.05\), \(n=3\), so \(\lambda n=0.15\); set the GP noise variance to the matched value \(\sigma^2=0.15\). Numbers from `checks/ch-gp-ex2.py`.
:::

1.  [Solve the ridge system.]{.wex-op} \(\alpha_{\mathrm{KRR}}=(K+0.15\,I)^{-1}y=(0.9855,\ 0,\ -0.9855)\). The middle coefficient vanishes because both \(y\) and the geometry are antisymmetric about \(x=1\).
2.  [Solve the GP system.]{.wex-op} \(\alpha_{\mathrm{GP}}=(K+0.15\,I)^{-1}y=(0.9855,\ 0,\ -0.9855)\); the maximum coefficient difference is exactly \(0\).
3.  [Predict at two test points.]{.wex-op} At \(x_*=0.5\): \(\hat f_{\mathrm{KRR}}=\bar m=0.549782\). At \(x_*=1.5\): \(\hat f_{\mathrm{KRR}}=\bar m=-0.549782\). The differences are \(0\) to machine precision.

**Reading.** The two algorithms, derived from a loss-plus-penalty on one side and from Bayes' rule on the other, return the same numbers to the last digit. The Gaussian process adds only what ridge cannot supply: a variance at each test point.
::::

**Verification artifact.** checks/example-ch-gp-example-40-2.json records the example source hash and verification scope.
:::::

## The marginal likelihood and hyperparameter learning {#marginal-likelihood}

A kernel comes with knobs: the length scale of a Gaussian covariance, the noise level \(\sigma^2\), an overall amplitude. In risk minimization these are set by cross-validation. The Bayesian framework offers an internal alternative. Collect the hyperparameters into \(\theta\) and integrate the latent function out of the joint distribution; because the prior on \(t\) is Gaussian and the noise is Gaussian, the marginal distribution of the data is Gaussian too, \(y\mid\theta\sim\mathcal N(0,\,K_\theta+\sigma^2 I)\). Its density, viewed as a function of \(\theta\), is the marginal likelihood, or evidence, and maximizing it is the type-II maximum likelihood estimate (MacKay 1992).

:::: {.definition #def-40-4}
[Definition (log marginal likelihood)]{.box-title}

With \(A=K_\theta+\sigma^2 I\), the log marginal likelihood is

$$\mathcal L(\theta)=\ln p(y\mid X,\theta)=-\frac12\,y^\top A^{-1}y-\frac12\ln\det A-\frac{n}{2}\ln(2\pi).$$
::::

The three terms tell the whole story of why this works. The quadratic term \(-\tfrac12 y^\top A^{-1}y\) rewards fitting the data. The log-determinant term \(-\tfrac12\ln\det A\) is a complexity penalty: a flexible covariance that could explain almost anything has a large determinant and is charged for it. Maximizing their sum trades fit against complexity automatically, an instance of Occam's razor built into the arithmetic, with no held-out set required. Differentiating with respect to a hyperparameter \(\theta_j\) gives, by the standard identities \(\partial\ln\det A=\operatorname{tr}(A^{-1}\partial A)\) and \(\partial A^{-1}=-A^{-1}(\partial A)A^{-1}\),

$$\frac{\partial\mathcal L}{\partial\theta_j}=\frac12\,y^\top A^{-1}\frac{\partial A}{\partial\theta_j}A^{-1}y-\frac12\operatorname{tr}\!\Big(A^{-1}\frac{\partial A}{\partial\theta_j}\Big),$$

which is the gradient step in the GP regression algorithm above. Gradient ascent or a Newton method then finds a (local) evidence maximum. The cost is dominated by the \(O(n^3)\) factorization of \(A\), which is why scalable approximations, sparse and low-rank, are the subject of [[ch:large-scale-kernels|the large-scale chapter]].

A refinement of this idea is automatic relevance determination. Give each input dimension its own length scale, \(k(x,x')=\exp\!\big(-\sum_{d}(x_d-x'_d)^2/2\ell_d^2\big)\), and let the evidence choose them. A dimension that is irrelevant to the target drives its \(1/\ell_d^2\) toward zero, effectively removing that input. The same principle, applied not to input dimensions but to basis functions, is what makes the Relevance Vector Machine sparse.

## Gaussian process classification {#gp-classification}

For classification the target is a label \(y\in\{-1,+1\}\), and the additive-Gaussian story breaks: a Gaussian latent value cannot be a Bernoulli label. We keep the Gaussian process prior on a latent function \(t(x)\) but squash it through a non-Gaussian link, typically the logistic \(P(y=1\mid t)=\sigma(t)=(1+e^{-t})^{-1}\) or the probit \(P(y=1\mid t)=\Phi(t)\). The posterior over the latent values is now

$$p(t\mid X,Y)\ \propto\ \Big[\prod_{i=1}^m p(y_i\mid t(x_i))\Big]\exp\!\Big(-\tfrac12 t^\top K^{-1}t\Big),$$

which is no longer Gaussian, and the predictive integral has no closed form. Two approximations dominate.

The *Laplace approximation* (Williams and Rasmussen 1996) fits a Gaussian to the posterior at its mode. Writing the negative log posterior as \(\Psi(t)=-\sum_i\ln p(y_i\mid t(x_i))+\tfrac12 t^\top K^{-1}t\), one finds the mode by Newton's method: with \(c\) the gradient of the log likelihood and \(C=-\nabla^2\ln p(y\mid t)\) its (diagonal) negative Hessian, the update is

$$t_{\mathrm{new}}=(K^{-1}+C)^{-1}(C\,t_{\mathrm{old}}+c),$$

equivalent in coefficient form to \(\alpha_{\mathrm{new}}=(KC+I)^{-1}(KC\,\alpha_{\mathrm{old}}+c)\). The curvature \(C\) at the mode then supplies the covariance of the Gaussian approximation, from which predictive probabilities follow. Because the logistic likelihood is log-concave, \(\Psi\) is convex and the mode is unique. A second option replaces the intractable posterior with the best matching Gaussian in a moment-matching sense rather than a curvature sense; expectation propagation is the standard such scheme, and it is generally more accurate than Laplace for the probit link, at comparable cost. A variational method of Jaakkola and Jordan, which sandwiches the logistic between tractable quadratic bounds, is a third route (Schölkopf and Smola 2002). In every case the approximation reduces classification to a sequence of the same linear-algebra operations that regression required, an \(n\times n\) solve per iteration.

## Laplace mechanics: from mode to prediction {#laplace-mechanics}

The previous section stated the Laplace idea in one line, fit a Gaussian at the mode, and moved on. Since this approximation is the workhorse of Gaussian process classification, it deserves to be opened up: which Gaussian is being fitted, how the computation is organized so that it never forms \(K^{-1}\), and what the classifier finally reports. The program was carried through for GP classifiers by Williams and Barber (1998), including the multiclass case with a softmax link; the numerically stable formulation below follows Rasmussen and Williams (2006).

Start from the negative log posterior \(\Psi(t)=-\ln p(Y\mid t)+\tfrac12 t^\top K^{-1}t\), up to an additive constant, and let \(\hat t\) be its minimizer, found by the Newton iteration of the previous section. Because the gradient vanishes at the minimizer, the Taylor expansion of \(\Psi\) around \(\hat t\) has no linear term:

$$\Psi(t)=\Psi(\hat t)+\tfrac12\,(t-\hat t)^\top\big(K^{-1}+C\big)(t-\hat t)+O\big(\|t-\hat t\|^3\big),$$

where \(C=-\nabla^2\ln p(Y\mid\hat t)\) is the likelihood curvature at the mode, diagonal because the likelihood factorizes over points. Dropping the cubic remainder and exponentiating leaves a Gaussian, and that Gaussian is the Laplace approximation:

$$p(t\mid X,Y)\ \approx\ q(t\mid X,Y)=\mathcal N\big(\hat t,\ (K^{-1}+C)^{-1}\big).$$

The covariance adds two curvatures: the prior contributes \(K^{-1}\) and each observation contributes \(C_{ii}\) on the diagonal. For the logistic link \(C_{ii}=\pi_i(1-\pi_i)\le\tfrac14\) with \(\pi_i=\sigma(\hat t_i)\), so a single binary label is only mildly informative, and most informative exactly where the current prediction is undecided at \(\pi_i=\tfrac12\).

The stationarity condition \(\nabla\Psi(\hat t)=0\) reads \(\hat t=K\hat c\) with \(\hat c=\nabla\ln p(Y\mid\hat t)\): the mode is once more a kernel expansion over the training points, the classification counterpart of \(\alpha=(K+\sigma^2I)^{-1}y\) in regression. The coefficients are readable. For the logistic link, \(\hat c_i=y_i\,\sigma(-y_i\hat t_i)\), which is the probability the model currently assigns to the *wrong* class at point \(i\), signed by the label. Confidently fitted points contribute almost nothing to the expansion; uncertain and misfitted points carry it. The Laplace classifier has soft support vectors: the hard sparsity of the SVM is replaced by an exponential taper of influence.

Numerics next. The matrix \(K^{-1}+C\) should never be assembled from \(K^{-1}\), whose computation is exactly what ill-conditioned kernels punish. The right object is

$$B=I+C^{1/2}KC^{1/2},$$

whose eigenvalues lie in \([1,\ 1+\tfrac14\lambda_{\max}(K)]\) for the logistic link: bounded below by one no matter how close to singular \(K\) is. By the Woodbury identity, \((K^{-1}+C)^{-1}=K-KC^{1/2}B^{-1}C^{1/2}K\), so every Newton step and every predictive quantity reduces to Cholesky solves against \(B\), at \(O(n^3)\) per iteration and only a handful of iterations, since the log-concave likelihood makes \(\Psi\) convex.

Prediction assembles the same two Gaussian-conditioning moves as regression, now under the approximate posterior. The latent value at \(x_*\) has approximate mean and variance

$$\mu_*=k_*^\top\hat c,\qquad v_*=k_{**}-k_*^\top C^{1/2}B^{-1}C^{1/2}k_*,$$

the mean following from \(\mathbb E_q[f_*]=k_*^\top K^{-1}\hat t=k_*^\top\hat c\), the variance from \((K+C^{-1})^{-1}=C^{1/2}B^{-1}C^{1/2}\). The reported class probability then averages the link over the latent uncertainty,

$$\bar\pi_*=\int\sigma(f)\,\mathcal N\big(f\mid\mu_*,v_*\big)\,df,$$

a one-dimensional integral: closed form \(\bar\pi_*=\Phi\big(\mu_*/\sqrt{1+v_*}\big)\) for the probit link, a short quadrature for the logistic. The averaging is not decoration. Plugging \(\mu_*\) straight into the sigmoid ignores \(v_*\) and overstates confidence far from the data; averaging pulls \(\bar\pi_*\) toward \(\tfrac12\) exactly where the latent variance says the model does not know.

Finally, the same expansion prices the whole fit: integrating the Gaussian approximation gives an approximate evidence

$$\ln q(Y\mid X,\theta)=-\tfrac12\,\hat t^\top\hat c+\ln p(Y\mid\hat t)-\tfrac12\ln\det B,$$

using \(\hat t^\top K^{-1}\hat t=\hat t^\top\hat c\) at the mode. The three terms replay the regression marginal likelihood: a prior charge, a data fit, and a determinant acting as the Occam factor. Ascending this quantity tunes length scales and amplitudes for classification exactly as the exact evidence did for regression; Williams and Barber (1998) ran this adaptation with the multiclass softmax, where \(C\) becomes block structured but the pipeline is unchanged. The same mechanics reappear wherever a Gaussian prior meets a factorized non-Gaussian likelihood, from robust regression to point-process intensity models.

:::: {.algorithm #algo-40-2}
[Algorithm (Laplace GP classification)]{.box-title}

::: algo-io
[Input]{.algo-lab} Gram matrix \(K\); labels \(y\in\{-1,+1\}^n\); link likelihood \(p(y_i\mid t_i)\); test point \(x_*\).

[Output]{.algo-lab} Predictive class probability \(\bar\pi_*\); approximate evidence \(\ln q(Y\mid X,\theta)\).
:::

1.  Initialize \(t=0\).
2.  Compute the gradient \(c=\nabla\ln p(Y\mid t)\) and curvature \(C=-\nabla^2\ln p(Y\mid t)\); form \(B=I+C^{1/2}KC^{1/2}\) and its Cholesky factor.
3.  Newton step: set \(b=Ct+c\) and update \(t\leftarrow K\big(b-C^{1/2}B^{-1}C^{1/2}Kb\big)\).
4.  Repeat steps 2 and 3 until \(\Psi\) stops decreasing; call the results \(\hat t\), \(\hat c\), \(\hat C\), \(\hat B\).
5.  Predict: \(\mu_*=k_*^\top\hat c\), \(v_*=k_{**}-k_*^\top\hat C^{1/2}\hat B^{-1}\hat C^{1/2}k_*\), and \(\bar\pi_*=\int\sigma(f)\,\mathcal N(f\mid\mu_*,v_*)\,df\) (probit: \(\Phi\big(\mu_*/\sqrt{1+v_*}\big)\)).
6.  Report \(\ln q(Y\mid X,\theta)=-\tfrac12\hat t^\top\hat c+\ln p(Y\mid\hat t)-\tfrac12\ln\det\hat B\) for hyperparameter ascent.
::::

## Sparse Bayesian priors and the Relevance Vector Machine {#laplacian-rvm}

The Gaussian process prior spreads its belief smoothly, and its posterior mean uses every training point: \(\alpha=(K+\sigma^2 I)^{-1}y\) is generically dense. Sometimes we want the opposite, a prediction supported on a handful of basis functions. Sparsity is a statement about the prior, and there are two ways to build it in.

The first keeps a single prior and changes its shape. A *Laplacian process* (Schölkopf and Smola 2002) places an independent Laplace prior on the expansion coefficients of \(f(x)=\sum_i\alpha_i k(x_i,x)\), so that \(-\ln p(\alpha)\propto\sum_i|\alpha_i|\). The MAP estimate then minimizes a data-fit term plus an \(\ell_1\) penalty, a linear or quadratic program whose solutions are genuinely sparse, in the spirit of basis pursuit and the LASSO. Because the prior depends on the data locations through the kernel expansion it is a data-dependent prior, a mild departure from the textbook Bayesian setup, but a fruitful one: it imports \(\ell_1\) sparsity into the kernel world while retaining error bars.

The second way, and the one we develop in detail, is the Relevance Vector Machine of Tipping (2001). Rather than fix the prior variance of the coefficients, give every coefficient its own variance, controlled by its own hyperparameter, and let the evidence decide which variances to shrink to zero.

:::: {.definition #def-40-5}
[Definition (RVM prior)]{.box-title}

Model the target as \(t=K\alpha\) with independent Gaussian noise of variance \(\sigma^2\). Place on each coefficient an independent zero-mean Gaussian prior with its own precision \(s_i\gt 0\),

$$p(\alpha_i\mid s_i)=\sqrt{\tfrac{s_i}{2\pi}}\,\exp\!\Big(-\tfrac12 s_i\alpha_i^2\Big),\qquad p(\alpha\mid s)=\mathcal N(0,\,S^{-1}),\ \ S=\operatorname{diag}(s_1,\dots,s_m),$$

with a flat hyperprior on \(\ln s_i\) (or a broad Gamma prior). Each \(s_i\) is an automatic relevance determination hyperparameter for one basis function.
::::

The precision \(s_i\) is the inverse prior variance of \(\alpha_i\): a large \(s_i\) means \(\alpha_i\) is pinned tightly to zero, a small \(s_i\) lets it roam. With the noise variance \(\sigma^2\) and Gaussian prior fixed, the posterior over \(\alpha\) is Gaussian with covariance and mean

$$\Sigma=\big(\sigma^{-2}K^\top K+S\big)^{-1},\qquad \mu=\sigma^{-2}\Sigma K^\top y.$$

Learning the \(s_i\) proceeds by type-II maximum likelihood, exactly as for GP hyperparameters: the coefficients are integrated out to give the marginal likelihood \(p(y\mid s,\sigma^2)=\mathcal N(0,\ \sigma^2 I+K S^{-1}K^\top)\), which is then maximized over \(s\) and \(\sigma^2\). Setting derivatives to zero yields fixed-point updates.

:::: {.algorithm #algo-40-3}
[Algorithm (RVM update)]{.box-title}

::: algo-io
[Input]{.algo-lab} Design matrix \(K\) (\(K_{ij}=k(x_i,x_j)\)), targets \(y\); initial precisions \(s\), noise \(\sigma^2\).

[Output]{.algo-lab} Sparse mean coefficients \(\mu\); most \(\mu_i=0\).
:::

1.  Form the posterior covariance \(\Sigma=(\sigma^{-2}K^\top K+S)^{-1}\) and mean \(\mu=\sigma^{-2}\Sigma K^\top y\).
2.  For each \(i\), compute the well-determinedness \(\gamma_i=1-s_i\Sigma_{ii}\in[0,1]\).
3.  Update each precision \(s_i\leftarrow\dfrac{\gamma_i}{\mu_i^2}=\dfrac{1-s_i\Sigma_{ii}}{\mu_i^2}\).
4.  Update the noise \(\sigma^2\leftarrow\dfrac{\|y-K\mu\|^2}{m-\sum_i\gamma_i}\).
5.  Whenever \(s_i\to\infty\) (numerically, exceeds a large threshold), prune basis \(i\): delete its row and column and continue. Repeat from step 1 until \(\mu\) stabilizes.
::::

Why does this produce sparsity? The update \(s_i\leftarrow(1-s_i\Sigma_{ii})/\mu_i^2\) is the mechanism. If a basis function contributes little to explaining \(y\), its posterior mean \(\mu_i\) is small while its posterior variance \(\Sigma_{ii}\) stays near the prior variance \(1/s_i\), so \(\gamma_i=1-s_i\Sigma_{ii}\) is near zero and the numerator collapses faster than the denominator: \(s_i\) is driven upward, which shrinks \(\mu_i\) further, which pushes \(s_i\) higher still. The feedback runs away, \(s_i\to\infty\), the prior variance \(1/s_i\to 0\), and \(\alpha_i\) is clamped exactly at zero and removed. Only the basis functions the data genuinely demand survive; Tipping (2001) calls the survivors the relevance vectors. An equivalent view integrates the hyperparameter out: with a Gamma hyperprior, \(\int p(\alpha_i\mid s_i)p(s_i)\,ds_i\) is a Student-\(t\) distribution over \(\alpha_i\), sharply peaked at zero with heavy tails, and a MAP estimate under such a prior naturally sends most coefficients to zero while letting a few grow large. The effect is like the support vectors of an SVM, but the retained points need not lie near the decision boundary, and the solution is typically far sparser. In the large-sample limit the RVM converges to a Gaussian process with a data-dependent covariance, closing the loop with the start of the chapter (Schölkopf and Smola 2002).

The price of this elegance is that the marginal likelihood over the \(s_i\) is highly multimodal, so the fixed-point iteration finds a local optimum and the training is heavier than an SVM's on large data. The reward is a probabilistic, extremely sparse predictor that still carries the full apparatus of Bayesian error bars, and whose predictive mean and variance,

$$\bar y(x_*)=k_*^\top\mu,\qquad v(x_*)=\sigma^2+k_*^\top\Sigma k_*,$$

mirror the Gaussian process formulas of the regression section, now built on only the relevance vectors.

## Sparse and variational Gaussian processes {#sparse-variational-gp}

The Relevance Vector Machine made its predictor sparse because the prior wanted few active basis functions. A second, blunter pressure pushes in the same direction: cost. Every exact GP quantity in this chapter, the posterior mean, the variance, the evidence and its gradient, passes through a factorization of the \(n\times n\) matrix \(K+\sigma^2 I\), at \(O(n^3)\) time and \(O(n^2)\) memory, repeated at every step of hyperparameter learning. Beyond a few tens of thousands of points the exact equations stop being computable. [[ch:large-scale-kernels|The large-scale chapter]] develops the generic remedies, random features and Nyström-type low-rank factorizations, which approximate the *matrix*. The Gaussian process literature developed a complementary line that approximates the *model*: replace the full process by one whose information about the data is carried by \(m\ll n\) well-placed points. The prize for staying inside the probabilistic frame is that variances and evidences survive with their meaning intact, and the quality of the approximation itself becomes a measurable quantity. Throughout this section \(n\) counts training points and \(m\) counts inducing points.

:::: {.definition #def-40-6}
[Definition (inducing points and the Nyström surrogate)]{.box-title}

Let \(Z=(z_1,\dots,z_m)\) be *inducing inputs* in \(\mathcal X\) and let \(u=(t(z_1),\dots,t(z_m))\) be the latent process evaluated there. Write \(K_{uu}\), \(K_{uf}\), \(K_{ff}\) for the kernel matrices among inducing and training inputs (so \(K_{ff}=K\)), and define, for any two blocks of inputs \(a\) and \(b\),

$$Q_{ab}=K_{au}\,K_{uu}^{-1}\,K_{ub}.$$

\(Q_{ab}\) is the covariance that remains when all correlation between \(a\) and \(b\) is forced to pass through \(u\); as a matrix approximation it is exactly the Nyström approximation of \(K_{ab}\) built from the columns at \(Z\).
::::

Conditioning the joint Gaussian prior of \((f,u)\) gives \(p(f\mid u)=\mathcal N\big(K_{fu}K_{uu}^{-1}u,\ K_{ff}-Q_{ff}\big)\), by the same block formula that produced the predictive equations. The conditional covariance \(K_{ff}-Q_{ff}\) is a Schur complement, hence positive semidefinite, and its \(i\)th diagonal entry vanishes whenever \(x_i\) belongs to \(Z\): it measures exactly the part of the prior that the bottleneck \(u\) fails to carry. The classical sparse constructions, unified by Quiñonero-Candela and Rasmussen (2005), all keep the exact \(p(u)=\mathcal N(0,K_{uu})\) and tamper only with this conditional. The *subset of regressors* (SoR) approximation pretends the conditional is deterministic at training and test points alike, \(f=K_{fu}K_{uu}^{-1}u\); the model degenerates to a rank-\(m\) GP with kernel \(Q\), and its variances collapse wherever \(Q\) does. The *deterministic training conditional* (DTC) keeps the deterministic rule at the training points but restores the exact conditional at test points, repairing the predictive variance. The *fully independent training conditional* (FITC) of Snelson and Ghahramani (2006) is subtler: it keeps every training point's exact conditional variance and severs only their correlations, replacing \(K_{ff}-Q_{ff}\) by its diagonal. Each variant is exact Gaussian inference in its modified model, and all share one predictive template. Setting

$$\Lambda_{\mathrm{SoR}}=\Lambda_{\mathrm{DTC}}=\sigma^2 I,\qquad \Lambda_{\mathrm{FITC}}=\operatorname{diag}\big(K_{ff}-Q_{ff}\big)+\sigma^2 I,$$

and \(\Sigma=\big(K_{uu}+K_{uf}\Lambda^{-1}K_{fu}\big)^{-1}\), the predictive law at \(x_*\) is \(\mathcal N(\mu_*,v_*)\) with

$$\mu_*=k_{u*}^\top\,\Sigma\,K_{uf}\Lambda^{-1}y,\qquad v_*=k_{**}-Q_{**}+k_{u*}^\top\,\Sigma\,k_{u*},$$

where \(k_{u*}\) collects the covariances between \(Z\) and \(x_*\) and \(Q_{**}=k_{u*}^\top K_{uu}^{-1}k_{u*}\). SoR alone replaces the honest \(k_{**}\) by \(Q_{**}\), which is what deflates its error bars away from \(Z\). The cost is \(O(nm^2)\) once, then \(O(m)\) per test mean and \(O(m^2)\) per test variance: linear in \(n\), with \(m\) ours to choose.

:::: {.algorithm #algo-40-4}
[Algorithm (sparse GP prediction with inducing points)]{.box-title}

::: algo-io
[Input]{.algo-lab} Training data \(X,y\) (\(n\) points); inducing inputs \(Z\) (\(m\ll n\)); kernel \(k\); noise \(\sigma^2\); family (DTC or FITC).

[Output]{.algo-lab} Predictive mean and variance at any \(x_*\); a training objective for \(\theta\), \(\sigma^2\), \(Z\).
:::

1.  Form \(K_{uu}\) (\(m\times m\)) and \(K_{uf}\) (\(m\times n\)); factor \(K_{uu}\) by Cholesky.
2.  With \(k_{ui}\) the \(i\)th column of \(K_{uf}\), compute the Nyström diagonal \(q_{ii}=k_{ui}^\top K_{uu}^{-1}k_{ui}\) for all \(i\) in \(O(nm^2)\); never form the full \(Q_{ff}\).
3.  Set \(\Lambda=\sigma^2 I\) (DTC) or \(\Lambda=\operatorname{diag}\big(k(x_i,x_i)-q_{ii}\big)+\sigma^2 I\) (FITC); \(\Lambda\) is diagonal.
4.  Factor \(\Sigma^{-1}=K_{uu}+K_{uf}\Lambda^{-1}K_{fu}\) (\(m\times m\)) and cache the weights \(w=\Sigma K_{uf}\Lambda^{-1}y\).
5.  At a test point form \(k_{u*}\) and report \(\mu_*=k_{u*}^\top w\), \(v_*=k_{**}-k_{u*}^\top K_{uu}^{-1}k_{u*}+k_{u*}^\top\Sigma k_{u*}\).
6.  To train, ascend the sparse evidence \(\ln\mathcal N\big(y\mid 0,\ Q_{ff}+\Lambda\big)\) or the variational bound \(\mathcal L_T\) below over \(\theta\), \(\sigma^2\), and the positions \(Z\); each gradient costs \(O(nm^2)\).
::::

### The variational answer: sparsity without changing the model {#titsias-bound}

DTC and FITC are honest algorithms with an awkward epistemology: each is exact inference in a model that is not the one we wrote down. Their evidences are evidences *of the surrogates*, so ascending them over \(Z\) rewards whichever surrogate most flatters the data, not the best approximation to the true posterior; the inducing positions act as extra kernel parameters, free to overfit. Titsias (2009) recast sparsity as approximate inference in the *exact* model. Choose a variational posterior of the constrained form \(q(f,u)=p(f\mid u)\,q(u)\), which may reshape belief over the inducing values but must propagate it to \(f\) through the true conditional, and maximize a lower bound on the exact evidence. Maximizing over \(q(u)\) in closed form collapses the bound to a formula.

:::: {.proposition #prop-40-7}
[Proposition (Titsias, 2009)]{.box-title}

For any inducing inputs \(Z\), the collapsed variational bound

$$\mathcal L_T=\ln\mathcal N\big(y\mid 0,\ Q_{ff}+\sigma^2 I\big)-\frac{1}{2\sigma^2}\operatorname{tr}\big(K_{ff}-Q_{ff}\big)$$

satisfies \(\mathcal L_T\le\ln p(y\mid X)\), with equality whenever \(Q_{ff}=K_{ff}\), in particular when \(Z=X\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

Fix \(u\) and apply Jensen's inequality to the training conditional: \(\ln p(y\mid u)=\ln\mathbb E_{p(f\mid u)}[p(y\mid f)]\ \ge\ \mathbb E_{p(f\mid u)}[\ln p(y\mid f)]\). With \(p(f\mid u)=\mathcal N(a,\,K_{ff}-Q_{ff})\), \(a=K_{fu}K_{uu}^{-1}u\), and \(\ln p(y\mid f)=-\tfrac{1}{2\sigma^2}\|y-f\|^2-\tfrac n2\ln(2\pi\sigma^2)\), the expectation uses \(\mathbb E\|y-f\|^2=\|y-a\|^2+\operatorname{tr}(K_{ff}-Q_{ff})\), giving the pointwise inequality

$$p(y\mid u)\ \ge\ \mathcal N\big(y\mid a,\ \sigma^2 I\big)\,\exp\!\Big(-\tfrac{1}{2\sigma^2}\operatorname{tr}\big(K_{ff}-Q_{ff}\big)\Big).$$

The trace factor does not depend on \(u\). Averaging both sides over \(p(u)=\mathcal N(0,K_{uu})\) and using the linear-Gaussian marginalization \(\int\mathcal N\big(y\mid K_{fu}K_{uu}^{-1}u,\ \sigma^2 I\big)\,\mathcal N(u\mid 0,K_{uu})\,du=\mathcal N\big(y\mid 0,\ Q_{ff}+\sigma^2 I\big)\) gives \(p(y)\ge\mathcal N(y\mid 0,Q_{ff}+\sigma^2I)\,e^{-\operatorname{tr}(K_{ff}-Q_{ff})/(2\sigma^2)}\); take logarithms. If \(Q_{ff}=K_{ff}\) the trace vanishes and the first term is the exact log marginal likelihood, so the inequality is tight; \(Z=X\) gives \(Q_{ff}=K_{ff}K_{ff}^{-1}K_{ff}=K_{ff}\). [\(\square\)]{.qed}
::::

Read \(\mathcal L_T\) term by term. The first term is exactly the DTC evidence. The second charges it \(\tfrac{1}{2\sigma^2}\sum_i\operatorname{Var}[f_i\mid u]\), the total conditional variance the bottleneck throws away. That one trace changes the status of everything: the positions \(Z\), and \(m\) itself, become variational parameters, and improving \(\mathcal L_T\) provably tightens an approximation to one fixed model rather than drifting toward a more convenient model, with the bound only improving as inducing points are added (Titsias 2009). The optimal inducing posterior is available in closed form, \(q^*(u)=\mathcal N\big(\sigma^{-2}K_{uu}\Sigma K_{uf}\,y,\ K_{uu}\Sigma K_{uu}\big)\) with \(\Sigma=(K_{uu}+\sigma^{-2}K_{uf}K_{fu})^{-1}\), and the prediction it induces coincides with the DTC formulas above. Variationally trained sparse GPs therefore predict like DTC; what changes is the objective that places \(Z\) and tunes the hyperparameters.

::::: {.example #example-40-3}
[Example (two inducing points against the full GP)]{.box-title}

:::: wex
::: wex-setup
Training inputs \(x=(-2,\,-1,\,0.5,\,2)\), targets \(y=(-1.7,\,-0.8,\,0.7,\,1.5)\), squared-exponential kernel \(k(x,x')=e^{-(x-x')^2/8}\) (length scale \(\ell=2\)), noise \(\sigma^2=0.1\). Inducing inputs \(Z=(-1,\,0.5)\), the two middle training points, so \(n=4\) and \(m=2\). Predict at \(x_*=0.3\). All numbers from `checks/ch-gp-ex3.py`.
:::

1.  [Fit the exact GP for reference.]{.wex-op}

$$K_{ff}=\begin{pmatrix}1&0.8825&0.4578&0.1353\\0.8825&1&0.7548&0.3247\\0.4578&0.7548&1&0.7548\\0.1353&0.3247&0.7548&1\end{pmatrix},$$

    and the full predictive equations give \(\bar m(x_*)=0.4672\), \(v(x_*)=0.0648\).
2.  [Assemble the Nyström surrogate.]{.wex-op} Since \(Z=(x_2,x_3)\), \(K_{uu}=\begin{pmatrix}1&0.7548\\0.7548&1\end{pmatrix}\) is a principal submatrix of \(K_{ff}\) and \(K_{uf}\) is its rows 2 and 3. Then

$$Q_{ff}=K_{fu}K_{uu}^{-1}K_{uf}=\begin{pmatrix}0.8797&0.8825&0.4578&0.0397\\0.8825&1&0.7548&0.3247\\0.4578&0.7548&1&0.7548\\0.0397&0.3247&0.7548&0.7095\end{pmatrix}.$$

    Rows 2 and 3 reproduce \(K_{ff}\) exactly; the corners degrade, the \(x_1\)-\(x_4\) covariance dropping from \(0.1353\) to \(0.0397\) because it must be carried through \(u\). The discarded diagonal is \(\operatorname{diag}(K_{ff}-Q_{ff})=(0.1203,\,0,\,0,\,0.2905)\), zero exactly at the inducing points, and \(Q_{**}=0.998\) against \(k_{**}=1\).
3.  [Predict with \(\Lambda=\sigma^2 I\) (DTC, equally the Titsias posterior).]{.wex-op} The template gives \(\mu_*=0.5054\) and, keeping \(k_{**}\), \(v_*=0.0443\). SoR substitutes \(Q_{**}\) and reports \(0.0423\).
4.  [Predict with FITC.]{.wex-op} \(\Lambda=\operatorname{diag}(0.2203,\,0.1,\,0.1,\,0.3905)\): the noise floor \(0.1\) plus the discarded variance, inflated exactly at the two poorly covered inputs. The template gives \(\mu_*=0.4364\) and \(v_*=0.0608\).
5.  [Score the sparsification with the variational bound.]{.wex-op} The fit term is \(\ln\mathcal N(y\mid 0,\ Q_{ff}+\sigma^2 I)=-4.982\) and the trace penalty \(\operatorname{tr}(K_{ff}-Q_{ff})/(2\sigma^2)=2.0544\) (the trace itself is \(0.4109\)), so \(\mathcal L_T=-4.982-2.0544=-7.0364\). The exact log marginal likelihood is \(-5.3762\): the bound holds with gap \(1.6602\), while the fit term alone, DTC's own evidence, overshoots the truth.

**Reading.** Half the data serving as inducing set moves the predictive means by a few hundredths, but the error bars tell the finer story: \(0.0423\) (SoR) \(\lt 0.0443\) (DTC) \(\lt 0.0608\) (FITC) against the exact \(0.0648\); the more of the discarded conditional variance a scheme keeps, the closer its uncertainty comes to the truth. The bound line is the operational one. \(\mathcal L_T\) sits \(1.6602\) below the exact evidence, all of it charged by the trace over the two badly covered inputs \(x_1\) and \(x_4\), so an optimizer ascending \(\mathcal L_T\) over \(Z\) is pushed to cover exactly those points, and at \(Z=X\) the bound closes.
::::

**Verification artifact.** checks/example-ch-gp-example-40-3.json records the example source hash and verification scope.
:::::

### Stochastic variational Gaussian processes {#svgp-big-data}

One computational ceiling remains: evaluating \(\mathcal L_T\) or its gradient sweeps all \(n\) points through the \(O(nm^2)\) sums, once per optimizer step. Hensman, Fusi, and Lawrence (2013) removed it by refusing to collapse. Keep \(q(u)=\mathcal N(m_u,S)\) as explicit variational parameters; the uncollapsed evidence lower bound is

$$\mathcal L(m_u,S,Z,\theta)=\sum_{i=1}^{n}\mathbb E_{q(f_i)}\big[\ln p(y_i\mid f_i)\big]\;-\;\mathrm{KL}\big(q(u)\,\big\|\,p(u)\big),$$

where \(q(f_i)\) is Gaussian with mean \(k_{ui}^\top K_{uu}^{-1}m_u\) and variance \(k(x_i,x_i)-q_{ii}+k_{ui}^\top K_{uu}^{-1}SK_{uu}^{-1}k_{ui}\). For the Gaussian likelihood, maximizing over \((m_u,S)\) at fixed \(Z\) recovers \(\mathcal L_T\) exactly, so nothing is conceded. What is gained is the shape of the objective: the data enter only through a sum of independent per-point terms, so a random minibatch of size \(b\), rescaled by \(n/b\), gives an unbiased estimate of the sum, and stochastic gradient ascent applies jointly to \(m_u\), \(S\), \(Z\), and the kernel hyperparameters. One step costs \(O(bm^2+m^3)\), with no dependence on \(n\) at all; the bound moreover fits the framework of stochastic variational inference, where natural-gradient steps in the exponential-family parameterization of \(q(u)\) speed convergence. This is SVGP, demonstrated by Hensman, Fusi, and Lawrence (2013) on hundreds of thousands of flight records with a few hundred inducing points, and it is the template modern GP software implements for big data.

Two further dividends. Because each expectation in the sum is a one-dimensional Gaussian integral, any likelihood whose scalar expectations can be computed or quadratured slots straight in, so the same loop trains GP classifiers with the links of the classification section on data far beyond the Laplace approximation's reach. And the machinery composes: the sparse variational posterior is the substrate on which the deep constructions of the next section stand, and it is what makes Gaussian process surrogates affordable inside larger systems.

## Deep, multi-output, and spectral-mixture processes {#deep-multioutput-gp}

The prior we have used so far is restrictive in three separable ways: the kernel is picked from a small catalogue of fixed stationary forms, the process models a single scalar function, and the function is drawn in one shot rather than built out of simpler stages. Each restriction has a principled lift, and all three run on machinery this chapter has already built, evidence-driven hyperparameter learning and inducing-point inference.

### Learning the kernel: spectral mixtures {#spectral-mixture-kernels}

Bochner's theorem, developed in [[ch:kernel-families|the kernel-families chapter]], says a stationary kernel is the Fourier transform of a nonnegative spectral density: choosing a stationary covariance *is* choosing a distribution over frequencies. Seen through that lens the standard kernels are rigid commitments: the squared-exponential is a single Gaussian density centered at zero frequency, so it can say nothing beyond \"smooth at scale \(\ell\)\", and automatic relevance determination tunes one number per dimension without ever changing the shape. Wilson and Adams (2013) put the flexibility where the theorem says it lives, modelling the spectral density as a mixture of \(Q\) Gaussians. In one dimension the resulting kernel is closed form,

$$k_{\mathrm{SM}}(\tau)=\sum_{q=1}^{Q}w_q\,\exp\big(-2\pi^2\sigma_q^2\tau^2\big)\cos\big(2\pi\mu_q\tau\big),\qquad \tau=x-x',$$

with weights \(w_q\), center frequencies \(\mu_q\), and bandwidths \(\sigma_q\). Each component is a squared-exponential envelope times a cosine: a quasi-periodic pattern at frequency \(\mu_q\) that persists over a range of order \(1/\sigma_q\); taking \(Q=1\) and \(\mu_1=0\) recovers the squared-exponential itself. Since mixtures of Gaussians can approximate any spectral density, spectral-mixture kernels are dense among stationary covariances, and every parameter is learned by ascending the marginal likelihood: kernel learning reduced to the hyperparameter learning of this chapter. The practical payoff Wilson and Adams (2013) demonstrated is extrapolation, the evidence discovering periodic structure and continuing it beyond the data, where a fixed squared-exponential can only fall back to its mean.

### Many outputs: coregionalization {#multi-output-gps}

Often the target is vector valued: \(p\) sensors on one machine, \(p\) related tasks, \(p\) outputs of a simulator. Modelling each coordinate with an independent GP wastes exactly what makes the problem interesting, that the outputs are correlated and data on one should sharpen predictions of another. A Gaussian prior over vector-valued functions needs a matrix-valued covariance, \(\operatorname{cov}\big(f_i(x),f_j(x')\big)\), positive semidefinite in the joint sense. The simplest separable choice, long used in geostatistics under the name co-kriging, is the *intrinsic coregionalization model*

$$\operatorname{cov}\big(f_i(x),f_j(x')\big)=B_{ij}\,k(x,x'),$$

with \(k\) an ordinary kernel on inputs and \(B\) a positive semidefinite \(p\times p\) *coregionalization matrix*; the joint Gram matrix is the Kronecker product \(B\otimes K\), positive semidefinite by the product closure rules. Writing \(B=AA^\top\) with \(A\) of rank \(r\) exposes the generative reading: the \(p\) outputs are linear mixtures of \(r\) shared latent GPs, and the *linear model of coregionalization* sums several such terms with different base kernels. The evidence learns \(B\), which is a task-similarity matrix read off from data, and transfer happens mechanically: an observation of output \(j\) enters the posterior of output \(i\) with weight proportional to \(B_{ij}\). This is the construction behind multi-task surrogates in [[ch:bayesian-optimization-and-bandits|Bayesian optimization]], where cheap evaluations of a related objective narrow the posterior over an expensive one, and Kronecker algebra plus the inducing-point machinery keeps the joint model tractable.

### Depth: compositions of processes {#deep-gps}

Stationarity is a strong commitment: one length scale for the whole input space. A function that undulates in one region and is nearly flat in another fits no stationary prior well. The classical fix warps the inputs through a fixed map \(h\) and models \(f(h(x))\); the radical fix is to learn the warp as a function too, and then there is no reason to stop at one stage. A *deep Gaussian process* (Damianou and Lawrence 2013) composes layers \(f=f_L\circ\cdots\circ f_1\), each layer a GP prior over a map between adjacent feature spaces. The composite is no longer a Gaussian process: its marginals are non-Gaussian, can be multimodal, and its effective smoothness varies with position, which is precisely the expressive gain. The price is inference, since each layer's inputs are the previous layer's random outputs; Damianou and Lawrence (2013) made it tractable by placing inducing points at every layer and nesting the variational bound of the previous section, and later stochastic variants scale the same construction with minibatches. The contrast with Neal (1996) is worth savoring: widening a single hidden layer to infinity collapses a neural network onto one GP with a fixed kernel, while stacking finite GP layers keeps a learned, data-dependent composition. That dialogue, between kernels fixed by an architecture and representations learned through it, is where this book is heading; [[ch:the-frontier|the frontier chapter]] takes it up in earnest through the infinite-width limits of deep networks and the kernels that grow depth.

## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

For **Gaussian Processes and the RVM**, do not apply a displayed formula without checking its domain, statistical assumptions, and numerical conditioning. Avoid selecting kernels or hyperparameters on test data, and do not interpret an optimization residual as a generalization guarantee. When the method is computational, report preprocessing, kernel parameters, regularization, solver tolerance, condition diagnostics, runtime, and a non-kernel baseline. When the result is theoretical, distinguish sufficient conditions from necessary ones and finite-sample claims from asymptotic statements.

## Summary and further reading {#summary-and-further-reading}

This chapter established explain the central definitions and claims in Gaussian Processes and the RVM; Apply the chapter's principal methods and interpret their outputs; State the assumptions behind formal results and connect them to earlier chapters. Revisit the assumptions attached to each formal result before transferring it to a new setting. For primary and extended treatments, consult [@rasmussen2006], [@williams1996], [@neal1996].

## Exercises {#exercises}

1.  **(Conditioning a bivariate normal.)** Let \((f_1,f_2)\) be zero-mean Gaussian with covariance \(K=\begin{pmatrix}1&0.75\\0.75&1\end{pmatrix}\). Compute the conditional mean and variance of \(f_2\) given \(f_1=1\). Verify that observing \(f_1\) reduces the variance of \(f_2\) from \(1\) to \(1-0.75^2\).
2.  **(Noise-free limit.)** Show that as \(\sigma^2\to 0\) the GP posterior mean interpolates the data exactly, \(\bar m(x_i)=y_i\) for every training point, and that \(v(x_i)\to 0\) there. Why does this make \(\sigma^2\) the analogue of the ridge parameter \(\lambda n\)?
3.  **(KRR equals GP, symbolically.)** Starting from the ridge objective \(\tfrac1n\|K\alpha-y\|^2+\lambda\alpha^\top K\alpha\), derive the normal equations and confirm the solution \(\alpha=(K+\lambda n I)^{-1}y\). State precisely why the GP posterior mean coincides with this fit but the GP additionally yields a variance. *Hint: identify the loss with the negative log Gaussian likelihood and the penalty with the negative log Gaussian prior.*
4.  **(Marginal-likelihood gradient.)** With \(A=K_\theta+\sigma^2 I\), verify the identities \(\partial\ln\det A=\operatorname{tr}(A^{-1}\partial A)\) and \(\partial A^{-1}=-A^{-1}(\partial A)A^{-1}\), and use them to derive \(\partial\mathcal L/\partial\theta_j\). Then specialize to \(\theta_j=\sigma^2\), where \(\partial A/\partial\sigma^2=I\). *Hint: for the determinant identity differentiate \(\ln\det A=\operatorname{tr}\ln A\).*
5.  **(Occam factor.)** Take a Gaussian covariance with amplitude \(a\) and length scale \(\ell\), so \(K_\theta=a\,K_1(\ell)\). Explain qualitatively how the two terms of \(\mathcal L\) move as \(\ell\) shrinks toward zero (a very flexible prior) and as \(\ell\) grows large (a very rigid one), and argue that the maximizer sits between the extremes. *Difficulty: medium.*
6.  **(RVM fixed point.)** Show that the update \(s_i\leftarrow(1-s_i\Sigma_{ii})/\mu_i^2\) has \(s_i=\infty\), \(\mu_i=0\) as a fixed point, and argue that a basis function with small posterior mean and near-prior posterior variance is attracted to it. Interpret \(\gamma_i=1-s_i\Sigma_{ii}\) as the number of well-determined parameters contributed by basis \(i\). *Difficulty: hard. Hint: track how \(\gamma_i\to 0\) and \(\mu_i\to 0\) reinforce each other across iterations.*
7.  **(Laplace vs. Gaussian prior.)** Contrast the MAP estimate under a Gaussian coefficient prior (\(-\ln p\propto\sum_i\alpha_i^2\)) with that under a Laplace prior (\(-\ln p\propto\sum_i|\alpha_i|\)). Explain why only the second yields exactly-zero coefficients, using the shape of the penalty near the origin. *Difficulty: medium.*
8.  **(GP classification curvature.)** For the logistic likelihood \(p(y\mid t)=\sigma(yt)\), compute \(c_i=\partial_{t_i}\ln p(y_i\mid t_i)\) and \(C_{ii}=-\partial_{t_i}^2\ln p(y_i\mid t_i)\), and confirm \(0\lt C_{ii}\le\tfrac14\). Explain why log-concavity guarantees the Laplace-approximation mode is unique. *Difficulty: hard. Hint: \(\sigma'=\sigma(1-\sigma)\).*
9.  **(The Nyström gap.)** Show that \(K_{ff}-Q_{ff}\) is the covariance of the conditional \(p(f\mid u)\), hence positive semidefinite, and that if a training input \(x_i\) belongs to the inducing set then the \(i\)th row and column of \(K_{ff}-Q_{ff}\) vanish. Check both claims against the worked sparse example, where \(Z=(x_2,x_3)\) gave \(\operatorname{diag}(K_{ff}-Q_{ff})=(0.1203,\,0,\,0,\,0.2905)\). Conclude that the FITC correction \(\Lambda_{\mathrm{FITC}}-\sigma^2 I\) never subtracts variance. *Difficulty: medium. Hint: a coordinate of \(u\) has zero variance given \(u\), and a positive semidefinite matrix with a zero diagonal entry has a zero row.*
10. **(A bound and a non-bound.)** Show that at \(Z=X\) one has \(Q_{ff}=K_{ff}\), so both the DTC evidence \(\ln\mathcal N(y\mid 0,\ Q_{ff}+\sigma^2 I)\) and the Titsias bound \(\mathcal L_T\) equal the exact log marginal likelihood. Explain why for general \(Z\) only \(\mathcal L_T\) is guaranteed to sit below the exact value, and confirm it on the worked example, where the DTC evidence \(-4.982\) exceeds the exact \(-5.3762\) while \(\mathcal L_T=-7.0364\) respects the bound with gap \(1.6602\). Which of the two is the safe objective for optimizing the positions \(Z\), and why? *Difficulty: medium.*
11. **(Averaging the probit.)** Prove the closed form used in the Laplace predictive step: for \(v\ge 0\),

$$\int\Phi(f)\,\mathcal N(f\mid\mu,v)\,df=\Phi\!\left(\frac{\mu}{\sqrt{1+v}}\right).$$

    Conclude that latent uncertainty always pulls the reported class probability toward \(\tfrac12\) compared with plugging the mean into the link, and that the pull grows with \(v\). *Difficulty: hard. Hint: write \(\Phi(f)=P(Z\le f)\) for \(Z\sim\mathcal N(0,1)\) independent of \(f\sim\mathcal N(\mu,v)\), and read the left side as \(P(Z-f\le 0)\) with \(Z-f\sim\mathcal N(-\mu,\,1+v)\).*
