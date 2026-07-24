---
id: ch-modern
slug: kernels-now
title: Modern Generalization Theory
part: XII · Kernels Now
order: 44
tier: advanced
prerequisites:
  - bayesian-optimization-and-bandits
objectives:
  - >-
    Derive why ridgeless risk can peak at interpolation and fall again in the
    overparameterized regime.
  - State the spectral conditions that separate benign from harmful overfitting.
  - >-
    Read kernel learning curves mode by mode from eigenvalues and target
    coefficients.
  - >-
    Use PAC-Bayes, compression, and algorithm-dependent diagnostics without
    treating them as interchangeable certificates.
  - >-
    Keep proportional random-matrix predictions distinct from finite-sample,
    distribution-free guarantees.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-modern.yml
verification_date: null
bibliography:
  - belkin2019dd
  - belkin2018understand
  - hastie2019
  - mei2022
  - liang2020interpolate
  - bartlett2020benign
  - bordelon2020
  - canatar2021
  - sollich1999
  - caponnetto2007
  - spigler2020
  - cui2021
  - neyshabur2018
  - elkaroui2010
  - cheng2013
---
# Modern Generalization Theory

<p class="lead">The two textbooks this book synthesizes were finished around 2004, and their account of generalization rests on a single intuition: a model with more capacity than it needs will fit the noise and generalize badly, so the art is to stop short of interpolation. Deep learning broke that intuition in plain sight, and kernels turned out to be the cleanest setting in which to understand why. This chapter derives the three results that rebuilt the theory around interpolation rather than against it: the <em>double descent</em> risk curve and its peak at the interpolation threshold, the <em>benign overfitting</em> condition under which a minimum-norm interpolant of noisy data still generalizes, and the <em>spectral learning curves</em> that fix the power-law rate at which a kernel method learns. The common thread is the kernel's spectrum: the eigenvalue decay that [[ch:mercer-and-rates]] read as a smoothness scale is, in this modern light, also the dial that decides whether interpolation is safe and how fast learning proceeds. Feature learning, the neural tangent kernel, and attention, the other half of the modern story, are the subject of the closing chapter [[ch:the-frontier]]; here we stay with the fixed kernel and ask what it does when it interpolates.</p>

## The interpolation revolution {#interpolation}

The most consequential modern result is that a central lesson of Parts II and IV is incomplete. The classical theory says a model that fits the training data exactly, an interpolant, must generalize badly, because it has spent all its capacity memorizing noise; the RKHS-ball bounds of [[ch:learning-theory]] and the bias-variance decomposition of [[ch:mercer-and-rates]] both rest on that reading. Modern practice contradicts it. Overparameterized networks, and kernel machines too, routinely drive the training error to zero on noisy labels and still predict well on unseen data. Belkin, Ma, and Mandal (2018) argued the point in their title, that to understand deep learning we need to understand kernel learning, precisely because kernel machines also interpolate noisy data and still generalize, so whatever explains the phenomenon must already be visible in the linear algebra of kernels.

The reframing is that overfitting is a question about *which* interpolant, not *whether* to interpolate. Once a model is expressive enough to fit the data exactly there are typically infinitely many ways to do so, and the learning algorithm silently selects one. Gradient descent from a small initialization, and the normal equations with a pseudoinverse, both select the interpolant of smallest norm. That choice is the whole story, so we begin by writing it down.

### The minimum-norm interpolant {#min-norm}

Fix a feature map \(\varphi:\mathcal X\to\mathbb R^{D}\), which may be an explicit random-feature map or the (possibly infinite-dimensional) map of a kernel, and fit a linear predictor \(f(x)=\varphi(x)^\top\beta\) to data \((x_i,y_i)_{i=1}^n\). Stack the features into the design matrix \(\Phi\in\mathbb R^{n\times D}\) with rows \(\varphi(x_i)^\top\). When \(D\lt n\) the system \(\Phi\beta=y\) is overdetermined and least squares returns a unique \(\beta\); when \(D\gt n\) it is underdetermined and there is an affine space of exact interpolants. The learning algorithm picks the smallest one.

:::: {.definition #def-42-1}
[Definition (minimum-norm interpolant)]{.box-title}

Given features \(\Phi\in\mathbb R^{n\times D}\) and targets \(y\in\mathbb R^n\), the *minimum-norm least-squares* solution is

$$\hat\beta=\arg\min\big\{\|\beta\|_2:\ \beta\ \text{minimizes}\ \|\Phi\beta-y\|_2^2\big\}=\Phi^{+}y,$$

where \(\Phi^{+}\) is the Moore-Penrose pseudoinverse. When \(D\gt n\) and \(\Phi\) has full row rank this interpolates, \(\Phi\hat\beta=y\), and equals \(\hat\beta=\Phi^\top(\Phi\Phi^\top)^{-1}y\); when \(D\lt n\) and \(\Phi\) has full column rank it is the ordinary least-squares estimator \(\hat\beta=(\Phi^\top\Phi)^{-1}\Phi^\top y\).
::::

The overparameterized form \(\hat\beta=\Phi^\top(\Phi\Phi^\top)^{-1}y\) is exactly kernel ridge regression at ridge zero, with Gram matrix \(K=\Phi\Phi^\top\): the predictor is \(f(x)=k(x)^\top K^{-1}y\), the ridgeless kernel interpolant of [[ch:kernel-ridge-and-friends]]. So the question \"does the minimum-norm interpolant generalize\" is a question about kernels, and its answer is written in the spectrum of \(K\). Liang and Rakhlin (2020) showed that kernel ridgeless regression can indeed generalize when the eigenvalues decay at the right rate, and the two theorems that follow make the mechanism precise.

## Double descent {#double-descent}

Plot test error against model size and the classical picture predicts a U: error falls as the model grows expressive enough to capture the signal, then rises as it grows expressive enough to fit the noise, with a sweet spot in between. Belkin, Hsu, Ma, and Mandal (2019) showed that the real curve does not stop at the U. Past the point where the model has just enough parameters to interpolate, the error descends a *second* time, so the full curve descends, rises to a peak, and descends again. The peak sits exactly at the **interpolation threshold** \(D=n\), where the number of parameters first equals the number of samples.

The mechanism is visible in the minimum-norm interpolant. As \(D\to n\) from either side the design matrix \(\Phi\) becomes square and generically ill-conditioned: its smallest singular value approaches zero, so the interpolating coefficient vector \(\hat\beta=\Phi^{+}y\) has enormous norm and the predictor swings wildly between the data points. This is the peak. For \(D\gt n\) there are many interpolants and the minimum-norm one need not be wild, because the extra directions give it room to stay small; as \(D\) grows further the solution smooths out and the error falls again. The cleanest place to see all of this with exact formulas is the isotropic linear model, where the risk can be computed in closed form.

### The risk of ridgeless least squares {#dd-theorem}

What does the test error actually do as the model crosses the interpolation threshold? In the isotropic model the two regimes each have an exact closed form, and reading them side by side shows both branches blowing up at the boundary, which is the peak.

:::: {.theorem #thm-42-2}
[Theorem (double-descent risk of minimum-norm least squares; Belkin et al. 2019, Hastie et al. 2019, Mei and Montanari 2022)]{.box-title}

Let \(x\sim\mathcal N(0,I_p)\), \(y=x^\top\beta+\varepsilon\) with \(\varepsilon\sim\mathcal N(0,\sigma^2)\) independent and \(\|\beta\|^2=r^2\). From \(n\) samples form the minimum-norm least-squares estimator \(\hat\beta=X^{+}y\), and measure the excess prediction risk \(R(\hat\beta)=\mathbb E_x\big[(x^\top\hat\beta-x^\top\beta)^2\big]=\|\hat\beta-\beta\|^2\). Then its expectation over the training noise and design is

$$\mathbb E\,R(\hat\beta)=\begin{cases}\ \dfrac{\sigma^2 p}{\,n-p-1\,}, & p\lt n-1\quad(\text{underparameterized}),\\[2ex]\ r^2\,\dfrac{p-n}{p}+\dfrac{\sigma^2 n}{\,p-n-1\,}, & p\gt n+1\quad(\text{overparameterized}).\end{cases}$$

Both branches diverge as \(p\to n\), producing the double-descent peak at the interpolation threshold. As \(n,p\to\infty\) with \(p/n\to\gamma\) they converge to the Marchenko-Pastur limits \(\sigma^2\gamma/(1-\gamma)\) for \(\gamma\lt1\) and \(r^2(1-1/\gamma)+\sigma^2/(\gamma-1)\) for \(\gamma\gt1\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
::::

The underparameterized branch is pure variance: the estimator is unbiased, and every added feature is one more direction of noise to fit, so the variance climbs and blows up as \(p\to n\). The overparameterized branch has two terms with opposite trends. The bias \(r^2(p-n)/p\) is the signal energy lost to the \(p-n\) directions the interpolant cannot see; it shrinks as \(p\) grows, since a richer feature space captures more of \(\beta\). The variance \(\sigma^2 n/(p-n-1)\) also shrinks as \(p\) grows, because the fixed budget of noise energy is spread over more directions. Their sum descends from the peak, reaches a minimum in the overparameterized regime, and rises slowly back toward \(r^2\) as \(p\to\infty\) and the interpolant loses ever more signal. The proof of both branches is one application of the inverse-Wishart mean.

:::::: {.proof}
[Proof]{.box-title}

**Underparameterized case \(p\lt n\).** Here \(X\) has full column rank almost surely, so \(\hat\beta=(X^\top X)^{-1}X^\top y\) and, writing \(y=X\beta+\varepsilon\), \(\hat\beta-\beta=(X^\top X)^{-1}X^\top\varepsilon\). The estimator is unbiased, and its risk is

$$\mathbb E\,\|\hat\beta-\beta\|^2=\mathbb E\big[\varepsilon^\top X(X^\top X)^{-2}X^\top\varepsilon\big]=\sigma^2\,\mathbb E\,\mathrm{tr}\!\big((X^\top X)^{-2}X^\top X\big)=\sigma^2\,\mathbb E\,\mathrm{tr}\!\big((X^\top X)^{-1}\big),$$

taking the expectation over \(\varepsilon\) first. Now \(X^\top X\) is a \(p\times p\) Wishart matrix with \(n\) degrees of freedom and identity scale, whose inverse has the known mean \(\mathbb E\big[(X^\top X)^{-1}\big]=I_p/(n-p-1)\) for \(n-p-1\gt0\). Hence \(\mathbb E\,\mathrm{tr}\big((X^\top X)^{-1}\big)=p/(n-p-1)\), giving the first branch.

**Overparameterized case \(p\gt n\).** Here \(X\) has full row rank almost surely, \(\hat\beta=X^\top(XX^\top)^{-1}y\), and with \(\Pi=X^\top(XX^\top)^{-1}X\) the orthogonal projection onto the \(n\)-dimensional row space,

$$\hat\beta-\beta=(\Pi-I)\beta+X^\top(XX^\top)^{-1}\varepsilon.$$

The two terms are uncorrelated because \(\varepsilon\) is independent with mean zero, so the risk splits into bias and variance. For the bias, \(\|(\Pi-I)\beta\|^2=\beta^\top(I-\Pi)\beta\) since \(I-\Pi\) is idempotent; by rotational invariance of the isotropic Gaussian design the row space is a uniformly random \(n\)-dimensional subspace, so \(\mathbb E[I-\Pi]=\tfrac{p-n}{p}I\) and the bias equals \(r^2(p-n)/p\). For the variance,

$$\mathbb E\big[\varepsilon^\top(XX^\top)^{-1}XX^\top(XX^\top)^{-1}\varepsilon\big]=\sigma^2\,\mathbb E\,\mathrm{tr}\!\big((XX^\top)^{-1}\big).$$

Now \(XX^\top\) is an \(n\times n\) Wishart matrix with \(p\) degrees of freedom, so \(\mathbb E\big[(XX^\top)^{-1}\big]=I_n/(p-n-1)\) and the trace is \(n/(p-n-1)\). Adding the two terms gives the second branch. [\(\square\)]{.qed}
::::::

### Why regularization flattens the peak {#dd-ridge}

The divergence at \(p=n\) is a property of the *ridgeless* minimum-norm solution, not of overparameterization itself. Replace the pseudoinverse by ridge regression, \(\hat\beta_\lambda=(X^\top X+n\lambda I)^{-1}X^\top y\), and the near-zero singular values that blow up the interpolant are lifted away from zero by \(n\lambda\). The peak is exactly the place where a good ridge helps most, and a well-chosen \(\lambda\) removes it.

::: {.remark}
[Optimal ridge monotonizes the risk]{.box-title}

Mei and Montanari (2022) computed the risk of ridge regression in the same proportional limit \(p/n\to\gamma\) and showed that at the *optimal* ridge \(\lambda^\star(\gamma)\) the risk is a monotone function of the sample size: the double-descent peak disappears entirely, and the interpolation threshold loses its special status. Double descent is therefore the signature of under-regularization. It appears when one interpolates on purpose or lets an implicit bias interpolate for you, and it is invisible to a practitioner who tunes the ridge by cross-validation. The modern lesson is not that interpolation is good, but that interpolation is safe under conditions the next section makes precise, and that regularization interpolates between the two regimes.
:::

::::: {.example #example-42-1}
[Example (double descent in the isotropic model)]{.box-title}

:::: wex
::: wex-setup
Isotropic model with \(n=40\), noise \(\sigma^2=0.25\), signal \(\|\beta\|^2=r^2=1\). The minimum-norm least-squares risk is averaged over \(200\) deterministic seeds and compared to the exact formulas of the Theorem. Feature counts \(p\) sweep across the threshold \(p=n=40\).
:::

1.  [Climb toward the peak.]{.wex-op} Underparameterized, the exact risk \(\sigma^2 p/(n-p-1)\) rises from \(0.263\) at \(p=20\) (\(\gamma=0.5\)) to \(3.00\) at \(p=36\) (\(\gamma=0.9\)); simulation gives \(0.276\) and \(2.81\).
2.  [Cross the threshold.]{.wex-op} Just past interpolation, \(p=44\) (\(\gamma=1.1\)) has exact risk \(3.42\) and simulated \(3.12\); the risk is largest right at \(p=n\), where both formulas diverge.
3.  [Descend a second time.]{.wex-op} Overparameterized, \(r^2(p-n)/p+\sigma^2 n/(p-n-1)\) falls to \(0.756\) at \(p=80\) (\(\gamma=2\)), below the best underparameterized value; simulation gives \(0.760\).
4.  [Approach the ceiling.]{.wex-op} Far overparameterized, \(p=400\) (\(\gamma=10\)) has risk \(0.928\) (both exact and simulated), climbing back toward \(r^2=1\) as the interpolant sheds signal.

**Reading.** The curve is not a U. It descends, spikes at \(p=n\), and descends again into the overparameterized regime, where the best test error (\(0.76\) at \(\gamma=2\)) beats every underparameterized model. Every simulated number matches the closed form to sampling error, and the spike is exactly the two Wishart denominators \(n-p-1\) and \(p-n-1\) passing through zero.
::::

**Verification artifact.** checks/example-ch-modern-example-42-1.json records the example source hash and verification scope.
:::::

The geometry of the curve is easier to retain than its case-specific constants. Just below interpolation, the fit has almost no slack and noise is amplified; just above it, the minimum-norm rule can spread the interpolating solution over additional directions. Whether the right-hand branch keeps falling is then a spectral question, not a universal law.

<figure class="viz" data-figure="double-descent" data-alt="A ridgeless test-risk curve rises sharply at the interpolation threshold where parameters equal samples, then falls in the overparameterized regime as a minimum-norm solution spreads across extra directions.">
<figcaption>The interpolation peak is a variance singularity, while the second descent comes from the geometry of the selected minimum-norm solution. A falling right branch requires suitable signal and covariance assumptions; interpolation alone does not guarantee it.</figcaption>
</figure>

## Benign overfitting {#benign-overfitting}

The isotropic model of the last section overfits harmfully: at any fixed \(p\gt n\) the risk stays bounded away from zero, because isotropic features give the noise nowhere to hide. Real kernels are not isotropic. Their eigenvalues decay, so the feature space has a few high-variance directions carrying the signal and a long tail of low-variance directions. Bartlett, Long, Lugosi, and Tsigler (2020) showed that this tail is exactly what makes interpolation safe: it absorbs the label noise without distorting the signal, so the minimum-norm interpolant of noisy data can still generalize. Whether it does is decided by two notions of how many effective directions the covariance spectrum contains.

:::: {.definition #def-42-3}
[Definition (effective ranks)]{.box-title}

Let \(\Sigma\) have eigenvalues \(\lambda_1\ge\lambda_2\ge\cdots\gt0\). For \(k\ge0\) define the two effective ranks of the tail beyond index \(k\),

$$r_k(\Sigma)=\frac{\sum_{i\gt k}\lambda_i}{\lambda_{k+1}},\qquad R_k(\Sigma)=\frac{\big(\sum_{i\gt k}\lambda_i\big)^2}{\sum_{i\gt k}\lambda_i^2}.$$

The first, \(r_k\), is a count in units of the largest tail eigenvalue; the second, \(R_k\), is the participation ratio, the number of tail directions of comparable size (largest when the tail is flat, since \(R_k=m\) for a flat tail of \(m\) equal eigenvalues). The *critical index* is \(k^\star=\min\{k\ge0:\ r_k(\Sigma)\ge bn\}\) for a fixed constant \(b\).
::::

The index \(k^\star\) splits the spectrum into a head of at most \(k^\star\) strong directions, where the signal lives and is estimated as in classical low-dimensional regression, and a tail whose job is to soak up noise. The theorem attaches a precise cost to each part.

::::: {.theorem #thm-42-4}
[Theorem (effective-rank characterization; Bartlett, Long, Lugosi, and Tsigler 2020)]{.box-title}

Consider the well-specified linear model \(y=x^\top\beta^\star+\varepsilon\), with independent noise of variance \(\sigma^2\), and let \(\hat\beta\) be the minimum-Euclidean-norm interpolant of \(n\) independent samples. Under the paper's covariance regularity and sub-Gaussian design assumptions, there are universal constants \(b,c\gt0\) for which high-probability upper and lower bounds decompose the excess prediction risk into a signal-dependent bias term and a noise-dependent variance term of order

$$\mathrm{Var}\ \asymp\ \sigma^2\left(\frac{k^\star}{n}+\frac{n}{R_{k^\star}(\Sigma)}\right).$$

For a sequence of problems in which the paper's bias condition also vanishes, its effective-rank conditions characterize when the variance vanishes:

$$\frac{r_0(\Sigma)}{n}\to\infty,\qquad \frac{k^\star}{n}\to0,\qquad \frac{n}{R_{k^\star}(\Sigma)}\to0.$$

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
:::::

::: {.remark}
[Scope of the characterization]{.box-title}

The displayed conditions do not by themselves prove benign overfitting for an arbitrary nonlinear, misspecified, dependent, or heavy-tailed problem. They characterize the effective-rank mechanism within the stated linear model and must be combined with control of the signal-dependent bias. Later extensions change the assumptions and constants; this chapter uses the theorem only for the covariance mechanism it establishes.
:::

Read the variance as a competition. The first piece, \(\sigma^2 k^\star/n\), is the ordinary cost of fitting noise in the \(k^\star\) strong directions with \(n\) samples: it is small only when the number of high-variance directions is small compared to the sample size, exactly the classical demand. The second piece, \(\sigma^2 n/R_{k^\star}\), is the price of interpolating the noise using the tail. The interpolant must fit every label exactly, and it does so by spending the tail directions; because there are effectively \(R_{k^\star}\) of them, each carries only a \(1/R_{k^\star}\) share of the noise energy, and the total noise leaked back into the prediction is of order \(n/R_{k^\star}\). When the tail is high-dimensional, \(R_{k^\star}\gg n\), that leak is negligible: the noise is diluted across so many harmless directions that its footprint on any test point vanishes. The condition \(r_0(\Sigma)/n\to\infty\) ensures the tail also has enough total variance to interpolate at all without inflating the strong directions, which would bias the signal. We can turn the whole mechanism into numbers.

::::: {.example #example-42-2}
[Example (benign versus harmful interpolation)]{.box-title}

:::: wex
::: wex-setup
Spiked covariance: \(s=5\) signal eigenvalues equal to \(1\), then a flat junk tail of \(m\) eigenvalues at level \(\tau\); signal \(\beta^\star\) lies in the spike with \(\|\beta^\star\|_\Sigma^2=1\), noise \(\sigma^2=0.25\), \(n=100\), constant \(b=1\). The min-norm interpolant's excess risk is averaged over \(120\) seeds. Two tails are compared.
:::

1.  [Build a wide, low-energy tail.]{.wex-op} Benign case: \(m=20{,}000\), \(\tau=0.001\), so the tail variance \(\sum_{i\gt k^\star}\lambda_i=20\) is well below \(n\) while its dimension is huge. The critical index is \(k^\star=5\), and \(R_{k^\star}=20{,}000\).
2.  [Read off the variance budget.]{.wex-op} \(k^\star/n=0.05\) and \(n/R_{k^\star}=0.005\), so the BLLT variance term is \(\sigma^2(0.05+0.005)=0.0138\). The full simulated excess risk is \(0.040\), against a null-predictor risk of \(1.0\): the interpolant fits all \(100\) noisy labels and still predicts well.
3.  [Narrow the tail.]{.wex-op} Harmful case: \(m=150\), \(\tau=0.5\), so the tail dimension is comparable to \(n\). Now \(R_{k^\star}=150\), so \(n/R_{k^\star}=0.667\) and the variance term jumps to \(\sigma^2(0.05+0.667)=0.179\).
4.  [Watch overfitting turn harmful.]{.wex-op} The simulated excess risk is \(0.608\), most of the way to the null risk of \(1.0\): with only \(\sim n\) tail directions, the noise cannot be diluted and floods the prediction.

**Reading.** The same interpolation rule, on the same signal, is harmless with a wide tail (risk \(0.04\)) and ruinous with a narrow one (risk \(0.61\)). The single quantity that flips is the tail's effective rank \(R_{k^\star}\): \(20{,}000\) versus \(150\). A high-dimensional spectral tail is not a nuisance to be regularized away; it is the mechanism that makes interpolation safe.
::::

**Verification artifact.** checks/example-ch-modern-example-42-2.json records the example source hash and verification scope.
:::::

The connection to double descent is now exact. The isotropic model interpolates harmfully because \(\Sigma=I\) has no tail: every direction is a strong direction, \(k^\star\) grows with \(p\), and there is no low-variance subspace to absorb the noise. A decaying kernel spectrum is the opposite, and Liang and Rakhlin's (2020) result that kernel ridgeless regression can generalize is the infinite-dimensional face of the same theorem, with the effective ranks read off the Mercer eigenvalues of [[ch:mercer-and-rates]].

## Spectral learning curves and scaling laws {#spectral-learning-curves}

The spectrum that decides whether interpolation is safe also fixes the rate at which a kernel method learns. For kernel ridge regression the generalization error typically falls as a power law in the number of samples, \(\varepsilon(n)\sim n^{-\beta}\), and the exponent \(\beta\) is set by two spectral quantities: how fast the kernel's eigenvalues decay, and how fast the target's coefficients in the kernel eigenbasis decay. A target aligned with the top eigenfunctions is learned quickly; a rough target that excites the tail is learned slowly. The learning curve is a joint statement about the kernel and the function it is asked to fit, and the tool that makes it exact is a per-mode accounting of bias and variance.

Set up the eigenbasis. By Mercer's theorem the kernel integral operator on \(L^2(p)\) has orthonormal eigenfunctions \(\phi_i\) with eigenvalues \(\eta_1\ge\eta_2\ge\cdots\), and any target expands as \(f^\star=\sum_i\bar a_i\phi_i\). The decay of the coefficients \(\bar a_i\) is the **source condition** of Caponnetto and De Vito (2007): writing \(\bar a_i^2\sim i^{-b}\) and \(\eta_i\sim i^{-a}\), the pair \((a,b)\) is what the rate depends on. Kernel ridge regression at ridge \(\lambda\) with \(n\) points then has a remarkably clean predicted error, derived by the replica method of statistical physics.

:::::: {.remark}
[Replica prediction (Sollich 1999; Bordelon et al. 2020; Canatar et al. 2021)]{.box-title}

Let \(\kappa\gt0\) be the unique solution of the self-consistent equation

$$\kappa=\lambda+\sum_i\frac{\eta_i\,\kappa}{n\eta_i+\kappa},$$

and define the per-mode *learnability* and the overfitting factor

$$\mathcal L_i=\frac{n\eta_i}{n\eta_i+\kappa}\in(0,1),\qquad \gamma=\frac1n\sum_i\mathcal L_i^2\in(0,1).$$

In the typical-case random-design setting analyzed by the cited work, the replica calculation predicts the dataset-averaged generalization error

$$E_g=\frac{1}{1-\gamma}\left[\sum_i\big(1-\mathcal L_i\big)^2\,\bar a_i^2+\sigma^2\gamma\right],\qquad 1-\mathcal L_i=\frac{\kappa}{n\eta_i+\kappa}.$$

Mode \(i\) is learned once \(n\eta_i\) exceeds the effective ridge \(\kappa\) (then \(\mathcal L_i\to1\) and it drops out of the error) and unlearned while \(n\eta_i\ll\kappa\) (then \(\mathcal L_i\to0\) and it contributes its full energy \(\bar a_i^2\)).
::::::

The self-consistent \(\kappa\) is an *effective ridge*: even at \(\lambda=0\), having only \(n\) samples acts like a positive regularizer, because the finite design cannot resolve modes below a resolution set by the spectrum. The equation is what fixes that resolution. To see the structure, derive the per-mode error from the ridge bias-variance decomposition.

::: {.remark}
[Heuristic derivation (per-mode bias and variance)]{.box-title}

Work in the eigenbasis, where the population problem is diagonal: mode \(i\) has feature variance \(\eta_i\) and true coefficient \(\bar a_i\). Kernel ridge regression is ridge regression on these features, and with \(n\) samples the effective penalty on mode \(i\) is not \(\lambda\) but the self-consistent \(\kappa\), which aggregates the interference from all other modes competing for the same \(n\) data points. The ridge estimate of a coefficient with signal-to-penalty balance \(n\eta_i\) versus \(\kappa\) shrinks it by the factor \(\mathcal L_i=n\eta_i/(n\eta_i+\kappa)\), leaving the residual coefficient \((1-\mathcal L_i)\bar a_i=\kappa\,\bar a_i/(n\eta_i+\kappa)\). Squaring and weighting by the mode's contribution to the \(L^2\) error gives the per-mode bias \((1-\mathcal L_i)^2\bar a_i^2\). Summing over modes and adding the noise, which the estimator fits with the same shrinkage and which contributes \(\sigma^2\gamma\), gives the bracket. The prefactor \(1/(1-\gamma)\) is the variance amplification: \(\gamma=\frac1n\sum_i\mathcal L_i^2\) is the fraction of the \(n\) degrees of freedom already spent on the learned modes, and as it approaches \(1\) the estimator runs out of data to fit anything new and the error diverges, the learning-curve echo of the interpolation peak. That \(\kappa\) makes this accounting self-consistent is the replica prediction; the argument explains the formula but is not a rigorous finite-sample proof.
:::

The power law follows from the threshold. The learned modes are those with \(n\eta_i\gtrsim\kappa\), i.e. \(i\lesssim i^\star(n)\) where \(\eta_{i^\star}\approx\kappa/n\). Feeding \(\eta_i\sim i^{-a}\) into the self-consistent equation gives \(i^\star\) growing proportionally to \(n\), so the error is dominated by the still-unlearned tail,

$$E_g\ \approx\ \sum_{i\gt i^\star}\bar a_i^2\ \sim\ \sum_{i\gt i^\star}i^{-b}\ \sim\ (i^\star)^{-(b-1)}\ \sim\ n^{-(b-1)},$$

a power law whose exponent is set jointly by the eigenvalue decay \(a\) (through how fast \(i^\star\) advances) and the source decay \(b\) (through how much target energy the tail still holds). Spigler, Geiger, and Wyart (2020) and Cui et al. (2021) work out the exponent in each regime, including the crossover to the noise-limited rate once \(\sigma^2\gt0\). The prediction is not merely asymptotic folklore; it tracks real kernel ridge regression mode by mode.

::::: {.example #example-42-3}
[Example (learning curve of kernel ridge regression)]{.box-title}

:::: wex
::: wex-setup
A discrete domain of \(P=1200\) points with the uniform measure carries a kernel with eigenvalues \(\eta_i=i^{-1.5}\) and orthonormal eigenfunctions; the noiseless target has coefficients \(\bar a_i^2=i^{-2}\). Kernel ridge regression at \(\lambda=10^{-4}\) is run on \(n\) points drawn uniformly, averaged over \(150\) draws, and its test error is compared to the omniscient prediction.
:::

1.  [Solve for the effective ridge.]{.wex-op} The self-consistent \(\kappa\) falls from \(1.02\) at \(n=10\) to \(0.115\) at \(n=320\): more data resolves finer modes, lowering the effective penalty.
2.  [Read the overfitting factor.]{.wex-op} \(\gamma\) rises from \(0.32\) to \(0.49\) as the sample budget fills, staying safely below \(1\), so the amplification \(1/(1-\gamma)\) stays modest.
3.  [Compare theory to simulation.]{.wex-op} The omniscient \(E_g\) reads \(0.256,\ 0.133,\ 0.0665,\ 0.0323,\ 0.0150,\ 0.00643\) at \(n=10,20,40,80,160,320\); the measured KRR error reads \(0.259,\ 0.137,\ 0.0700,\ 0.0339,\ 0.0166,\ 0.00815\). Theory and experiment agree to a few percent at every sample size.
4.  [Fit the power law.]{.wex-op} The large-\(n\) points give a slope \(\beta\approx1.17\) on log-log axes, near the source-limited value \(b-1=1\) predicted by the tail estimate.

**Reading.** The generalization error is not a single opaque number but a sum over modes, each switched on when the sample size crosses its eigenvalue. The replica formula reproduces the whole KRR learning curve from the spectrum and the target alone, and its power-law tail is the kernel version of a neural scaling law.
::::

**Verification artifact.** checks/example-ch-modern-example-42-3.json records the example source hash and verification scope.
:::::

These curves are the theoretical backbone of the empirical scaling laws seen in large models, and they close the loop with double descent. The near-singular direction that produces the interpolation peak is exactly the mode whose eigenvalue the sample size has just reached, where \(n\eta_i\approx\kappa\) and \(\mathcal L_i\approx\tfrac12\): the peak and the power-law tail are two views of one spectral accounting. The exponent depends on the target, so the same kernel learns a smooth function fast and a rough one slowly, which is why the \"right\" kernel for a task is the one whose eigenfunctions align with the target, the alignment that Canatar, Bordelon, and Pehlevan (2021) quantify.

## PAC-Bayes and data-dependent certificates {#pac-bayes-certificates}

Uniform convergence asks for one event on which every hypothesis in a class behaves well. PAC-Bayes takes a different route. It places a prior distribution \(P\) over predictors before observing the sample, allows learning to return a posterior distribution \(Q\), and charges the learner for moving from \(P\) to \(Q\) through \(\mathrm{KL}(Q\|P)\). The resulting certificate concerns the randomized predictor obtained by drawing \(f\sim Q\). For a bounded loss, a representative bound has the shape

$$R(Q)\ \lesssim\ \widehat R(Q)+\sqrt{\frac{\mathrm{KL}(Q\|P)+\ln(1/\delta)}{n}},$$

with constants and lower-order terms depending on the chosen PAC-Bayes inequality. This is not a license to choose the prior after looking at all labels. A data-dependent prior needs either an independent data split, a hierarchical argument, or a theorem that explicitly pays for the dependence.

For a kernel machine, take \(Q\) to be a Gaussian perturbation around the learned coefficient vector or RKHS function. A flat minimum tolerates a broader perturbation without increasing empirical loss, while a sharp minimum does not. The KL term measures how far the learned center and covariance moved from the prior. This turns the informal claim that flat solutions generalize into an auditable optimization: choose the posterior scale that minimizes the sum of perturbed empirical risk and information cost. Spectrally normalized neural bounds use the same logic in parameter space [@neyshabur2018], while an RKHS version exposes the kernel eigenbasis directly.

::: {.algorithm #alg-modern-pac-bayes}
[Algorithm (a PAC-Bayes reporting protocol)]{.box-title}

1. Declare the prior, loss range, confidence level, and the exact theorem before using validation or test labels.
2. Fit the predictor and define a posterior family around it.
3. estimate the empirical Gibbs risk using fresh perturbation draws with a fixed random seed.
4. Compute the KL term analytically where possible and optimize only parameters permitted by the theorem.
5. Report the certificate beside the ordinary test estimate, including Monte Carlo error and every data-dependent choice.
:::

## Compression and algorithm-dependent complexity {#compression-and-stability}

A second escape from worst-case class capacity is to describe the output of the learning algorithm rather than the entire class it could have returned. If a classifier can be reconstructed from a small subset of \(s\) training examples plus a short side message, sample-compression bounds scale with the information in that description rather than the nominal feature dimension. Sparse support-vector expansions make the connection immediate: the support vectors are a compression set, although their number can itself approach \(n\) and must not be treated as small without measurement.

Stability gives a complementary view. Replace one training example and ask how much the learned predictor or loss changes. Strong regularization and a well-conditioned Gram system make kernel ridge regression stable; ridgeless interpolation can be unstable near small empirical eigenvalues even when its average risk is benign. Compression, stability, PAC-Bayes, and spectral effective dimension therefore answer different questions:

  View                 Data-dependent object                 Main failure mode
  -------------------- ------------------------------------- --------------------------------------
  PAC-Bayes            posterior-to-prior information        invalid data-dependent prior
  Compression          reconstruction description length     a large or unstable compression set
  Stability            sensitivity to replacing one example  ill-conditioning or weak curvature
  Spectral analysis    covariance and kernel eigenvalues      mismatched source or tail assumptions

Agreement among these views is useful evidence. Disagreement is diagnostic, not paradoxical: each theorem controls a different property under different assumptions.

## Random-matrix equivalents and finite-sample diagnostics {#random-matrix-diagnostics}

Modern proportional asymptotics replace random traces of resolvents by deterministic equivalents. In ridge regression the key object is

$$m_n(-\lambda)=\frac1n\operatorname{tr}(K+n\lambda I)^{-1},$$

and derivatives of this resolvent trace control variance and effective degrees of freedom. When \(n\) and feature dimension grow together under a specified random-design model, \(m_n\) can converge to the solution of a fixed-point equation. The resulting formulas predict test risk, but they are not distribution-free finite-sample guarantees. Their validity depends on the design ensemble, aspect ratio, spectral convergence, noise model, and whether the target is deterministic or random.

Where do such deterministic equivalents come from for a kernel matrix, whose entries are nonlinear functions of the data? The founding result is El Karoui's: when \(n\) and \(d\) grow proportionally and the entries of \(K\) are a smooth function of inner products \(x_i^\top x_j / d\) or of scaled distances, the kernel matrix is asymptotically indistinguishable in operator norm from a linear surrogate, a weighted sum of the all-ones matrix, the sample Gram matrix \(XX^\top/d\), and the identity, with weights given by the first Taylor coefficients of the kernel profile at its concentration point [@elkaroui2010]. In this regime the nonlinearity survives only through a handful of scalars: the bulk spectrum is a shifted and scaled Marchenko-Pastur law, and the identity term acts as an implicit ridge that the kernel adds on its own, before any explicit regularization. This is a sharp warning and a useful tool at once. The warning: with high-dimensional near-orthogonal inputs, a Gaussian kernel machine behaves like linear ridge regression, and no bandwidth tuning will recover the low-dimensional intuition. The tool: that same self-induced ridge is one mechanism behind the benign bulk behavior met in the interpolation sections, and the Taylor weights predict exactly how strong it is. Beyond the linear surrogate, the spectrum of inner-product kernel matrices admits finer descriptions that track the nonlinearity's higher moments and separate bulk from outlier eigenvalues [@cheng2013]; those refinements matter when the kernel profile is not smooth at the origin or the data carry spiked structure.

The practical use is a three-way diagnostic. Compare the observed learning curve with the deterministic-equivalent prediction, a nonparametric bootstrap over examples, and held-out risk. If the first fails while the latter two agree, the asymptotic model is misspecified. If the bootstrap is erratic, influential observations or dependence may invalidate iid reasoning. If all three disagree, first audit leakage, preprocessing, and numerical conditioning before telling a new generalization story.

## Limits, lower bounds, and what cannot be universal {#limits-and-lower-bounds}

No spectrum-only condition can characterize every regression problem. The same covariance eigenvalues can be paired with different target alignments, noise distributions, leverage profiles, or train-test shifts. Benign overfitting conditions must therefore separate signal assumptions from variance conditions, and sufficient conditions must not be restated as necessary ones. Likewise, a typical-case replica prediction is not a worst-case theorem, and a proportional limit can conceal slow convergence at the sample sizes used in practice.

Three negative checks should accompany any modern generalization claim:

1. **Null-target check.** Set the signal to zero and verify that the claimed mechanism still controls fitted noise.
2. **Adversarial-alignment check.** Move target energy into weak eigenfunctions and see which source condition breaks.
3. **Shift check.** Change the test covariate distribution while preserving training risk and document whether the guarantee survives.

These checks make the chapter's main lesson precise: interpolation is neither automatically harmful nor automatically benign. Its behavior is conditional on algorithm, spectrum, signal, noise, and evaluation distribution.

## Where this connects {#connections}

The three phenomena are one story told from three angles: interpolation is safe or not, learning is fast or slow, according to how the kernel's eigenvalues decay and how the target's energy is spread across the eigenbasis. That spectrum is the object [[ch:mercer-and-rates]] built and [[ch:learning-theory]] first used for bounds, now doing double duty as the arbiter of overfitting. The algorithms that make interpolation with a fixed kernel practical at scale are the Nystrom and random-feature methods of [[ch:large-scale-kernels]], and the Bayesian twin of every learning curve here is a Gaussian-process posterior, the subject of [[ch:gaussian-processes-and-rvm]]. What this chapter deliberately leaves out is the case where the kernel itself is learned: when features move, a network can beat its own tangent kernel, and the separation between the fixed-kernel world and feature learning is the closing story of [[ch:the-frontier]].

## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

Interpolation is a property of the training fit, not a guarantee of benign overfitting. State the noise model, covariance spectrum, target alignment, aspect ratio, and minimum-norm rule before transferring a double-descent or benign-overfitting result. A proportional random-matrix equivalent predicts a specified ensemble as \(n,d\to\infty\); it is not a distribution-free finite-sample bound.

Do not diagnose a spectral tail from eigenvalues alone. Compare target coefficients with those eigenfunctions, report finite-sample uncertainty in the empirical spectrum, and test whether ridge changes the apparent peak. PAC-Bayes, compression, stability, and spectral formulas answer different questions; presenting several does not make them mutually validating certificates.

## Summary and further reading {#summary-and-further-reading}

Modern generalization theory replaces the slogan “capacity causes overfitting” with a more precise question: which solution does the algorithm select inside an interpolating class? At the interpolation threshold noise amplification creates a peak; beyond it, minimum norm can distribute the fit across additional directions. Benign overfitting requires a spectrum with enough tail dimension to absorb noise without corrupting signal, and learning curves require target alignment as well as eigenvalue decay. Double descent was crystallized by [@belkin2019dd] and [@belkin2018understand], with detailed linear-model analysis in [@hastie2019]. The spectrum unifies the phenomena, but it does not make their assumptions universal.

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} In the isotropic double-descent theorem, take \(\sigma^2=0\) (no label noise). Show the underparameterized risk is exactly zero and the overparameterized risk is pure bias \(r^2(p-n)/p\). Explain why the peak at \(p=n\) disappears, and what this says about the source of the double-descent spike.
2.  [computation]{.ex-tag} Verify the underparameterized branch numerically-by-hand for \(n=10\), \(p=4\), \(\sigma^2=1\): compute \(\sigma^2 p/(n-p-1)\) and compare with the asymptotic \(\sigma^2\gamma/(1-\gamma)\) at \(\gamma=0.4\). Which is larger, and why does the finite-sample formula exceed the limit? [Hint: compare \(p/(n-p-1)\) with \(\gamma/(1-\gamma)=p/(n-p)\); the extra \(-1\) in the denominator is the finite-sample correction.]{.ex-hint}
3.  [computation]{.ex-tag} Show directly that ridge regression removes the divergence at \(p=n\). For the estimator \(\hat\beta_\lambda=(X^\top X+n\lambda I)^{-1}X^\top y\), argue that the smallest eigenvalue of \(X^\top X+n\lambda I\) is at least \(n\lambda\), so no factor \(1/(n-p)\) can appear, and the variance stays finite through \(p=n\). [Hint: bound \(\mathrm{tr}\big((X^\top X+n\lambda I)^{-2}X^\top X\big)\) using the eigenvalues \(s_j\) of \(X^\top X\) and the inequality \(s_j/(s_j+n\lambda)^2\le 1/(4n\lambda)\).]{.ex-hint}
4.  [computation]{.ex-tag} For the benign-overfitting effective ranks, take the spiked spectrum \(\lambda_1=\cdots=\lambda_s=1\) and \(\lambda_{s+1}=\cdots=\lambda_{s+m}=\tau\). Compute \(r_k\) and \(R_k\) for \(s\le k\lt s+m\), and confirm that at the spike cutoff \(R_s=m\) and \(r_s=m\tau/\tau=m\). Deduce the condition on \(m\) and \(\tau\) for benign overfitting with \(n\) samples. [Hint: a flat tail of \(m\) equal eigenvalues has participation ratio exactly \(m\); benign needs \(m\gg n\) and tail variance \(m\tau\) neither vanishing nor swamping the signal.]{.ex-hint}
5.  [challenge]{.ex-tag} Prove the isotropic model is never benign. With \(\Sigma=I_p\) and \(p\gt n\), show \(k^\star\) is forced to be of order \(p\) and \(R_{k^\star}\) of order \(p-k^\star\), so the variance term \(k^\star/n+n/R_{k^\star}\) cannot tend to zero as \(p\) grows at fixed \(n\). Relate this to the overparameterized branch \(r^2(p-n)/p+\sigma^2 n/(p-n-1)\) staying bounded away from zero. [Hint: for a flat spectrum \(r_k(\Sigma)=p-k\), so \(r_k\ge bn\) fails only once \(k\) is within \(bn\) of \(p\).]{.ex-hint}
6.  [challenge]{.ex-tag} In the omniscient learning curve, take the ridgeless limit \(\lambda\to0\) with a finite number \(M\) of nonzero modes and \(n\gt M\). Show the self-consistent equation forces \(\kappa\to0\), hence \(\mathcal L_i\to1\) for every mode, \(\gamma\to M/n\), and \(E_g\to\sigma^2\gamma/(1-\gamma)=\sigma^2 M/(n-M)\). Interpret this as ordinary least squares with \(M\) features. [Hint: divide the self-consistent equation by \(\kappa\) and let \(\kappa\to0\); the equation \(1=\sum_i \eta_i/(n\eta_i+\kappa)\) has no positive root when \(n\gt M\).]{.ex-hint}
7.  [challenge]{.ex-tag} Show that when \(n\lt M\) the ridgeless self-consistent equation \(1=\sum_{i=1}^M\eta_i/(n\eta_i+\kappa)\) has a unique positive root \(\kappa\gt0\), so the effective ridge is nonzero even at \(\lambda=0\). Explain why this is the learning-curve statement of the same implicit regularization that keeps the minimum-norm interpolant well-behaved. [Hint: the right side is continuous, strictly decreasing in \(\kappa\), equals \(M/n\gt1\) at \(\kappa=0^+\) when \(M\gt n\), and tends to \(0\) as \(\kappa\to\infty\), so it crosses \(1\) exactly once.]{.ex-hint}
8.  [synthesis]{.ex-tag} Explain in one paragraph, using the effective ranks and the per-mode learnability, why a kernel with a slowly decaying spectrum (large tail effective rank) both interpolates benignly and learns with a shallow power-law exponent, so that safety at interpolation and slowness of learning are two consequences of the same heavy tail. What does this trade-off imply for choosing a kernel's bandwidth?
