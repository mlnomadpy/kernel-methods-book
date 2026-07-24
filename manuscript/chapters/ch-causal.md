---
id: ch-causal
slug: causal-inference-with-kernels
title: Causal Inference with Kernels
part: IX · Kernel Probabilistic Inference
order: 37
tier: advanced
prerequisites:
  - kernel-stein-discrepancy
objectives:
  - Explain the central definitions and claims in Causal Inference with Kernels.
  - Apply the chapter's principal methods and interpret their outputs.
  - >-
    State the assumptions behind formal results and connect them to earlier
    chapters.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-causal.yml
verification_date: null
bibliography:
  - gretton2005hsic
  - fukumizu2008
  - sriperumbudur2010
  - gretton2012
  - song2013cme
  - muandet2017
  - szabo2016dr
  - spirtes2000
  - pearl2009
  - zhang2011kcit
  - newey2003
  - darolles2011
  - angrist1996
  - singh2019kiv
  - muandet2020dualiv
  - miao2018
  - mastouri2021
  - muandet2021cme
---
# Causal Inference with Kernels

<p class="lead">Data record what the world did, not what it would have done: two variables can move in lockstep for years while the question that decides policy, whether changing one would move the other, stays open. Which variable causes which, and what an intervention would do, cannot be settled by any measure of dependence alone, because dependence is symmetric and observation is passive. Earlier chapters of this part turned distributions into RKHS points through [[ch:kernel-mean-embeddings|mean embeddings]] and conditional distributions into operators through [[ch:conditional-mean-embeddings|conditional mean embeddings]], making statistical questions linear algebra; this chapter spends that machinery on the causal ones. First we build a nonparametric test of conditional independence, the engine that constraint-based algorithms use to reconstruct a causal graph from observational data. Second we estimate the effect of an intervention when hidden confounders make ordinary regression lie, using instruments and proxies lifted into an RKHS. Throughout, the kernel buys freedom from assumptions about functional form, but it cannot buy the assumptions that make a causal quantity identifiable. A running theme is which of those assumptions the data can check and which it cannot.</p>

## From dependence to intervention {#dependence-to-intervention}

Every method so far in this part of the book measures how two distributions differ or how two variables depend. Dependence, though, is silent about direction and about intervention. If ice-cream sales and drowning deaths rise together, an embedding reports their dependence faithfully and says nothing about whether banning ice cream would save swimmers. The gap between the statement that \(X\) and \(Y\) are dependent and the statement that \(X\) causes \(Y\) is filled by two extra ingredients, and this chapter supplies a kernel version of each.

The first ingredient is structure. A dependence between \(X\) and \(Z\) can arise because \(X\) causes \(Z\), because \(Z\) causes \(X\), or because a common cause \(Y\) drives both. These are told apart, when they can be at all, by patterns of conditional independence: in the chain \(X \to Y \to Z\) and in the fork \(X \leftarrow Y \to Z\), the variables \(X\) and \(Z\) are dependent but become independent once \(Y\) is held fixed, whereas in the collider \(X \to Y \leftarrow Z\) the reverse happens. Constraint-based causal discovery (Spirtes, Glymour, and Scheines 2000; Pearl 2009) turns a list of such conditional-independence facts into the set of graphs compatible with them. It needs a test that can certify \(X \perp Z \mid Y\) without assuming the variables are Gaussian or the dependence linear, and the next section builds one.

The second ingredient is intervention, which we must define before we can estimate it.

::: {.definition #def-35-1}
[Definition (intervention and structural function)]{.box-title}

In a structural causal model each variable is produced by a deterministic mechanism from its direct causes and an independent noise term. The intervention \(\operatorname{do}(X=x)\) replaces the mechanism for \(X\) by the constant \(x\), leaving every other mechanism intact, and induces the interventional distribution \(P(Y \mid \operatorname{do}(X=x))\). When \(Y = f(X) + e\) with \(e\) the aggregate noise, the *structural function* \(f(x) = \mathbb{E}[Y \mid \operatorname{do}(X=x)]\) is the object of interest, and it need not equal the observational regression \(\mathbb{E}[Y \mid X=x]\).
:::

The inequality \(f(x) \ne \mathbb{E}[Y\mid X=x]\) is the entire difficulty of effect estimation. It fails exactly when a confounder makes \(X\) and the noise \(e\) dependent, so that the regression reads off a mixture of the causal response and the confounder's shadow. Sections on treatment effects below recover \(f\) despite this, first through instruments and then through proxies.

## Dependence as a distance between embeddings {#hsic}

Before conditioning, recall the unconditional measure of dependence from [[ch:kernel-mean-embeddings|the mean-embedding chapter]]. Independence of \((X,Y)\) is the statement \(P_{XY} = P_X \otimes P_Y\), an equality of distributions, so any distance between embeddings is automatically a dependence measure. Feeding the joint and the product of the marginals into the maximum mean discrepancy gives the Hilbert-Schmidt Independence Criterion of Gretton, Bousquet, Smola, and Schölkopf (2005).

:::: {.definition #def-35-2}
[Definition (HSIC, recap)]{.box-title}

For a kernel \(k\) on \(\mathcal X\) and a kernel \(\ell\) on \(\mathcal Y\), the Hilbert-Schmidt Independence Criterion is the squared MMD between the joint and the product of the marginals, \(\operatorname{HSIC}(X,Y) = \operatorname{MMD}^2(P_{XY},\, P_X \otimes P_Y)\). Given \(N\) paired samples it is estimated by

$$ \widehat{\operatorname{HSIC}} = \frac{1}{N^2}\operatorname{Tr}(K H L H), \qquad H = I - \tfrac1N \mathbf 1 \mathbf 1^\top, $$

with \(K, L\) the two Gram matrices and \(H\) the centering matrix. When \(k\) and \(\ell\) are characteristic (Sriperumbudur et al. 2010), \(\operatorname{HSIC}(X,Y)=0\) if and only if \(X \perp Y\).
::::

HSIC is the right tool for marginal independence, and it already powers feature selection and independent-component analysis. But causal discovery asks a harder question. To decide whether \(Y\) screens off \(X\) from \(Z\) we must remove from \(X\) and \(Z\) everything that \(Y\) explains and test only what remains, and that is not a marginal operation. The next section builds the conditional version.

## Conditional independence in an RKHS {#kernel-ci}

The natural RKHS notion of \"everything that \(Y\) explains\" is a regression of feature maps onto the features of \(Y\); the residuals of that regression carry the conditional dependence. Fukumizu, Gretton, Sun, and Schölkopf (2008) make this precise through covariance operators, the RKHS analogues of covariance matrices.

:::: {.definition #def-35-3}
[Definition (cross-covariance and conditional cross-covariance operators)]{.box-title}

Let \(\phi, \psi, \upsilon\) be the feature maps of \(X, Y, Z\). The cross-covariance operator \(\Sigma_{XY}\colon \mathcal H_Y \to \mathcal H_X\) is the centered second moment satisfying \(\langle g, \Sigma_{XY}\, h\rangle_{\mathcal H_X} = \operatorname{Cov}\!\big(g(X), h(Y)\big)\) for all \(g \in \mathcal H_X\) and \(h \in \mathcal H_Y\). The conditional cross-covariance operator partials out \(Z\):

$$ \Sigma_{XY \mid Z} = \Sigma_{XY} - \Sigma_{XZ}\,\Sigma_{ZZ}^{-1}\,\Sigma_{ZY}, $$

the exact operator analogue of the partial covariance of Gaussian theory.
::::

The hope is that \(\Sigma_{XY\mid Z}=0\) captures conditional independence. It very nearly does, with one repair.

:::: {.theorem #thm-35-4}
[Theorem (kernel characterization of conditional independence; Fukumizu et al., 2008)]{.box-title}

Write \(\ddot X = (X, Z)\) for \(X\) augmented by the conditioning variable, with feature map \(\ddot\phi\). Under characteristic kernels and the regularity needed for the operators to be well defined,

$$ \Sigma_{\ddot X\, Y \mid Z} = 0 \quad\Longleftrightarrow\quad X \perp Y \mid Z. $$

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
::::

The augmentation \(\ddot X = (X,Z)\) is not cosmetic. Without it the operator equation \(\Sigma_{XY\mid Z}=0\) states only that the *expected* conditional covariance \(\mathbb{E}_Z\!\big[\operatorname{Cov}(g(X),h(Y)\mid Z)\big]\) vanishes, a strictly weaker condition than conditional independence, since positive and negative conditional dependence at different values of \(Z\) can cancel in the average. Building \(Z\) into the first argument removes that loophole and upgrades the characterization to the full statement (Fukumizu et al. 2008).

The empirical statistic of Zhang, Peters, Janzing, and Schölkopf (2011), called KCIT, estimates \(\Sigma_{\ddot X\, Y\mid Z}\) by kernel ridge regression. Center every Gram matrix, \(\tilde K = HKH\). Regressing a feature onto the span of the instruments' features \(\{\upsilon(z_i)\}\) with ridge \(\varepsilon\) predicts through the smoother \(\tilde K_Z(\tilde K_Z + \varepsilon I)^{-1}\), so the residual is applied by the operator

$$ R_Z = I - \tilde K_Z(\tilde K_Z + \varepsilon I)^{-1} = \varepsilon\,(\tilde K_Z + \varepsilon I)^{-1}, $$

the second equality because \((\tilde K_Z+\varepsilon I) - \tilde K_Z = \varepsilon I\). The residualized matrices \(\tilde K_{\ddot X\mid Z} = R_Z \tilde K_{\ddot X} R_Z\) and \(\tilde K_{Y\mid Z} = R_Z \tilde K_Y R_Z\) hold the parts of \(\ddot X\) and \(Y\) that \(Z\) does not explain, and their normalized inner product is the test statistic

$$ T_{\mathrm{CI}} = \frac{1}{n}\operatorname{Tr}\!\big(\tilde K_{\ddot X\mid Z}\,\tilde K_{Y\mid Z}\big). $$

Under \(H_0\colon X\perp Y\mid Z\) the statistic \(T_{\mathrm{CI}}\) has no fixed null law: it converges to a weighted sum of independent \(\chi^2_1\) variables whose weights are the products \(\lambda_i \mu_j\) of the eigenvalues of the two residual operators (Zhang et al. 2011), the same spectral shape that governs the two-sample MMD null (Gretton et al. 2012). In practice one either matches a two-parameter Gamma distribution to the first two moments of that mixture or samples it directly from the empirical eigenvalues. A permutation test, the standard calibration for marginal HSIC, is *not* valid here: shuffling the pairing destroys the dependence on \(Z\) along with any conditional dependence, so it does not draw from the conditional null. That conditional independence admits no simple resampling scheme is one honest reason nonparametric CI testing is genuinely hard.

:::: {.algorithm #algo-35-1}
[Algorithm (kernel conditional-independence test, KCIT)]{.box-title}

::: algo-io
[Input]{.algo-lab} Samples \(\{(x_i,y_i,z_i)\}_{i=1}^n\); kernels for \(X, Y, Z\); ridge \(\varepsilon \gt 0\).

[Output]{.algo-lab} Statistic \(T_{\mathrm{CI}}\) and an approximate \(p\)-value for \(H_0\colon X\perp Y\mid Z\).
:::

1.  Form Gram matrices \(K_{\ddot X}\) on the augmented points \(\ddot x_i=(x_i,z_i)\), \(K_Y\) on \(y_i\), and \(K_Z\) on \(z_i\); center each as \(\tilde K = HKH\).
2.  Build the residual-maker \(R_Z = \varepsilon(\tilde K_Z + \varepsilon I)^{-1}\).
3.  Residualize: \(\tilde K_{\ddot X\mid Z} = R_Z \tilde K_{\ddot X} R_Z\) and \(\tilde K_{Y\mid Z} = R_Z \tilde K_Y R_Z\).
4.  Compute \(T_{\mathrm{CI}} = \tfrac1n \operatorname{Tr}(\tilde K_{\ddot X\mid Z}\,\tilde K_{Y\mid Z})\).
5.  Approximate the null by a Gamma fit to its first two moments, or by Monte Carlo from the eigenvalues \(\{\lambda_i\},\{\mu_j\}\); report \(p = \Pr[\text{null} \ge T_{\mathrm{CI}}]\).
::::

::::: {.example #example-35-1}
[Example (a conditional-independence test on a chain)]{.box-title}

:::: wex
::: wex-setup
A tiny sample of \(n=8\) from the chain \(X \to Y \to Z\), designed so \(X \perp Z \mid Y\) holds exactly: the conditioner \(Y\) has two strata, and inside each stratum the \((X,Z)\) pairs form a balanced \(2\times2\) grid, so \(X\) and \(Z\) are independent given \(Y\) while both drift upward with \(Y\).

  row                                    1   2   3   4   5   6   7   8
  -------------------------------------- --- --- --- --- --- --- --- ---
  \(X\)   0   0   1   1   2   2   3   3
  \(Y\)   0   0   0   0   1   1   1   1
  \(Z\)   0   1   0   1   2   3   2   3

Gaussian kernels with bandwidth set by the median heuristic, which equals \(1\) for each of \(X, Y, Z\) here; ridge \(\varepsilon=10^{-3}\).
:::

1.  [Measure marginal dependence with HSIC.]{.wex-op} The biased estimator \(\tfrac1{n^2}\operatorname{Tr}(\tilde K_A \tilde K_B)\) gives \(\operatorname{HSIC}(X,Z)=0.0844\), \(\operatorname{HSIC}(X,Y)=0.0572\), and \(\operatorname{HSIC}(Y,Z)=0.0572\). All three are clearly positive: every adjacent pair is dependent, and so are the endpoints \(X\) and \(Z\).
2.  [Build the residual-maker for \(Y\).]{.wex-op} Center \(K_Y\) and form \(R_Y = \varepsilon(\tilde K_Y + \varepsilon I)^{-1}\), the operator that strips out whatever the features of \(Y\) can predict.
3.  [Residualize and take the trace.]{.wex-op} With \(\ddot X = (X,Y)\), compute \(\tilde K_{\ddot X\mid Y} = R_Y \tilde K_{\ddot X} R_Y\) and \(\tilde K_{Z\mid Y} = R_Y \tilde K_Z R_Y\). Their trace statistic is \(T_{\mathrm{CI}}(X,Z\mid Y) = 0.000000\), a ratio of \(1.3\times 10^{-12}\) to the marginal \(\operatorname{HSIC}(X,Z)\): the dependence between \(X\) and \(Z\) is entirely explained by \(Y\).
4.  [Confirm the statistic is not vacuous.]{.wex-op} Had \(Z\) instead tracked \(X\) inside each stratum, breaking conditional independence, the identical computation returns \(T_{\mathrm{CI}} = 0.097\), so the test does discriminate.

**Reading.** The endpoints of a chain are marginally dependent but conditionally independent given the middle variable, and the kernel statistic reports exactly that, dropping from \(0.0844\) to numerical zero once \(Y\) is partialled out. Note the honest limit: the fork \(X \leftarrow Y \to Z\) and the reversed chain \(Z \to Y \to X\) imply the same conditional independence, so this test constrains the graph without orienting every edge.
::::

**Verification artifact.** checks/example-ch-causal-example-35-1.json records the example source hash and verification scope.
:::::

## Estimating effects under confounding {#treatment-effects}

Turn now from discovery to estimation. When every confounder of \(X\) and \(Y\) is observed, the back-door adjustment reduces the causal effect to a regression, and the [[ch:conditional-mean-embeddings|conditional mean embedding]] estimates it directly. The interesting and common case is a confounder we cannot observe, where regression is biased and we need extra leverage from the data. Two kinds of leverage admit clean kernel treatments: an instrument that perturbs the treatment, and a pair of proxies that shadow the hidden confounder.

### Kernel instrumental variables {#kernel-iv}

Write the structural model \(Y = f(X) + e\), where the noise \(e\) absorbs the unobserved confounder and is therefore correlated with the treatment \(X\). Because \(\mathbb{E}[e\mid X]\ne 0\), the regression of \(Y\) on \(X\), kernel ridge included, estimates \(f\) plus the confounder's imprint, not \(f\). An instrument breaks the deadlock.

::: {.definition #def-35-5}
[Definition (instrument)]{.box-title}

A variable \(Z\) is an *instrument* for the effect of treatment \(X\) on outcome \(Y\) in the model \(Y=f(X)+e\) if it is (i) *relevant*, meaning \(Z\) is dependent on \(X\); (ii) *exogenous*, meaning \(\mathbb{E}[e\mid Z]=0\), so \(Z\) is unconfounded with the outcome noise; and (iii) subject to the *exclusion* restriction, meaning \(Z\) influences \(Y\) only through \(X\).
:::

Exogeneity converts the unobservable structural equation into an observable one. Taking the conditional expectation of \(Y=f(X)+e\) given \(Z=z\) and using \(\mathbb{E}[e\mid Z]=0\) lands on the crux identity of the whole method.

:::: {.proposition #prop-35-6}
[Proposition (the two-stage target)]{.box-title}

If \(Y = f(X) + e\) with \(f\in\mathcal H_X\) and \(\mathbb{E}[e\mid Z]=0\), then for every \(z\),

$$ \mathbb{E}[Y\mid Z=z] = \langle f,\ \mu_{X\mid Z=z}\rangle_{\mathcal H_X}, \qquad \mu_{X\mid Z=z} = \mathbb{E}[\phi(X)\mid Z=z]. $$

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

By the [[ch:kernels-and-rkhs|reproducing property]], \(f(X)=\langle f,\phi(X)\rangle_{\mathcal H_X}\). Since \(f\) is a fixed element, the inner product against it is a continuous linear functional and commutes with the (Bochner) conditional expectation, so

$$ \mathbb{E}[f(X)\mid Z=z] = \big\langle f,\ \mathbb{E}[\phi(X)\mid Z=z]\big\rangle_{\mathcal H_X} = \langle f,\ \mu_{X\mid Z=z}\rangle_{\mathcal H_X}. $$

Adding \(\mathbb{E}[e\mid Z=z]=0\) gives the claim. [\(\square\)]{.qed}
::::

The unknown \(f\) is thus pinned down by an equation relating two observable objects, the outcome regression on the left and the conditional mean embedding of the treatment on the right. Solving it for \(f\) is a Fredholm integral equation of the first kind, and it is ill-posed: the conditional-expectation operator \(z \mapsto \mu_{X\mid Z=z}\) is compact, so its inverse is unbounded and amplifies sampling noise without limit (Newey and Powell 2003; Darolles et al. 2011). Regularization is not a convenience here, it is a necessity.

Singh, Sahani, and Gretton (2019) solve the equation in two ridge-regularized stages, the RKHS lift of classical two-stage least squares. Stage one estimates the conditional mean embedding \(\mu_{X\mid Z=z}\) from a first sample \(\{(x_i,z_i)\}_{i=1}^n\) by kernel ridge regression, giving \(\hat\mu(z)=\sum_i \beta_i(z)\,\phi(x_i)\) with weights \(\beta(z)=(K_Z+n\lambda I)^{-1} k_Z(\cdot,z)\). Stage two regresses the outcome on those embeddings, using a second sample \(\{(y_j,\tilde z_j)\}_{j=1}^m\) to fit

$$ \hat f = \arg\min_{f\in\mathcal H_X}\ \frac1m\sum_{j=1}^m \big(y_j - \langle f, \hat\mu(\tilde z_j)\rangle\big)^2 + \xi\|f\|_{\mathcal H_X}^2. $$

By the representer theorem \(\hat f = \sum_i \alpha_i \phi(x_i)\), and collecting the stage-one weights into a matrix \(B\in\mathbb{R}^{n\times m}\) with \(B_{ij}=\beta_i(\tilde z_j)\), the prediction at \(\tilde z_j\) is \(\langle \hat f,\hat\mu(\tilde z_j)\rangle = (B^\top K_X \alpha)_j\). The stage-two problem is then a ridge regression whose inputs are the estimated embeddings, with Gram matrix \(G = B^\top K_X B\), solved in closed form by \(c=(G+m\xi I)^{-1} y\) and \(\alpha = Bc\).

:::: {.algorithm #algo-35-2}
[Algorithm (kernel instrumental-variable regression, KIV)]{.box-title}

::: algo-io
[Input]{.algo-lab} Stage-1 sample \(\{(x_i,z_i)\}_{i=1}^n\), stage-2 sample \(\{(y_j,\tilde z_j)\}_{j=1}^m\); kernels \(k_X, k_Z\); ridges \(\lambda,\xi \gt 0\).

[Output]{.algo-lab} Structural function \(\hat f(x)=\sum_i \alpha_i\, k_X(x_i,x)\).
:::

1.  Stage 1: form \(K_Z\) on the stage-1 instruments and \(\Gamma_{ij}=k_Z(z_i,\tilde z_j)\); solve \(B = (K_Z + n\lambda I)^{-1}\Gamma\) for the conditional-mean-embedding weights.
2.  Form the stage-1 treatment Gram \(K_X\) and the stage-2 embedding Gram \(G = B^\top K_X B\).
3.  Stage 2: solve \(c = (G + m\xi I)^{-1} y\) and set \(\alpha = B c\).
4.  Return \(\hat f(\cdot) = \sum_i \alpha_i\, k_X(x_i,\cdot)\); the effect of moving \(X\) from \(x\) to \(x'\) is \(\hat f(x')-\hat f(x)\).
::::

::::: {.example #example-35-2}
[Example (recovering a structural slope with an instrument)]{.box-title}

:::: wex
::: wex-setup
A confounded linear model, all variables centered, \(n=6\). Instrument \(z=(-1,-1,-1,1,1,1)\); hidden confounder \(u=(-1,1,0,-1,1,0)\), chosen sample-uncorrelated with \(z\); treatment \(x = z + u\); outcome \(y = \beta x + \gamma u\) with structural slope \(\beta=2\) and confounding strength \(\gamma=3\). The sample covariances are \(\widehat{\operatorname{Cov}}(z,u)=0\) (valid instrument) and \(\widehat{\operatorname{Cov}}(x,u)=0.667\) (the treatment is confounded). Linear kernels \(k_X(x,x')=xx'\), \(k_Z(z,z')=zz'\), for which KIV reduces to two-stage least squares.
:::

1.  [See the bias in plain regression.]{.wex-op} The OLS slope is \(\hat\beta_{\mathrm{OLS}} = \widehat{\operatorname{Cov}}(x,y)/\widehat{\operatorname{Var}}(x) = 3.2\), far from \(\beta=2\). The gap is the confounding term \(\gamma\,\widehat{\operatorname{Cov}}(x,u)/\widehat{\operatorname{Var}}(x) = 3\times 0.667/1.667 = 1.2\).
2.  [Stage 1: regress treatment on instrument.]{.wex-op} The slope is \(\hat a = \widehat{\operatorname{Cov}}(z,x)/\widehat{\operatorname{Var}}(z) = 1.0\), giving fitted treatments \(\hat x = \hat a\, z = z\). This \(\hat x\) is the conditional mean embedding in the linear case, and it is purged of \(u\) because \(z\) is uncorrelated with \(u\).
3.  [Stage 2: regress outcome on the fitted treatment.]{.wex-op} The slope is \(\hat\beta_{\mathrm{2SLS}} = \widehat{\operatorname{Cov}}(\hat x,y)/\widehat{\operatorname{Var}}(\hat x) = 2.0\), recovering the structural slope exactly.
4.  [Confirm with the full KIV formula.]{.wex-op} Running the matrix algorithm with linear-kernel Grams and ridges \(\lambda=\xi=10^{-6}\) yields \(\hat\beta_{\mathrm{KIV}} = 2.0\), agreeing with the two-stage computation.

**Reading.** The confounder biases ordinary regression by \(1.2\), from the truth \(2\) up to \(3.2\). Routing the fit through the instrument, which is clean of the confounder, restores the causal slope. The kernel version replaces the two scalar regressions by two ridge regressions in feature space, so the same logic recovers a nonlinear structural function \(f\).
::::

**Verification artifact.** checks/example-ch-causal-example-35-2.json records the example source hash and verification scope.
:::::

The two-sample split, stage one on one half and stage two on the other, is what keeps the estimator honest, preventing the stage-two fit from chasing the stage-one estimation error. An alternative avoids the split entirely: Muandet, Mehrjou, Lee, and Raj (2020) rewrite instrumental-variable regression as a single convex-concave saddle-point problem (DualIV), replacing the nested regression by a minimax objective that is often easier to optimize and to analyze.

### Proximal causal learning {#proximal}

Instruments are not always available. Proximal causal learning asks a different favor of the data: instead of a variable that pokes the treatment, it uses two *proxies* of the unmeasured confounder \(U\), a treatment-side proxy \(W\) and an outcome-side proxy \(Z\), which are noisy views of \(U\) satisfying conditional-independence restrictions (Miao, Geng, and Tchetgen Tchetgen 2018). Identification runs through a *bridge function* \(h\) that solves the conditional moment equation

$$ \mathbb{E}\big[\, Y - h(W, X) \mid Z, X \,\big] = 0, $$

after which the interventional mean is recovered by averaging out the proxy, \(\mathbb{E}[Y\mid \operatorname{do}(X=x)] = \mathbb{E}_W[h(W,x)]\). This moment equation has the same first-kind, ill-posed structure as the IV equation, and Mastouri et al. (2021) solve it with the same kernel two-stage regression and maximum-moment-restriction machinery, producing a proximal estimator that inherits the RKHS consistency theory. Proximal learning thus extends kernel causal estimation to settings where the confounder is entirely hidden and only its shadows are observed.

## Beyond the mean: distributional and counterfactual effects {#distributional-effects}

A structural function returns the mean outcome under intervention, but a mean can hide the story: a treatment that helps half the population and harms the other half has zero average effect. Because the entire apparatus is built on embeddings, nothing forces us to stop at the mean. Replacing the scalar outcome \(Y\) by its feature map \(\phi_Y(Y)\) and running the same estimators embeds the whole interventional distribution as a mean embedding \(\mu_{Y\mid \operatorname{do}(x)} = \mathbb{E}[\phi_Y(Y)\mid \operatorname{do}(X=x)]\), the counterfactual mean embedding of Muandet, Kanagawa, Saengkyongam, and Marukatat (2021). From it one reads any smooth functional of the interventional law, its variance, its quantiles, or the probability of exceeding a threshold, by the ordinary embedding calculus of [[ch:kernel-mean-embeddings|the mean-embedding chapter]]. When the treatment itself is a distribution or a function rather than a point, the same lift connects to [[ch:distribution-regression|distribution regression]] (Szabó et al. 2016), closing the loop with the rest of this part of the book.

## What kernels buy, and what they do not {#assumptions}

It is worth stating plainly what the kernel has and has not done. In every method of this chapter the RKHS bought exactly one thing: freedom from parametric form. KCIT tests conditional independence without assuming linearity or Gaussianity; KIV and the proximal estimator recover a structural function without assuming it is a straight line. That freedom is substantive, because a linearity assumption quietly built into a causal estimate is itself an untestable causal claim, and the nonparametric test avoids the false negatives that a linear correlation test suffers when the dependence is real but curved.

Identification, however, is a separate layer, and the kernel is silent there.

::: {.remark}
[The assumptions the data cannot check]{.box-title}

Constraint-based discovery reads structure from conditional independences only under *faithfulness*, the assumption that the sole independences present are those the graph forces; and even then it returns a Markov equivalence class, not a unique graph, so the chain \(X\to Y\to Z\) and the fork \(X\leftarrow Y\to Z\) stay indistinguishable by conditional-independence tests alone (Spirtes et al. 2000; Pearl 2009). Instrumental-variable estimation is consistent only if the instrument is valid, yet exogeneity and exclusion are statements about the unobserved noise that no amount of data can verify (Angrist, Imbens, and Rubin 1996). Proximal learning trades those for completeness conditions on the proxies, equally uncheckable. A perfect kernel fit to a misidentified estimand is a precise answer to the wrong question. The honest summary is that kernels supply consistent, assumption-lean estimators of causal quantities *once those quantities are identified*, and identification always rests on domain assumptions imported from outside the sample.
:::

## Summary {#summary}

Causal inference splits into two problems, and embeddings answer a piece of each. For discovery, independence is a distance between a joint embedding and a product of marginals (HSIC), and conditional independence is the vanishing of a conditional cross-covariance operator, estimated by the residual-trace statistic of KCIT; this is the nonparametric engine of constraint-based structure learning. For effect estimation under hidden confounding, exogeneity turns the structural equation into an ill-posed integral equation whose solution is the two-stage kernel instrumental-variable regression, with proximal learning covering the case where only proxies of the confounder are seen, and counterfactual mean embeddings lifting the answer from the mean effect to the whole interventional distribution. In every case the kernel removes assumptions about functional form but not the identifying assumptions, faithfulness, instrument validity, proxy completeness, that make the causal question answerable at all. The next chapter turns from these inferential uses of embeddings to [[ch:distribution-regression|learning when the training example is itself a distribution]].

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

For **Causal Inference with Kernels**, do not apply a displayed formula without checking its domain, statistical assumptions, and numerical conditioning. Avoid selecting kernels or hyperparameters on test data, and do not interpret an optimization residual as a generalization guarantee. When the method is computational, report preprocessing, kernel parameters, regularization, solver tolerance, condition diagnostics, runtime, and a non-kernel baseline. When the result is theoretical, distinguish sufficient conditions from necessary ones and finite-sample claims from asymptotic statements.

## Summary and further reading {#summary-and-further-reading}

This chapter established explain the central definitions and claims in Causal Inference with Kernels; Apply the chapter's principal methods and interpret their outputs; State the assumptions behind formal results and connect them to earlier chapters. Revisit the assumptions attached to each formal result before transferring it to a new setting. For primary and extended treatments, consult [@gretton2005hsic], [@fukumizu2008], [@sriperumbudur2010].

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} The worked example certified \(X \perp Z \mid Y\) for data generated by the chain \(X\to Y\to Z\). Show that the fork \(X \leftarrow Y \to Z\) and the reversed chain \(Z\to Y\to X\) imply the same single conditional independence and no other. Conclude that a conditional-independence test alone cannot orient the edges among these three variables, and name one extra source of information, from the modeling assumptions or the data-collection design, that would break the tie.
2.  [computation]{.ex-tag} Verify the residual-maker identity \(I - \tilde K_Z(\tilde K_Z + \varepsilon I)^{-1} = \varepsilon(\tilde K_Z + \varepsilon I)^{-1}\) used in KCIT. Then describe the two limits of \(R_Z\): what operator does it approach as \(\varepsilon \to 0\), and what as \(\varepsilon \to \infty\), and explain why each extreme destroys the test, one by removing too much and the other by partialling out nothing.
3.  [proof]{.ex-tag} Show that with linear kernels \(k_X(x,x')=xx'\) and \(k_Z(z,z')=zz'\) on centered scalar data, the KIV algorithm reduces to two-stage least squares, \(\hat\beta = \widehat{\operatorname{Cov}}(\hat X, Y)/\widehat{\operatorname{Var}}(\hat X)\) with \(\hat X\) the least-squares projection of \(X\) on \(Z\).
    Hint

    ::: hint-body
    The linear-kernel Gram \(K_Z = zz^\top\) is rank one, so the stage-1 embedding \(\hat\mu(z)\) is proportional to the fitted value \(\hat x = \hat a z\); as \(\lambda,\xi\to 0\) the stage-2 ridge regression on that scalar embedding is ordinary least squares of \(Y\) on \(\hat X\). Compare with Worked Example 2, where these steps give \(1.0\) then \(2.0\).
    :::
4.  [computation]{.ex-tag} For the confounded model of Worked Example 2, derive the omitted-variable bias of ordinary least squares, \(\hat\beta_{\mathrm{OLS}} \to \beta + \gamma\,\widehat{\operatorname{Cov}}(X,U)/\widehat{\operatorname{Var}}(X)\), starting from \(Y=\beta X + \gamma U\). Check it against the sample: with \(\gamma=3\), \(\widehat{\operatorname{Cov}}(X,U)=0.667\), and \(\widehat{\operatorname{Var}}(X)=1.667\), confirm the bias is \(1.2\) and hence \(\hat\beta_{\mathrm{OLS}}=3.2\).
5.  [exploration]{.ex-tag} Re-run the check script for Worked Example 1 after replacing \(Z\) by values that track \(X\) inside each stratum of \(Y\) (so the chain becomes a structure in which \(Y\) no longer screens off \(X\) from \(Z\)). Confirm that the KCIT statistic jumps from numerical zero to \(0.097\), and explain in one sentence why a causal-discovery algorithm needs exactly this behavior, a near-zero statistic under independence and a clearly positive one otherwise.
6.  [proof]{.ex-tag} Interpret the KCIT statistic as a partial dependence. Dropping the augmentation, show that \(\tfrac1n\operatorname{Tr}(\tilde K_{X\mid Z}\,\tilde K_{Y\mid Z})\) is, up to normalization, the HSIC between the RKHS residuals of \(X\) and of \(Y\) after each is regressed on \(Z\), and relate this to the partial-covariance reading of \(\Sigma_{XY\mid Z}=\Sigma_{XY}-\Sigma_{XZ}\Sigma_{ZZ}^{-1}\Sigma_{ZY}\).
    Hint

    ::: hint-body
    The residual-maker \(R_Z\) applied to a centered feature matrix returns the regression residual in feature space; the trace of the product of two such residual Grams is the empirical HSIC of the residuals. The subtracted term \(\Sigma_{XZ}\Sigma_{ZZ}^{-1}\Sigma_{ZY}\) is precisely the part of the cross-covariance mediated by \(Z\).
    :::
7.  [challenge]{.ex-tag} Suppose the exclusion restriction fails, so the instrument has a direct effect: \(Y = \beta X + \delta Z + \gamma U\) with \(\delta \ne 0\), while stage 1 is \(X = aZ + U\) with \(\operatorname{Cov}(Z,U)=0\). Show that two-stage least squares now estimates \(\beta + \delta/a\) rather than \(\beta\), so a small direct effect masquerades as a change in the structural slope. Discuss why no test on the observed \((X,Y,Z)\) can detect this bias.
    Hint

    ::: hint-body
    Project \(Y\) on the fitted \(\hat X = aZ\): the stage-2 slope is \(\operatorname{Cov}(aZ,\ \beta X + \delta Z + \gamma U)/\operatorname{Var}(aZ)\). Use \(\operatorname{Cov}(Z,U)=0\) and \(\operatorname{Cov}(Z,X)=a\operatorname{Var}(Z)\) to get \((a\beta+\delta)/a\). The extra term involves only the instrument's observed relations to \(X\) and \(Y\), which are equally consistent with a valid but stronger or weaker instrument.
    :::
:::
