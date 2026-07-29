---
example_code_policy: visible-for-executable
id: ch-causal
slug: causal-inference-with-kernels
title: Causal Inference with Kernels
part: 'VIII · Conditional, Stein, and Causal Inference'
order: 47
tier: advanced
prerequisites:
  - kernel-stein-discrepancy
objectives:
  - >-
    Explain why dependence does not identify intervention and separate
    statistical from causal assumptions.
  - >-
    Construct HSIC and KCIT statistics while stating when their null
    calibrations are valid.
  - >-
    Derive adjustment and RKHS balancing estimators under consistency,
    exchangeability, and positivity.
  - >-
    Derive kernel instrumental-variable regression and diagnose weak
    instruments, invalid exclusion, regularization, and ill-posedness.
  - >-
    Distinguish pointwise effects, interventional mean embeddings, and nonlinear
    distributional summaries.
  - >-
    State proximal bridge completeness conditions and perform sensitivity and
    nonidentification analyses.
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
narrative_link_policy: exact
---
# Causal Inference with Kernels

<p class="lead">Data record what the world did, not what it would have done: two variables can move in lockstep for years while the question that decides policy, whether changing one would move the other, stays open. Which variable causes which, and what an intervention would do, cannot be settled by any measure of dependence alone, because dependence is symmetric and observation is passive. Earlier chapters of this part turned distributions into RKHS points through [[ch:kernel-mean-embeddings|mean embeddings]] and conditional distributions into operators through [[ch:conditional-mean-embeddings|conditional mean embeddings]], making statistical questions linear algebra; this chapter spends that machinery on the causal ones. First we build a nonparametric test of conditional independence, the engine that constraint-based algorithms use to reconstruct a causal graph from observational data. Second we estimate the effect of an intervention when hidden confounders make ordinary regression lie, using instruments and proxies lifted into an RKHS. Throughout, the kernel buys freedom from assumptions about functional form, but it cannot buy the assumptions that make a causal quantity identifiable. A running theme is which of those assumptions the data can check and which it cannot.</p>

## From dependence to intervention {#dependence-to-intervention}

Every method so far in this part of the book measures how two distributions differ or how two variables depend. Dependence, though, is silent about direction and about intervention. If ice-cream sales and drowning deaths rise together, an embedding reports their dependence faithfully and says nothing about whether banning ice cream would save swimmers. The gap between the statement that \(X\) and \(Y\) are dependent and the statement that \(X\) causes \(Y\) is filled by two extra ingredients, and this chapter supplies a kernel version of each.

The first ingredient is structure. A dependence between \(X\) and \(Z\) can arise because \(X\) causes \(Z\), because \(Z\) causes \(X\), or because a common cause \(Y\) drives both. These are told apart, when they can be at all, by patterns of conditional independence: in the chain \(X \to Y \to Z\) and in the fork \(X \leftarrow Y \to Z\), the variables \(X\) and \(Z\) are dependent but become independent once \(Y\) is held fixed, whereas in the collider \(X \to Y \leftarrow Z\) the reverse happens. Constraint-based causal discovery (Spirtes, Glymour, and Scheines 2000; Pearl 2009) turns a list of such conditional-independence facts into the set of graphs compatible with them. It needs a test that can certify \(X \perp Z \mid Y\) without assuming the variables are Gaussian or the dependence linear, and the next section builds one.

The second ingredient is intervention, which we must define before we can estimate it.

::: {.definition #def-35-1}
[Definition (intervention and structural function)]{.box-title}

In a structural causal model each variable is produced by a deterministic mechanism from its direct causes and an independent noise term. The intervention \(\operatorname{do}(X=x)\) replaces the mechanism for \(X\) by the constant \(x\), leaving every other mechanism intact, and induces the interventional distribution \(P(Y \mid \operatorname{do}(X=x))\). When \(Y = f(X) + e\) with centered aggregate noise \(\mathbb E[e]=0\), the *structural function* \(f(x) = \mathbb{E}[Y \mid \operatorname{do}(X=x)]\) is the object of interest, and it need not equal the observational regression \(\mathbb{E}[Y \mid X=x]\).
:::

The possible inequality \(f(x) \ne \mathbb{E}[Y\mid X=x]\) is the entire difficulty of effect estimation. The equality fails when a confounder makes \(X\) and the noise \(e\) dependent, so that the regression reads off a mixture of the causal response and the confounder's shadow. Sections on treatment effects below recover \(f\) despite this, first through instruments and then through proxies.

<figure class="viz" data-figure="confounding-intervention" data-alt="A confounded observational sample has an upward fitted regression line even though the structural intervention response slopes downward. The comparison shows that association and causal effect can have opposite signs."><figcaption>Hidden confounding can reverse the sign: the observational regression rises because \(U\) drives both treatment and outcome, while the intervention curve falls because setting \(X\) breaks the arrow from \(U\) into \(X\). A more flexible kernel can fit either curve, but only the causal assumptions determine which curve is the estimand.</figcaption></figure>

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

Let \(\mathcal X,\mathcal Y,\mathcal Z\) be standard Borel spaces and let \(P_{XYZ}\) be a Borel probability law. Write \(\ddot X=(X,Z)\). Assume the kernels on \(\mathcal X\times\mathcal Z\), \(\mathcal Y\), and \(\mathcal Z\) are measurable and bounded, the product kernel on \((\mathcal X\times\mathcal Z)\times\mathcal Y\) is characteristic to the signed measures used in the proof, and the conditional covariance operator is defined through the Moore-Penrose inverse on \(\overline{\operatorname{ran}\Sigma_{ZZ}}\). Equivalently, require the relevant covariance functions to lie in that range. Then

$$ \Sigma_{\ddot X\, Y \mid Z} = 0 \quad\Longleftrightarrow\quad X \perp Y \mid Z. $$

**Assumptions.** The boundedness assumption ensures all feature maps are Bochner square-integrable. Characteristicness is required for the relevant product-domain measure class, not merely for each marginal kernel in isolation. The inverse is not assumed bounded on the whole RKHS.

**Proof status.** The equivalence is imported from Fukumizu et al. (2008), Theorem 4. The chapter explains the augmentation but does not reproduce the operator-theoretic proof.
::::

The augmentation \(\ddot X = (X,Z)\) is not cosmetic. Without it the operator equation \(\Sigma_{XY\mid Z}=0\) states only that the *expected* conditional covariance \(\mathbb{E}_Z\!\big[\operatorname{Cov}(g(X),h(Y)\mid Z)\big]\) vanishes, a strictly weaker condition than conditional independence, since positive and negative conditional dependence at different values of \(Z\) can cancel in the average. Building \(Z\) into the first argument removes that loophole and upgrades the characterization to the full statement (Fukumizu et al. 2008).

The empirical statistic of Zhang, Peters, Janzing, and Schölkopf (2011), called KCIT, estimates \(\Sigma_{\ddot X\, Y\mid Z}\) by kernel ridge regression. Center every Gram matrix, \(\tilde K = HKH\). Regressing a feature onto the span of the instruments' features \(\{\upsilon(z_i)\}\) with ridge \(\varepsilon\) predicts through the smoother \(\tilde K_Z(\tilde K_Z + \varepsilon I)^{-1}\), so the residual is applied by the operator

$$ R_Z = I - \tilde K_Z(\tilde K_Z + \varepsilon I)^{-1} = \varepsilon\,(\tilde K_Z + \varepsilon I)^{-1}, $$

the second equality because \((\tilde K_Z+\varepsilon I) - \tilde K_Z = \varepsilon I\). The residualized matrices \(\tilde K_{\ddot X\mid Z} = R_Z \tilde K_{\ddot X} R_Z\) and \(\tilde K_{Y\mid Z} = R_Z \tilde K_Y R_Z\) hold the parts of \(\ddot X\) and \(Y\) that \(Z\) does not explain, and their normalized inner product is the test statistic

$$ T_{\mathrm{CI}} = \frac{1}{n}\operatorname{Tr}\!\big(\tilde K_{\ddot X\mid Z}\,\tilde K_{Y\mid Z}\big). $$

Under \(H_0\colon X\perp Y\mid Z\) the statistic \(T_{\mathrm{CI}}\) has no fixed null law: it converges to a weighted sum of independent \(\chi^2_1\) variables whose weights are the products \(\lambda_i \mu_j\) of the eigenvalues of the two residual operators (Zhang et al. 2011), the same spectral shape that governs the two-sample MMD null (Gretton et al. 2012). In practice one either matches a two-parameter Gamma distribution to the first two moments of that mixture or samples it directly from the empirical eigenvalues. A permutation test, the standard calibration for marginal HSIC, is *not* valid here: shuffling the pairing destroys the dependence on \(Z\) along with any conditional dependence, so it does not draw from the conditional null. That conditional independence admits no simple resampling scheme is one honest reason nonparametric CI testing is genuinely hard.

The asymptotic statement requires more than the population equivalence. The observations must be i.i.d.; kernels must be bounded and measurable; the empirical residual operators must converge to their population counterparts; the ridge sequence must tend to zero slowly enough to control inverse amplification; and the empirical spectral mixture must consistently approximate the null law. Zhang et al. (2011), Sections 2.2 and 2.3 and Theorems 1 and 2, give the construction and null approximation. A Gamma fit is a computational approximation, not an exact finite-sample calibration. Rejecting \(H_0\) establishes residual statistical dependence under the tested conditioning set. It does not estimate an intervention, establish an edge direction, or quantify a treatment effect.

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

Here is the executable core. The construction deliberately includes a
conditional-dependence witness, because a tiny statistic is meaningful only
if the same pipeline reacts when the null is false.

```python
import numpy as np

X = np.array([0, 0, 1, 1, 2, 2, 3, 3.], float)
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1.], float)
Z = np.array([0, 1, 0, 1, 2, 3, 2, 3.], float)
n = len(X)
H = np.eye(n) - np.ones((n, n)) / n

def gram(a):
    a = np.atleast_2d(a).T if a.ndim == 1 else a
    d2 = np.maximum(
        np.sum(a*a, 1)[:, None] + np.sum(a*a, 1)[None, :]
        - 2*np.matmul(a, a.T), 0
    )
    sigma = np.median(np.sqrt(d2[np.triu_indices(n, 1)]))
    return np.exp(-d2 / (2*sigma**2))

def centered(a):
    return np.linalg.multi_dot([H, gram(a), H])

Kx, Ky, Kz = centered(X), centered(Y), centered(Z)
hsic_xz = np.trace(np.matmul(Kx, Kz)) / n**2
ridge = 1e-3
Ry = ridge * np.linalg.solve(Ky + ridge*np.eye(n), np.eye(n))
Kxy = centered(np.column_stack([X, Y]))
residual_xy = np.linalg.multi_dot([Ry, Kxy, Ry])
residual_z = np.linalg.multi_dot([Ry, Kz, Ry])
t_null = np.trace(np.matmul(residual_xy, residual_z)) / n

Z_dep = X.copy()
residual_dep = np.linalg.multi_dot([Ry, centered(Z_dep), Ry])
t_alt = np.trace(np.matmul(residual_xy, residual_dep)) / n
assert np.isclose(hsic_xz, 0.0844, atol=5e-5)
assert t_null < 1e-10
assert np.isclose(t_alt, 0.097063, atol=1e-6)
print(hsic_xz, t_null, t_alt)
```
::::
:::::

## Estimating effects under confounding {#treatment-effects}

Turn now from discovery to estimation. A test asks whether a conditional independence is compatible with the observed law. An effect estimator targets a functional of an interventional law and is valid only after identification assumptions connect that law to observations. A small KCIT \(p\)-value cannot be converted into an average treatment effect, and failure to reject cannot certify ignorability.

When every confounder is observed, adjustment and balancing identify effects. When a confounder is hidden, extra variables must carry identifying information: an instrument can perturb treatment without entering the outcome mechanism, or a pair of proxies can shadow the hidden confounder. These are different models and should never be pooled into one vague claim that kernels "handle confounding."

### Adjustment and kernel balancing {#kernel-balancing}

Kernel balancing can drive a rich RKHS discrepancy toward zero, but overlap determines the price. When the treated and comparison covariates barely overlap, exact balance concentrates the weights and destroys effective sample size.

<figure class="viz" data-figure="kernel-balance-variance-frontier" data-alt="A frontier plots RKHS covariate imbalance against effective sample size as the weight penalty changes. A second panel shows maximum weight and inverse effective sample size."><figcaption>Balance and variance are competing objectives, not separate diagnostics. Moving toward exact RKHS balance reduces discrepancy but concentrates the weights and lowers effective sample size; regularization retreats along this frontier. A small imbalance alone is not evidence of a reliable causal estimate.</figcaption></figure>

Let \(A\in\{0,1\}\) be treatment, \(C\) observed pretreatment covariates, and \(Y(a)\) the potential outcome under \(A=a\). Assume i.i.d. sampling, consistency \(Y=Y(A)\), conditional exchangeability \((Y(0),Y(1))\perp A\mid C\), and positivity \(0\lt P(A=a\mid C=c)\lt1\) almost surely on the target population. For finite-variance estimation one usually strengthens positivity to \(P(A=a\mid C)\ge\eta\gt0\). Then

$$
\theta_a=\mathbb E[Y(a)]
 =\mathbb E_C\!\left[\mathbb E(Y\mid A=a,C)\right]
 =\mathbb E\!\left[\frac{\mathbf 1\{A=a\}Y}{P(A=a\mid C)}\right].
$$

The regression and weighting expressions are two representations of the same identified mean. Kernels enter either by estimating \(c\mapsto\mathbb E[Y\mid A=a,C=c]\), or by choosing weights that balance rich functions of \(C\).

:::: {.proposition #prop-causal-balance}
[Proposition (RKHS balance controls confounding bias)]{.box-title}

Let \(k_C\) be measurable and bounded by \(k_C(c,c)\le\kappa^2\), with RKHS \(\mathcal H_C\). For nonnegative treated weights \(w_i\) summing to one and target weights \(q_j\) summing to one, define

$$
\Delta_{\mathcal H}
=\left\|\sum_{i:A_i=1}w_i k_C(C_i,\cdot)
-\sum_j q_j k_C(C_j,\cdot)\right\|_{\mathcal H_C}.
$$

If the treated conditional outcome regression \(m_1(c)=\mathbb E[Y(1)\mid C=c]\) belongs to \(\mathcal H_C\) with \(\|m_1\|_{\mathcal H_C}\le B\), then the conditional design bias of the weighted treated mean relative to the target covariate distribution is at most \(B\Delta_{\mathcal H}\).

**Assumptions.** This bound concerns balance bias only. Identification additionally requires consistency, exchangeability, and overlap; sampling error and outcome noise require separate control.

**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

By the reproducing property, the imbalance in expected conditional outcomes equals

$$
\sum_{i:A_i=1}w_i m_1(C_i)-\sum_jq_jm_1(C_j)
=\left\langle m_1,\sum_{i:A_i=1}w_i k_C(C_i,\cdot)-\sum_jq_jk_C(C_j,\cdot)\right\rangle_{\mathcal H_C}.
$$

Cauchy-Schwarz bounds its absolute value by \(B\Delta_{\mathcal H}\). [\(\square\)]{.qed}
::::

Exact balance on a non-characteristic or poorly tuned kernel may miss important confounding directions. Conversely, driving empirical MMD to zero with extreme weights can explode variance. A defensible analysis reports balance on held-out or prespecified features, effective sample size \((\sum_iw_i^2)^{-1}\), maximum weights, overlap diagnostics, and estimates across a ridge or weight-cap path. Balance is a design diagnostic under the causal model, not a test of exchangeability.

### Kernel instrumental variables {#kernel-iv}

Write the structural model \(Y = f(X) + e\), where the noise \(e\) absorbs the unobserved confounder and is therefore correlated with the treatment \(X\). Because \(\mathbb{E}[e\mid X]\ne 0\), the regression of \(Y\) on \(X\), kernel ridge included, estimates \(f\) plus the confounder's imprint, not \(f\). An instrument breaks the deadlock.

::: {.definition #def-35-5}
[Definition (instrument)]{.box-title}

A variable \(Z\) is an *instrument* for the structural equation \(Y=f(X)+e\) if it is (i) *relevant* for the function class, meaning the conditional-expectation operator \(T f=\mathbb E[f(X)\mid Z=\cdot]\) does not erase the causal directions of interest; (ii) *exogenous*, \(\mathbb E[e\mid Z]=0\); and (iii) subject to *exclusion*, so \(Z\) affects \(Y\) only through \(X\). Point identification of \(f\) additionally requires *completeness*: \(Tf=0\) almost surely implies \(f=0\) in the chosen function class. Mere dependence of \(Z\) and \(X\) is relevance for a scalar linear model, but is not enough for nonlinear identification.
:::

Exogeneity converts the unobservable structural equation into an observable one. Taking the conditional expectation of \(Y=f(X)+e\) given \(Z=z\) and using \(\mathbb{E}[e\mid Z]=0\) lands on the crux identity of the whole method.

:::: {.proposition #prop-35-6}
[Proposition (the two-stage target)]{.box-title}

Let \(k_X\) be measurable with \(\mathbb E\,k_X(X,X)\lt\infty\). If \(Y=f(X)+e\), \(f\in\mathcal H_X\), \(\mathbb E|Y|\lt\infty\), and \(\mathbb E[e\mid Z]=0\) almost surely, then for \(P_Z\)-almost every \(z\),

$$ \mathbb{E}[Y\mid Z=z] = \langle f,\ \mu_{X\mid Z=z}\rangle_{\mathcal H_X}, \qquad \mu_{X\mid Z=z} = \mathbb{E}[\phi(X)\mid Z=z]. $$

**Assumptions.** The conditional mean embedding must exist as a Bochner conditional expectation. This proposition gives a moment equation, not uniqueness. Identification requires injectivity of \(T\) on the candidate class; stable estimation further requires source, capacity, and regularization conditions.
**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

By the [[ch:kernels-and-rkhs|reproducing property]], \(f(X)=\langle f,\phi(X)\rangle_{\mathcal H_X}\). Since \(f\) is a fixed element, the inner product against it is a continuous linear functional and commutes with the (Bochner) conditional expectation, so

$$ \mathbb{E}[f(X)\mid Z=z] = \big\langle f,\ \mathbb{E}[\phi(X)\mid Z=z]\big\rangle_{\mathcal H_X} = \langle f,\ \mu_{X\mid Z=z}\rangle_{\mathcal H_X}. $$

Adding \(\mathbb{E}[e\mid Z=z]=0\) gives the claim. [\(\square\)]{.qed}
::::

The equation relates two observable objects, but it pins down \(f\) only when \(T\) is injective. Newey and Powell (2003), Theorem 2.1, use bounded completeness to obtain uniqueness in nonparametric IV regression; Darolles et al. (2011), Sections 2 and 3, formulate the inverse problem and its regularization. When \(T\) is compact with singular values approaching zero, inversion amplifies sampling noise. Completeness rules out an exact null direction; it does not prevent near-null directions, so regularization remains necessary.

Singh, Sahani, and Gretton (2019) solve the equation in two ridge-regularized stages, the RKHS lift of classical two-stage least squares. Stage one estimates the conditional mean embedding \(\mu_{X\mid Z=z}\) from a first sample \(\{(x_i,z_i)\}_{i=1}^n\) by kernel ridge regression, giving \(\hat\mu(z)=\sum_i \beta_i(z)\,\phi(x_i)\) with weights \(\beta(z)=(K_Z+n\lambda I)^{-1} k_Z(\cdot,z)\). Stage two regresses the outcome on those embeddings, using a second sample \(\{(y_j,\tilde z_j)\}_{j=1}^m\) to fit

$$ \hat f = \arg\min_{f\in\mathcal H_X}\ \frac1m\sum_{j=1}^m \big(y_j - \langle f, \hat\mu(\tilde z_j)\rangle\big)^2 + \xi\|f\|_{\mathcal H_X}^2. $$

The consistency analysis in Singh, Sahani, and Gretton (2019), Section 4 and Appendices B and C, assumes two independent samples for the two stages, bounded measurable kernels, existence of the relevant conditional embedding operator, a well-specified structural function, source conditions controlling how the first- and second-stage targets align with covariance-operator ranges, effective-dimension or eigenvalue control, and regularization sequences coupled to both sample sizes. These assumptions are part of the statistical theorem, not consequences of the matrix formula. Cross-fitting can reuse observations while preserving out-of-fold nuisance predictions, but a same-sample two-stage fit without analysis changes the dependence structure of the errors.

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

The two ridge systems solve different inverse problems, so tune and diagnose them separately. Reuse Cholesky factorizations instead of explicit inverses, report the spectra of \(K_Z+n\lambda I\) and \(G+m\xi I\), and compare first-stage predictive strength with a weak-instrument baseline. A numerically stable second stage cannot recover directions the instrument barely excites; that is an identification-strength failure, not an optimizer failure.

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

**Reproduce the calculation.**

```python
import numpy as np

z = np.array([-1, -1, -1, 1, 1, 1], float)
u = np.array([-1, 1, 0, -1, 1, 0], float)   # sample-uncorrelated with z
x = z + u
beta, gamma = 2.0, 3.0
y = beta * x + gamma * u
n = len(x)

xc, yc, zc = x - x.mean(), y - y.mean(), z - z.mean()
print("Cov_hat(z,u) = %.4f   (instrument uncorrelated with confounder)"
      % (zc @ (u - u.mean()) / n))
print("Cov_hat(x,u) = %.4f   (treatment IS confounded)"
      % (xc @ (u - u.mean()) / n))

# --- naive OLS slope (biased) ---
b_ols = (xc @ yc) / (xc @ xc)
varX = (xc @ xc) / n
covXU = xc @ (u - u.mean()) / n
print("Var_hat(X) = %.4f" % varX)
print("beta_OLS  = %.4f  (= beta + gamma*Cov(X,U)/Var(X) = 2 + 3*%.4f/%.4f)"
      % (b_ols, covXU, varX))
print("OLS bias  = %.4f" % (b_ols - beta))

# --- two-stage least squares = KIV with linear kernels ---
a_hat = (zc @ xc) / (zc @ zc)          # stage 1: regress X on Z
xhat = a_hat * zc                      # fitted treatment (the CME, linear case)
b_2sls = (xhat @ yc) / (xhat @ xhat)   # stage 2: regress Y on fitted treatment
print("stage-1 slope a_hat = %.4f" % a_hat)
print("beta_2SLS = %.4f" % b_2sls)

# --- full KIV matrix formula with linear kernels + tiny ridge ---
lam, xi = 1e-6, 1e-6
Kz = np.outer(zc, zc)                   # linear-kernel Gram of the instrument
Kx = np.outer(xc, xc)                   # linear-kernel Gram of the treatment
B = np.linalg.solve(Kz + n * lam * np.eye(n), Kz)   # stage-1 CME coefficients
K2 = B.T @ Kx @ B                       # Gram of the estimated embeddings
c = np.linalg.solve(K2 + n * xi * np.eye(n), yc)    # stage-2 ridge weights
alpha = B @ c                           # structural-function coefficients
slope_kiv = alpha @ xc                  # f(x) = (sum_i alpha_i xc_i) * x
print("beta_KIV(linear kernels, ridge->0) = %.4f" % slope_kiv)
```
:::::

The two independent samples simplify propagation of stage-one error into stage two; they do not validate the instrument. An alternative avoids the nested regression: Muandet, Mehrjou, Lee, and Raj (2020), Sections 3 and 4, rewrite IV regression as a convex-concave saddle-point problem. Its identification still rests on the same moment restrictions and injectivity boundary.

### Proximal causal learning {#proximal}

Instruments are not always available. Proximal causal learning instead uses two noisy views of an unmeasured confounder \(U\): an outcome-inducing proxy \(W\) and a treatment-inducing proxy \(Z\). Let \(A\) denote treatment and \(X\) observed baseline covariates. The negative-control restrictions are

$$
Y\perp Z\mid(A,U,X),\qquad W\perp(A,Z)\mid(U,X).
$$

They say that \(Z\) carries information about \(U\) but has no direct outcome information after \((A,U,X)\), while \(W\) is not caused by treatment and has no extra dependence on \(Z\) after \((U,X)\). Consistency, positivity for \(A\) conditional on \((U,X)\), and well-defined regular conditional distributions are also required. Identification runs through an outcome bridge \(h\) satisfying

$$ \mathbb{E}\big[\, Y - h(W,A,X) \mid Z,A,X \,\big] = 0. $$

Existence of a bridge is not automatic. Uniqueness requires completeness of the conditional law of \(W\) given \((Z,A,X)\): for the bridge class \(\mathcal G\), \(\mathbb E[g(W)\mid Z,A,X]=0\) almost surely must imply \(g(W)=0\) almost surely. To transport the observed bridge to the latent-confounder equation, proximal identification also uses completeness of \(Z\) relative to \(U\), for example \(\mathbb E[q(U)\mid Z,A,X]=0\Rightarrow q(U)=0\). These are injectivity statements about two conditional-expectation operators. Proxy association alone is not completeness.

:::: {.theorem #thm-causal-proximal}
[Theorem (proximal identification by an outcome bridge)]{.box-title}

Assume the proxy restrictions above, consistency, positivity, integrability, existence of a measurable bridge \(h\), and the stated completeness condition that transfers the observed bridge equation to \(U\). Then, for every treatment \(a\) in the positivity region,

$$
\mathbb E[Y(a)]=\mathbb E\!\left[h(W,a,X)\right].
$$

If the \(W\)-given-\((Z,A,X)\) operator is complete on the bridge class, the bridge is unique up to null sets.

**Assumptions.** The proxy conditional-independence restrictions,
consistency, treatment positivity, integrability, bridge existence, and the
two completeness roles stated in the preceding paragraphs hold on the
declared bridge class. The mean formula is asserted only for treatments in the
positivity region.

**Proof status.** The identification argument follows Miao, Geng, and Tchetgen Tchetgen (2018), identification results in Sections 2 and 3. The kernel estimator follows Mastouri et al. (2021), Section 3 and Theorems 1 and 2; their consistency conditions include bounded kernels, RKHS well-specification or approximation control, operator-range/source assumptions, and vanishing regularization.
::::

The proof idea is a chain of conditional expectations. The observed bridge equation and completeness imply \(\mathbb E[Y-h(W,A,X)\mid U,A,X]=0\). Under intervention \(A=a\), proxy stability gives \(\mathbb E[Y(a)\mid U,X]=\mathbb E[h(W,a,X)\mid U,X]\); averaging over \((U,X)\) yields the formula. Every arrow uses a causal or completeness assumption. The RKHS only regularizes the inverse problem.

## Beyond the mean: distributional and counterfactual effects {#distributional-effects}

A structural function returns a mean, but a mean can hide offsetting benefit and harm. Let \(\ell\) be a bounded characteristic kernel on outcomes with feature map \(\psi\). Under observed-confounder consistency, exchangeability, and positivity, the interventional mean embedding is identified by the embedding-valued g-formula

$$
\mu_a
=\mathbb E[\psi(Y(a))]
=\mathbb E_C\!\left[\mu_{Y\mid A=a,C}\right].
$$

Given a training sample, fit the regularized conditional embedding

$$
\widehat\mu_{Y\mid a,c}
=\Psi^\top(K_{AC}+n\lambda I)^{-1}k_{AC}((a,c),\cdot),
$$

where \(\Psi^\top b=\sum_i b_i\psi(Y_i)\). On an independent evaluation fold \(c_1^\star,\ldots,c_m^\star\), estimate

$$
\widehat\mu_a=\frac1m\sum_{j=1}^m\widehat\mu_{Y\mid a,c_j^\star}.
$$

The distributional effect between \(a\) and \(a'\) can then be summarized by

$$
D_\ell(a,a')=\|\mu_a-\mu_{a'}\|_{\mathcal H_\ell},
$$

with the empirical squared norm computed entirely from outcome Gram matrices and the two coefficient vectors. If \(\ell\) is characteristic on the outcome measure class, \(D_\ell=0\) exactly when the two interventional outcome laws agree. For any \(g\in\mathcal H_\ell\), \(\mathbb E[g(Y(a))]=\langle g,\mu_a\rangle\). Quantiles and threshold probabilities need approximation or distribution reconstruction when their indicator functions are not in the RKHS. This is an estimator of marginal interventional distributions, not of individual counterfactual pairs \((Y(0),Y(1))\), whose joint law is generally unidentified. Muandet et al. (2021), Sections 3 and 4, develop the counterfactual mean-embedding construction and empirical estimator.

### Sensitivity and nonidentification {#causal-sensitivity}

Identification assumptions deserve a perturbation path, not a binary footnote. For IV, replace exact exogeneity by \(\mathbb E[e\mid Z]=r(Z)\) and bound \(\|r\|_{L^2(P_Z)}\le\rho\). The observed equation becomes

$$
g=Tf+r.
$$

For any regularized inverse \(R_\xi\), the change attributable to exclusion or exogeneity violation is \(R_\xi r\), bounded by \(\rho\|R_\xi\|_{\mathrm{op}}\). Small singular values make that bound large. A practical sensitivity plot varies \(\rho\), the first-stage ridge, and the second-stage ridge, then reports the range of target contrasts. Analogously, balancing analyses vary weight caps and overlap truncation; proximal analyses vary bridge penalties and report near-null singular directions. No one-dimensional sensitivity parameter proves validity, but it shows how much hidden violation the substantive conclusion can tolerate.

::::: {.example #example-causal-nonidentification}
[Example (identical observations, different causal slopes)]{.box-title}

:::: wex
::: wex-setup
Let \(Z,U\) be independent Rademacher variables, \(X=Z+U\), and observe \(Y=2Z+5U\). Two structural decompositions generate exactly this same observed quadruple \((Z,U,X,Y)\):

$$
\mathcal M_1:\quad Y=2X+3U,
\qquad
\mathcal M_2:\quad Y=3X-Z+2U.
$$
:::

1.  [Check observational equivalence.]{.wex-op} Substitution gives \(2(Z+U)+3U=2Z+5U\) and \(3(Z+U)-Z+2U=2Z+5U\). Every observed sample is identical under the two decompositions.
2.  [Compare interventions.]{.wex-op} Under \(\mathcal M_1\), setting \(X=x\) gives \(\mathbb E[Y\mid\operatorname{do}(X=x)]=2x\). Under \(\mathcal M_2\), it gives \(3x\), because \(Z\) and \(U\) retain mean zero.
3.  [Read the failure.]{.wex-op} Model \(\mathcal M_1\) satisfies exclusion; \(\mathcal M_2\) has the direct term \(-Z\). The observed law cannot choose between causal slopes \(2\) and \(3\). A universal kernel and infinite data still cannot repair the missing exclusion restriction.

**Sensitivity calculation.** In \(Y=\beta X+\delta Z+\gamma U\) with first-stage coefficient \(a=1\), the IV estimand is \(\beta+\delta\). Bounding \(|\delta|\le\rho\) gives the identified sensitivity interval \([\widehat\beta_{\mathrm{IV}}-\rho,\widehat\beta_{\mathrm{IV}}+\rho]\).
::::

**Reproduce the calculation.**

```python
from itertools import product

for z, u in product((-1, 1), repeat=2):
    x = z + u
    observed = 2 * z + 5 * u
    model_1 = 2 * x + 3 * u
    model_2 = 3 * x - z + 2 * u
    assert model_1 == observed == model_2

# Under intervention, the retained exogenous terms have mean zero.
do_x = 1.25
intervention_mean_1 = 2 * do_x
intervention_mean_2 = 3 * do_x
assert intervention_mean_1 == 2.5
assert intervention_mean_2 == 3.75

beta_iv, rho = 2.0, 0.4
sensitivity_interval = (beta_iv - rho, beta_iv + rho)
assert sensitivity_interval == (1.6, 2.4)
```
:::::

## What kernels buy, and what they do not {#assumptions}

It is worth stating plainly what the kernel has and has not done. In every method of this chapter the RKHS bought exactly one thing: freedom from parametric form. KCIT tests conditional independence without assuming linearity or Gaussianity; KIV and the proximal estimator recover a structural function without assuming it is a straight line. That freedom is substantive, because a linearity assumption quietly built into a causal estimate is itself an untestable causal claim, and the nonparametric test avoids the false negatives that a linear correlation test suffers when the dependence is real but curved.

Identification, however, is a separate layer, and the kernel is silent there.

::: {.remark}
[The assumptions the data cannot check]{.box-title}

Constraint-based discovery reads structure from conditional independences only under *faithfulness*, and even then returns a Markov equivalence class rather than a unique graph (Spirtes et al. 2000; Pearl 2009). Balancing requires consistency, conditional exchangeability, and positivity. Instrumental-variable estimation requires relevance, exogeneity, exclusion, and completeness; exogeneity and exclusion concern unobserved errors and are not generally testable from one just-identified observational law (Angrist, Imbens, and Rubin 1996). Proximal learning requires valid proxy restrictions, bridge existence, positivity, and completeness. Distributional embeddings identify marginal interventional laws but not individual-level counterfactual coupling. A perfect kernel fit to a misidentified estimand is a precise answer to the wrong question.
:::

## Summary {#summary}

Causal inference splits statistical evidence from causal identification. KCIT tests a conditional-independence null under an asymptotic spectral calibration, but does not estimate an effect. Under observed confounding, adjustment and kernel balancing identify mean or distributional effects only with consistency, exchangeability, and overlap. Under hidden confounding, KIV solves an ill-posed moment equation only when instrument validity and completeness identify the structural function; proximal kernels replace the instrument by proxy restrictions, bridge existence, and two completeness requirements. Counterfactual mean embeddings lift estimation from means to marginal interventional distributions. Sensitivity paths and explicit observationally equivalent models show where no kernel can recover information absent from the observed law.

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

For **Causal Inference with Kernels**, name the task first: testing, mean-effect estimation, or distributional-effect estimation. KCIT needs conditional-null calibration rather than ordinary row permutation. Balancing needs exchangeability and overlap, with weight diagnostics. KIV needs relevance, exogeneity, exclusion, completeness, and regularized inversion. Proximal methods need negative-control proxy restrictions, bridge existence, positivity, and completeness. Report ridge values, spectra, effective sample size, support limits, and sensitivity to plausible violations.

## Summary and further reading {#summary-and-further-reading}

Gretton et al. [@gretton2005hsic] provide the embedding view of dependence, Fukumizu et al. [@fukumizu2008] the conditional operator, and Zhang et al. [@zhang2011kcit] the conditional-null test. Characteristicness and two-sample calibration delimit what those embeddings identify [@sriperumbudur2010; @gretton2012], while conditional embeddings and their review connect the operator view to regression [@song2013cme; @muandet2017]. Constraint-based discovery requires the causal and graphical assumptions stated by [@spirtes2000; @pearl2009]. Newey and Powell [@newey2003] and Darolles et al. [@darolles2011] establish the nonparametric-IV inverse-problem boundary; the classical instrument assumptions are explicit in [@angrist1996]. Singh et al. [@singh2019kiv] and Muandet et al. [@muandet2020dualiv] develop primal and dual kernel IV estimators. Miao et al. [@miao2018] give proximal identification, Mastouri et al. [@mastouri2021] construct kernel bridge estimators, and Muandet et al. [@muandet2021cme] develop counterfactual mean embeddings. The next chapter on [[ch:distribution-regression|distribution regression]] changes the sampling unit from a person to a distribution and makes the resulting two-stage sampling error explicit.

## Exercises {#exercises}

For the neighboring problem in which each input is itself an empirical
distribution, see the two-stage sampling analysis of [@szabo2016dr].

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
8.  [proof]{.ex-tag} Prove the RKHS balancing bound in Proposition \(\ref{prop-causal-balance}\). Then construct a two-dimensional covariate example in which a linear kernel on the first coordinate exactly balances that coordinate but misses confounding through the second coordinate. Explain why zero empirical kernel imbalance is only as meaningful as the chosen RKHS.
9.  [synthesis]{.ex-tag} State the assumptions needed to identify \(\mathbb E[Y(a)]\) using an outcome bridge \(h(W,a,X)\). Separate proxy restrictions, bridge existence, positivity, and both completeness roles. For each assumption, state whether it is an observed-law restriction, a causal restriction, or an operator injectivity condition.
10. [computation]{.ex-tag} For the nonidentification example, verify algebraically that \(\mathcal M_1\) and \(\mathcal M_2\) generate the same \(Y\), but intervention slopes \(2\) and \(3\). If \(\widehat\beta_{\mathrm{IV}}=2\) and the direct effect is bounded by \(|\delta|\le0.4\) with first-stage coefficient \(a=1\), report the sensitivity interval.
11. [synthesis]{.ex-tag} Derive the empirical squared distance \(\|\widehat\mu_1-\widehat\mu_0\|_{\mathcal H_\ell}^2\) when \(\widehat\mu_a=\sum_i q_i^{(a)}\psi(Y_i)\). State what characteristicness identifies, and explain why the result does not identify the joint counterfactual law of \((Y(0),Y(1))\).
:::
