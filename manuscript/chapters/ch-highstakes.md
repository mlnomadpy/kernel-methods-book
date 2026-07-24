---
id: ch-highstakes
slug: kernels-in-science-and-space
title: Kernels in Science and Space
part: XVIII · Kernels You Can Defend
order: 58
tier: advanced
prerequisites:
  - accountable-kernels
objectives:
  - Match a scientific task to the kernel structure that encodes its physics.
  - >-
    Interpret credible intervals, matched-filter scores, and active-learning
    variances within their assumptions.
  - Distinguish model-based uncertainty from validated operational risk.
  - >-
    Design domain-appropriate checks for astronomy, molecular simulation, and
    Earth observation.
  - >-
    Choose a scalability structure without hiding the approximation it
    introduces.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-highstakes.yml
verification_date: null
bibliography:
  - aigrain2023gp
  - foremanmackey2017celerite
  - angus2018rotation
  - rajpaul2015rv
  - haywood2014corot7
  - allen2012findchirp
  - almosallam2016gpz
  - fujimaki2005telemetry
  - peng2018orbit
  - deringer2021gpr
  - bartok2010gap
  - bartok2013soap
  - bartok2018silicon
  - rupp2012coulomb
  - ramakrishnan2014qm9
  - faber2017errors
  - faber2018fchl
  - christensen2020fchl
  - chmiela2017gdml
  - vandermause2020flare
  - jinnouchi2019onthefly
  - sacks1989dace
  - shahriari2016
  - frazier2018
  - shields2021bo
  - xue2016adaptive
  - lookman2019active
  - kennedy2001calibration
  - lawrence2010coyote
  - campsvalls2005hyper
  - campsvalls2006composite
  - campsvalls2016gpsurvey
  - verrelst2012retrieval
  - verrelst2013uncertainty
  - verrelst2016emulation
  - perezsuay2017fair
---
# Kernels in Science and Space

<p class="lead">An orbit prediction is useless to a collision-avoidance system without a covariance whose calibration has been tested; a planet's mass is not a result until it carries an interval conditional on an explicit noise model; a molecular simulation must know when its cheap surrogate has left validated territory; a global crop map needs a confidence layer and a rule for pixels outside support. These are domains where a bare point estimate is operationally incomplete. Kernel methods earned a role here because the covariance can encode quasi-periodicity, symmetry, spatial dependence, or a detector's noise geometry, while the fitted model can return uncertainty and identify expensive places to sample next. The mathematics does not make those outputs trustworthy by itself. Each case in this chapter therefore pairs the useful kernel structure with the validation, approximation, and failure condition required to act on it.</p>

## Why these domains demand accountable models {#why-these-domains}

It is worth being precise about why these particular sciences adopted kernel methods, because the reasons extend beyond raw predictive accuracy. A stellar light curve or radial-velocity series is short, irregularly sampled, and dominated by nuisance structure that must be separated from the quantity of interest. A potential-energy surface is smooth and expensive, sampled at quantum-mechanical points that may cost hours, so the model must interpolate from little data and expose when it has entered an unfamiliar configuration. A satellite retrieval turns reflectance into a biophysical quantity over millions of pixels, some outside the training support. In each case the kernel provides an explicit place to encode quasi-periodicity, rotational symmetry, spatial context, or noise weighting, while the GP or kernel estimator provides quantities that can be checked against held-out reality. “How sure?” is answered only after that check, not by covariance algebra alone.

## Space and astronomy {#space-and-astronomy}

Astronomy was among the first fields to make Gaussian processes standard equipment, and the review of Aigrain and Foreman-Mackey (2023) is the map of that adoption across exoplanets, stellar rotation, and asteroseismology. The through-line is that the covariance kernel is a statement of stellar physics, and the posterior is a distribution over the quantity a mission actually needs.

### Stellar rotation and exoplanet light curves {#light-curves}

A star's brightness is modulated as spots rotate into and out of view, but spots form and decay, so the signal is quasi-periodic rather than strictly periodic. The kernel writes that down directly. The quasi-periodic covariance

$$k(\tau)=A\,\exp\!\Big(-\frac{\tau^2}{2\ell^2}-\Gamma\,\sin^2\frac{\pi\tau}{P}\Big),$$

with rotation period \(P\), spot-coherence length \(\ell\), and modulation depth \(\Gamma\), combines a periodic term with a decay whose length scale is literally the spot lifetime (Angus et al. 2018). This is the [[ch:kernel-families|kernel-families]] construction put to physical use: an interpretable hyperparameter for each physical quantity. The payoff over a classical periodogram is a posterior for the period conditional on the quasi-periodic model rather than a point estimate alone. The celerite method of Foreman-Mackey et al. (2017) makes a restricted semiseparable covariance family affordable with linear scaling in the number of time points for fixed representation rank, so probabilistic time-series models can reach light curves far beyond a dense exact factorization.

::::: {.example #example-56-1}
[Example (a rotation period, with a credible interval)]{.box-title}

:::: wex
::: wex-setup
We synthesize \(120\) irregularly sampled brightness measurements over \(60\) days from a quasi-periodic GP with true period \(P=10\) days, coherence \(\ell=30\) days, and modulation \(\Gamma=2\), plus white noise. We recover \(P\) by evaluating the GP log marginal likelihood over a grid and reading off the posterior, and compare with a periodogram peak. All numbers from `checks/ch-highstakes-ex1.py`.
:::

1.  [Marginalize over the period.]{.wex-op} With a flat prior on \(P\), the posterior peaks at a maximum-a-posteriori value of \(10.10\) days with posterior mean also \(10.10\) days, recovering the true \(10.00\) days.
2.  [Report the interval, not just the peak.]{.wex-op} The \(68\%\) credible interval is \([10.00,\,10.20]\) days: the deliverable is the period and its uncertainty together.
3.  [Contrast the periodogram.]{.wex-op} A Lomb-Scargle-style periodogram returns a single peak at \(10.73\) days, a point estimate with no native interval and biased here by the evolving spot phase the quasi-periodic kernel models and the sinusoid does not.

**Reading.** The kernel encodes the physics (quasi-periodicity with a finite coherence time) and the posterior delivers the accountable answer (a period with a credible interval). The honest limitation is that the period posterior is often multimodal, with harmonics of \(P\) as competing peaks, so careful sampling matters; and the exact GP is cubic, which is exactly the cost celerite's semi-separable structure removes for this kernel family.
::::

**Verification artifact.** checks/example-ch-highstakes-example-56-1.json records the example source hash and verification scope.
:::::

<figure class="viz" data-widget="quasiperiodic-gp">

<figcaption>A quasi-periodic Gaussian process conditioned live on a synthetic light curve. Drag the period and coherence length; the posterior mean and credible band recompute and the log marginal likelihood peaks as the period locks onto the truth, tightening the band. The readout reports the recovered period and its credible interval. Real GP conditioning on every change.</figcaption>
</figure>

### Disentangling a planet from stellar activity {#radial-velocity}

The same quasi-periodic signal that reveals rotation is a nuisance when the goal is a planet's mass from radial velocities, because spots induce apparent velocity shifts that mimic or mask a Keplerian signal. The framework of Rajpaul et al. (2015) models the radial velocity and several activity indicators as different linear combinations of one shared latent quasi-periodic GP and its time derivative, so the activity component is constrained by the indicators and can be subtracted, leaving the planet. Because the fit is a posterior, the planet's mass comes out as a credible interval rather than a point. The CoRoT-7 analysis of Haywood et al. (2014) is the textbook case: treating the activity as a GP whose covariance is inherited from the light curve, they recover the mass of CoRoT-7b as \(4.73\pm0.95\) Earth masses. The error bar is the result, and it is an error bar precisely because a kernel model produced it.

### Gravitational waves: the noise-weighted inner product is a kernel {#matched-filter}

The cleanest illustration in all of physics that detection is a projection in an inner-product space is matched filtering, the method behind gravitational-wave detection (Allen et al. 2012). The optimal statistic for a known waveform \(h\) buried in noise is the noise-weighted inner product

$$\langle d,h\rangle=4\,\mathrm{Re}\!\int_0^\infty\frac{\tilde d(f)\,\tilde h^*(f)}{S_n(f)}\,df,$$

where \(S_n(f)\) is the noise power spectral density. This is a kernel: it defines the geometry of a whitened Hilbert space in which the data and the template have a well-defined angle, and the detection statistic is the projection of the data onto the template. In white noise it reduces to the plain dot product, which is all we need to see the mechanism on numbers.

::::: {.example #example-56-2}
[Example (finding a chirp by projection)]{.box-title}

:::: wex
::: wex-setup
A chirp template \(h\) of \(512\) samples with rising frequency and a tapered envelope, injected into a white-noise stream at a known arrival lag. We scale the noise so the optimal signal-to-noise ratio \(\rho=\lVert h\rVert/\sigma\) equals \(8\), then slide the template and compute the normalized statistic \(\rho(t_0)=\langle d,h_{t_0}\rangle/(\sigma\lVert h\rVert)\), whose noise has unit variance. All numbers from `checks/ch-highstakes-ex2.py`.
:::

1.  [Set the difficulty.]{.wex-op} The template norm is \(\lVert h\rVert=7.372\); the optimal SNR of \(8\) fixes the noise level \(\sigma=0.9215\). The signal is invisible in the raw stream at this noise.
2.  [Project and read the peak.]{.wex-op} The matched-filter statistic peaks at \(7.37\) at exactly the true arrival lag, near the optimal value of \(8\) (the realized peak is the optimal SNR plus a standard-normal draw).
3.  [Compare to noise.]{.wex-op} The loudest pure-noise excursion elsewhere is only \(1.71\). The projection onto the template lifts the signal far above the noise floor, which is the entire idea of an optimal detector.

**Reading.** Detection is a projection onto a template in a whitened inner-product space, and that inner product is a kernel in the exact sense the book has used throughout. The optimality is a genuine theorem for a known waveform in stationary Gaussian noise, and the SNR has a known null distribution under those assumptions. Real pipelines must estimate the noise spectrum, account for many correlated template trials, add non-Gaussian-glitch vetoes, and calibrate significance empirically; a raw matched-filter peak is not itself a detection probability.
::::

**Verification artifact.** checks/example-ch-highstakes-example-56-2.json records the example source hash and verification scope.
:::::

<figure class="viz" data-widget="matched-filter">

<figcaption>Matched filtering as projection. A chirp is injected into a noisy stream, invisible by eye. Drag the template's frequency; the lower panel shows the live matched-filter statistic, the real noise-weighted inner product slid over time, which spikes sharply at the true arrival when the template matches and flattens into the noise when it is mistuned.</figcaption>
</figure>

### Spacecraft telemetry and orbit uncertainty {#telemetry-orbits}

Two more space applications make the accountability argument in engineering terms. Spacecraft almost never have labeled fault data, so anomaly detection is naturally a one-class problem: Fujimaki et al. (2005) build a kernel-PCA subspace of nominal telemetry and score a new sample by its reconstruction error in that feature space. The score is inspectable, but its alarm threshold still needs a null or operational false-alarm study. For conjunction assessment, where two objects might collide, a predicted position is incomplete without a covariance validated against tracking residuals; Peng and Bai (2018) learn the residual of a physics-based orbit propagator with a Gaussian process, keeping the interpretable dynamics and adding an input-dependent model uncertainty that fuses with the filter covariance.

## Molecules and materials {#molecules-and-materials}

If there is one domain where kernels, not neural networks, are the canonical data-efficient and uncertainty-aware workhorse, it is the machine learning of interatomic potentials and molecular properties, and the review of Deringer et al. (2021) is its reference. The reason is the accountability triple again: data efficiency because each label is a density-functional-theory calculation, a physics prior written into the kernel, and a posterior variance that decides when to compute the next expensive label.

### Potentials from a kernel that knows the symmetries {#force-fields}

A Gaussian Approximation Potential (Bartók et al. 2010) is Gaussian-process regression of the energy on quantum-mechanical training data, with no fixed functional form; it interpolates the potential-energy surface and is systematically improvable as more data arrive. What made it general is the kernel. The Smooth Overlap of Atomic Positions descriptor (Bartók, Kondor, and Csányi 2013) defines the similarity between two atomic neighborhoods as the rotationally integrated overlap of their smeared neighbor densities, so the kernel is invariant to rotation, translation, and permutation of like atoms by construction. Those are hard physical symmetries encoded in the kernel itself rather than learned from data, a vivid instance of the book's theme that the kernel is where domain knowledge enters. One such model reproduces density-functional theory across crystalline, liquid, and amorphous silicon and its defects (Bartók et al. 2018), phases where classical force fields fail.

The molecular-property lineage runs in parallel. Kernel ridge regression on the Coulomb-matrix descriptor (Rupp et al. 2012) predicted atomization energies of \(7165\) small organic molecules with a mean absolute error near \(9.9\) kcal/mol at a tiny fraction of the quantum cost, and the descriptors that followed, culminating in FCHL (Faber et al. 2018; Christensen et al. 2020), reach chemical accuracy on the QM9 benchmark of Ramakrishnan et al. (2014). The systematic learning curves of Faber et al. (2017) are the evidence for the headline claim: kernel models dominate the small-data regime and reach below hybrid-DFT error. And by learning in the gradient domain with energy conservation imposed exactly, the GDML force fields of Chmiela et al. (2017) reach roughly \(0.3\) kcal/mol from only a thousand molecular configurations, a data efficiency that comes from pairing the kernel with a physics prior.

::::: {.example #example-56-3}
[Example (a bond length from a handful of points)]{.box-title}

:::: wex
::: wex-setup
The smallest instance of a learned potential. We sample about a dozen points of the Morse potential for the hydrogen molecule (dissociation energy \(D_e=4.75\) eV, equilibrium length \(r_e=0.7416\) angstrom, width \(a=1.942\) inverse angstrom), denser near the well, fit kernel ridge regression with an RBF kernel, and recover the equilibrium bond length as the minimum of the predicted curve. All numbers from `checks/ch-highstakes-ex3.py`.
:::

1.  [Recover the physical quantity.]{.wex-op} The minimum of the kernel-ridge curve sits at \(0.7425\) angstrom, within \(0.9\) milliangstrom of the true \(0.7416\), from a dozen points.
2.  [Read the leave-one-out error as a sensitivity diagnostic.]{.wex-op} The closed-form leave-one-out errors of [[ch:accountable-kernels|the accountability chapter]] are largest on the steep repulsive wall, \(4.365\) eV at \(r=2.40\) angstrom, and small in the smooth well, with a median of \(0.0234\) eV over the set.
3.  [See where to sample next.]{.wex-op} The interpolation is hardest exactly where the leave-one-out error is largest, on the repulsive wall, which is precisely where an active learner would place its next expensive point.

**Reading.** Even at toy scale the pattern of the real force fields is visible: a kernel interpolates a physical curve from few points, and leave-one-out error identifies training locations on which that fit is sensitive. It is not predictive uncertainty at a new configuration. The active-learning loop below instead uses model-based posterior variance, calibrated against held-out force and energy errors.
::::

**Verification artifact.** checks/example-ch-highstakes-example-56-3.json records the example source hash and verification scope.
:::::

### The killer app: variance triggers a calculation {#active-learning-chemistry}

The property that makes kernels useful here is that, for a fixed GP covariance and observation model, posterior variance depends on the configuration and training locations rather than the unknown energy. It can therefore flag unfamiliar geometry before the expensive label is computed. The on-the-fly force fields of Vandermause et al. (2020) turn this into a method: a molecular-dynamics simulation runs on the cheap GP energy, and when a model-based uncertainty criterion crosses a threshold, a density-functional-theory calculation is triggered, added to the training set, and the simulation continues. The same idea in a Bayesian linear-regression form bypasses more than ninety-nine percent of the quantum calculations in the reported setting of Jinnouchi et al. (2019). The variance becomes a control signal only after threshold calibration and out-of-distribution stress tests; otherwise a misspecified model can be confidently wrong at precisely the rare event the loop must catch.

::::: {.example #example-56-4}
[Example (uncertainty sampling reaches accuracy sooner)]{.box-title}

:::: wex
::: wex-setup
A one-dimensional toy potential-energy surface with two wells. A Gaussian process (RBF length scale \(0.35\)) starts from three points and, at each step, either places its next sample at the point of maximum posterior variance (uncertainty sampling) or at a random location. We count how many points each needs to reach a target root-mean-square error of \(0.05\). All numbers from `checks/ch-highstakes-ex4.py`.
:::

1.  [Sample where the model is unsure.]{.wex-op} Uncertainty sampling reaches the target with \(19\) points total.
2.  [Sample blindly.]{.wex-op} Random sampling needs \(26.6\) points on average over thirty seeds to reach the same error.
3.  [Count the saving.]{.wex-op} In this one-dimensional experiment, active learning uses \(0.72\) of the random budget, about a quarter fewer expensive evaluations. A more expensive label makes that saving more valuable; higher dimension does not guarantee that the same strategy remains effective.

**Reading.** Posterior variance can be a decision rule rather than decoration. In this smooth, well-specified toy it spends fewer labels than random sampling, illustrating the mechanism of on-the-fly force fields and autonomous materials discovery. It is not a universal optimality result: model misspecification, a disconnected domain, or a variance that ignores the important failure direction can reverse the comparison. The computational cost is also real, forcing sparse or low-rank approximations at simulation scale.
::::

**Verification artifact.** checks/example-ch-highstakes-example-56-4.json records the example source hash and verification scope.
:::::

<figure class="viz" data-widget="onthefly-mlip">

<figcaption>An on-the-fly potential in miniature. A simulated trajectory runs across a one-dimensional energy surface on the cheap Gaussian-process energy; the predictive variance is plotted along the path, and when it spikes at an unseen barrier crossing a quantum calculation is triggered, marked, and added, after which the variance drops and the training set grows. The trajectory, the variance spike, and the retrain are all real computed events.</figcaption>
</figure>

### Bayesian optimization for discovery {#bo-discovery}

When the goal is not a potential but the best material or reaction, the same posterior drives Bayesian optimization, from [[ch:bayesian-optimization-and-bandits|its chapter]], whose modern form traces to modeling a deterministic simulator as a Gaussian process (Sacks et al. 1989) and whose review is Shahriari et al. (2016). The acquisition function balances exploiting the posterior mean against exploring its variance, so each costly experiment is chosen to be maximally informative. In reaction chemistry, Bayesian optimization over reaction conditions outperformed expert chemists in both efficiency and consistency (Shields et al. 2021). In materials, the adaptive-design loop of Xue et al. (2016) searched roughly eight hundred thousand candidate compositions and, by synthesizing only thirty-six of them chosen by expected improvement, found a new nickel-titanium shape-memory alloy with the smallest thermal hysteresis in the family. The uncertainty, not just the mean, chose each alloy to make, which is what turned a combinatorial search into a dozen experiments, and the methodology is consolidated in the review of Lookman et al. (2019).

## Earth observation {#earth-observation}

The third domain turns satellite measurements into maps, and its adoption of kernels tracks the accountability properties once more, through the long line of work surveyed for Gaussian processes by Camps-Valls et al. (2016). Hyperspectral pixels live in hundreds of spectral dimensions but labeled examples are scarce, the small-sample high-dimensional regime where the margin and regularization of a support vector machine shine; Camps-Valls and Bruzzone (2005) made kernel classifiers the remote-sensing standard for exactly this reason. Spatial context enters through the algebra of kernels: a composite kernel \(k=\mu\,k_{\text{spectral}}+(1-\mu)\,k_{\text{spatial}}\) is valid because sums of valid kernels are valid (Camps-Valls et al. 2006), and the mixing weight \(\mu\) is an interpretable, auditable dial for how much neighborhood texture versus pure spectrum drives the decision, a direct use of the Mercer closure properties of [[ch:kernel-families|the kernel-families chapter]].

The flagship is retrieval with a spatial uncertainty layer. Gaussian-process regression from reflectance to biophysical variables such as leaf area index and chlorophyll content (Verrelst et al. 2012, 2013) delivers a per-pixel predictive variance, which can become a confidence map after calibration against geographically and temporally separated ground truth. Unusual spectra or poor atmospheric correction may then be flagged, but distance in the learned covariance is not by itself proof of being out of distribution. The reported GP systems matched or beat neural comparators on accuracy while supplying a model-based variance, and learned per-band length scales offered a sensitivity diagnostic rather than a causal relevance ranking. Because radiative-transfer models are slow, the same kernels emulate them: a GP surrogate over a principal-component basis of the model output (Verrelst et al. 2016) is smooth and differentiable for inversion and sensitivity analysis, with speed and extrapolation claims that must be measured on the target grid. Fair Kernel Learning (Pérez-Suay et al. 2017) uses HSIC to penalize dependence on a sensor or acquisition variable, while the remaining task and group performance still require direct audit.

## What the cases share {#what-the-cases-share}

Read together, the domains make one argument three times. First, a validated uncertainty quantity is part of the product: a planet mass with a credible interval, a per-pixel confidence map, an orbit covariance checked against tracking residuals, or a variance threshold that decides when to run quantum mechanics. Second, data efficiency is a requirement when one label is a mission, a supercomputer run, or scarce ground truth. Third, the kernel is an explicit place to write physics: quasi-periodicity for evolving spots, rotational and permutation invariance for atomic environments, spatial-spectral coupling for imagery, and noise weighting for matched filtering. The accountability chapter supplies the audit pattern; each science must supply the domain-specific calibration test.

The honest limitation is also the same everywhere, and stating it once is the fair close. The exact Gaussian process is cubic in the number of points and assumes a stationary, correctly chosen kernel. That is why every mature application here pairs the method with a scalability structure, the semi-separable celerite covariance, sparse or inducing-point approximations, principal-component bases for emulation, or the Nyström and random-feature methods of [[ch:large-scale-kernels|the large-scale chapter]], and why kernel design and cross-validation are never skipped. The accountability is not free; it costs a solve and a modeling choice, and these sciences pay that cost gladly for an error bar they can defend.

## Common mistakes and practical implications {#highstakes-practice}

- A calibrated covariance is only as trustworthy as the kernel that produced it; a stationary kernel imposed on a nonstationary field, or a length scale left unfitted, yields an error bar that looks defensible and is not.
- Data efficiency is not license to skip validation: with a handful of expensive samples, an over-flexible kernel interpolates the noise, so cross-validation and a smoothness prior are what keep few-point interpolation honest.
- Physics written into the kernel must be the physics of the task; a rotation-invariant atomic kernel applied where orientation carries signal, or a matched filter built on a mis-specified template, encodes the wrong prior and misses what matters.
- Matched-filter and anomaly scores are ranking statistics, not calibrated probabilities, until their null distribution over correlated trials is worked out; reading a raw peak as a detection confidence overstates significance.
- The exact Gaussian process is cubic in the number of points, so every mature deployment here pairs it with a scalability structure (celerite, inducing points, principal-component emulation, Nystrom or random features), and treating the exact solve as the only option caps the problem size artificially.
- An error bar is a claim about the model, not about reality; when the training set no longer resembles the deployment regime, the covariance stops meaning what the report says it means, and a drift check must guard it.

The practical implication is consistent across all three domains: the accountability is not free, it costs a solve and a modeling choice, and it is worth stating in every deliverable which kernel encodes which physics, which hyperparameters were fitted, and which scalability approximation was accepted.

## Summary and further reading {#summary}

Kernel methods earn their place in these sciences when three conditions meet: labels are expensive, useful structure can be written as a covariance or inner product, and the resulting uncertainty can be checked against the decisions it will control. Quasi-periodic kernels encode evolving stellar signals; noise-weighted inner products define matched filters; symmetry-aware kernels reduce the sample burden of molecular potentials; spatial-spectral kernels connect satellite pixels to context. The output is never “an error bar for free.” It is a model-based covariance, score, or interval whose credibility depends on held-out residuals, shift tests, support, numerical approximation, and the cost of a wrong action. The scalable method must be named as carefully as the kernel: semiseparable structure, inducing points, sparse precision, or low-rank features change what is computed. In high-stakes work, the kernel is valuable because the modeling claim is explicit enough to test and reject.

For further reading, Rasmussen and Williams (2006) remains the reference for Gaussian-process regression and its covariance; Angus et al. (2018) and Foreman-Mackey et al. (2017) develop the quasi-periodic and celerite models behind the astronomy cases; Bartok et al. (2013) and Rupp et al. (2012) introduce the SOAP and Coulomb-matrix representations for molecules and materials, and Jinnouchi et al. (2019) the on-the-fly force fields; Camps-Valls and Bruzzone (2005) and Verrelst et al. (2016) survey kernel methods and Gaussian-process retrieval for Earth observation.

::: {.exercises}
## Exercises {#exercises}

1.  [warm-up]{.ex-tag} For each application, name the kernel property that earned it its place (calibrated uncertainty, data efficiency, or physics encoded in the kernel), in one phrase: (a) an on-the-fly interatomic potential; (b) matched filtering for gravitational waves; (c) a per-pixel satellite chlorophyll map; (d) recovering a stellar rotation period; (e) a Bayesian-optimization search for a new alloy. Some have more than one; give the dominant one.
2.  [computation]{.ex-tag} In the matched-filter example the optimal SNR is \(\rho=\lVert h\rVert/\sigma\) and the normalized statistic at the true lag is \(\rho\) plus a standard normal. With \(\lVert h\rVert=7.372\) and \(\sigma=0.9215\), verify \(\rho=8\), and compute the probability that a single pure-noise sample of the statistic exceeds the observed peak of \(7.37\). Why does the loudest of many correlated noise lags still stay far below \(7.37\)?
    Hint

    ::: hint-body
    \(7.372/0.9215=8.0\). A unit-normal exceeds \(7.37\) with probability about \(9\times10^{-14}\). The template width makes adjacent lags highly correlated, so the number of effectively independent noise trials is far smaller than the number of lags, and the expected maximum is a few, not eight.
    :::
3.  [proof]{.ex-tag} Show that the composite kernel \(k=\mu\,k_1+(1-\mu)\,k_2\) with \(\mu\in[0,1]\) is positive definite whenever \(k_1,k_2\) are, and explain why the mixing weight \(\mu\) is therefore a safe, interpretable dial in the spatial-spectral remote-sensing kernel. What goes wrong if \(\mu\) is allowed outside \([0,1]\)?
    Hint

    ::: hint-body
    A nonnegative combination of positive definite kernels is positive definite, since the Gram matrix is the same nonnegative combination of positive semidefinite matrices. Outside \([0,1]\) a negative coefficient can destroy positive definiteness, so the \"kernel\" is no longer a valid inner product.
    :::
4.  [exploration]{.ex-tag} Reproduce the Morse-potential fit and study data efficiency: as you thin the training points from twelve to five, track the recovered bond length and the median leave-one-out error, and identify which points are most costly to remove. Relate your finding to why an active learner samples the repulsive wall first.
    Hint

    ::: hint-body
    Removing points on the steep wall degrades the fit fastest because the curvature is highest there and the kernel must work hardest; the well is nearly flat and tolerates sparser sampling. The leave-one-out error you computed already ranks the wall points as the least redundant.
    :::
5.  [challenge]{.ex-tag} The on-the-fly force field trusts the GP prediction while the posterior variance is below a threshold and calls quantum mechanics when it exceeds it. Frame this as a decision rule with a cost for a wrong prediction and a cost for a quantum call, and derive how the threshold should depend on the two costs. What failure mode arises if the kernel is misspecified so the variance is not calibrated?
    Hint

    ::: hint-body
    Call quantum mechanics when the expected cost of trusting the GP (roughly the miscoverage probability times the error cost) exceeds the call cost; the threshold falls as the error cost rises. If the variance is miscalibrated low, the rule accepts predictions it should not, and the simulation drifts into unphysical regions unnoticed, the deployment analogue of the previous chapter's under-coverage.
    :::
6.  [exploration]{.ex-tag} The chapter claims the exact GP is cubic and every mature application pairs it with a scalability structure. Pick two, celerite's semi-separable covariance and inducing-point sparse GPs, and explain what structural assumption each exploits and what it gives up, connecting them to the Nyström and random-feature trade-offs of [[ch:large-scale-kernels|the large-scale chapter]].
    Hint

    ::: hint-body
    Celerite restricts the kernel to a sum of (complex) exponentials on one-dimensional inputs, buying an exact linear-time likelihood at the cost of generality; inducing points summarize the data by \(m\ll n\) pseudo-points, an approximation whose fidelity is governed by the effective dimension, the same quantity that sets how many Nyström landmarks suffice.
    :::
:::
