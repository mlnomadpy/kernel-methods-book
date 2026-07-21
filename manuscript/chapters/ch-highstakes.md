---
id: ch-highstakes
slug: kernels-in-science-and-space
title: Kernels in Science and Space
part: XVIII · Kernels You Can Defend
order: 56
tier: advanced
prerequisites:
  - accountable-kernels
objectives:
  - Explain the central definitions and claims in Kernels in Science and Space.
  - Apply the chapter's principal methods and interpret their outputs.
  - >-
    State the assumptions behind formal results and connect them to earlier
    chapters.
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

<p class="lead">The previous chapter argued that a kernel machine carries its own accountability: a calibrated error bar, an exact attribution to training data, a reproducible solve, and audit statistics with p-values. This chapter is the payoff, in three domains that demand exactly those properties because a wrong answer with no error bar is unacceptable. In astronomy and space engineering the deliverable is often the uncertainty itself: a planet's mass with a credible interval, a spacecraft-telemetry anomaly score with a false-alarm rate, an orbit prediction with a covariance a collision-avoidance system can trust. In the physical sciences each training label is a quantum-mechanical calculation or a wet-lab experiment, so data efficiency and a variance that tells you where to spend the next one are not conveniences but the method. In Earth observation the product is a global map, and every pixel needs a confidence. Across all three the same three kernel properties recur as requirements, not features: a calibrated posterior covariance, data efficiency when samples are precious, and a kernel that is the explicit, auditable place to write down the physics. We work each domain to a real, cited result, name the kernel and the property that earned it, and state the honest limitation, which turns out to be the same one everywhere.</p>

## Why these domains demand accountable models {#why-these-domains}

It is worth being precise about why these particular sciences adopted kernel methods, because the reasons are the accountability properties of the last chapter rather than raw predictive accuracy. A stellar light curve or a radial-velocity series is short, irregularly sampled, and dominated by a nuisance signal that must be separated from the thing of interest; the model has to report how much of its answer is signal and how much is the model's own uncertainty. A potential-energy surface is smooth and expensive, sampled at a handful of quantum-mechanical points that cost hours each, so the model must interpolate from little data and, critically, must know where it is interpolating badly so a simulation can decide when to pay for another point. A satellite retrieval turns reflectance into a biophysical quantity over millions of pixels, some of which fall outside anything in the training set, and the operational product is only usable if each pixel carries a trustworthy confidence. In every case the Gaussian process and its kernel relatives were adopted because they answer \"how sure, from what, and is this still valid\" as part of the computation, and because the kernel is where the domain's physics, quasi-periodicity, rotational symmetry, spatial context, enters explicitly rather than being learned opaquely.

## Space and astronomy {#space-and-astronomy}

Astronomy was among the first fields to make Gaussian processes standard equipment, and the review of Aigrain and Foreman-Mackey (2023) is the map of that adoption across exoplanets, stellar rotation, and asteroseismology. The through-line is that the covariance kernel is a statement of stellar physics, and the posterior is a distribution over the quantity a mission actually needs.

### Stellar rotation and exoplanet light curves {#light-curves}

A star's brightness is modulated as spots rotate into and out of view, but spots form and decay, so the signal is quasi-periodic rather than strictly periodic. The kernel writes that down directly. The quasi-periodic covariance

$$k(\tau)=A\,\exp\!\Big(-\frac{\tau^2}{2\ell^2}-\Gamma\,\sin^2\frac{\pi\tau}{P}\Big),$$

with rotation period \(P\), spot-coherence length \(\ell\), and modulation depth \(\Gamma\), combines a periodic term with a decay whose length scale is literally the spot lifetime (Angus et al. 2018). This is the [[ch:kernel-families|kernel-families]] construction put to physical use: an interpretable hyperparameter for each physical quantity. The payoff over a classical periodogram is the subject of the last chapter, a full posterior on the period rather than a point estimate, and the celerite method of Foreman-Mackey et al. (2017) makes it affordable by giving the semi-separable covariance an exact likelihood in linear rather than cubic time, so the fully probabilistic model runs on the hundred-thousand-point light curves of Kepler and TESS.

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

**Reading.** Detection is a projection onto a template in a whitened inner-product space, and that inner product is a kernel in the exact sense the book has used throughout. The optimality is a genuine theorem, the Neyman-Pearson optimal linear detector for a known waveform in stationary Gaussian noise, and the recovered SNR is an auditable, calibrated number. The honest limitation is that the optimality assumes stationary Gaussian noise and a known template, so real pipelines add non-Gaussian-glitch vetoes and must tile the template bank densely, which scales badly in parameter dimension.
::::

**Verification artifact.** checks/example-ch-highstakes-example-56-2.json records the example source hash and verification scope.
:::::

<figure class="viz" data-widget="matched-filter">

<figcaption>Matched filtering as projection. A chirp is injected into a noisy stream, invisible by eye. Drag the template's frequency; the lower panel shows the live matched-filter statistic, the real noise-weighted inner product slid over time, which spikes sharply at the true arrival when the template matches and flattens into the noise when it is mistuned.</figcaption>
</figure>

### Spacecraft telemetry and orbit uncertainty {#telemetry-orbits}

Two more space applications make the accountability argument in engineering terms. Spacecraft almost never have labeled fault data, so anomaly detection is naturally a one-class problem: Fujimaki et al. (2005) build a kernel-PCA subspace of nominal telemetry and score a new sample by its reconstruction error in that feature space, a deterministic, reproducible distance rather than a black-box classifier, which is what a mission-review board can act on. And for conjunction assessment, where two objects might collide, a predicted position is useless without a trustworthy covariance; Peng and Bai (2018) learn the residual of a physics-based orbit propagator with a Gaussian process, keeping the interpretable dynamics and adding an input-dependent uncertainty that fuses with the filter covariance. In both, the kernel model is chosen because its output is an auditable uncertainty, not merely a number.

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
2.  [Read the leave-one-out error as uncertainty.]{.wex-op} The closed-form leave-one-out errors of [[ch:accountable-kernels|the accountability chapter]] are largest on the steep repulsive wall, \(4.365\) eV at \(r=2.40\) angstrom, and small in the smooth well, with a median of \(0.0234\) eV over the set.
3.  [See where to sample next.]{.wex-op} The interpolation is hardest exactly where the leave-one-out error is largest, on the repulsive wall, which is precisely where an active learner would place its next expensive point.

**Reading.** Even at toy scale the pattern of the real force fields is visible: a kernel interpolates a physical curve from few points, and its own leave-one-out error is a map of where it is least trustworthy. Scaled up with the SOAP kernel and a real potential-energy surface, that error map becomes the control signal for the active-learning loop below.
::::

**Verification artifact.** checks/example-ch-highstakes-example-56-3.json records the example source hash and verification scope.
:::::

### The killer app: variance triggers a calculation {#active-learning-chemistry}

The property that makes kernels indispensable here is that the Gaussian-process posterior variance depends only on where a configuration sits relative to the training data, not on its unknown energy, so it can flag an unfamiliar configuration before the expensive label is computed. The on-the-fly force fields of Vandermause et al. (2020) turn this into a method: a molecular-dynamics simulation runs on the cheap GP energy, and at each step the predictive variance is checked; while it stays low the GP prediction is accepted, and when it spikes at a configuration the model has never seen, a rare event such as a bond breaking, a single density-functional-theory calculation is triggered, added to the training set, and the simulation continues. The same idea in a Bayesian linear-regression form bypasses more than ninety-nine percent of the quantum calculations that a brute-force simulation would require (Jinnouchi et al. 2019). The calibrated variance is the entire control signal, converting a fixed compute budget into an adaptive one, and it is the same variance whose honest reading opened the previous chapter.

::::: {.example #example-56-4}
[Example (uncertainty sampling reaches accuracy sooner)]{.box-title}

:::: wex
::: wex-setup
A one-dimensional toy potential-energy surface with two wells. A Gaussian process (RBF length scale \(0.35\)) starts from three points and, at each step, either places its next sample at the point of maximum posterior variance (uncertainty sampling) or at a random location. We count how many points each needs to reach a target root-mean-square error of \(0.05\). All numbers from `checks/ch-highstakes-ex4.py`.
:::

1.  [Sample where the model is unsure.]{.wex-op} Uncertainty sampling reaches the target with \(19\) points total.
2.  [Sample blindly.]{.wex-op} Random sampling needs \(26.6\) points on average over thirty seeds to reach the same error.
3.  [Count the saving.]{.wex-op} Active learning uses \(0.72\) of the random budget, about a quarter fewer expensive evaluations, and the saving grows with the cost and dimension of the label.

**Reading.** The posterior variance is not decoration; it is a decision rule that spends the experimental budget where it buys the most accuracy. This toy loop is the mechanism of the on-the-fly force fields and, in the next section, of autonomous materials discovery. The honest cost is the cubic GP solve, which forces sparse or low-rank approximations at the scales real simulations reach.
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

The flagship is calibrated retrieval. Gaussian-process regression from reflectance to biophysical variables such as leaf area index and chlorophyll content (Verrelst et al. 2012, 2013) delivers a per-pixel predictive variance, which becomes a spatial confidence map: pixels whose spectra fall outside the training distribution, unusual canopies or poor atmospheric correction, are flagged as low-confidence, and for an operational product used in crop monitoring or carbon accounting that confidence is the deliverable, not an extra. The GP matched or beat neural networks on accuracy and supplied the uncertainty for free, and its learned per-band length scales double as a relevance ranking over the spectral bands. Because radiative-transfer models that map biophysical parameters to spectra are slow, the same kernels emulate them: a GP surrogate over a principal-component basis of the model output (Verrelst et al. 2016) runs orders of magnitude faster, is smooth and differentiable for inversion and sensitivity analysis, and flags where it is extrapolating. And the audit tools of the last chapter reappear here in their birthplace: Fair Kernel Learning (Pérez-Suay et al. 2017) uses the HSIC dependence measure to remove a sensor or acquisition bias from a retrieval, the same statistic that turned a fairness check into a hypothesis test.

## What the cases share {#what-the-cases-share}

Read together, the domains make one argument three times. First, a calibrated posterior covariance is the actual product: a planet mass of \(4.73\pm0.95\) Earth masses, a per-pixel confidence map, an orbit covariance a collision-avoidance system trusts, a variance that decides when to run a quantum calculation. Second, data efficiency is a requirement, not a nicety, when a single sample is a space mission, a supercomputer run, or a scarce ground-truth label, and the kernel's smoothness prior is what makes interpolation from few points defensible. Third, the kernel is the explicit, auditable place where the physics is written: a quasi-periodic kernel for evolving spots, a semi-separable kernel for stellar granulation, a rotation-and-permutation-invariant kernel for atomic environments, a composite spatial-spectral kernel for imagery, and a noise-weighted inner product for matched filtering. These are the three structural facts of [[ch:accountable-kernels|the accountability chapter]] cashed out in real science.

The honest limitation is also the same everywhere, and stating it once is the fair close. The exact Gaussian process is cubic in the number of points and assumes a stationary, correctly chosen kernel. That is why every mature application here pairs the method with a scalability structure, the semi-separable celerite covariance, sparse or inducing-point approximations, principal-component bases for emulation, or the Nyström and random-feature methods of [[ch:large-scale-kernels|the large-scale chapter]], and why kernel design and cross-validation are never skipped. The accountability is not free; it costs a solve and a modeling choice, and these sciences pay that cost gladly for an error bar they can defend.

## Summary {#summary}

Kernel methods earned their place in astronomy, the physical sciences, and Earth observation for the reasons the previous chapter laid out, not despite them. In space and astronomy the quasi-periodic Gaussian process recovers a stellar rotation period with a credible interval, a shared latent GP separates a planet's mass from stellar activity and reports it with an error bar, matched filtering realizes detection as a projection in a kernel inner-product space, and kernel models supply auditable anomaly scores and orbit covariances. In molecules and materials the SOAP kernel writes rotational and permutational symmetry into a Gaussian Approximation Potential, kernel ridge regression reaches below hybrid-DFT error in the small-data regime, and the posterior variance drives on-the-fly force fields and autonomous discovery by triggering an expensive calculation exactly where the model is unsure. In Earth observation kernel classifiers handle the high-dimensional few-label regime, composite kernels fold in spatial context through Mercer closure, and Gaussian-process retrieval turns every pixel into a value with a confidence. The recurring requirement is a calibrated covariance, the recurring enabler is data efficiency, and the recurring design principle is that the kernel holds the physics; the recurring cost is the cubic solve and the need for a well-chosen, stationary kernel, paid with scalability structure. The book's through-line reaches its most consequential form here: fix a positive definite kernel that encodes what you know, and a great deal of high-stakes science becomes a linear-algebra problem whose answer arrives with the error bar attached.

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
