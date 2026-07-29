---
id: ch-highstakes
slug: kernels-in-science-and-space
title: Kernels in Science and Space
part: XII · Reliable Practice
order: 61
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
  - kernelbook-code-ch-highstakes-ex1
  - kernelbook-code-ch-highstakes-ex5
  - perezsuay2017fair
narrative_link_policy: exact
example_code_policy: visible-for-executable
---
# Kernels in Science and Space

The [[ch:accountable-kernels|accountability chapter]] supplied uncertainty,
influence, drift, and audit tools. High-stakes science supplies the missing
criterion: whether those tools protect the scientific or operational quantity
that motivated the model.

<p class="lead">An orbit prediction is useless to a collision-avoidance system without a covariance whose calibration has been tested; a planet's mass is not a result until it carries an interval conditional on an explicit noise model; a molecular simulation must know when its cheap surrogate has left validated territory; a global crop map needs a confidence layer and a rule for pixels outside support. These are domains where a bare point estimate is operationally incomplete. Kernel methods earned a role here because the covariance can encode quasi-periodicity, symmetry, spatial dependence, or a detector's noise geometry, while the fitted model can return uncertainty and identify expensive places to sample next. The mathematics does not make those outputs trustworthy by itself. Each case in this chapter therefore pairs the useful kernel structure with the validation, approximation, and failure condition required to act on it.</p>

## Why these domains demand accountable models {#why-these-domains}

It is worth being precise about why these sciences adopted kernel methods, because the reasons extend beyond predictive accuracy. A stellar light curve is irregularly sampled and dominated by nuisance structure. A potential-energy surface is expensive to label and dangerous to extrapolate. A satellite retrieval is deployed over locations, seasons, and sensors that may not resemble its training set. In each case the kernel provides an explicit place to encode structure, while the estimator produces quantities that can be tested against held-out reality.

This chapter uses one protocol template throughout. A scientific result is not reproducible unless its report fixes all eight fields.

| Field | Question the report must answer |
|---|---|
| Population and sampling | What process or named dataset generated each observation, and what is the unit of independence? |
| Split or experimental design | Which time blocks, objects, trajectories, sites, or campaigns are held out? |
| Kernel and baseline | What representation, kernel, hyperparameter-selection rule, and non-kernel comparator are frozen before testing? |
| Uncertainty currency | Is the output a posterior interval, predictive interval, score, covariance, or decision probability? |
| Metrics | Which point, calibration, ranking, and computational metrics are reported, with units? |
| Negative controls | Which deliberately wrong kernel, shuffled target, off-template signal, or shifted domain should fail? |
| Failure criterion | What predeclared observation rejects the model or stops deployment? |
| Operational validation | Is the final check performed on the quantity that drives an action? |

The split must follow the scientific unit. Randomly splitting nearby pixels, adjacent molecular-dynamics frames, or observations from the same star leaks information and produces optimistic results. Hyperparameters and uncertainty thresholds belong to the training and validation sets. The test set is opened once.

## Space and astronomy {#space-and-astronomy}

Astronomy was among the first fields to make Gaussian processes standard equipment, and the review of Aigrain and Foreman-Mackey (2023) is the map of that adoption across exoplanets, stellar rotation, and asteroseismology. The through-line is that the covariance kernel is a statement of stellar physics, and the posterior is a distribution over the quantity a mission actually needs.

**Stellar rotation and exoplanet light curves.** {#light-curves}

A star's brightness is modulated as spots rotate into and out of view, but spots form and decay. The quasi-periodic covariance

$$k(\tau)=A\,\exp\!\Big(-\frac{\tau^2}{2\ell^2}-\Gamma\,\sin^2\frac{\pi\tau}{P}\Big),$$

with period \(P\), coherence \(\ell\), and modulation \(\Gamma\), combines periodic recurrence with finite memory. Angus et al. evaluate this model on simulated and Kepler light curves, while explicitly sampling the period posterior [@angus2018rotation, Secs. 2--4]. Restricted semiseparable covariance families permit linear-time one-dimensional GP operations for fixed representation rank [@foremanmackey2017celerite, Secs. 2--5].

**Study protocol: stellar rotation.** The observational unit is one light curve with timestamps, fluxes, quality flags, and known measurement uncertainties. Simulations must publish the cadence, gap process, noise law, spot-evolution generator, and true period. Real-data evaluation must split by star, never by timestamp within a star. Fit the quasi-periodic GP on a development set; choose priors, detrending, and convergence criteria there; compare against a sinusoidal periodogram and an autocorrelation baseline. The uncertainty currency is the posterior distribution of \(P\), conditional on the covariance and priors. Report absolute or fractional period error, interval coverage and width on simulations, multimodality, convergence diagnostics, and wall time. Include a strictly periodic kernel, time-shuffled flux, and injected harmonics as negative controls. Reject the model when chains do not mix, the posterior piles against a prior boundary, simulation coverage misses its tolerance, or plausible aliases carry material posterior mass. Operationally validate on independently measured rotation periods or on downstream quantities whose sensitivity to period error is stated.

::::: {.example #example-56-1}
[Example (a rotation period, with a credible interval)]{.box-title}

```python
import numpy as np
rng = np.random.default_rng(7)
def qp(t1,t2,p):
    d=t1[:,None]-t2[None,:]
    return np.exp(-d*d/(2*30**2)-2*np.sin(np.pi*np.abs(d)/p)**2)
t=np.sort(rng.uniform(0,60,120))
y=np.linalg.cholesky(qp(t,t,10)+1e-8*np.eye(120))@rng.normal(size=120)
y+=.15*rng.normal(size=120)
grid=np.linspace(5,20,601)
def log_evidence(p):
    L=np.linalg.cholesky(qp(t,t,p)+.15**2*np.eye(120))
    a=np.linalg.solve(L.T,np.linalg.solve(L,y))
    return -.5*y@a-np.log(np.diag(L)).sum()
logp=np.array([log_evidence(p) for p in grid])
mass=np.exp(logp-logp.max()); mass/=mass.sum()
interval=grid[np.searchsorted(np.cumsum(mass),[.16,.84])]
print(grid[np.argmax(logp)],interval)
assert np.allclose(interval,[10.,10.2])
```

:::: wex
::: wex-setup
This is an **illustrative deterministic simulation**, not a published benchmark. We synthesize \(120\) irregular observations over \(60\) days from the stated GP with \(P=10\), \(\ell=30\), and \(\Gamma=2\), plus Gaussian noise. A grid posterior uses a flat prior on \(P\in[5,20]\). The simulation and its assertions are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-highstakes-ex1].
:::

1.  [Marginalize over the period.]{.wex-op} With a flat prior on \(P\), the posterior peaks at a maximum-a-posteriori value of \(10.10\) days with posterior mean also \(10.10\) days, recovering the true \(10.00\) days.
2.  [Report the interval, not just the peak.]{.wex-op} The \(68\%\) credible interval is \([10.00,\,10.20]\) days: the deliverable is the period and its uncertainty together.
3.  [Contrast the periodogram.]{.wex-op} A Lomb-Scargle-style periodogram returns a single peak at \(10.73\) days, a point estimate with no native interval and biased here by the evolving spot phase the quasi-periodic kernel models and the sinusoid does not.

**Reading.** This fixture verifies a calculation under a correctly specified generator. It does not establish frequentist coverage, robustness to detrending, or performance on real stars.
::::
:::::

<figure class="viz" data-widget="quasiperiodic-gp">

<figcaption>A quasi-periodic Gaussian process conditioned live on a synthetic light curve. Drag the period and coherence length; the posterior mean and credible band recompute and the log marginal likelihood peaks as the period locks onto the truth, tightening the band. The readout reports the recovered period and its credible interval. Real GP conditioning on every change.</figcaption>
</figure>

**Disentangling a planet from stellar activity.** {#radial-velocity}

The same signal becomes a nuisance when the target is a planet's mass. Rajpaul et al. model radial velocity and activity indicators as linear functionals of a shared latent GP [@rajpaul2015rv, Secs. 2--3]. Haywood et al. combine a Keplerian component with an activity covariance in the CoRoT-7 analysis [@haywood2014corot7, Secs. 3--5]. We do not transfer a reported mass or interval into this chapter because it is conditional on that paper's data reduction, priors, activity model, and instrument treatment.

**Study protocol: radial velocity.** Publish timestamps, instruments, velocities, quoted errors, activity indicators, exclusion rules, and injection generator. Hold out complete observing seasons or instruments; tune priors and kernels without the final block. Compare a shared latent GP plus Keplerian model with a Keplerian-only model, a GP without activity channels, and an activity-only model. The uncertainty currency is a posterior over orbital parameters and derived mass, not an unconditional confidence statement. Report injection-recovery error, interval coverage, false inclusion under no-planet injections, posterior predictive residuals by instrument, and sensitivity to priors and kernels. Negative controls include phase-shuffled injections, an activity indicator decoupled from velocity, and a planet period near the activity period. Failure means non-identifiability, poor injection coverage, instrument-specific residual structure, or conclusions that reverse under a defensible activity kernel. Operational validation requires recovery of blind injections before interpreting a real candidate.

**Gravitational waves: the noise-weighted inner product is a kernel.** {#matched-filter}

Matched filtering makes detection a projection in a noise-weighted space. For stationary Gaussian noise and a known waveform \(h\), the statistic is

$$\langle d,h\rangle=4\,\mathrm{Re}\!\int_0^\infty\frac{\tilde d(f)\,\tilde h^*(f)}{S_n(f)}\,df,$$

where \(S_n\) is the noise power spectral density. FINDCHIRP defines the discrete filtering, normalization, template-bank search, and signal-consistency test [@allen2012findchirp, Secs. II--IX, especially Eqs. (2.3), (4.1), and (8.1)]. Calling this inner product a kernel identifies its PSD geometry; it does not make the maximum over time and templates a calibrated probability.

**Study protocol: matched-filter search.** Start from strain segments, data-quality flags, a PSD-estimation interval disjoint from the tested event, a fixed template bank, and an injection population. Split by observing time so tuning and final false-alarm estimation use disjoint background. Compare the full noise-weighted statistic with an unweighted correlation and mismatched-template controls. The uncertainty currency is initially a ranking statistic. Convert it to a false-alarm rate only through the complete search pipeline using time slides or another valid background design. Report detection efficiency against false-alarm rate, parameter recovery on injections, veto survival, latency, and sensitivity to PSD drift. Negative controls include time-reversed or off-family templates, hardware-free null segments, and artificial glitches. Failure occurs when background tails are unstable, injections are missed above the declared amplitude, vetoes respond differently between tuning and test time, or PSD drift invalidates normalization. Operational validation is end-to-end blind injection recovery with the production bank, vetoes, and trials factor.

::::: {.example #example-56-2}
[Example (finding a chirp by projection)]{.box-title}

```python
import numpy as np
rng=np.random.default_rng(11); n=512; t=np.linspace(0,1,n)
env=np.exp(-(t-.55)**2/(2*.12**2))
h=env*np.sin(2*np.pi*(8*t+20*t**2)); h-=h.mean()
sigma=np.linalg.norm(h)/8; arrival=60
data=sigma*rng.normal(size=3*n); data[n+arrival:n+arrival+n]+=h
lags=np.arange(-n//2,n//2)
def filter(template):
    scale=sigma*np.linalg.norm(template)
    return np.array([data[n+j:n+j+n]@template/scale for j in lags])
rho=filter(h)
wrong=env*np.sin(2*np.pi*(15*t+4*t**2)); wrong-=wrong.mean()
rho_wrong=filter(wrong)
print(lags[np.argmax(rho)],rho.max(),rho_wrong[lags==arrival][0])
assert lags[np.argmax(rho)]==arrival and rho_wrong[lags==arrival][0]<2
```

:::: wex
::: wex-setup
This is an **illustrative deterministic white-noise simulation**, not a detection pipeline. A \(512\)-sample chirp is injected at a known lag and scaled to optimal SNR \(8\). The check also runs a mismatched template as a negative control.
:::

1.  [Set the difficulty.]{.wex-op} The template norm is \(\lVert h\rVert=7.372\); the optimal SNR of \(8\) fixes the noise level \(\sigma=0.9215\). The signal is invisible in the raw stream at this noise.
2.  [Project and read the peak.]{.wex-op} The matched-filter statistic peaks at \(7.37\) at exactly the true arrival lag, near the optimal value of \(8\) (the realized peak is the optimal SNR plus a standard-normal draw).
3.  [Run the negative control.]{.wex-op} The matched template peaks at the injected lag; the frequency-mismatched template does not reproduce that recovery. Exact deterministic values are asserted in the check.

**Reading.** The fixture checks projection geometry only. It deliberately does not report a detection probability.
::::
:::::

<figure class="viz" data-widget="matched-filter">

<figcaption>Matched filtering as projection. A chirp is injected into a noisy stream, invisible by eye. Drag the template's frequency; the lower panel shows the live matched-filter statistic, the real noise-weighted inner product slid over time, which spikes sharply at the true arrival when the template matches and flattens into the noise when it is mistuned.</figcaption>
</figure>

**Spacecraft telemetry and orbit uncertainty.** {#telemetry-orbits}

Kernel-PCA reconstruction error is a score for nominal telemetry, not a fault probability [@fujimaki2005telemetry, Secs. 2--4]. A learned correction to a physical orbit propagator is likewise useful only if residual covariance is validated [@peng2018orbit, Secs. 2--4].

**Study protocol: orbit residuals and telemetry.** Define one trajectory arc or mission day as the independent unit. Publish sensors, sampling cadence, propagator, coordinate frame, maneuver flags, and the rule producing residuals. Split chronologically, reserve complete maneuver and solar-activity regimes, and never place adjacent samples from one arc on both sides. Compare physics-only propagation, GP residual correction, and a simple linear residual model. Report position error by horizon, normalized innovation squared, empirical coverage of covariance ellipsoids, false alarms per mission day, detection delay, and compute latency. Negative controls include timestamp permutation, omitted maneuver covariates, and synthetic faults outside the training family. Reject deployment if long-horizon errors grow beyond the operational tolerance, covariance ellipsoids under-cover, false alarms exceed staffing capacity, or alarms arrive after the intervention deadline.

## Molecules and materials {#molecules-and-materials}

Kernel models remain important in molecular simulation because labels can be expensive, symmetry can be built into the representation, and a model-based variance can drive data acquisition. Deringer et al. review the GP construction, descriptors, sparse approximations, and validation problems [@deringer2021gpr, Secs. 2--6].

**Potentials from a kernel that knows the symmetries.** {#force-fields}

A Gaussian Approximation Potential regresses energies and derivatives using local atomic environments [@bartok2010gap, method and Eqs. (1)--(4)]. SOAP obtains rotational invariance by integrating overlap between smooth neighbor densities [@bartok2013soap, Secs. II--IV]. Gradient-domain learning imposes energy conservation through derivatives of a scalar kernel [@chmiela2017gdml, method and Eq. (1)]. Published benchmark numbers are not repeated here because comparisons depend on electronic-structure level, molecular split, conformation overlap, force weighting, and units.

**Study protocol: interatomic potential.** Name the chemical systems, electronic-structure code, functional, basis or cutoff, convergence thresholds, temperature and pressure schedule, and configuration-selection mechanism. Split by complete trajectories, molecules, compositions, phases, and rare-event pathways, not by individual atoms or adjacent frames. Compare the symmetry-aware kernel against the physics baseline it replaces, a simpler descriptor, and an ablation without force labels. The uncertainty currency is GP posterior variance or an explicitly calibrated residual score. Report energy error per atom, force-component RMSE, stress error, energy conservation, force consistency, stability duration, rare-event barrier error, calibration by chemical regime, and cost per molecular-dynamics step. Negative controls include rotation/permutation tests, withheld phases, compressed structures, and deliberately disconnected configuration families. Fail when symmetry tests break, trajectories become unstable, high-force tails exceed tolerance, uncertainty fails to rank errors, or the model crosses a barrier without triggering the reference calculation.

::::: {.example #example-56-3}
[Example (a bond length from a handful of points)]{.box-title}

```python
import numpy as np
De,re,a=4.75,.7416,1.942
morse=lambda r: De*(1-np.exp(-a*(r-re)))**2
def rbf(x,z,ell=.18):
    return np.exp(-(np.asarray(x)[:,None]-np.asarray(z)[None,:])**2/(2*ell**2))
r=np.array([.55,.60,.66,.72,.74,.78,.85,.95,1.1,1.4,1.8,2.4]); y=morse(r)
A=rbf(r,r)+1e-6*np.eye(r.size); coef=np.linalg.solve(A,y)
grid=np.linspace(.5,2.5,4001); prediction=rbf(grid,r)@coef
r_hat=grid[np.argmin(prediction)]
H=rbf(r,r)@np.linalg.solve(A,np.eye(r.size))
loo=(y-H@y)/(1-np.diag(H))
thin=r[2:]
thin_pred=rbf(grid,thin)@np.linalg.solve(
    rbf(thin,thin)+1e-6*np.eye(thin.size),morse(thin))
compressed=grid<=.72
print(r_hat,np.max(np.abs(loo)))
assert np.isclose(r_hat,.7425)
assert np.max(np.abs(thin_pred[compressed]-morse(grid[compressed]))) > (
  2*np.max(np.abs(prediction[compressed]-morse(grid[compressed])))
)
```

:::: wex
::: wex-setup
This is an **illustrative deterministic simulation**, not an ab initio benchmark. Twelve noiseless samples from a declared Morse curve are fit with fixed RBF-KRR hyperparameters. A held-out grid evaluates both the equilibrium location and worst-case energy error.
:::

1.  [Recover the physical quantity.]{.wex-op} The minimum of the kernel-ridge curve sits at \(0.7425\) angstrom, within \(0.9\) milliangstrom of the true \(0.7416\), from a dozen points.
2.  [Separate diagnostics.]{.wex-op} The check reports held-out grid error and leave-one-out sensitivity separately. Leave-one-out residuals are not predictive intervals.
3.  [Run a failure control.]{.wex-op} Removing the shortest-bond samples worsens error on the compressed region; the split exposes extrapolation that a random point split could hide.

**Reading.** The fixture verifies interpolation arithmetic and an extrapolation failure. It does not validate a molecular potential.
::::
:::::

**The killer app: variance triggers a calculation.** {#active-learning-chemistry}

For a fixed covariance and observation model, posterior variance can flag geometry unsupported by the design. Vandermause et al. use a GP uncertainty threshold to decide whether to accept a force prediction or request a first-principles calculation [@vandermause2020flare, Methods: “Bayesian active learning”]. Jinnouchi et al. use a related on-the-fly loop [@jinnouchi2019onthefly, Sec. II]. We omit savings percentages because they are protocol-specific.

**Study protocol: on-the-fly learning.** Calibrate the trigger only on trajectories withheld from fitting. Define the hazardous event, error cost, reference-call cost, maximum latency, and threshold before the final simulation. Compare uncertainty sampling with random sampling, geometric-distance sampling, and a fixed offline design at equal reference-call budgets. Report force-error recall above the hazard threshold, false-trigger rate, reference calls per simulated time, maximum untriggered error, trajectory stability, and wall-clock overhead. Include a disconnected-domain generator where an RBF GP is confidently wrong as a negative control. Stop the simulation if the uncertainty-error calibration leaves tolerance, a NaN or energy drift appears, or a high-error state passes without a trigger.

::::: {.example #example-56-4}
[Example (uncertainty sampling reaches accuracy sooner)]{.box-title}

```python
import numpy as np
f=lambda x:(x*x-1)**2+.3*np.sin(4*x)
grid=np.linspace(-2,2,400); truth=f(grid)
def rbf(x,z,ell=.35):
    return np.exp(-(np.asarray(x)[:,None]-np.asarray(z)[None,:])**2/(2*ell**2))
def predict(x):
    L=np.linalg.cholesky(rbf(x,x)+1e-4*np.eye(len(x))); cross=rbf(grid,x)
    mean=cross@np.linalg.solve(L.T,np.linalg.solve(L,f(x)))
    v=np.linalg.solve(L,cross.T)
    return mean,np.sqrt(np.maximum(1-(v*v).sum(0),0))
def run(seed=None):
    rng=np.random.default_rng(seed); x=np.array([-2.,0.,2.])
    for _ in range(25):
        mean,sd=predict(x)
        if np.sqrt(np.mean((mean-truth)**2))<.05:return len(x)
        x=np.append(x,grid[np.argmax(sd)] if seed is None else rng.uniform(-2,2))
    return len(x)
active=run(); random_mean=np.mean([run(s) for s in range(30)])
miss_x=np.array([-2.,-1.,0.,.8,1.,1.2,2.]); miss_mean,miss_sd=predict(miss_x)
i=np.argmin(np.abs(grid-1.05))
miss_truth=truth+1.5*np.exp(-((grid-1.05)/.035)**2)
print(active,random_mean,abs(miss_mean[i]-miss_truth[i]),miss_sd[i])
assert active==19 and abs(random_mean-26.6)<.05
assert abs(miss_mean[i]-miss_truth[i])>1 and miss_sd[i]<.2
```

:::: wex
::: wex-setup
This is an **illustrative deterministic simulation** on a declared one-dimensional function. It compares policies at equal stopping criteria and includes a misspecified-variance control.
:::

1.  [Sample where the model is unsure.]{.wex-op} Uncertainty sampling reaches the target with \(19\) points total.
2.  [Sample blindly.]{.wex-op} Random sampling needs \(26.6\) points on average over thirty seeds to reach the same error.
3.  [Count the saving.]{.wex-op} In this one-dimensional experiment, active learning uses \(0.72\) of the random budget, about a quarter fewer expensive evaluations. A more expensive label makes that saving more valuable; higher dimension does not guarantee that the same strategy remains effective.

**Reading.** The result is conditional on this generator, grid, kernel, seed set, and stopping rule. The negative control demonstrates that low model variance need not imply low error under misspecification.
::::
:::::

<figure class="viz" data-widget="onthefly-mlip">

<figcaption>An on-the-fly potential in miniature. A simulated trajectory runs across a one-dimensional energy surface on the cheap Gaussian-process energy; the predictive variance is plotted along the path, and when it spikes at an unseen barrier crossing a quantum calculation is triggered, marked, and added, after which the variance drops and the training set grows. The trajectory, the variance spike, and the retrain are all real computed events.</figcaption>
</figure>

**Bayesian optimization for discovery.** {#bo-discovery}

When the target is the best material or reaction, the posterior drives Bayesian optimization. The underlying computer-experiment design is described by Sacks et al. [@sacks1989dace, Secs. 2--4], and the sequential workflow by Shahriari et al. [@shahriari2016, Secs. 2--4]. Chemistry and materials case studies demonstrate the design pattern [@shields2021bo, Methods; @xue2016adaptive, Methods; @lookman2019active, Secs. 2--4], but their headline counts are not transferable performance guarantees.

**Study protocol: discovery.** Freeze the candidate space, feasibility constraints, assay or simulator, replicate policy, and total budget. Use repeated historical replay only when unobserved counterfactual outcomes are available; otherwise run a prospective randomized comparison across acquisition policies. Compare expected improvement or another declared acquisition with random, space-filling, and expert-selected designs. Report best-so-far value versus cost, simple regret when ground truth is known, feasibility violations, replicate variability, and time to decision. Negative controls include shuffled outcomes and a kernel that removes chemically meaningful structure. Failure means an acquisition repeatedly proposes infeasible points, uncertainty is anti-correlated with error, or improvement disappears under matched budgets and seeds. Operational validation requires independent synthesis or assay of the selected candidate.

## Earth observation {#earth-observation}

Remote-sensing kernels combine spectral and spatial information through

$$k=\mu k_{\mathrm{spectral}}+(1-\mu)k_{\mathrm{spatial}},\qquad 0\leq\mu\leq1,$$

a PSD construction used in composite-kernel classification [@campsvalls2006composite, Eqs. (1)--(4)]. GP retrieval maps reflectance to variables such as leaf area index or chlorophyll [@verrelst2012retrieval, Secs. II--IV; @verrelst2013uncertainty, Secs. 2--4]. Model variance becomes a confidence layer only after spatial and temporal calibration.

**Study protocol: Earth-observation retrieval.** Publish the sensor product and version, atmospheric correction, bands, cloud mask, ground-truth instrument, geolocation tolerance, acquisition dates, sites, target units, and missing-data rules. Hold out complete regions and seasons; prevent neighboring pixels or repeated measurements of one plot from crossing splits. Compare the GP with a linear or physical retrieval baseline and an RBF model without spatial context. The uncertainty currency is a predictive standard deviation conditional on the retrieval model. Report MAE and RMSE in physical units, bias by site and season, interval coverage and width, spatial residual autocorrelation, support or drift score, map completeness, and inference cost. Negative controls include coordinate permutation, withheld sensor or season, corrupted atmospheric correction, and target permutation. Failure occurs when geographic coverage falls below tolerance, residuals retain spatial structure, uncertainty is narrower out of support, or errors exceed the action threshold for a declared land-management decision.

::::: {.example #example-highstakes-spatial}
[Example (why pixel-wise splitting lies)]{.box-title}

:::: wex
::: wex-setup
This **illustrative deterministic simulation** creates spatially correlated groups with group-specific offsets. It compares a random pixel split with leave-one-region-out evaluation and audits Gaussian interval coverage. The simulation and its assertions are independently reproducible from the chapter's computational reference [@kernelbook-code-ch-highstakes-ex5].
:::

1. [Leak by splitting pixels.]{.wex-op} Nearby training pixels reveal the regional offset, so the random split looks artificially accurate.
2. [Hold out a region.]{.wex-op} Grouped evaluation exposes the unseen offset and larger error.
3. [Audit the uncertainty.]{.wex-op} A standard deviation estimated from leaked residuals under-covers the held-out region, so the confidence layer fails its declared currency.

**Reading.** The unit of independence determines the credibility of both accuracy and uncertainty. This fixture is not a satellite benchmark.

```python
import numpy as np

rng = np.random.default_rng(29)
region = np.repeat(np.arange(8), 40)
x = rng.uniform(-1.0, 1.0, region.size)
offset = np.array([-1.4, -0.9, -0.5, -0.1, 0.2, 0.6, 1.0, 1.5])
y = 2.0 * x + offset[region] + 0.12 * rng.standard_normal(x.size)

def evaluate(train, test):
    design = lambda idx: np.column_stack(
        [np.ones(idx.size), x[idx]] +
        [(region[idx] == group).astype(float) for group in range(8)]
    )
    X_train = design(train)
    coef = np.linalg.solve(
        X_train.T @ X_train + 1e-8 * np.eye(10), X_train.T @ y[train]
    )
    residual = y[train] - X_train @ coef
    prediction = design(test) @ coef
    sigma = np.sqrt(np.mean(residual**2))
    return (np.sqrt(np.mean((y[test] - prediction)**2)),
            np.mean(np.abs(y[test] - prediction) <= 1.96 * sigma))

permutation = rng.permutation(y.size)
random_result = evaluate(permutation[80:], permutation[:80])
held_out = region == 7
group_result = evaluate(np.flatnonzero(~held_out), np.flatnonzero(held_out))
assert random_result[0] < 0.15 and group_result[0] > 1.0
print(random_result, group_result)
```
::::
:::::

## What the cases share {#what-the-cases-share}

Read together, the protocols make one argument three times. A useful uncertainty output is part of the product, but each currency has a different validation route: simulation coverage for a parameter posterior, search-background calibration for a detection score, withheld-regime error recall for an active-learning trigger, covariance consistency for orbit residuals, and geographic-temporal coverage for a map. None can borrow validity from another.

<figure class="viz" data-figure="uncertainty-decomposition-highstakes" data-alt="A stacked variance budget over an operating condition separates aleatoric noise, Gaussian-process posterior epistemic variance, and an explicit shift allowance. The shift allowance grows outside the shaded training support."><figcaption>Posterior variance is only one term in a high-stakes uncertainty budget. Within support it can measure finite-data uncertainty conditional on the model; outside support, an explicit shift allowance must record that the covariance itself is no longer calibrated. The decomposition prevents model confidence from masquerading as operational confidence.</figcaption></figure>

Scalability is part of the estimand. Semiseparable covariance, inducing points, principal-component emulation, Nyström landmarks, and random features change what is computed. A study must report approximation rank, residual or likelihood tolerance, timing hardware, and whether calibration survives approximation.

## Common mistakes and practical implications {#highstakes-practice}

- Do not call a covariance calibrated until empirical coverage or a proper-score audit has been run on the deployment split.
- Data efficiency is not license to skip validation: with a handful of expensive samples, an over-flexible kernel interpolates the noise, so cross-validation and a smoothness prior are what keep few-point interpolation honest.
- Physics written into the kernel must be the physics of the task; a rotation-invariant atomic kernel applied where orientation carries signal, or a matched filter built on a mis-specified template, encodes the wrong prior and misses what matters.
- Matched-filter and anomaly scores are ranking statistics, not calibrated probabilities, until their null distribution over correlated trials is worked out; reading a raw peak as a detection confidence overstates significance.
- The exact Gaussian process is cubic in the number of points, so every mature deployment here pairs it with a scalability structure (celerite, inducing points, principal-component emulation, Nystrom or random features), and treating the exact solve as the only option caps the problem size artificially.
- An error bar is a claim about the model, not about reality; when the training set no longer resembles the deployment regime, the covariance stops meaning what the report says it means, and a drift check must guard it.

The practical implication is consistent across all domains: every deliverable should name the data version, split unit, kernel, baseline, uncertainty currency, calibration test, negative controls, failure threshold, and operational decision.

Each case therefore has four ledgers. Kernel validity, symmetry, and
conservative-force construction are theorem-level claims. Recovery curves and
benchmark errors are empirical evidence. GP variance and acquisition scores
are model-based uncertainty. Operational validation measures missed events,
trajectory stability, simulator discrepancy, spatial transfer, and the cost
of an unnecessary intervention.

## Summary and further reading {#summary}

Kernel methods earn their place in these sciences when three conditions meet: labels are expensive, useful structure can be written as a covariance or inner product, and the resulting uncertainty can be checked against the decisions it will control. Quasi-periodic kernels encode evolving stellar signals; noise-weighted inner products define matched filters; symmetry-aware kernels reduce the sample burden of molecular potentials; spatial-spectral kernels connect satellite pixels to context. The output is never “an error bar for free.” It is a model-based covariance, score, or interval whose credibility depends on held-out residuals, shift tests, support, numerical approximation, and the cost of a wrong action. The scalable method must be named as carefully as the kernel: semiseparable structure, inducing points, sparse precision, or low-rank features change what is computed. In high-stakes work, the kernel is valuable because the modeling claim is explicit enough to test and reject.

For further reading, use the astronomy review [@aigrain2023gp, Secs. 3--5], the molecular GP review [@deringer2021gpr, Secs. 2--6], and the Earth-observation survey [@campsvalls2016gpsurvey, Secs. II--V]. The individual protocol sections above point to the primary method descriptions.

The complete workflow returns to
[[ch:applications-and-practice|the applications chapter]]: encode the domain,
fit within a leakage-safe split, test the operational quantity, and retain the
evidence needed to reconstruct every decision.

::: {.exercises}
## Exercises {#exercises}

1.  [warm-up]{.ex-tag} For stellar rotation, matched filtering, an on-the-fly potential, orbit correction, and satellite retrieval, name the uncertainty currency and one operational validation test.
2.  [computation]{.ex-tag} In the white-noise matched-filter fixture, \(\rho=\lVert h\rVert/\sigma\). With \(\lVert h\rVert=7.372\) and \(\sigma=0.9215\), verify \(\rho=8\), and compute the one-sided probability that one standard normal exceeds \(7.37\). Explain why this is not the search false-alarm probability.
    Hint

    ::: hint-body
    \(7.372/0.9215=8.0\). Use \(\tfrac12\operatorname{erfc}(7.37/\sqrt2)\). A search maximizes over correlated times and templates and is exposed to non-Gaussian noise, so its background must be calibrated end to end.
    :::
3.  [proof]{.ex-tag} Show that the composite kernel \(k=\mu\,k_1+(1-\mu)\,k_2\) with \(\mu\in[0,1]\) is positive definite whenever \(k_1,k_2\) are, and explain why the mixing weight \(\mu\) is therefore a safe, interpretable dial in the spatial-spectral remote-sensing kernel. What goes wrong if \(\mu\) is allowed outside \([0,1]\)?
    Hint

    ::: hint-body
    A nonnegative combination of positive definite kernels is positive definite, since the Gram matrix is the same nonnegative combination of positive semidefinite matrices. Outside \([0,1]\) a negative coefficient can destroy positive definiteness, so the \"kernel\" is no longer a valid inner product.
    :::
4.  [computation]{.ex-tag} Reproduce the Morse fixture and compare the full design with the design that removes the shortest-bond samples. Report equilibrium error and maximum compressed-region error. Why is leave-one-out sensitivity not a predictive interval?
    Hint

    ::: hint-body
    The executable check gives the exact values. Leave-one-out changes the fitted dataset and measures sensitivity at observed sites; it does not supply a probability model or coverage guarantee at a new configuration.
    :::
5.  [challenge]{.ex-tag} An on-the-fly force field trusts the GP below a score threshold. Let \(C_e\) be the cost of a hazardous error and \(C_q\) the cost of a reference call. Derive the decision rule if \(p(s)\) is a calibrated hazard probability given score \(s\). State what cannot be derived from an uncalibrated posterior variance.
    Hint

    ::: hint-body
    Request the reference calculation when \(C_e p(s)\gt C_q\), or \(p(s)\gt C_q/C_e\). A variance threshold follows only after estimating a monotone calibration map from variance to hazard probability on representative held-out trajectories.
    :::
6.  [exploration]{.ex-tag} The chapter claims the exact GP is cubic and every mature application pairs it with a scalability structure. Pick two, celerite's semi-separable covariance and inducing-point sparse GPs, and explain what structural assumption each exploits and what it gives up, connecting them to the Nyström and random-feature trade-offs of [[ch:large-scale-kernels|the large-scale chapter]].
    Hint

    ::: hint-body
    Celerite restricts the kernel to a sum of (complex) exponentials on one-dimensional inputs, buying an exact linear-time likelihood at the cost of generality; inducing points summarize the data by \(m\ll n\) pseudo-points, an approximation whose fidelity is governed by the effective dimension, the same quantity that sets how many Nyström landmarks suffice.
    :::
7.  [synthesis]{.ex-tag} Design an Earth-observation retrieval study using the eight-field template. Explain why random pixel splitting is invalid when residuals are spatially correlated, and give one predeclared coverage failure criterion.
    Hint

    ::: hint-body
    Hold out complete regions and seasons, fit all preprocessing and hyperparameters inside the development data, and audit physical error plus interval coverage by region. A possible failure rule is coverage below \(90\%\) for a nominal \(95\%\) interval in any operational region with adequate sample size.
    :::
:::
