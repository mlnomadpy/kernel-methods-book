---
id: ch-apps
slug: applications-and-practice
title: Applications and Practice
part: XIII · Practice
order: 46
tier: core
prerequisites:
  - the-frontier
objectives:
  - >-
    Translate vector, sequence, graph, distributional, spatial, and scientific
    data into an explicit kernel-design choice.
  - >-
    Run a leakage-safe model-selection loop with normalization, nested
    validation, and non-kernel baselines.
  - >-
    Diagnose capacity and conditioning from the Gram spectrum before changing
    bandwidth or regularization.
  - >-
    Choose exact, Nyström, random-feature, or matrix-free computation from the
    deployment budget.
  - >-
    Produce a reproducibility packet and model card that expose uncertainty,
    shift, influence, and failure conditions.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-apps.yml
verification_date: null
bibliography:
  - leslie2002
  - leslie2004
  - jaakkola2000
  - jaakkola1999fisher
  - saigo2004
  - haussler1999
  - joachims1998text
  - lodhi2002
  - cristianini2002lsk
  - ralaivola2005
  - gartner2003
  - borgwardt2005
  - shervashidze2011
  - scholkopf2002
  - shawe2004
  - chapelle2002
  - steinwart2008
  - rasmussen2006
  - rahimi2007
  - drineas2005
  - williams2001
  - rudi2015
  - rudi2017falkon
  - meanti2020
  - le2013fastfood
  - chang2011libsvm
  - fan2008liblinear
  - pedregosa2011sklearn
  - matthews2017gpflow
  - gardner2018gpytorch
  - charlier2021keops
  - sonnenburg2010shogun
---
# Applications and Practice

<p class="lead">A protein sequence arrives with one question attached: which structural family does it belong to? Two members of the same superfamily can share as little as a fifth of their amino acids, so counting matches gets you nowhere, and the object is a variable-length string, not a vector any classifier expects. The same shape recurs for a document to be filed by topic and a molecule whose toxicity must be read off its bonds. Every chapter so far has built one piece of the answer: a construction that turns structure into an inner product, a convex learner that fits once the inner product is fixed, a theorem that says when the fit will generalize. This closing chapter assembles them the way a practitioner does. We walk three classic applications end to end, distill the choices that decide whether a kernel method works in practice, from picking a kernel for your data type to model selection, normalization, and the spectrum diagnostics that separate under from overfitting, and close with a map of the software that puts all of this a few lines of code away. The aim is not new theory but judgment: given a problem, which of the book's kernels do you reach for, how do you tune it honestly, and when do you trade the exact solve for an approximation?</p>

## The anatomy of a kernel-method project {#anatomy}

A kernel method has a rigid skeleton, and seeing it once makes every application below a variation on one theme. You choose a kernel \(k\) that encodes similarity for your data, form or implicitly access the Gram matrix \(K_{ij}=k(x_i,x_j)\), feed it to a convex learner (a support vector machine, kernel ridge regression, a Gaussian process), and tune a few hyperparameters by cross-validation. The kernel is where domain knowledge enters, the learner is off the shelf, and the tuning is where honesty enters. Almost everything separating a method that works from one that does not lives in three decisions: whether the kernel matches the structure of the data, whether the data and kernel are normalized so the similarity means what you think, and whether the model's capacity is matched to the data, which is what the spectrum of \(K\) measures.

Two facts frame the enterprise. There is no universally best kernel: the representer theorem and the RKHS geometry hold for any positive definite \(k\), so choosing \(k\) is choosing what \"similar\" means, a modeling decision no theorem makes for you. And once \(k\) is fixed the problem is convex, so the only real freedom left is the handful of hyperparameters. The art is front-loaded into kernel design and back-loaded into model selection, with a mechanical optimization between: the applications show the former, the recipe the latter.

A defensible project is therefore a loop rather than a one-way training script. Geometry and normalization are checked before optimization; spectrum and conditioning determine the compute route; nested validation chooses the model; and deployment evidence can send the project back to the kernel itself.

<figure class="viz" data-figure="kernel-workflow" data-alt="A six-step kernel workflow moves from encoding structure through normalization, geometry diagnostics, compute choice, nested validation, and deployment audit, with feedback arrows returning failures to earlier design decisions.">
<figcaption>A kernel result is defensible only when modeling, computation, selection, and auditing form one evidence loop. Validation failure revisits capacity and compute; shift or calibration failure revisits the geometry itself.</figcaption>
</figure>

## Three applications, three kernels {#three-applications}

The kernels for structured data earn their keep where the raw objects, protein sequences, documents, molecules, are not vectors at all. In each the kernel is the only bridge from the object to the RKHS machinery, and in each a specific kernel plus a plain support vector machine set the state of the art for years.

### Remote protein homology with string and mismatch kernels {#protein-homology}

The task is detection of distant evolutionary relatives. Two proteins in the same structural superfamily can share as little as a fifth of their amino acids, so a raw identity count fails, and the goal is to decide, for a query sequence, whether it belongs to a given family despite that low overlap. The discriminative formulation, following Jaakkola, Diekhans, and Haussler (2000), trains a support vector machine to separate family members from non-members, which turns the problem into the design of a kernel between variable-length strings over the twenty-letter amino-acid alphabet.

The spectrum kernel of Leslie, Eskin, and Noble (2002) is the simplest such kernel and already very strong. For a fixed length \(k\), its feature map counts the occurrences of every \(k\)-mer (contiguous substring of length \(k\)), so a protein becomes a vector of \(20^k\) counts and the kernel is the inner product of two count vectors. The mismatch kernel of Leslie et al. (2004) relaxes the exact match: a \(k\)-mer contributes to every \(k\)-mer within \(m\) substitutions of it, giving the \((k,m)\)-mismatch kernel, with \((5,1)\) a standard choice. Both are string kernels from [[ch:string-kernels|the chapter on sequence kernels]], computed without materializing the \(20^k\)-dimensional vector by the trie and suffix-tree techniques of [[ch:efficient-string-and-tree-kernels|the efficient string-kernel chapter]] in time linear in sequence length. Since a long protein has more \(k\)-mers than a short one, the raw kernel scales with length and is always cosine-normalized before use, the normalization step discussed below.

A complementary route lets a generative model design the features. The Fisher kernel of Jaakkola and Haussler (1999) fits a profile hidden Markov model to the positive family, represents each sequence by the gradient of its log-likelihood under that model, and feeds those Fisher scores to the SVM; this is the generative-model kernel of [[ch:generative-and-marginalization-kernels|the chapter on kernels from generative models]] and the original SVM-Fisher method for the benchmark. The string and Fisher kernels bracket the two philosophies of the book: build similarity from the object, or read it off a fitted model.

The pipeline is identical in either case: build the Gram matrix over the training sequences, normalize it, train one SVM per family, score held-out sequences. On the standard SCOP benchmark, in which each of thirty-three families is held out in turn and performance is measured by the ROC and ROC50 scores (the area under the ROC curve, and under its first fifty false positives), the string kernels match or exceed the SVM-Fisher method while being computed directly from sequence with no model to fit (Leslie et al. 2004; Jaakkola et al. 2000). Alignment kernels scoring pairs by their optimal local alignment (Saigo et al. 2004) push detection further still, and all are convolution kernels in the sense of Haussler (1999). The lesson that carried into practice: a cheap, exact, sequence-derived kernel can beat an expensive model-based one, and the string kernels became the default for biological sequence classification.

### Text categorization with the vector-space and string kernels {#text-categorization}

Assigning topic labels to documents was the problem on which support vector machines first proved themselves in the discrete world. The classical representation is the bag of words: fix a vocabulary and represent a document by its (TF-IDF weighted) term counts, ignoring word order. The kernel is the inner product in this space, which once the vectors are length-normalized is exactly the cosine similarity of the vector-space model. The SVM is the right learner because of the geometry the earlier chapters built: the feature space has tens of thousands of dimensions, almost all irrelevant to any given category, and the margin bounds of [[ch:support-vector-machines|the support vector machine chapter]] guarantee that a large-margin separator there generalizes from few examples with no explicit feature selection.

Joachims (1998) established this on the Reuters-21578 benchmark, where the SVM was the best of the methods tried across essentially all categories. The reported microaveraged precision-recall breakeven points make the gap concrete.

  Method                                                  Microaveraged breakeven (%)
  ------------------------------------------------------- -----------------------------
  Naive Bayes                                             72.0
  Rocchio                                                 79.9
  C4.5 decision tree                                      79.4
  \(k\)-nearest neighbors   82.3
  SVM (RBF kernel, best setting)                          86.4

The numbers are those reported by Joachims (1998) on the Reuters-21578 \"ModApte\" split; the SVM improves on the strongest classical baseline, the nearest-neighbor rule, by about four points and on naive Bayes by more than fourteen. Word order can be recovered without leaving the framework: the string subsequence kernel of Lodhi et al. (2002) works directly on character sequences, scoring documents by their shared (possibly gapped) substrings, competitive with the word kernels while needing no tokenization, and latent semantic kernels (Cristianini et al. 2002) fold a low-rank semantic smoothing into the same inner product. Both are developed in [[ch:kernels-for-text|the chapter on kernels for text]]. The takeaway is that for text a normalized linear kernel on a good weighting is a strong baseline that is hard to beat, with the string kernels held in reserve for when subword structure matters.

### Molecular property prediction with graph kernels {#molecular-prediction}

A small molecule is naturally a labeled graph: atoms are vertices labeled by element, bonds are edges labeled by type. Predicting a property (mutagenicity, toxicity, activity) from that graph is the cheminformatics problem, and graph kernels are the bridge to the SVM. The difficulty is fundamental: Gärtner et al. (2003) showed that a kernel able to separate any two non-isomorphic graphs is as hard to compute as graph isomorphism, so every practical graph kernel compares tractable substructures. The marginalized graph kernel counts label sequences along random walks, a marginalized kernel in the sense of [[ch:generative-and-marginalization-kernels|the generative-kernels chapter]]; the shortest-path kernel of Borgwardt and Kriegel (2005) compares multisets of shortest-path lengths; and the Weisfeiler-Lehman subtree kernel of Shervashidze et al. (2011), comparing iteratively refined subtree patterns, is the scalable modern default. All belong to [[ch:graph-kernels|the chapter on kernels for and on graphs]].

The application that named the field is Ralaivola et al. (2005), who built marginalized and fingerprint-based (Tanimoto and MinMax) graph kernels, paired them with an SVM, and reported performance competitive with the best hand-engineered molecular descriptors of the time on standard structure-activity benchmarks: mutagenicity, carcinogenicity, and the NCI anticancer screens. Because a random-walk kernel sums over walks of every length, its raw value grows with molecule size, so cosine normalization is again mandatory. This is what moved cheminformatics onto the kernel framework: the same SVM code that classifies proteins and documents classifies molecules, once a graph kernel supplies the inner product. The worked examples below are the small numeric surrogate for such a pipeline.

## The practical recipe {#the-recipe}

The applications share a skeleton, and that skeleton is the recipe: choose a kernel, normalize the inputs and the kernel, select the hyperparameters by nested cross-validation, and diagnose the fit. We take these in turn and collect them into one algorithm.

### Choosing a kernel for your data {#choosing-a-kernel}

The first decision is the least mechanical, and the book has been a catalog of answers to it. The following guide maps a data type to a sensible first kernel and to the chapter that builds it. It is a starting point, not a verdict: the honest test is always the cross-validated error of the section that follows.

  Your data                                    A good first kernel                                                   Where it is built
  -------------------------------------------- --------------------------------------------------------------------- -------------------------------------------------------------------------------------------------------------------------------------------
  Fixed-length real vectors, smooth response   Gaussian (RBF); Matern if you want to tune smoothness                 [[ch:kernel-families|Kernel families]], fit with [[ch:support-vector-machines|SVM]] / [[ch:kernel-ridge-and-friends|KRR]] / [[ch:gaussian-processes-and-rvm|GP]]
  Vectors, interactions of bounded order       Polynomial or ANOVA kernel                                            [[ch:kernel-families|Kernel families]]
  Biological sequences (variable length)       Spectrum, mismatch, or alignment string kernel                        [[ch:string-kernels|Sequence kernels]], [[ch:efficient-string-and-tree-kernels|efficient evaluation]]
  Free-text documents                          Normalized bag-of-words linear kernel, or string subsequence kernel   [[ch:kernels-for-text|Kernels for text]]
  Molecules, parse trees, networks             Weisfeiler-Lehman, random-walk, or shortest-path graph kernel         [[ch:graph-kernels|Graph kernels]]
  Objects with a fitted generative model       Fisher or marginalized kernel                                         [[ch:generative-and-marginalization-kernels|Generative-model kernels]]
  Data with a known invariance                 Invariant / jittered kernel, tangent distance                         [[ch:invariances-and-pre-images|Invariances]]
  Sets, bags, or probability distributions     Mean-embedding or distribution kernel                                 [[ch:distribution-regression|Distribution regression]]
  Several heterogeneous sources                A learned combination of base kernels                                 [[ch:multiple-kernel-learning|Multiple kernel learning]]

Two rules of thumb cut across the table. On vectors in doubt, start with the Gaussian kernel and one well-tuned bandwidth: it is universal, and its single length scale is easy to search. On structured data, spend your effort on the kernel, not the classifier, since the SVM downstream is interchangeable. And if you truly cannot choose, do not: multiple kernel learning turns the choice into a convex problem, as [[ch:multiple-kernel-learning|its chapter]] develops.

### Normalization and centering {#normalization}

A kernel is a statement about similarity, and that statement is only as trustworthy as the units it is made in. Three normalizations recur, and skipping them is the most common way a sound method underperforms.

*Feature standardization.* For a distance-based kernel such as the Gaussian, a coordinate measured in large units dominates \(\lVert x-x'\rVert^2\) and silently sets the similarity, so raw features are first standardized to zero mean and unit variance, \(x\mapsto (x-\hat\mu)/\hat\sigma\) coordinatewise. The estimates \(\hat\mu,\hat\sigma\) must be computed on the training fold alone; computing them on the full data before splitting leaks information and biases the cross-validated error downward.

*Kernel (cosine) normalization.* Any kernel can be rescaled to unit diagonal by

$$\tilde k(x,x')=\frac{k(x,x')}{\sqrt{k(x,x)\,k(x',x')}},$$

which sends every point to the unit sphere of feature space, so \(\tilde k(x,x')=\cos\angle(\Phi(x),\Phi(x'))\). This is a conformal transformation of the kernel and preserves positive definiteness, as [[ch:kernel-families|the closure properties]] guarantee. It is cosmetic for the Gaussian, whose diagonal is already \(1\), but essential for string and graph kernels, whose raw magnitude tracks object size rather than content, and for polynomial kernels, whose magnitude tracks input scale.

*Centering in feature space.* Some methods need the data centered at the origin of feature space. The centered Gram matrix is

$$\tilde K = H K H,\qquad H = I-\tfrac1n\mathbf{1}\mathbf{1}^\top,$$

which subtracts the feature-space mean without ever computing it, and it is the first step of kernel PCA in [[ch:kernel-pca|its chapter]]. Supervised machines often skip it: the constraint \(\sum_i\alpha_i y_i=0\) in the SVM dual already makes the solution invariant to a constant offset of the kernel, the conditional-positive-definiteness invariance of [[ch:kernel-families|the kernel-families chapter]].

### The model-selection loop {#model-selection}

With the kernel and its normalizations fixed, a kernel machine still has a small grid of dials: the kernel hyperparameters (the Gaussian bandwidth \(\sigma\), the polynomial degree) and the regularization (the SVM penalty \(C\), the ridge \(\lambda\)). These are not learned by the convex fit but chosen by estimating out-of-sample error. The most common methodological error in applied kernel work is to let one cross-validation both choose the hyperparameters and estimate the final error: reusing the selection folds to report performance is optimistic, because the grid was tuned to those very folds. The fix is nested cross-validation, an inner loop that selects and an outer loop that scores, so no data point ever helps choose the model later graded on it. Chapelle et al. (2002) give gradient-based alternatives to the grid; the grid remains the robust default.

:::: {.algorithm #algo-44-1}
[Algorithm (End-to-end kernel method)]{.box-title}

::: algo-io
[Input]{.algo-lab} data \(\{(x_i,y_i)\}_{i=1}^n\); a kernel family \(k_\theta\) with hyperparameter grid \(\Theta\); a learner with regularization grid \(\mathcal C\); outer/inner fold counts \(K_1,K_2\).

[Output]{.algo-lab} a deployed predictor and an unbiased estimate of its error.
:::

1.  Encode each object into the kernel's input form (standardize vectors; leave sequences, graphs, and sets as is).
2.  Split the data into \(K_1\) outer folds.
3.  For each outer fold, holding out \((\text{test})\) and keeping \((\text{train})\):
4.  fit the feature standardization on \((\text{train})\) only;
5.  for each \((\theta,C)\in\Theta\times\mathcal C\), estimate the error on \((\text{train})\) by \(K_2\)-fold inner cross-validation, rebuilding and cosine-normalizing the Gram matrix inside each inner fold;
6.  pick \((\theta^\star,C^\star)\) minimizing the inner error, refit on all of \((\text{train})\), and record the error on \((\text{test})\).
7.  Report the mean and spread of the outer-fold errors as the generalization estimate.
8.  Refit once on all the data at the \((\theta^\star,C^\star)\) selected on the full set, and deploy that predictor.
::::

The worked example makes the inner loop concrete on a dataset small enough to check every number by hand.

::::: {.example #example-44-1}
[Example (a model-selection loop by hand)]{.box-title}

:::: wex
::: wex-setup
Six points on a line, \(x_{\text{raw}}=(0,1,2,3,4,5)\), with regression target \(y=\sin(x_{\text{raw}})=(0,\,0.8415,\,0.9093,\,0.1411,\,-0.7568,\,-0.9589)\). We standardize the inputs (mean \(2.5\), standard deviation \(1.7078\)) to \(x=(-1.4639,\,-0.8783,\,-0.2928,\,0.2928,\,0.8783,\,1.4639)\), fit kernel ridge regression with the RBF kernel \(k(x,x')=e^{-(x-x')^2/2\sigma^2}\) and fixed ridge \(\lambda=0.1\), and select the bandwidth by 2-fold cross-validation over \(\sigma\in\{0.5,\,2.0\}\). The interleaved folds are \(A=\{0,2,4\}\) and \(B=\{1,3,5\}\). All numbers from `checks/ch-apps-ex1.py`.
:::

1.  [Score the narrow bandwidth.]{.wex-op} At \(\sigma=0.5\), training on \(A\) and testing on \(B\) gives held-out MSE \(0.1786\); training on \(B\) and testing on \(A\) gives \(0.1686\); the mean CV error is \(0.1736\).
2.  [Score the wide bandwidth.]{.wex-op} At \(\sigma=2.0\), the two folds give \(0.0935\) and \(0.3724\), averaging to \(0.2330\). The wide kernel is excellent on one split and poor on the other, a reminder that averaging folds, not trusting a single split, is what stabilizes the estimate.
3.  [Pick the winner.]{.wex-op} The narrow bandwidth has the smaller CV error, \(0.1736\lt 0.2330\), so \(\sigma^\star=0.5\).
4.  [Refit and read the gap.]{.wex-op} Refitting at \(\sigma^\star=0.5\) on all six points drives the training MSE to \(0.0025\), far below the cross-validated \(0.1736\). The near-interpolation of the training data alongside a much larger held-out error is the signature of a high-capacity fit, which is exactly what the spectrum will quantify next.

**Reading.** Cross-validation, not a smoothness heuristic, chooses the bandwidth: here it prefers the narrower, higher-capacity kernel because the target genuinely wiggles over the range, even though \"wider is smoother is safer\" would have guessed otherwise. The wide-bandwidth fold swing and the narrow-bandwidth train-versus-CV gap are the two things to watch, and both are visible in numbers on six points.
::::

**Verification artifact.** checks/example-ch-apps-example-44-1.json records the example source hash and verification scope.
:::::

### Diagnosing under and overfitting through the spectrum {#spectrum-diagnosis}

When a fit disappoints, the eigenvalues of the Gram matrix say why. Mercer's theorem in [[ch:mercer-and-rates|its chapter]] reads the spectrum of \(K\) as the energy the kernel places along each eigen-direction, and how fast the eigenvalues decay is the effective number of directions the model can use. A useful scalar summary at ridge level \(\lambda\) is the effective dimension

$$d_{\text{eff}}(\lambda)=\sum_{i=1}^n\frac{\lambda_i}{\lambda_i+\lambda},$$

which counts eigenvalues large compared to \(\lambda\) (each near \(1\)) and discounts the small ones (near \(0\)). It is the number of parameters the machine actually spends, running from \(0\) when \(\lambda\) swamps every eigenvalue to \(\operatorname{rank}(K)\) as \(\lambda\to 0\), and it is the degrees of freedom of kernel ridge regression that govern the rates of [[ch:kernel-ridge-and-friends|the ridge chapter]] and [[ch:gaussian-processes-and-rvm|the Gaussian-process chapter]].

::::: {.example #example-44-2}
[Example (reading the spectrum)]{.box-title}

:::: wex
::: wex-setup
The same six standardized points. We form the RBF Gram matrix at the two bandwidths and read its eigenvalues and effective dimension at \(\lambda=0.1\). The trace is \(6\) in both cases (unit diagonal), so the spectra differ only in how that fixed energy is distributed. All numbers from `checks/ch-apps-ex2.py`.
:::

1.  [Take the wide-bandwidth spectrum.]{.wex-op} At \(\sigma=2.0\) the eigenvalues are \((4.8828,\,1.0143,\,0.0976,\,0.0052,\,0.0002,\,0.0000)\): the top two already hold \(98.3\%\) of the trace, and the tail is numerically negligible. The kernel exposes very few directions.
2.  [Take the narrow-bandwidth spectrum.]{.wex-op} At \(\sigma=0.5\) the eigenvalues are \((1.9971,\,1.6210,\,1.1422,\,0.6959,\,0.3666,\,0.1773)\): the energy is spread across all six directions, with the top two holding only \(60.3\%\).
3.  [Convert to effective dimension.]{.wex-op} At \(\lambda=0.1\) the wide kernel has \(d_{\text{eff}}=2.43\), the narrow one \(d_{\text{eff}}=5.11\). The wide kernel is a roughly two-parameter model; the narrow one spends five of its six possible parameters.
4.  [Match the diagnosis to the fit.]{.wex-op} The narrow kernel's high effective dimension is why it nearly interpolated the six points in the previous example, and why its train error collapsed while its CV error did not. A wide kernel with \(d_{\text{eff}}=2.43\) would risk the opposite failure, too few directions to bend to the target, which is underfitting.

**Reading.** The spectrum turns a vague worry about capacity into a number. A fast-decaying spectrum (small \(d_{\text{eff}}\)) means a simple model prone to underfitting: narrow the bandwidth or lower the ridge to expose more directions. A slow-decaying spectrum (large \(d_{\text{eff}}\)) with a train-versus-validation gap means overfitting: widen the bandwidth or raise the ridge. Both knobs act on the same quantity, \(d_{\text{eff}}\), and matching it to the data is the whole game.
::::

**Verification artifact.** checks/example-ch-apps-example-44-2.json records the example source hash and verification scope.
:::::

### Exact, Nystrom, or random features {#exact-nystrom-rff}

Everything above assumes the \(n\times n\) Gram matrix can be formed and factored, at \(O(n^2)\) memory and \(O(n^3)\) time. That is fine up to roughly \(n\sim 10^4\); beyond it the exact solve, not the statistics, becomes the bottleneck. [[ch:large-scale-kernels|The large-scale chapter]] develops two escapes, and the choice follows the data type.

The Nystrom method (Williams and Seeger 2001; Drineas and Mahoney 2005) samples \(m\ll n\) landmark points and approximates \(K\) by its best rank-\(m\) reconstruction from the landmark columns, a low-rank approximation of the top of the spectrum. It shines when the spectrum decays fast, that is, when \(d_{\text{eff}}\) is small, because then a few landmarks capture the signal; the effective dimension of the previous section is the right guide to how many landmarks suffice, and Rudi et al. (2015) show that \(m\) of order \(d_{\text{eff}}\) preserves the statistical rate. It works for any kernel, including string and graph kernels, needing only to evaluate \(k\) on sampled pairs. Random Fourier features (Rahimi and Recht 2007) instead sample the Bochner spectral measure of a translation-invariant kernel to build an explicit feature map whose inner product approximates \(k\), and Fastfood (Le et al. 2013) accelerates the sampling; this route is specific to shift-invariant kernels on vectors but feeds a fast linear solver directly. As a rule of thumb: exact solve up to \(n\sim 10^4\); Nystrom for structured data or fast-decaying spectra; random features for the Gaussian and its relatives on vectors. For the largest problems the preconditioned Nystrom solver Falkon (Rudi et al. 2017; Meanti et al. 2020) fits kernel ridge regression on \(10^7\) to \(10^9\) points on a single machine.

## Case study: spatial exposure mapping {#case-study-spatial}

Suppose monitoring stations measure an environmental exposure and the goal is a map with uncertainty. Random train-test splitting is optimistic because nearby stations are correlated. The unit of validation should be a held-out region or buffered spatial block. Start with a mean model for known covariates, then model residual dependence with the spatial kernels of [[ch:spatial-and-spatiotemporal-kernels]]. Compare a Matern GP, a compactly supported approximation, and a simple regional baseline.

The deliverable is not only RMSE. Report spatially blocked error, interval coverage by region, variogram diagnostics, sensitivity to coordinate system and range parameter, and maps of both posterior mean and standard deviation. State whether the map interpolates within the sampled domain or extrapolates beyond its support. Never present narrow posterior bands as total uncertainty when sensor bias, preferential site placement, or mean misspecification are omitted.

## Case study: a scientific inverse problem {#case-study-scientific}

Consider inferring a forcing field from noisy measurements of a physical response. Encode the forward operator and boundary conditions before choosing an optimizer. The workflows in [[ch:inverse-learning-and-spectral-regularization]] and [[ch:scientific-computing-and-operator-learning]] separate observation fit from the regularity and physics assumptions that identify the inverse.

Create synthetic recovery tests where the truth is known, including a truth outside the assumed RKHS. Plot recovered spectra, forward residuals, parameter error, and uncertainty calibration. Compare Tikhonov, early-stopped Krylov iteration, and a mesh-based baseline under equal compute. A low observation residual can coexist with a wrong forcing field; the report must expose null spaces and resolution limits.

## Case study: dynamics and off-policy evaluation {#case-study-dynamics}

For a controlled dynamical system, split by complete trajectories and time, never by individual transitions. A one-step kernel model can achieve excellent random-split error while compounding bias over a rollout. Evaluate one-step prediction, multi-step rollout, invariant or constraint violations, and performance under policies different from the behavior policy. Use [[ch:dynamical-systems-control-and-reinforcement-learning]] to distinguish state-transition regression, Koopman approximation, value-function learning, and kernelized policy evaluation.

Coverage is the central diagnostic. Plot importance weights or a kernel discrepancy between behavior-policy and target-policy state-action distributions. If support overlap fails, report the target value as unidentified rather than stabilizing a meaningless estimate with stronger regularization.

## Case study: distribution shift and calibrated decisions {#case-study-shift}

Train a predictor on a source population and deploy it on a changing target population. First distinguish covariate, label, and concept shift; each permits different corrections. Use a classifier-based density-ratio diagnostic and witness functions to localize change, then compare unweighted training, clipped importance weighting, and a representation chosen without target labels. The framework in [[ch:distribution-shift-robustness-and-conformal-prediction]] supplies the assumptions and conformal layer.

Report source and target discrimination, effective sample size of weights, subgroup performance, and conformal coverage with interval width. Marginal coverage can hide subgroup failure, and online recalibration spends labels. State the label-acquisition delay and the exact exchangeability or weighted-exchangeability argument behind each interval.

## Case study: censored and time-to-event outcomes {#case-study-survival}

Time-to-event data add right censoring and competing risks. A kernel can enter through a nonlinear risk score, an accelerated failure-time regression, or a kernel on longitudinal histories. Splits must respect patient and calendar time. Censoring weights require a model for the censoring mechanism and can become unstable in the tail.

Report time-dependent discrimination, integrated Brier score, calibration at clinically meaningful horizons, and the number still at risk. Compare against regularized Cox and non-kernel survival baselines. Do not tune on a concordance measure and claim calibrated survival probabilities; ranking and calibration are separate targets.

## Model cards and reproducibility packets {#model-cards-and-reproducibility}

Every applied result should ship with a compact reproducibility packet:

- task definition, intended use, excluded use, and decision costs;
- dataset version, license, cohort construction, missingness, and split unit;
- preprocessing fit boundaries and leakage audit;
- kernel formula, all learned hyperparameters, regularization, solver, tolerance, and random seeds;
- compute environment, wall time, memory, and a competitive non-kernel baseline;
- uncertainty or calibration procedure and the assumptions it needs;
- subgroup, shift, and failure-mode analysis;
- links to the exact notebook, artifacts, and book sections.

This packet is the applied analogue of a theorem's assumption block. It makes clear what was established, on which population, and what would have to remain true for the result to transfer.

## A map of the software {#software}

None of this needs to be written from scratch. The table is a dated reproducibility record, not a timeless endorsement. Project links and status were checked on **2026-07-19**, matching this edition's software cutoff; re-check the linked release notes before reproducing an environment.

  Library        Verified role and maintenance note                                                                                                                                                    Project
  -------------- ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ ------------------------------------------------------------
  LIBSVM         Exact dual SVM and SVR via SMO; stable reference implementation for small-to-medium dense kernel problems                                                                                 [project](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
  LIBLINEAR      Linear SVM and logistic regression after an explicit feature map; maintained separately from kernel solvers                                                                               [project](https://www.csie.ntu.edu.tw/~cjlin/liblinear/)
  scikit-learn   Version 1.9.0 documents SVC/SVR, KernelRidge, KernelPCA, Nystroem, RBFSampler, pipelines, and model selection                                                                             [stable API](https://scikit-learn.org/stable/api/index.html)
  GPy            Classical Python GP library with sparse models; maintenance and compatibility should be checked before starting a new project                                                            [repository](https://github.com/SheffieldML/GPy)
  GPflow         Actively maintained TensorFlow GP library with variational and inducing-point models                                                                                                     [project](https://www.gpflow.org/)
  GPyTorch       Version 1.15.2 documents exact, variational, multitask, deep-kernel, and matrix-multiply-based GPU inference                                                                               [stable docs](https://docs.gpytorch.ai/en/stable/)
  KeOps          Symbolic and lazy kernel matrix-vector products with accelerator support, useful when a dense Gram matrix cannot be stored                                                               [docs](https://www.kernel-operations.io/keops/)
  Falkon         Preconditioned Nystrom kernel ridge regression with accelerator support                                                                                                                   [project](https://www.falkon.org/)
  SHOGUN         Broad historical kernel collection; the latest GitHub release shown at the cutoff is 6.1.4 from 2019, so treat it as a legacy option and verify toolchain compatibility                    [repository](https://github.com/shogun-toolbox/shogun)

For the companion environment, create a clean virtual environment and run `python -m pip install -r requirements-notebooks.txt`; the exact pins are versioned with the book. For a minimal classical workflow, `python -m pip install scikit-learn==1.9.0` matches the checked API. Accelerator libraries have platform-specific wheels and must follow their linked installation pages. Record Python, package, BLAS, accelerator, and driver versions with every benchmark.

A division of labor follows the table. Prototype classical estimators and approximations in scikit-learn; move to a linear solver once features are explicit, to KeOps or Falkon when a Gram matrix cannot be stored, and to GPflow or GPyTorch when predictive uncertainty is part of the task. For string and graph kernels, evaluate whether a maintained domain library or a small tested implementation is safer than adopting a legacy umbrella package.

## Summary {#summary}

A kernel method is a pipeline with a rigid shape and a few load-bearing choices. The kernel encodes what \"similar\" means, and the book has supplied one for nearly every data type: Gaussian and polynomial kernels for vectors, string and mismatch kernels for [[ch:string-kernels|sequences]], graph kernels for [[ch:graph-kernels|molecules and networks]], Fisher and marginalized kernels when a [[ch:generative-and-marginalization-kernels|generative model]] is at hand. The three applications showed one skeleton in three domains: choose the kernel, normalize it, hand the Gram matrix to a support vector machine, and the state of the art followed. The recipe made the tuning honest, through nested cross-validation, the normalizations that make similarity mean what it should, and the spectrum diagnostics that read under and overfitting off the eigenvalues of \(K\) as an effective dimension. When the exact solve runs out of room, the Nystrom method, random features, and preconditioned solvers of [[ch:large-scale-kernels|the large-scale chapter]] carry the same kernels to millions of points. The book's through-line returns one last time: fix a positive definite kernel and a convex loss, and a great deal of learning is a linear-algebra problem in disguise. What remains is judgment, and judgment is what practice teaches.

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

The most expensive practical errors happen before the solver starts: choosing a similarity that erases the relevant structure, normalizing outside the training split, or selecting hyperparameters on the reported test set. Fit every data-dependent transform inside the resampling loop, compare against a simple linear or tree baseline, and inspect group, time, or site splits whenever random folds would leak related examples.

At deployment, a good validation score is incomplete evidence. Record the spectrum and condition diagnostics, uncertainty calibration, influential training points, shift monitors, compute budget, and a retraining trigger. Approximation choice is part of the model card because Nyström landmarks, random-feature seeds, and dictionary eviction rules decide which geometry survives.

## Summary and further reading {#summary-and-further-reading}

The reusable lesson of the applications is a workflow, not a favorite kernel. Encode the object's structure, normalize the induced geometry, diagnose spectrum and conditioning, choose a compute route, select the full pipeline by nested validation, and audit the deployed decision. Sequence applications such as [@leslie2002] and [@leslie2004] and the Fisher-kernel route of [@jaakkola2000] differ in representation but share that evidence chain. A kernel method becomes a strong practical result only when its similarity, optimization, approximation, validation, uncertainty, and failure conditions are reported as one system.

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} Using the decision guide, name a sensible first kernel and the chapter that builds it for each: (a) a house price from ten standardized numeric features; (b) proteins classified into structural families from their sequences; (c) the toxicity of small molecules given as labeled graphs; (d) news articles categorized by topic; (e) two probability distributions given as samples. In one sentence, say why \"there is no universally best kernel\" makes this a modeling choice rather than a lookup.
2.  [computation]{.ex-tag} Take the \(3\times 3\) Gram matrix \(K=\begin{pmatrix}4 & 2 & 0\\ 2 & 9 & 3\\ 0 & 3 & 2\end{pmatrix}\). (a) Cosine-normalize it, \(\tilde K_{ij}=K_{ij}/\sqrt{K_{ii}K_{jj}}\), and verify the diagonal is all ones. (b) Double-center the original \(K\) with \(H=I-\tfrac13\mathbf 1\mathbf 1^\top\), forming \(HKH\), and verify every row sum is \(0\). State in one line what each operation does to the feature-space picture.
3.  [proof]{.ex-tag} Show that cosine normalization preserves positive definiteness: if \(k\) is a positive definite kernel with \(k(x,x)\gt 0\) for all \(x\), then \(\tilde k(x,x')=k(x,x')/\sqrt{k(x,x)k(x',x')}\) is positive definite. Then identify the feature map of \(\tilde k\) in terms of the feature map \(\Phi\) of \(k\).
    Hint

    ::: hint-body
    This is the conformal-transformation closure property with \(f(x)=1/\sqrt{k(x,x)}\): \(\tilde k(x,x')=f(x)\,k(x,x')\,f(x')\). The feature map is \(\tilde\Phi(x)=\Phi(x)/\lVert\Phi(x)\rVert\), the projection of \(\Phi(x)\) onto the unit sphere.
    :::
4.  [exploration]{.ex-tag} Rerun the model-selection example with a third bandwidth \(\sigma=1.0\) in the grid. Before computing, predict where its \(d_{\text{eff}}(0.1)\) falls relative to \(2.43\) (at \(\sigma=2.0\)) and \(5.11\) (at \(\sigma=0.5\)), and whether its CV error beats the winner \(0.1736\); then check with a short numpy script and explain any surprise using the spectrum.
    Hint

    ::: hint-body
    The effective dimension is monotonic in the bandwidth: a smaller \(\sigma\) spreads the spectrum and raises \(d_{\text{eff}}\), so \(\sigma=1.0\) should land between \(2.43\) and \(5.11\). Cross-validated error need not be monotonic, though, since it trades bias against variance; the point of the check is to see that the best bandwidth is an interior compromise, not an endpoint.
    :::
5.  [challenge]{.ex-tag} Explain precisely why selecting the bandwidth and reporting the error on the same folds gives an optimistically biased estimate, and how nested cross-validation removes the bias. Then count the model fits: for outer \(K_1\) folds, inner \(K_2\) folds, and a grid of \(G\) settings, how many times is the base learner trained, and how does that compare with the biased single-loop count?
    Hint

    ::: hint-body
    The grid is chosen to minimize the very quantity you then report, so the minimum of \(G\) noisy estimates is biased below the truth, and the bias grows with \(G\). Nested CV grades the selected model on outer folds it never saw during selection. The fit count is \(K_1\cdot G\cdot K_2\) for selection plus \(K_1\) refits, versus \(G\cdot K_2\) for the single loop; the extra factor \(K_1\) is the price of an honest estimate.
    :::
6.  [exploration]{.ex-tag} Study how \(d_{\text{eff}}(\lambda)=\sum_i\lambda_i/(\lambda_i+\lambda)\) depends on the ridge \(\lambda\) for a fixed Gram matrix, and connect it to Nystrom. (a) Show \(d_{\text{eff}}(\lambda)\to\operatorname{rank}(K)\) as \(\lambda\to 0^+\) and \(\to 0\) as \(\lambda\to\infty\), and argue it decreases in \(\lambda\). (b) Explain why a Nystrom approximation with \(m\approx d_{\text{eff}}(\lambda)\) landmarks loses little accuracy at ridge level \(\lambda\).
    Hint

    ::: hint-body
    Each term \(\lambda_i/(\lambda_i+\lambda)\) is decreasing in \(\lambda\), tends to \(\mathbf 1[\lambda_i\gt 0]\) as \(\lambda\to 0\), and to \(0\) as \(\lambda\to\infty\). Nystrom approximates the top of the spectrum; the directions it drops have \(\lambda_i\ll\lambda\), so they already contribute almost nothing to \(d_{\text{eff}}\) and almost nothing to the ridge-regularized fit. This is the argument of Rudi et al. (2015).
    :::
:::
