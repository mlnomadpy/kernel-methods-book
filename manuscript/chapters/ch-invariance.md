---
id: ch-invariance
slug: invariances-and-pre-images
title: Invariances and the Pre-Image Problem
part: VII · Designing Kernels for Data
order: 20
tier: advanced
prerequisites:
  - kernel-families
objectives:
  - >-
    Explain the central definitions and claims in Invariances and the Pre-Image
    Problem.
  - Apply the chapter's principal methods and interpret their outputs.
  - >-
    State the assumptions behind formal results and connect them to earlier
    chapters.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-invariance.yml
verification_date: null
bibliography:
  - lecun1995
  - scholkopf1996prior
  - simard1998
  - decoste2002
  - mika1999
  - burges1996
  - romdhani2001
  - scholkopf2002
  - shawe2004
---
# Invariances and the Pre-Image Problem

<p class="lead">The original support vector machine would classify a handwritten digit exactly the same way if you first scrambled every pixel by a fixed permutation. It knew nothing of images, and on a task that is all about spatial layout that blindness is at once striking and damning. Nothing in the kernel machinery built so far tells the machine that a digit survives a one-pixel shift or a small rotation, and nothing lets it hand back a cleaned-up image once its answer lives in feature space as an expansion \(\sum_i \alpha_i \Phi(x_i)\). This chapter takes up both gaps. The first half teaches a kernel machine the invariances we already know, through virtual examples, tangent-vector penalties, and kernels that fold the transformation into their evaluation, the ingredient that turned support vector machines into a benchmark-winning tool. The second solves the pre-image problem of recovering a point in input space from a feature-space vector, which has no exact solution in general and a workable approximate one that powers kernel PCA denoising and reduced-set compression. The two themes meet at one place: both are about the map \(\Phi\) between input and feature space, one pushing prior structure forward through it, the other pulling solutions back.</p>

## Prior knowledge and invariance {#prior-knowledge}

In 1995 LeCun et al. (1995) compared learning algorithms on handwritten digits and remarked that the optimal margin classifier, the early support vector machine, was remarkable precisely because it used no prior knowledge at all: it would perform identically if the image pixels were scrambled by a fixed permutation. That is a striking property, and also a damning one. A method blind to the spatial layout of an image is throwing away most of what makes the problem solvable. Within a few years the same group had closed the gap, and the tool that closed it was the systematic incorporation of prior knowledge, following Schölkopf, Burges, and Vapnik (1996).

By prior knowledge we mean everything about the task that is available beyond the training examples themselves. Some of it is generic. Every kernel already encodes a smoothness assumption: using a kernel \(k\) as a regularizer penalizes \(\|\Upsilon f\|^2\) for an operator \(\Upsilon\) whose Green function is \(k\), so that patterns close in input space are pushed toward the same label. Some prior knowledge is more specific, such as knowing that in digit images the correlations between nearby pixels are more reliable than those between distant ones, which one builds in through kernels that emphasize local products of pixels. The sharpest and most useful form, and the subject of this chapter, is knowledge of a transformation that must not change the answer.

:::: {.definition #def-20-1}
[Definition (invariance under a transformation)]{.box-title}

Let \(\{L_t\}_{t}\) be a family of transformations of the input space \(\mathcal{X}\), indexed by a parameter \(t\) with \(L_0=\mathrm{id}\). A decision function \(f:\mathcal{X}\to\mathbb{R}\) is *invariant* under \(\{L_t\}\) if

$$f(L_t x)=f(x)\qquad\text{for all }x\in\mathcal{X}\text{ and all admissible }t.$$

Common examples are the group of translations, rotations, or line-thickness changes of a digit image. When \(\{L_t\}\) is a differentiable one-parameter group, the *tangent vector* at \(x\) is \(\left.\tfrac{d}{dt}\right|_{t=0} L_t x\), the local direction in which the transformation moves the pattern.
::::

Schölkopf, Burges, and Vapnik (1996) distinguish three ways to exploit such knowledge, and the rest of the first half of the chapter is an unpacking of these three. First, one can generate artificial training examples, virtual examples, by applying the transformations to the data and hoping the machine learns the invariance from the enlarged set. Second, one can change the learning algorithm itself, modifying the objective so the estimated function is forced to have small derivative along the tangent directions. Third, one can change the representation, mapping the data into a space where the invariance becomes trivial, for instance replacing each point by a feature that the transformation leaves fixed. The third is the most powerful when the required map can be found, but it is often unavailable or too global for a locally defined invariance, so in practice the first two dominate. A fourth, hybrid idea folds the transformation directly into the kernel.

## The Virtual Support Vector method {#virtual-sv}

Generating virtual examples from the entire training set is simple but expensive: multiplying a large database by the number of desired transforms produces an unwieldy training set, and much of the added data is redundant because it sits far from the decision boundary where it cannot affect the solution. The Virtual Support Vector (VSV) method of Schölkopf, Burges, and Vapnik (1996) exploits a structural fact about support vector machines to avoid that waste: the support vectors alone carry all the information the machine used, so a classifier retrained on just the support vector set matches one trained on the full data. If the support vectors contain the whole solution, then the transformations worth applying are the ones applied to them.

The logic is geometric. Small transformations move a support vector a short distance, and support vectors sit on or near the margin, so their transformed copies land near the decision surface too, exactly the region where new examples sharpen the boundary. Transformed copies of interior points, by contrast, stay interior and teach the machine nothing. This is why generating virtual examples from the full database, in the experiments of Schölkopf, Burges, and Vapnik (1996), gave no accuracy gain over generating them from the support vectors alone.

:::: {.algorithm #algo-20-1}
[Algorithm (Virtual Support Vector method)]{.box-title}

::: algo-io
[Input]{.algo-lab} Training set \(\{(x_i,y_i)\}_{i=1}^m\), kernel \(k\), penalty \(C\), a family of invariance transforms \(\{L_t\}\) with chosen parameter values \(t_1,\dots,t_r\).

[Output]{.algo-lab} An invariant SVM decision function \(f\).
:::

1.  Train a standard SVM on \(\{(x_i,y_i)\}\); extract the support vector set \(S=\{x_i:\alpha_i\gt 0\}\).
2.  For each support vector \(x\in S\) and each transform \(L_{t_1},\dots,L_{t_r}\), form the virtual support vector \(L_{t_j}x\) with the same label as \(x\).
3.  Collect the virtual support vectors together with the original support vectors into an enlarged set \(S'\).
4.  Train a second SVM on \(S'\); return its decision function.
5.  Optionally iterate steps 1 to 4, with care: iterating a local invariance can accumulate into an unwanted global one, for instance rotating a 6 into a 9.
::::

The one-pixel translation instance is the workhorse. On the USPS database, a degree-3 polynomial SVM had a test error of 4.0%; generating virtual support vectors by shifting each support vector one pixel in the four principal directions and retraining cut the error to 3.2%, a substantial gain from a nearly free preprocessing step. The table records how the support vector set changes.

  Classifier trained on           Training size   Avg. no. of SVs   Test error
  ------------------------------- --------------- ----------------- ------------
  Full training set               7291            274               4.0%
  Overall SV set                  1677            268               4.1%
  Virtual SV set                  8385            686               3.2%
  Virtual patterns from full DB   36455           719               3.4%

Two readings stand out. Training on the overall support vector set (row two) barely changes the error, confirming that the support vectors carry the solution. And generating virtual patterns from the whole database (row four) is far more expensive yet no better than the VSV row, confirming that transforms of non-support-vectors add little. On the larger MNIST database, where the method has more room to work, a degree-9 polynomial with twelve translated virtual examples per support vector reached 0.56% test error, the record for that benchmark at the time (DeCoste and Schölkopf 2002). The price is a larger final support vector set, hence slower evaluation, which is exactly the motivation for the reduced-set methods of the second half of this chapter. Because it manipulates only the data, the VSV method also handles discrete symmetries that derivative-based schemes cannot, such as the mirror reflection of a bilaterally symmetric object.

## Tangent vectors, invariance kernels, and jittering {#invariance-kernels}

The second route modifies the algorithm rather than the data. Instead of asking the machine to infer the invariance from examples, we penalize any variation of the decision function along the tangent directions. Locally, invariance of \(g\) at a pattern \(x_j\) under the group \(\{L_t\}\) is the statement \(\left.\tfrac{d}{dt}\right|_{t=0} g(L_t x_j)=0\), and summing the squared derivative over the data gives a regularizer that pulls the solution toward invariant functions.

Carrying this through for a linear decision function \(g(x)=\sum_i \alpha_i y_i \langle x, x_i\rangle + b\), following Schölkopf and Smola (2002), the tangent regularizer reduces to a modified quadratic form governed by the *tangent covariance matrix*

$$T=\frac{1}{m}\sum_{j=1}^m \Big(\left.\tfrac{d}{dt}\right|_{0} L_t x_j\Big)\Big(\left.\tfrac{d}{dt}\right|_{0} L_t x_j\Big)^\top,$$

the empirical covariance of the tangent vectors, which have zero mean. The whole invariant training problem then collapses back to a standard SVM once we replace the ordinary dot product by \(\langle x, x'\rangle_A = x^\top A\,x'\) with \(A=T_\gamma^{-1}\) and \(T_\gamma=(1-\gamma)T+\gamma I\). The trade-off parameter \(\gamma\in(0,1]\) interpolates between the plain SVM at \(\gamma=1\) and full invariance as \(\gamma\to 0\); the regularized \(T_\gamma\) is strictly positive definite, hence invertible.

The preprocessing has a clean interpretation. Diagonalize \(T_\gamma=UDU^\top\); then \(A^{1/2}=UD^{-1/2}U^\top\) projects a pattern onto the eigenvectors of the tangent covariance and rescales each by the inverse square root of its eigenvalue. Directions in which the data vary a lot under the transformation, the large-eigenvalue directions of \(T\), are scaled down, so the classifier leans on the features that the transformation leaves nearly fixed. This is a whitening of the tangent covariance, and it connects invariance to principal component analysis: for translations, it downweights the absolute positions of strokes and emphasizes the relative amounts of ink. Computing \(T\) from the support vectors alone turns it into a task-dependent covariance that concentrates invariance where it matters, near the boundary.

### Tangent distance and translation-invariant kernels {#tangent-distance}

The tangent-vector idea has an older cousin in the tangent distance of Simard et al. (1998). The set of all transforms \(\{L_t x\}\) of a single pattern traces out a low-dimensional manifold in input space, the orbit of \(x\) under the invariance. Two patterns should be judged similar if their orbits come close, not if the raw points do, since a one-pixel shift can move a digit far in Euclidean distance while leaving its identity untouched. Computing the true distance between two curved manifolds is hard, so tangent distance replaces each orbit by its tangent plane at the pattern, spanned by the tangent vectors, and measures the distance between those planes. To first order in \(t\) this is exactly invariant under the transformation.

This is where the two halves of the book's kernel story touch. A translation-invariant kernel \(k(x,x')=\varphi(x-x')\) is invariant by construction: it reads only the difference of its arguments, so translating both patterns together leaves it unchanged. Tangent distance builds the same idea locally and for a general transformation group, by making the metric, and hence any kernel built from it, insensitive to first-order motion along the orbit. A genuinely translation-invariant kernel is the global, exact version of what tangent distance achieves approximately and for one specific pair of patterns; both encode the invariance into the geometry itself rather than into extra training data. See [[ch:kernel-families]] for the translation-invariant families and [[ch:kernels-and-deep-learning]] for invariance as the prior that convolutional architectures hard-wire.

### Jittering kernels {#jittering}

The hybrid method folds the transformation into the kernel evaluation itself, following DeCoste and Schölkopf (2002). Given any admissible kernel with Gram entries \(K_{ij}=k(x_i,x_j)\), the jittering kernel considers all jittered forms of one argument, including the untransformed one, and keeps the closest match in feature space.

:::: {.definition #def-20-2}
[Definition (jittering kernel)]{.box-title}

Let \(\{x_i^{(1)},\dots,x_i^{(J)}\}\) be the jittered forms of \(x_i\) (its transforms plus itself). The jittering kernel is

$$k_J(x_i,x_j)=K_{qj},\qquad q=\arg\min_{1\le p\le J}\big(K^{(p)}_{ii}-2K^{(p)}_{ij}+K_{jj}\big),$$

where the minimized quantity is the squared feature-space distance \(\|\Phi(x_i^{(p)})-\Phi(x_j)\|^2\) between the jittered pattern and \(x_j\). For a normalized RBF kernel, where \(k(x,x)=1\), this reduces to selecting the jitter that maximizes \(K^{(p)}_{ij}\).
::::

Jittering trades space for time relative to VSV. The VSV method inflates the training set by the number of jitters \(J\) and pays the quadratic training cost of that larger set; the jittering kernel keeps the training set at its original size but makes each kernel evaluation \(J\) times more expensive, so training can scale only linearly in \(J\), and kernel caching amortizes much of even that. The cost is that a jittering kernel need not be positive definite, and the induced distance can violate the triangle inequality: for the three single-pixel images \(A=(1,0,0)\), \(B=(0,1,0)\), \(C=(0,0,1)\) under one-pixel translation, the jittered distances \(d(A,B)\) and \(d(B,C)\) are \(0\) while \(d(A,C)\) is positive, so \(d(A,C)\gt d(A,B)+d(B,C)\). In practice such violations are rare and SMO tolerates them, and a jittering kernel must keep jittering at test time, unlike VSV which compiles the invariance into its final support vectors.

## The pre-image problem {#pre-image-problem}

We now turn the map around. A kernel algorithm produces a vector in feature space as an expansion in mapped inputs,

$$\Psi=\sum_{i=1}^{m}\alpha_i\,\Phi(x_i),$$

whether it is an SVM weight vector or a kernel PCA feature extractor (see [[ch:kernel-pca]]). For most purposes this is enough: taking a dot product with a test point \(\Phi(x)\) turns into the kernel expansion \(\sum_i \alpha_i k(x_i,x)\), computable even when \(\Phi\) lands in an infinite-dimensional space. But sometimes we need the point in input space that \(\Psi\) represents: a denoised image reconstructed by kernel PCA, or a compact synthetic pattern that stands in for a whole expansion. A point \(z\) with \(\Phi(z)=\Psi\) is a *pre-image* of \(\Psi\).

When a pre-image exists and the kernel is an invertible function of the dot product, it is easy to recover, following Schölkopf and Smola (2002).

:::: {.proposition #prop-20-3}
[Exact pre-images (Schölkopf and Smola, 2002)]{.box-title}

Let \(\Psi=\sum_{j=1}^m \alpha_j \Phi(x_j)\) with \(x_j\in\mathcal{X}\subseteq\mathbb{R}^N\), and suppose there is an invertible \(f_k\) with \(k(x,x')=f_k(\langle x,x'\rangle)\). If a pre-image \(z\in\mathbb{R}^N\) with \(\Phi(z)=\Psi\) exists, then for any orthonormal basis \(e_1,\dots,e_N\) of input space,

$$z=\sum_{i=1}^N f_k^{-1}\!\Big(\sum_{j=1}^m \alpha_j\, k(x_j,e_i)\Big)\,e_i.$$

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

Expand \(z\) in the orthonormal basis and read each coordinate as a dot product, which \(f_k^{-1}\) recovers from the kernel:

$$z=\sum_{i=1}^N \langle z,e_i\rangle\,e_i=\sum_{i=1}^N f_k^{-1}\big(k(z,e_i)\big)\,e_i=\sum_{i=1}^N f_k^{-1}\Big(\sum_{j=1}^m \alpha_j\,k(x_j,e_i)\Big)\,e_i,$$

where the last step uses \(k(z,e_i)=\langle\Phi(z),\Phi(e_i)\rangle=\langle\Psi,\Phi(e_i)\rangle=\sum_j \alpha_j k(x_j,e_i)\), valid because \(\Phi(z)=\Psi\). Invertible \(f_k\) covers polynomial kernels \((\langle x,x'\rangle+c)^d\) with \(c\ge 0\) and \(d\) odd, and sigmoid kernels; a variant using polarization handles RBF kernels. [\(\square\)]{.qed}
::::

The catch is the hypothesis that a pre-image exists, and for the most-used kernel it usually does not. Under a Gaussian kernel \(\Phi\) maps each input to a Gaussian bump centered on it, and no Gaussian is a linear combination of Gaussians centered elsewhere; so no expansion with more than one term has an exact pre-image. The problem as posed is unsolvable in general, which forces us to relax it.

We call \(z\) an *approximate pre-image* of \(\Psi\) when

$$\rho(z)=\|\Psi-\Phi(z)\|^2$$

is small. Good approximate pre-images do exist for the vectors we care about. Kernel PCA projects \(\Phi(x)\) onto its top \(n\) eigenvectors, \(P_n\Phi(x)=\sum_{j=1}^n \langle\Phi(x),v_j\rangle v_j\), the best rank-\(n\) approximation of the mapped data in the mean-square sense of Mika et al. (1999). If \(x\) is drawn from the same distribution as the training data, \(P_n\Phi(x)\) stays close to the mapped-data manifold and has a good approximate pre-image; \(x\) itself is already one. This is the engine of kernel PCA *denoising*: map a noisy pattern \(x\) into feature space, discard the low-variance components that mostly capture noise, and take the pre-image of the projection as a cleaned-up \(z\). On the USPS digits, kernel PCA denoising beats linear PCA once enough components are used, because the nonlinear map can devote more features to structure rather than noise (Mika et al. 1999).

## Finding approximate pre-images {#finding-preimages}

To minimize \(\rho(z)\) we could differentiate \(\|\Psi-\Phi(z)\|^2\) directly, but a lower-dimensional and better-scaled problem is to maximize the length of the projection of \(\Psi\) onto the ray through \(\Phi(z)\),

$$\frac{\langle\Psi,\Phi(z)\rangle^2}{\langle\Phi(z),\Phi(z)\rangle},$$

after which the optimal scaling is set separately. For normalized RBF kernels, where \(\langle\Phi(z),\Phi(z)\rangle=k(z,z)=1\), this is simply the maximization of \(\langle\Psi,\Phi(z)\rangle^2\), and its stationarity condition yields a fixed-point iteration.

Write \(\Psi=\sum_i \alpha_i \Phi(x_i)\), so \(\langle\Psi,\Phi(z)\rangle=\sum_i \alpha_i k(x_i,z)\). At an extremum the gradient in \(z\) vanishes,

$$0=\nabla_z\,\langle\Psi,\Phi(z)\rangle=\sum_{i=1}^m \alpha_i\,\nabla_z\,k(x_i,z).$$

For a Gaussian kernel \(k(x_i,z)=\exp\!\big(-\|x_i-z\|^2/2\sigma^2\big)\) we have \(\nabla_z k(x_i,z)=k(x_i,z)\,(x_i-z)/\sigma^2\), so

$$0=\sum_{i=1}^m \alpha_i\,\exp\!\Big(-\tfrac{\|x_i-z\|^2}{2\sigma^2}\Big)(x_i-z).$$

Solving the linear-in-\(z\) relation, \(z\sum_i \alpha_i e_i(z)=\sum_i \alpha_i e_i(z)\,x_i\) with \(e_i(z)=\exp(-\|x_i-z\|^2/2\sigma^2)\), gives \(z\) as a weighted average of the \(x_i\), which we read as an iteration.

:::: {.algorithm #algo-20-2}
[Algorithm (pre-image fixed-point iteration, Gaussian kernel)]{.box-title}

::: algo-io
[Input]{.algo-lab} Expansion \(\Psi=\sum_{i=1}^m \alpha_i\Phi(x_i)\), Gaussian width \(\sigma\), starting point \(z_0\), tolerance \(\tau\).

[Output]{.algo-lab} Approximate pre-image \(z\) with \(\Phi(z)\approx\Psi\).
:::

1.  Set \(n\leftarrow 0\); choose \(z_0\), for instance the weighted mean \(\sum_i\alpha_i x_i/\sum_i\alpha_i\).
2.  Compute the weights \(w_i=\alpha_i\exp\!\big(-\|x_i-z_n\|^2/2\sigma^2\big)\).
3.  Update \(\displaystyle z_{n+1}=\frac{\sum_{i=1}^m w_i\,x_i}{\sum_{i=1}^m w_i}\).
4.  Repeat steps 2 and 3 until \(\|z_{n+1}-z_n\|\lt\tau\); return \(z_{n+1}\).
::::

The denominator equals \(\langle\Psi,\Phi(z_n)\rangle\) up to normalization, so it is nonzero near a nontrivial extremum; instability only arises when the projection of \(\Psi\) onto the span of the mapped inputs is near zero, in which case there is nothing to approximate and one restarts from a different \(z_0\). The update has a vivid reading as clustering: with all \(\alpha_i\gt 0\) it finds the center of a single Gaussian cluster that captures as much weight as possible, and when the \(\alpha_i\) carry signs, as in an SVM where the sign is the label, it seeks a point where one class outweighs the other, estimating the difference of two densities rather than one.

:::: {.example #example-20-1}
[Example (pre-image of a convex combination)]{.box-title}

::: wex
Three points in \(\mathbb{R}^2\): \(x_1=(0,0)\), \(x_2=(2,0)\), \(x_3=(1,1.5)\). Target \(\Psi=\sum_i\beta_i\Phi(x_i)\) with \(\beta=(0.5,0.3,0.2)\), Gaussian width \(\sigma=1\). We seek the approximate pre-image \(z\) minimizing \(\|\Psi-\Phi(z)\|^2\), equivalently maximizing \(J(z)=\langle\Psi,\Phi(z)\rangle=\sum_i\beta_i k(x_i,z)\).

1.  [Start at the weighted mean.]{.wex-op} \(z_0=\beta_1 x_1+\beta_2 x_2+\beta_3 x_3=(0.8,\,0.3)\), with \(J(z_0)=0.5821\).
2.  [Reweight and average.]{.wex-op} The weights \(w_i=\beta_i e^{-\|x_i-z_0\|^2/2}\) pull the estimate toward the heavier, nearer points, giving \(z_1=(0.6435,\,0.2459)\), \(J=0.5959\).
3.  [Iterate.]{.wex-op} \(z_2=(0.5328,\,0.2152)\), \(z_3=(0.4621,\,0.1956)\), \(z_4=(0.4204,\,0.1833)\), \(z_5=(0.3970,\,0.1759)\), with \(J\) climbing \(0.6024,\,0.6050,\,0.6059,\,0.6062\).
4.  [Converge.]{.wex-op} By \(z_6=(0.3842,\,0.1717)\) the objective has settled at \(J=0.6062\); the iterates have stopped moving to four places.

**Reading.** The pre-image \(z^\star\approx(0.384,\,0.172)\) lands between the three points but nearest \(x_1\), the one with the largest weight, exactly where a single Gaussian bump best matches the weighted mixture. The projection \(J\) increases at every step, confirming the iteration is climbing toward the extremum.
:::

**Verification artifact.** checks/example-ch-invariance-example-20-1.json records the example source hash and verification scope.
::::

## Reduced set methods {#reduced-set}

The pre-image problem is the extreme case, one term, of a more general and more useful one. An SVM's decision function \(\sum_i \alpha_i k(x_i,\cdot)\) costs one kernel evaluation per support vector, and after the VSV method the support vector set is large, so evaluation is slow, a real obstacle in tasks like face detection where a classifier is scanned over every image location. We would like to approximate the exact expansion \(\Psi=\sum_{i=1}^{N_x}\alpha_i\Phi(x_i)\) by a shorter *reduced set* expansion

$$\Psi'=\sum_{i=1}^{N_z}\beta_i\,\Phi(z_i),\qquad N_z\ll N_x,$$

minimizing \(\|\Psi-\Psi'\|^2\), which is fully expressible in kernels. The problem splits into finding the reduced-set vectors \(z_i\) and finding the coefficients \(\beta_i\); the coefficients have a closed form.

:::: {.proposition #prop-20-4}
[Optimal expansion coefficients (Schölkopf and Smola, 2002)]{.box-title}

Let \(\Psi=\sum_{i=1}^{N_x}\alpha_i\Phi(x_i)\), and fix reduced-set vectors \(z_1,\dots,z_{N_z}\) whose images \(\Phi(z_i)\) are linearly independent. The coefficients \(\beta=(\beta_1,\dots,\beta_{N_z})\) minimizing \(\|\Psi-\sum_i\beta_i\Phi(z_i)\|^2\) are

$$\beta=(K^z)^{-1}K^{zx}\alpha,$$

where \(K^z_{ij}=\langle\Phi(z_i),\Phi(z_j)\rangle\) and \(K^{zx}_{ij}=\langle\Phi(z_i),\Phi(x_j)\rangle\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

:::: {.proof}
[Proof]{.box-title}

Setting the derivative of \(\|\Psi-\sum_i\beta_i\Phi(z_i)\|^2\) with respect to \(\beta_j\) to zero gives, for every \(j\),

$$0=\Big\langle\Phi(z_j),\ \Psi-\sum_{i}\beta_i\Phi(z_i)\Big\rangle=\big(K^{zx}\alpha\big)_j-\big(K^z\beta\big)_j.$$

In matrix form \(K^z\beta=K^{zx}\alpha\), and since the \(\Phi(z_i)\) are linearly independent \(K^z\) has full rank, so \(\beta=(K^z)^{-1}K^{zx}\alpha\). No reduced-set method using the feature-space norm can beat this choice of coefficients for given \(z_i\). [\(\square\)]{.qed}
::::

What differs between methods is how the \(z_i\) are chosen. *Selection* methods pick a subset of the original \(x_i\); this is worthwhile because an SVM expansion is not as sparse as it could be, its coefficients being pinned into \([-C,C]\) by the quadratic program. One can select via the near-null space of the Gram matrix, a computation closely tied to kernel PCA, or via an \(\ell_1\) penalty that shrinks coefficients to zero (Schölkopf and Smola 2002). On USPS, removing about 40% of the support vectors this way leaves the test error practically unchanged. *Construction* methods do better by allowing synthetic \(z_i\) that need not be training points, and they get them by iterating the pre-image algorithm.

:::: {.algorithm #algo-20-3}
[Algorithm (reduced set construction, Burges 1996)]{.box-title}

::: algo-io
[Input]{.algo-lab} Expansion \(\Psi=\Psi_1=\sum_{i=1}^{N_x}\alpha_i\Phi(x_i)\), Gaussian kernel, target size \(N_z\) or residual threshold \(\epsilon\).

[Output]{.algo-lab} Reduced set \(\{(z_i,\beta_i)\}_{i=1}^{N_z}\) with \(\sum_i\beta_i\Phi(z_i)\approx\Psi\).
:::

1.  Set \(m\leftarrow 1\) and let the current residual be \(\Psi_1\).
2.  Find a one-term pre-image \(z_m\) of the residual \(\Psi_m=\Psi-\sum_{i=1}^{m-1}\beta_i\Phi(z_i)\) by the fixed-point iteration above.
3.  Recompute all coefficients \(\beta_1,\dots,\beta_m\) optimally from Proposition (optimal expansion coefficients).
4.  Increment \(m\) and repeat steps 2 and 3 until \(m=N_z\) or \(\|\Psi_{m}\|\lt\epsilon\).
5.  Optionally run a phase II: jointly optimize all \((z_i,\beta_i)\) by gradient descent, then recompute \(\beta\) once more.
::::

On USPS, the construction method beats selection because it draws on vectors off the training set; a tenfold speedup (25 reduced-set vectors in place of 254 support vectors) costs only a rise from 4.4% to 5.1% error, competitive with the convolutional networks of the day, and the synthetic vectors even look like digits (Burges 1996; Schölkopf and Smola 2002). The next example shows the single-term version, the base case of the loop, in miniature.

:::: {.example #example-20-2}
[Example (one-term reduced set of a three-term expansion)]{.box-title}

::: wex
An SVM-style expansion \(\Psi=\sum_{i=1}^3\alpha_i\Phi(x_i)\) with \(x_1=(0,0)\), \(x_2=(1,0)\), \(x_3=(0.5,1)\), coefficients \(\alpha=(1.0,\,0.8,\,0.6)\), Gaussian width \(\sigma=1\). Approximate it by a single term \(\beta\,\Phi(z)\).

1.  [Measure the target.]{.wex-op} With Gram entries \(k(x_1,x_2)=0.6065\), \(k(x_1,x_3)=k(x_2,x_3)=0.5353\), the squared length is \(\|\Psi\|^2=\alpha^\top K\alpha=4.1266\), so \(\|\Psi\|=2.0314\).
2.  [Find the pre-image.]{.wex-op} Starting at the weighted mean \(z_0=(0.4583,\,0.25)\), the fixed-point iteration converges in a handful of steps to \(z^\star=(0.4465,\,0.2218)\).
3.  [Set the optimal coefficient.]{.wex-op} Since \(k(z^\star,z^\star)=1\), \(\beta=\sum_i\alpha_i k(x_i,z^\star)=1.0\cdot 0.8831+0.8\cdot 0.8371+0.6\cdot 0.7377=1.9955\).
4.  [Measure the error.]{.wex-op} With \(\langle\Psi,\Phi(z^\star)\rangle=\beta\), the residual is \(\|\Psi-\beta\Phi(z^\star)\|^2=\|\Psi\|^2-\beta^2=4.1266-3.9818=0.1448\), so \(\|\Psi-\beta\Phi(z^\star)\|=0.3805\).

**Reading.** One synthetic term captures the three-term vector with a relative error of \(0.3805/2.0314=18.7\%\). Collapsing three bumps into one loses a fifth of the vector's length; adding a second reduced-set vector on the residual, the next pass of the construction loop, would shrink it further. This is the speed-accuracy dial that reduced-set methods turn.
:::

**Verification artifact.** checks/example-ch-invariance-example-20-2.json records the example source hash and verification scope.
::::

### Sequential evaluation and face detection {#sequential-evaluation}

Because reduced-set vectors are found one at a time, in decreasing order of importance, a classifier can be evaluated *sequentially*: run the first reduced-set vector everywhere, discard the locations it confidently rejects, and only spend the second vector on what remains, and so on (Romdhani et al. 2001). In face detection, where a binary face-versus-nonface classifier is scanned over every patch of an image at many scales, most patches are obvious non-faces that the first one or two vectors already reject. Romdhani et al. (2001) reduced an SVM with 1742 support vectors to 60 reduced-set vectors and, evaluating them sequentially, spent on average fewer than three kernel evaluations per image location, turning a slow classifier into one competitive with the fastest detectors of the time. This is the payoff for the reduced-set machinery and the natural complement to the VSV method that made the expansion large in the first place; see [[ch:support-vector-machines]] for the base classifier and [[ch:large-scale-kernels]] for the broader problem of making kernel evaluation cheap.

## Summary {#summary}

Two problems, one map. Incorporating invariances pushes prior knowledge forward through \(\Phi\): the Virtual Support Vector method generates transformed copies of the support vectors and retrains, invariance kernels whiten the tangent covariance so the classifier ignores motion along the orbit, tangent distance and translation-invariant kernels build the invariance into the metric, and jittering kernels fold the transformation into each evaluation. The pre-image problem pulls solutions back through \(\Phi\): exact pre-images rarely exist for Gaussian kernels, so a fixed-point iteration finds an approximate one, powering kernel PCA denoising and, iterated, the reduced-set expansions that compress a slow kernel machine into a fast one. The through-line is that a kernel is not just a similarity but a map with two directions, and controlling both is what makes kernel methods practical. The [[ch:kernel-families]] supply the invariant kernels this chapter assumed, and [[ch:kernels-and-deep-learning]] returns to invariance as the prior knowledge that modern architectures learn or hard-wire.

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

For **Invariances and the Pre-Image Problem**, do not apply a displayed formula without checking its domain, statistical assumptions, and numerical conditioning. Avoid selecting kernels or hyperparameters on test data, and do not interpret an optimization residual as a generalization guarantee. When the method is computational, report preprocessing, kernel parameters, regularization, solver tolerance, condition diagnostics, runtime, and a non-kernel baseline. When the result is theoretical, distinguish sufficient conditions from necessary ones and finite-sample claims from asymptotic statements.

## Summary and further reading {#summary-and-further-reading}

This chapter established explain the central definitions and claims in Invariances and the Pre-Image Problem; Apply the chapter's principal methods and interpret their outputs; State the assumptions behind formal results and connect them to earlier chapters. Revisit the assumptions attached to each formal result before transferring it to a new setting. For primary and extended treatments, consult [@lecun1995], [@scholkopf1996prior], [@simard1998].

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} Explain in two sentences why the Virtual SV method generates virtual examples only from the support vectors rather than from the full training set, citing the property of the SVM solution that makes this sound. What is the one experimental observation from the USPS table that confirms nothing is lost by this restriction?
2.  [computation]{.ex-tag} Using the setup of the pre-image example (points \(x_1=(0,0)\), \(x_2=(2,0)\), \(x_3=(1,1.5)\), weights \(\beta=(0.5,0.3,0.2)\), \(\sigma=1\)), carry out one step of the fixed-point iteration by hand starting from \(z_0=(0.8,0.3)\): compute the three weights \(w_i=\beta_i e^{-\|x_i-z_0\|^2/2}\) and the updated \(z_1=\sum_i w_i x_i/\sum_i w_i\). Check that you recover \(z_1\approx(0.644,0.246)\).
3.  [proof]{.ex-tag} Derive the Gaussian-kernel fixed-point update from scratch. Starting from the objective \(J(z)=\langle\Psi,\Phi(z)\rangle=\sum_i\alpha_i\exp(-\|x_i-z\|^2/2\sigma^2)\), compute \(\nabla_z J\), set it to zero, and solve the resulting equation for \(z\) to obtain the weighted-average form. Where in the derivation do you use that the kernel depends on \(z\) only through \(\|x_i-z\|^2\)?
    Hint

    ::: hint-body
    Use \(\nabla_z \exp(-\|x_i-z\|^2/2\sigma^2)=\exp(-\|x_i-z\|^2/2\sigma^2)\,(x_i-z)/\sigma^2\). The common factor \(1/\sigma^2\) drops out of the equation \(\sum_i\alpha_i e_i(z)(x_i-z)=0\), which rearranges to \(z=\sum_i\alpha_i e_i(z)x_i/\sum_i\alpha_i e_i(z)\).
    :::
4.  [proof]{.ex-tag} Prove the optimal-coefficient formula \(\beta=(K^z)^{-1}K^{zx}\alpha\) of Proposition (optimal expansion coefficients) by expanding \(\|\Psi-\sum_i\beta_i\Phi(z_i)\|^2\) as a quadratic in \(\beta\) and minimizing. Then verify that in the one-term case \(N_z=1\) with a normalized kernel, this reduces to \(\beta=\sum_i\alpha_i k(x_i,z)\), the value used in the reduced-set worked example.
    Hint

    ::: hint-body
    The objective is \(\alpha^\top K^x\alpha-2\beta^\top K^{zx}\alpha+\beta^\top K^z\beta\); its gradient in \(\beta\) is \(-2K^{zx}\alpha+2K^z\beta\). For \(N_z=1\), \(K^z=k(z,z)=1\).
    :::
5.  [proof]{.ex-tag} Show that a Gaussian kernel expansion \(\Psi=\sum_{j=1}^m\alpha_j\Phi(x_j)\) with at least two distinct \(x_j\) and all \(\alpha_j\neq 0\) has no exact pre-image. Use the fact that distinct Gaussians are linearly independent as functions, so \(\Phi(z)=\Psi\) would express one Gaussian as a nontrivial combination of Gaussians centered elsewhere.
    Hint

    ::: hint-body
    If \(\Phi(z)=k(z,\cdot)\) equalled \(\sum_j\alpha_j k(x_j,\cdot)\) as functions, then the linear independence of Gaussians centered at distinct points forces a contradiction unless the expansion has a single term.
    :::
6.  [computation]{.ex-tag} Take the three single-pixel images \(A=(1,0,0)\), \(B=(0,1,0)\), \(C=(0,0,1)\) and a linear kernel. Under one-pixel translation the jittered distances give \(d(A,B)=d(B,C)=0\). Compute \(d(A,C)^2=\|A\|^2-2\langle A,C\rangle+\|C\|^2\) directly and confirm the triangle inequality \(d(A,C)\le d(A,B)+d(B,C)\) fails. What does this tell you about whether a jittering kernel is positive definite?
7.  [exploration]{.ex-tag} The invariance-kernel preprocessing replaces the dot product by \(\langle x,x'\rangle_{T_\gamma^{-1}}\) with \(T_\gamma=(1-\gamma)T+\gamma I\) and \(T\) the tangent covariance. Explain what the two limits \(\gamma\to 1\) and \(\gamma\to 0\) do, and why diagonalizing \(T_\gamma\) and rescaling by inverse square-root eigenvalues amounts to downweighting the directions in which the data vary most under the transformation. Relate this to whitening in principal component analysis.
    Hint

    ::: hint-body
    At \(\gamma=1\), \(T_\gamma=I\) and the ordinary SVM is recovered. As \(\gamma\to 0\), \(T_\gamma\to T\) and the large-eigenvalue (high tangent-variance) directions are most strongly suppressed by \(T^{-1/2}\).
    :::
8.  [challenge]{.ex-tag} In the reduced-set construction algorithm, argue that iterating one-term pre-images greedily need not reach the globally optimal \(N_z\)-term reduced set, which motivates the phase-II joint optimization. Then explain why, for classification specifically, it is acceptable that the residual \(\|\Psi-\Psi'\|\) does not fall to zero, by contrasting the quantity the reduced set minimizes with the quantity that actually governs test error.
    Hint

    ::: hint-body
    Greedy pre-imaging fixes earlier \(z_i\) before later ones are chosen, so it optimizes a restricted problem. For classification the object of interest is \(\mathrm{sgn}(\sum_j\beta_j k(z_j,x)+\tilde b)\) integrated against the data distribution, not \(\|\Psi-\Psi'\|\); re-optimizing the threshold \(\tilde b\) recovers much of the accuracy even when the feature-space residual is nonzero.
    :::
:::
