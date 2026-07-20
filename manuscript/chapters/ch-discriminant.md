---
id: ch-discriminant
slug: kernel-discriminants-and-projections
title: Kernel Discriminants and Projections
part: V · Structure and Subspaces
order: 16
tier: advanced
prerequisites:
  - kernel-cca-and-correlation
objectives:
  - >-
    Explain the central definitions and claims in Kernel Discriminants and
    Projections.
  - Apply the chapter's principal methods and interpret their outputs.
  - >-
    State the assumptions behind formal results and connect them to earlier
    chapters.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-discriminant.yml
verification_date: null
bibliography:
  - shawe2004
  - fisher1936
  - mika1999kfd
  - baudat2000
  - rosipal2001
  - wold1975
---
# Kernel Discriminants and Projections

<p class="lead">The subspace methods of [[ch:kernel-pca|kernel PCA]] and [[ch:kernel-cca-and-correlation|kernel CCA]] read structure off the data without ever looking at a label. This chapter turns the same Gram matrix to supervised ends. When a label or a target is available, the question is no longer where the data spreads out the most, but which directions in feature space separate the classes or predict the response. We build three answers, all from one algebraic engine. Kernel Fisher discriminant analysis maximizes a ratio of between-class to within-class scatter and lands on a regularized generalized eigenproblem. Principal components regression keeps the eigenvectors of the kernel matrix but reweights them by how well each predicts the target. Kernel partial least squares abandons variance altogether and greedily extracts the directions of maximum covariance with the response, deflating the Gram matrix after each. Seen together, they are the supervised face of a single idea: optimize a Rayleigh quotient, or iterate a deflation, on the kernel matrix.</p>

## From variance to prediction {#supervised-subspaces}

Unsupervised subspace analysis chooses directions by an internal criterion. Principal components analysis picks the directions of largest variance; it never asks whether those directions have anything to do with a label. That is often exactly the wrong instinct for a supervised task. As Shawe-Taylor and Cristianini (2004) put it plainly, what matters for prediction is not the size of the variance of the data but how well a direction can be used to predict the output, and it can happen that the high-variance directions found by PCA are uncorrelated with the target while a direction of relatively low variance carries all the predictive signal. A supervised subspace method must therefore let the label into the objective.

Remarkably, doing so does not change the shape of the computation. Almost every method in this chapter reduces to one of two canonical operations on symmetric matrices built from the Gram matrix: solving a generalized eigenvalue problem \(Aw=\lambda Bw\), or iterating a deflation that strips off one direction and repeats. Fisher discriminant analysis is a generalized eigenproblem where \(A\) encodes class separation and \(B\) encodes within-class spread. Partial least squares is a deflation loop where the extracted direction maximizes covariance with the target. Principal components regression sits between them, an eigenproblem in the inputs followed by a target-driven reweighting. Because each operation touches the data only through inner products, all three pass into a kernel-defined feature space through the same dual representation used for [[ch:kernel-pca|kernel PCA]]. We begin with the shared engine.

## The generalized eigenvalue problem and deflation {#generalized-eigenproblem}

A Rayleigh quotient is a ratio of two quadratic forms whose maximizer is an eigenvector. We met the simplest version in PCA, where maximizing \(w^\top C w\) subject to \(\lVert w\rVert=1\) is the same as maximizing \(\rho(w)=w^\top C w / w^\top w\), solved by the top eigenvector of the covariance \(C\). The supervised methods below all optimize a two-matrix generalization.

:::: {.definition #def-16-1}
[Definition (generalized Rayleigh quotient)]{.box-title}

Given symmetric matrices \(A\) and \(B\) with \(B\) positive definite, the generalized Rayleigh quotient is

$$\rho(w)=\frac{w^\top A w}{w^\top B w}.$$

Its stationary points are the solutions of the generalized eigenvalue problem \(Aw=\lambda Bw\), and the maximum of \(\rho\) is the largest generalized eigenvalue \(\lambda_1\), attained at the corresponding eigenvector.
::::

The claim is proved by turning the generalized problem into an ordinary one. Because \(B\) is positive definite it has a symmetric positive definite square root \(B^{1/2}\) with \(B^{1/2}B^{1/2}=B\), and \(B^{1/2}\) is a bijection of \(\mathbb{R}^\ell\). Substitute \(v=B^{1/2}w\).

::::: {.proposition #prop-16-2}
[Reduction to a symmetric eigenproblem]{.box-title}

The generalized problem \(Aw=\lambda Bw\) is equivalent, under \(w=B^{-1/2}v\), to the ordinary symmetric eigenvalue problem

$$B^{-1/2}A B^{-1/2}\,v=\lambda v,$$

and the quotient transforms as

$$\rho(w)=\frac{w^\top A w}{w^\top B w}=\frac{\big(B^{1/2}w\big)^\top B^{-1/2}AB^{-1/2}\big(B^{1/2}w\big)}{\big\lVert B^{1/2}w\big\rVert^2},$$

which is the ordinary Rayleigh quotient of the symmetric matrix \(B^{-1/2}AB^{-1/2}\) in the variable \(v\).

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
:::::

::: {.proof}
[Proof]{.box-title}

Premultiply \(Aw=\lambda Bw\) by \(B^{-1/2}\) and insert \(B^{-1/2}B^{1/2}=I\) before \(w\): \(B^{-1/2}A B^{-1/2}(B^{1/2}w)=\lambda B^{1/2}w\), which is \(B^{-1/2}AB^{-1/2}v=\lambda v\). The matrix \(B^{-1/2}AB^{-1/2}\) is symmetric because \(A\) and \(B^{-1/2}\) are, so its eigenvectors \(v_1,\dots,v_\ell\) are orthonormal and its eigenvalues real. The quotient identity is a direct substitution. Since \(B^{1/2}\) is a bijection, maximizing \(\rho(w)\) over \(w\) is the same as maximizing the ordinary quotient over \(v\), whose maximum is \(\lambda_1\) at \(v_1\); hence the top generalized eigenvector is \(w_1=B^{-1/2}v_1\) (Shawe-Taylor and Cristianini 2004). [\(\square\)]{.qed}
:::

The eigenvectors of the generalized problem are not orthogonal in the usual sense, but in the geometry set by \(A\) and \(B\) they are. This conjugacy is what lets us extract several directions without their interfering.

:::: {.proposition #prop-16-3}
[Generalized orthogonality]{.box-title}

If the generalized eigenvalues are distinct, the eigenvectors \(w_i\) of \(Aw=\lambda Bw\) satisfy

$$w_i^\top B w_j=\delta_{ij},\qquad w_i^\top A w_j=\delta_{ij}\lambda_i,$$

that is, they are orthonormal in the \(B\)-metric and orthogonal in the \(A\)-metric.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.
::::

Once a leading eigenpair \((\lambda,w)\) is found, the next is obtained by removing that direction and repeating. For the two-matrix problem the correct removal is a generalized deflation that respects the \(B\)-geometry.

:::: {.definition #def-16-4}
[Generalized deflation]{.box-title}

After finding a nonzero generalized eigenpair \(\lambda,w\) of \(Aw=\lambda Bw\), deflate the numerator matrix by

$$A\ \longleftarrow\ A-\lambda\,Bw\,(Bw)^\top,$$

leaving \(B\) unchanged. The deflated problem has the same remaining eigenpairs with the found direction removed.
::::

Two facts about \(B\) drive everything that follows. First, if \(B\) is only positive semidefinite, its square root is not invertible and the reduction above breaks down: the quotient can be made arbitrarily large along the null space of \(B\), so the problem is ill posed. This is exactly what happens in feature space, where the within-class scatter is singular, and it forces the regularization \(B\leftarrow B+\mu I\) that appears in every algorithm below. Second, when \(A\) has rank one, the reduced matrix \(B^{-1/2}AB^{-1/2}\) has a single nonzero eigenvalue, so only one eigenvector carries information and it can be read off by a single linear solve rather than a full eigendecomposition. Fisher discriminant analysis is precisely this rank-one case.

## Kernel Fisher discriminant analysis {#kernel-fisher}

Fisher (1936) asked for the one-dimensional projection of a two-class dataset that best separates the classes. Good separation means two things at once: the projected class means should be far apart, and each class should be tightly concentrated around its own mean. Fisher captured both in a single ratio, and Mika et al. (1999) carried the construction into a kernel feature space, where the projection becomes nonlinear in the input.

### Between-class and within-class scatter {#kfd-scatter}

Work in the feature space with map \(\phi\). Write the two classes as index sets of sizes \(\ell_+\) and \(\ell_-\), and let the class means in feature space be

$$\mu_c^{\phi}=\frac{1}{\ell_c}\sum_{i\in c}\phi(x_i),\qquad c\in\{+,-\}.$$

A projection direction \(w\) sends a point \(\phi(x)\) to the scalar \(w^\top\phi(x)\). The projected class means are \(w^\top\mu_c^{\phi}\), and the between-class scatter is the squared gap

$$\big(w^\top\mu_+^{\phi}-w^\top\mu_-^{\phi}\big)^2=w^\top S_B\,w,\qquad S_B=(\mu_+^{\phi}-\mu_-^{\phi})(\mu_+^{\phi}-\mu_-^{\phi})^\top.$$

The within-class scatter is the total spread of each class about its own projected mean,

$$w^\top S_W\,w,\qquad S_W=\sum_{c\in\{+,-\}}\sum_{i\in c}\big(\phi(x_i)-\mu_c^{\phi}\big)\big(\phi(x_i)-\mu_c^{\phi}\big)^\top.$$

Fisher's criterion is the ratio of the two, a generalized Rayleigh quotient in \(w\),

$$J(w)=\frac{w^\top S_B\,w}{w^\top S_W\,w}.$$

Maximizing \(J\) is the generalized eigenproblem \(S_B w=\lambda S_W w\). Because \(S_B\) has rank one, its range is spanned by \(\mu_+^{\phi}-\mu_-^{\phi}\), and the optimal direction is \(w\propto S_W^{-1}(\mu_+^{\phi}-\mu_-^{\phi})\): a single linear solve, not a full eigendecomposition.

### The dual, and why regularization is unavoidable {#kfd-dual}

In a high-dimensional feature space we can neither form \(\phi(x_i)\) nor \(S_W\) explicitly, so we pass to the dual. Any useful direction lies in the span of the data, \(w=\sum_{i=1}^{\ell}\alpha_i\phi(x_i)\), and every quantity in \(J\) then depends on \(\phi\) only through the kernel. Projecting the direction onto a class mean gives

$$w^\top\mu_c^{\phi}=\sum_{i=1}^{\ell}\alpha_i\,\frac{1}{\ell_c}\sum_{j\in c}\kappa(x_i,x_j)=\alpha^\top m_c,\qquad (m_c)_i=\frac{1}{\ell_c}\sum_{j\in c}K_{ij},$$

so the between-class term becomes \(w^\top S_B w=\alpha^\top M\alpha\) with \(M=(m_+-m_-)(m_+-m_-)^\top\). A parallel computation writes the within-class term as \(\alpha^\top N\alpha\) with

$$N=\sum_{c\in\{+,-\}}K_c\Big(I_{\ell_c}-\tfrac{1}{\ell_c}\mathbf{1}\mathbf{1}^\top\Big)K_c^\top,$$

where \(K_c\) is the \(\ell\times\ell_c\) block of columns of the Gram matrix belonging to class \(c\), and \(I_{\ell_c}-\tfrac{1}{\ell_c}\mathbf{1}\mathbf{1}^\top\) is the centering projector for that class. The kernel Fisher objective is thus

$$J(\alpha)=\frac{\alpha^\top M\alpha}{\alpha^\top N\alpha}.$$

Now the singularity bites. The matrix \(N\) is built from \(\ell\) feature vectors centered within their classes, so it is at most of rank \(\ell-2\); in a feature space of dimension larger than the sample it is always singular. Its null space contains directions along which \(\alpha^\top N\alpha=0\) while \(\alpha^\top M\alpha\gt 0\), which would send \(J\) to infinity and pick out a meaningless direction that fits the training split perfectly and generalizes not at all. The cure, exactly the regularization of the generalized eigenproblem above, is to add a multiple of the identity, replacing \(N\) by \(N+\mu I\). This penalizes the RKHS norm \(\lVert w\rVert^2=\alpha^\top K\alpha\) of the direction (a ridge on \(\alpha\) in practice), restores invertibility, and stabilizes the solution. Since \(M\) is rank one, the maximizer is the single linear solve

$$\alpha\ \propto\ (N+\mu I)^{-1}(m_+-m_-).$$

This is the kernel Fisher discriminant of Mika et al. (1999); the equivalent generalized discriminant analysis of Baudat and Anouar (2000) casts the same computation as an eigendecomposition of a kernel scatter matrix and extends it to more than two classes.

:::: {.algorithm #algo-16-1}
[Algorithm (kernel Fisher discriminant)]{.box-title}

::: algo-io
[Input]{.algo-lab} Gram matrix \(K\in\mathbb{R}^{\ell\times\ell}\), class index sets \(+,-\) of sizes \(\ell_+,\ell_-\), regularization \(\mu\gt 0\).

[Output]{.algo-lab} dual direction \(\alpha\) and threshold \(b\); projection \(t(x)=\sum_i\alpha_i\kappa(x_i,x)\).
:::

1.  Form the class-mean vectors \((m_c)_i=\tfrac{1}{\ell_c}\sum_{j\in c}K_{ij}\) for \(c\in\{+,-\}\).
2.  Form the within-class matrix \(N=\sum_{c}K_c\big(I_{\ell_c}-\tfrac{1}{\ell_c}\mathbf{1}\mathbf{1}^\top\big)K_c^\top\).
3.  Solve the regularized system \((N+\mu I)\,\alpha=(m_+-m_-)\); since the between-class matrix \(M\) has rank one, this single solve is the top generalized eigenvector.
4.  Set the threshold midway between the projected class means, \(b=\tfrac12\,\alpha^\top(m_++m_-)\), and classify \(x\) by the sign of \(t(x)-b\).
::::

The same discriminant appears in [[ch:kernel-cca-and-correlation|the correlation chapter]] as a generalized eigenproblem in the input covariances. Writing \(y\) for the \(\pm\) label vector and \(X\) for the (feature) data matrix, Shawe-Taylor and Cristianini (2004) show the regularized Fisher discriminant maximizes \(w^\top E w/w^\top F w\) with \(E=X^\top y y^\top X\) and \(F=\lambda I+\tfrac{\ell}{2\ell_+\ell_-}X^\top B X\); the numerator matrix \(E=(X^\top y)(y^\top X)\) has rank one because \(X^\top y\) is a single column, which is again why only the first eigenvector matters and why the solution reduces to a matrix inversion. The kernel form above is the dual of this statement.

:::::: {.example #example-16-1}
[Example (kernel Fisher on four points)]{.box-title}

::::: wex
:::: wex-setup
Two points per class in \(\mathbb{R}^2\) with the linear kernel \(\kappa(x,x')=\langle x,x'\rangle\): class \(+\) is \(x_1=(1,0),\,x_2=(0,1)\); class \(-\) is \(x_3=(-1,0),\,x_4=(0,-1)\). Regularization \(\mu=1\). The Gram matrix is

$$K=\begin{pmatrix}1&0&-1&0\\0&1&0&-1\\-1&0&1&0\\0&-1&0&1\end{pmatrix}.$$
::::

1.  [Average kernel columns within each class.]{.wex-op} The class-mean vectors are \(m_+=(0.5,\,0.5,\,-0.5,\,-0.5)\) and \(m_-=(-0.5,\,-0.5,\,0.5,\,0.5)\), so the between-class direction is \(m_+-m_-=(1,\,1,\,-1,\,-1)\).
2.  [Assemble the within-class matrix.]{.wex-op} Centering each class block and summing gives

$$N=\begin{pmatrix}1&-1&-1&1\\-1&1&1&-1\\-1&1&1&-1\\1&-1&-1&1\end{pmatrix},$$

    which is singular (rank one), confirming that without regularization the quotient is ill posed.
3.  [Solve the regularized system.]{.wex-op} Solving \((N+I)\alpha=m_+-m_-\) yields \(\alpha=(1,1,-1,-1)\), that is \(\alpha=(0.5,0.5,-0.5,-0.5)\) after normalization to unit length.
4.  [Project the training points.]{.wex-op} The projections \(t=K\alpha=(1,\,1,\,-1,\,-1)\) place both \(+\) points at \(+1\) and both \(-\) points at \(-1\); the objective value is \(\alpha^\top M\alpha/\alpha^\top(N+I)\alpha=4\).

**Reading.** The discriminant sends the two classes to cleanly separated points \(\pm 1\) with the threshold at \(0\), and the whole computation used only the Gram matrix. The singular \(N\) makes concrete why the \(+\mu I\) term is not a numerical convenience but a necessity.
:::::

**Verification artifact.** checks/example-ch-discriminant-example-16-1.json records the example source hash and verification scope.
::::::

## Principal components regression {#pcr}

Suppose the target is real-valued and we want to regress. A natural pipeline is to run [[ch:kernel-pca|kernel PCA]] first, project the data onto its leading principal directions, and fit a linear regressor there. This is principal components regression (PCR), and its motivation is denoising: by discarding the low-variance directions, presumed to be noise, we reduce the variance of the regression estimate. Shawe-Taylor and Cristianini (2004) also flag its danger, the same warning that opened the chapter: PCA orders directions by variance, not by relevance to the target, so a low-variance direction that happens to predict the response well can be thrown away before the regression ever sees it.

The dual form is elegant. Let \(K=\sum_{j}\lambda_j v_j v_j^\top\) be the eigendecomposition of the (centered) kernel matrix, eigenvalues in descending order. Keeping the first \(k\) components, the dual regression coefficients are a target-weighted sum of the leading eigenvectors,

$$\alpha=\sum_{j=1}^{k}\frac{1}{\lambda_j}\big(v_j^\top y\big)\,v_j,$$

and the fitted function is \(f(x)=\sum_{i}\alpha_i\kappa(x_i,x)\). Each eigenvector contributes in proportion to its covariance \(v_j^\top y\) with the target, tempered by the inverse eigenvalue \(1/\lambda_j\). The construction is incremental: adding a component appends one term to \(\alpha\) without disturbing the earlier ones, because the eigenvectors are orthogonal.

:::: {.algorithm #algo-16-2}
[Algorithm (dual principal components regression)]{.box-title}

::: algo-io
[Input]{.algo-lab} centered Gram matrix \(K\in\mathbb{R}^{\ell\times\ell}\), target vector \(y\in\mathbb{R}^{\ell}\), number of components \(k\).

[Output]{.algo-lab} dual coefficients \(\alpha\); predictor \(f(x)=\sum_i\alpha_i\kappa(x_i,x)\).
:::

1.  Eigendecompose \(K=\sum_j\lambda_j v_j v_j^\top\) with \(\lambda_1\ge\lambda_2\ge\cdots\).
2.  For each retained component compute the target covariance \(v_j^\top y\).
3.  Repeat for each component \(j=1,\dots,k\): accumulate \(\alpha\leftarrow\alpha+\tfrac{1}{\lambda_j}(v_j^\top y)\,v_j\).
4.  Predict new points via \(f(x)=\sum_i\alpha_i\kappa(x_i,x)\), centering \(x\) against the training kernel as in kernel PCA.
::::

:::::: {.example #example-16-2}
[Example (PCR and a low-variance predictive direction)]{.box-title}

::::: wex
:::: wex-setup
Three points in \(\mathbb{R}^2\), linear kernel, targets \(y=(1,-1,0)\): \(x_1=(2,0),\,x_2=(0,1),\,x_3=(-2,-1)\). The Gram matrix is already centered,

$$K=\begin{pmatrix}4&0&-4\\0&1&-1\\-4&-1&5\end{pmatrix},\qquad y=(1,-1,0).$$
::::

1.  [Eigendecompose the centered kernel.]{.wex-op} The nonzero eigenvalues are \(\lambda_1=5+\sqrt{13}\approx 8.6056\) and \(\lambda_2=5-\sqrt{13}\approx 1.3944\), with a zero third eigenvalue (three centered points span a plane).
2.  [Measure covariance of each direction with the target.]{.wex-op} The projections are \(v_1^\top y\approx -0.5537\) and \(v_2^\top y\approx -1.3013\): the second, low-variance direction is the one more aligned with \(y\).
3.  [Fit with only the top component.]{.wex-op} With \(k=1\), \(\alpha\approx(0.0420,\,0.0064,\,-0.0483)\) and the fitted values \(K\alpha\approx(0.361,\,0.055,\,-0.416)\) are far from \(y\).
4.  [Add the second component.]{.wex-op} With \(k=2\), \(\alpha=(0.5,\,-0.75,\,0.25)\) and \(K\alpha=(1,\,-1,\,0)\) recovers the target exactly.

**Reading.** The predictive direction here is the one of smaller variance. Truncating PCR at the single leading component discards precisely the direction that carries the signal; only keeping the second eigenvector recovers the response. This is the concrete failure mode that motivates partial least squares.
:::::

**Verification artifact.** checks/example-ch-discriminant-example-16-2.json records the example source hash and verification scope.
::::::

## Kernel partial least squares {#kernel-pls}

Partial least squares fixes PCR's blind spot by choosing directions using the target from the start. Instead of the directions of maximum variance, it extracts the directions of maximum covariance with the response, then deflates and repeats. It originated in the chemometrics work of Wold (1975), where high-dimensional and highly correlated predictors are the norm, and it was kernelized by Rosipal and Trejo (2001). Because covariance couples inputs to outputs, PLS aligns its features with what actually predicts \(y\).

### Directions of maximum covariance {#max-covariance}

The building block is the direction pair that maximizes covariance between two views of the data. For paired data with cross-covariance \(C_{xy}\), the problem

$$\max_{w_x,w_y}\ \frac{w_x^\top C_{xy}\,w_y}{\lVert w_x\rVert\,\lVert w_y\rVert}$$

is solved by the first singular vectors of \(C_{xy}=U\Sigma V^\top\): \(w_x=u_1\), \(w_y=v_1\), with the covariance equal to the largest singular value \(\sigma_1\) (Shawe-Taylor and Cristianini 2004). Unlike the Rayleigh quotient, \(C_{xy}\) is neither square nor symmetric, and we optimize over two vectors; but the mechanism to extract more than one direction is again deflation. After taking the leading pair, project both views onto the orthogonal complement,

$$X\leftarrow X\big(I-u_1u_1^\top\big),\qquad Y\leftarrow Y\big(I-v_1v_1^\top\big),$$

and recompute. When the second view is the target, this is the engine of PLS regression.

### Primal PLS and its deflation {#pls-deflation}

Primal PLS runs the covariance step against the target and deflates the inputs. At stage \(j\) it takes \(u_j\), the first singular vector of \(X_j^\top Y\) (found by a short power iteration \(u\leftarrow X_j^\top Y Y^\top X_j u\)), forms the score \(\tau_j=X_j u_j\), and then deflates. The key object is the loading

$$p_j=\frac{X_j^\top X_j\,u_j}{u_j^\top X_j^\top X_j\,u_j},$$

which is the direction the score \(\tau_j\) explains in the input space. Deflation removes exactly that,

$$X_{j+1}=X_j\big(I-u_j p_j^\top\big).$$

This choice is what makes the extracted scores conjugate. One checks that \(u_i^\top p_j=0\) for \(i\lt j\) and \(u_j^\top p_j=1\), so the successive scores \(X_j u_j\) are mutually orthogonal, which in turn lets the regression coefficients be computed one component at a time without interaction (Shawe-Taylor and Cristianini 2004). Because PLS deflates only \(X\) and removes only the covariance already explained, deflating \(Y\) as well changes nothing, and the restriction \(k\le m\) that limits maximum-covariance features does not apply: PLS can extract as many components as it likes.

:::::: {.example #example-16-3}
[Example (one PLS deflation step)]{.box-title}

::::: wex
:::: wex-setup
Centered inputs and target,

$$X=\begin{pmatrix}1&2\\1&-1\\-2&-1\end{pmatrix},\qquad y=(1,-1,0),$$

with columns of \(X\) and entries of \(y\) each summing to zero.
::::

1.  [Find the first covariance direction.]{.wex-op} \(X^\top y=(0,3)\), so the unit direction is \(u_1=(0,1)\); the score is \(\tau_1=X u_1=(2,-1,-1)\).
2.  [Form the loading.]{.wex-op} With \(X^\top X=\left(\begin{smallmatrix}6&3\\3&6\end{smallmatrix}\right)\) and \(u_1^\top X^\top X u_1=6\), the loading is \(p_1=X^\top X u_1/6=(0.5,\,1)\), which differs from \(u_1\).
3.  [Read off the regression weight.]{.wex-op} \(c_1=y^\top X u_1/6=0.5\), so this component's fitted contribution is \(\tau_1 c_1=(1,\,-0.5,\,-0.5)\).
4.  [Deflate the inputs.]{.wex-op} \(X_2=X(I-u_1p_1^\top)=\left(\begin{smallmatrix}0&0\\1.5&0\\-1.5&0\end{smallmatrix}\right)\); the check \(X_2^\top\tau_1=(0,0)\) confirms the new residual is orthogonal to the extracted score.

**Reading.** One step selects the direction of maximum covariance with \(y\), fits a regression weight along it, and subtracts what it explained. The residual \(X_2\) is orthogonal to the score just used, so the next component is extracted from a genuinely new subspace: this orthogonality is what makes the deflation loop terminate cleanly and the coefficients accumulate independently.
:::::

**Verification artifact.** checks/example-ch-discriminant-example-16-3.json records the example source hash and verification scope.
::::::

### The dual and kernel PLS {#kernel-pls-dual}

Everything above touches \(X\) only through inner products, so PLS kernelizes. In the dual we never form \(u_j\); we track the score \(\tau_j\) directly. A power iteration on the kernel matrix, \(\beta\leftarrow K_j Y_j Y_j^\top\beta\) (normalized), converges to a dual vector \(\beta_j\), from which \(\tau_j=K_j\beta_j\). The input deflation \(X_{j+1}=(I-\tau_j\tau_j^\top/\tau_j^\top\tau_j)X_j\) becomes a two-sided deflation of the Gram matrix,

$$K_{j+1}=\Big(I-\frac{\tau_j\tau_j^\top}{\tau_j^\top\tau_j}\Big)K_j\Big(I-\frac{\tau_j\tau_j^\top}{\tau_j^\top\tau_j}\Big),$$

computable with no explicit feature vectors (Rosipal and Trejo 2001). After \(k\) steps, collecting the dual vectors as columns of \(B\), the scores as columns of \(T\), and the targets in \(Y\), the dual regression coefficients are

$$\alpha=B\big(T^\top K B\big)^{-1}T^\top Y,\qquad f_r(x)=\sum_{i=1}^{\ell}\alpha^r_i\,\kappa(x_i,x),$$

where the matrix \(T^\top K B\) is upper triangular, so the inverse is cheap. This combines the two advantages the earlier methods split between them: like PLS it aligns features with the target through covariance, and like PCR it is not limited to \(k\le m\) components.

:::: {.algorithm #algo-16-3}
[Algorithm (kernel PLS)]{.box-title}

::: algo-io
[Input]{.algo-lab} centered kernel matrix \(K\in\mathbb{R}^{\ell\times\ell}\), target matrix \(Y\in\mathbb{R}^{\ell\times m}\), number of components \(k\).

[Output]{.algo-lab} dual regression coefficients \(\alpha\); predictor \(f_r(x)=\sum_i\alpha^r_i\kappa(x_i,x)\).
:::

1.  Initialize \(K_1=K\), \(Y_1=Y\).
2.  Repeat for each component \(j=1,\dots,k\):
3.  power-iterate \(\beta\leftarrow K_j Y_j Y_j^\top\beta\), renormalizing, until \(\beta_j\) converges;
4.  form the score \(\tau_j=K_j\beta_j\) and the output weight \(c_j=Y_j^\top\tau_j/(\tau_j^\top\tau_j)\);
5.  deflate \(K_{j+1}=(I-P_j)K_j(I-P_j)\) and \(Y_{j+1}=Y_j-\tau_j c_j^\top\), where \(P_j=\tau_j\tau_j^\top/\tau_j^\top\tau_j\).
6.  Assemble \(\alpha=B(T^\top K B)^{-1}T^\top Y\) from the stored \(B=[\beta_1\cdots\beta_k]\) and \(T=[\tau_1\cdots\tau_k]\).
::::

## One engine, three methods {#unification}

The three algorithms differ only in which matrices they feed to the shared engine and whether they solve one eigenproblem or iterate a deflation. All read the data through the kernel matrix alone, and all can be viewed as choosing directions that optimize a quadratic criterion under a quadratic constraint. The table makes the parallel explicit.

  Method          Criterion optimized                                    Engine                                                                                                   Uses target?
  --------------- ------------------------------------------------------ -------------------------------------------------------------------------------------------------------- ----------------------------
  Kernel PCA      variance \(w^\top C w\)          eigenproblem on \(K\)                                                     no
  Kernel Fisher   scatter ratio \(\dfrac{\alpha^\top M\alpha}{\alpha^\top (N+\mu I)\alpha}\)     regularized generalized eigenproblem (rank-one \(M\): one solve)          yes (class labels)
  PCR             variance, then target reweighting                      eigenproblem on \(K\) plus \(\tfrac{1}{\lambda_j}(v_j^\top y)\) weights   yes (in the weights)
  Kernel PLS      covariance with \(y\)   deflation loop on \(K\)                                                   yes (drives the direction)

Reading down the last two columns tells the whole story. PCA and PCR both diagonalize the kernel matrix, but PCR then lets the target choose how much of each eigenvector to keep. Fisher and PLS both let the target choose the direction itself, one through a generalized eigenproblem, the other through a deflation. The regularization that rescues Fisher from its singular scatter matrix is the same \(+\mu I\) that the generalized eigenproblem needs whenever the denominator is only positive semidefinite, which in a rich feature space is always. What began as unsupervised subspace analysis becomes, with the label admitted into \(A\) or into the deflation target, a supervised one, at no change to the underlying linear algebra.

## Summary {#summary}

Supervised subspace methods choose directions in a kernel feature space by how well they separate classes or predict a target, and they all reduce to generalized eigenproblems or deflations on the Gram matrix. Kernel Fisher discriminant analysis maximizes the ratio of between-class to within-class scatter; because the within-class scatter is singular in feature space, it must be regularized, after which the rank-one numerator makes the solution a single linear solve. Principal components regression keeps the kernel eigenvectors but reweights them by their covariance with the target, and its worked failure mode, discarding a low-variance predictive direction, motivates the covariance-driven alternative. Kernel partial least squares extracts the directions of maximum covariance with the response and deflates the kernel matrix after each, combining target alignment with an unbounded number of components. The next parts put these projections to work alongside [[ch:support-vector-machines|the supervised margin methods]] and revisit their statistical footing through [[ch:mercer-and-rates|Mercer's theorem and spectral rates]].

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

For **Kernel Discriminants and Projections**, do not apply a displayed formula without checking its domain, statistical assumptions, and numerical conditioning. Avoid selecting kernels or hyperparameters on test data, and do not interpret an optimization residual as a generalization guarantee. When the method is computational, report preprocessing, kernel parameters, regularization, solver tolerance, condition diagnostics, runtime, and a non-kernel baseline. When the result is theoretical, distinguish sufficient conditions from necessary ones and finite-sample claims from asymptotic statements.

## Summary and further reading {#summary-and-further-reading}

This chapter established explain the central definitions and claims in Kernel Discriminants and Projections; Apply the chapter's principal methods and interpret their outputs; State the assumptions behind formal results and connect them to earlier chapters. Revisit the assumptions attached to each formal result before transferring it to a new setting. For primary and extended treatments, consult [@shawe2004], [@fisher1936], [@mika1999kfd].

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} Explain in one or two sentences why maximizing the variance \(w^\top C w\) subject to \(\lVert w\rVert=1\) has the same solution as maximizing the Rayleigh quotient \(w^\top C w/w^\top w\) without any constraint. What role does the homogeneity of the quotient under rescaling \(w\mapsto t w\) play?
2.  [computation]{.ex-tag} Repeat the kernel Fisher worked example but with the polynomial kernel \(\kappa(x,x')=(\langle x,x'\rangle+1)^2\) on the same four points \(x_1=(1,0),x_2=(0,1),x_3=(-1,0),x_4=(0,-1)\) and \(\mu=1\). Build \(K\), the class means \(m_\pm\), the matrices \(M\) and \(N\), solve \((N+\mu I)\alpha=m_+-m_-\), and report the four projections \(t=K\alpha\). Are the classes still separated?
    Hint

    ::: hint-body
    Only the Gram matrix changes; the pipeline is identical to Example (kernel Fisher on four points). Compute \(K_{ij}=(\langle x_i,x_j\rangle+1)^2\) first, then reuse the steps. A short numpy script mirrors the check for the linear case.
    :::
3.  [proof]{.ex-tag} Prove the reduction of the generalized eigenproblem: if \(B\) is symmetric positive definite with square root \(B^{1/2}\), then \(Aw=\lambda Bw\) is equivalent to \(B^{-1/2}AB^{-1/2}v=\lambda v\) under \(v=B^{1/2}w\), and conclude that the maximum of \(w^\top A w/w^\top B w\) is the largest eigenvalue of \(B^{-1/2}AB^{-1/2}\). Where in the argument does positive definiteness, rather than mere positive semidefiniteness, get used?
    Hint

    ::: hint-body
    Premultiply by \(B^{-1/2}\) and insert \(I=B^{-1/2}B^{1/2}\). Positive definiteness is what makes \(B^{1/2}\) invertible and a bijection of \(\mathbb{R}^\ell\), so the change of variables is one-to-one and no direction is lost.
    :::
4.  [proof]{.ex-tag} Show that the between-class scatter matrix \(S_B=(\mu_+^{\phi}-\mu_-^{\phi})(\mu_+^{\phi}-\mu_-^{\phi})^\top\) has rank one, and deduce that the Fisher direction is \(w\propto S_W^{-1}(\mu_+^{\phi}-\mu_-^{\phi})\) with no eigendecomposition required. Why does the rank-one structure mean that only one discriminant direction carries information in the two-class problem?
    Hint

    ::: hint-body
    An outer product \(dd^\top\) has range \(\operatorname{span}\{d\}\), hence rank one. In \(S_B w=\lambda S_W w\), the left side is always a multiple of \(d=\mu_+^{\phi}-\mu_-^{\phi}\), so \(w\propto S_W^{-1}d\); all other generalized eigenvalues are zero.
    :::
5.  [proof]{.ex-tag} In primal PLS the loading is \(p_j=X_j^\top X_j u_j/(u_j^\top X_j^\top X_j u_j)\). Verify the two conjugacy relations \(u_j^\top p_j=1\) and \(u_i^\top p_j=0\) for \(i\lt j\), and explain why they imply that the scores \(\tau_j=X_j u_j\) extracted at different stages are mutually orthogonal.
    Hint

    ::: hint-body
    The first is immediate from the definition of \(p_j\). For the second, use that the deflation \(X_{j+1}=X_j(I-u_j p_j^\top)\) makes columns of later \(X_i\) orthogonal to the already-removed loading directions; expand \(u_i^\top p_j\) using the definition of \(p_j\) and the orthogonality of \(X_i u_i\) to \(X_j u_j\).
    :::
6.  [computation]{.ex-tag} Take the PCR worked example and confirm the incremental property: compute \(\alpha\) for \(k=1\), then show that adding the \(j=2\) term \(\tfrac{1}{\lambda_2}(v_2^\top y)v_2\) gives the \(k=2\) coefficients without recomputing the first. Verify numerically that \(K\alpha=(1,-1,0)\) at \(k=2\), and explain in one sentence why the third (zero-eigenvalue) direction can never be used.
    Hint

    ::: hint-body
    Orthogonality of the \(v_j\) makes the sum \(\alpha=\sum_j\tfrac{1}{\lambda_j}(v_j^\top y)v_j\) a running total. The \(1/\lambda_j\) weight is undefined for \(\lambda_3=0\); that direction has no variance and cannot enter a variance-based projection.
    :::
7.  [exploration]{.ex-tag} Contrast PCR and kernel PLS on the situation of the PCR worked example, where the predictive direction has the smaller eigenvalue. Argue that PLS, extracting the direction of maximum covariance with \(y\), would select that predictive direction first, whereas PCR ordered by variance selects it second. What does this say about when to prefer each method?
8.  [challenge]{.ex-tag} The regularized kernel Fisher objective penalizes the RKHS norm \(\lVert w\rVert^2=\alpha^\top K\alpha\) of the discriminant direction, which is why \(N\) is replaced by \(N+\mu K\) in some formulations and by \(N+\mu I\) in others. Derive both variants from a penalized quotient \(\alpha^\top M\alpha/(\alpha^\top N\alpha+\mu\,\Omega(\alpha))\), taking \(\Omega(\alpha)=\alpha^\top K\alpha\) in one case and \(\Omega(\alpha)=\alpha^\top\alpha\) in the other, and discuss which penalty corresponds to controlling the feature-space norm of \(w\) and which to a plain ridge on the dual coefficients.
    Hint

    ::: hint-body
    Since \(w=\sum_i\alpha_i\phi(x_i)\), the feature-space norm is \(\lVert w\rVert^2=\sum_{i,j}\alpha_i\alpha_j\kappa(x_i,x_j)=\alpha^\top K\alpha\), giving the \(N+\mu K\) form; the plain \(\alpha^\top\alpha\) ridge gives \(N+\mu I\). Both restore invertibility, but they penalize different geometries.
    :::
:::
