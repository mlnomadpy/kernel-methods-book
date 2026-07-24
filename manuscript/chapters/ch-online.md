---
id: ch-online
slug: online-kernel-learning
title: Online Kernel Learning
part: III · Optimization and Implementation
order: 10
tier: advanced
prerequisites:
  - solving-the-svm
objectives:
  - Derive the kernel perceptron as a mistake-driven dual expansion.
  - Prove Novikoff's mistake bound and identify every assumption it uses.
  - Relate the kernel adatron to coordinate ascent on the SVM dual.
  - Build regularized online updates for regression and margin losses.
  - >-
    Diagnose support-set growth and compare removal, projection, merging, and
    decay.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-online.yml
verification_date: null
bibliography:
  - rosenblatt1958
  - novikoff1962
  - aizerman1964
  - friess1998
  - freund1999voted
  - kivinen2004
  - crammer2006pa
  - shawe2004
  - scholkopf2002
---
# Online Kernel Learning

<p class="lead">Data may arrive as a stream, one example at a time, faster than any solver can re-fit; the target may drift while we watch; or the sample may simply be too large for its Gram matrix to fit in memory. Every kernel machine we have built so far is a batch machine, one that reads the whole training set, assembles the Gram matrix, and solves one optimization problem to convergence, and that is a luxury the world does not always grant. This chapter develops the kernel machines that learn from a stream, updating a little on each example and never forming the full Gram matrix. The two classical engines are the kernel perceptron, whose mistake-driven update is one line and whose convergence is governed by the margin through Novikoff's two-sided bound, and the kernel adatron, which turns the support vector dual into a coordinate-wise gradient ascent that climbs to the very same max-margin solution the batch [[ch:support-vector-machines|support vector machine]] finds. We close with the price of streaming, the budget problem: an online kernel expansion tends to grow one term per mistake without limit, and taming it is what separates a toy from a deployable system.</p>

## The online learning setting {#online-setting}

In the batch setting a learner is handed a fixed sample \(S=\{(x_1,y_1),\dots,(x_\ell,y_\ell)\}\) and produces a single hypothesis. Online learning replaces this with a game played over time. At each round \(t\) the learner holds a current hypothesis \(f_t\); it receives an input \(x_t\), predicts \(\hat y_t=\operatorname{sgn} f_t(x_t)\), then sees the true label \(y_t\) and suffers a loss. Only after that does it update, producing \(f_{t+1}\). The data may be an unending stream, the distribution generating \((x_t,y_t)\) may change over time, and so the goal is not to minimize an expected risk against one fixed distribution but to predict well, round by round, as the stream unfolds (Kivinen, Smola, and Williamson 2004).

Three constraints shape everything that follows. First, the learner processes one example at a time and cannot revisit the past at will. Second, and decisively for us, it never assembles the full \(\ell\times\ell\) Gram matrix: a kernel evaluation \(K(x_j,x_t)\) is computed on demand against the examples the learner has chosen to remember, and nothing more. Third, updates should be cheap, ideally constant work per round beyond the kernel evaluations. The methods below meet all three, and the reason a kernel can appear at all is the dual representation that has run through this whole book: the hypothesis is stored not as a weight vector in a feature space we cannot touch, but as a weighted sum of kernel functions centered on remembered examples.

## The kernel perceptron {#kernel-perceptron}

The oldest online learner is the perceptron of Rosenblatt (1958). In its primal form it maintains a weight vector \(w\), predicts \(\operatorname{sgn}\langle w,x\rangle\), and on a mistake nudges \(w\) toward the offending example: \(w\leftarrow w+y_t x_t\). The nudge is exactly right in direction, since adding \(y_t x_t\) raises \(y_t\langle w,x_t\rangle\) by \(\|x_t\|^2\), pushing the example toward the correct side. Aizerman, Braverman, and Rozonoer (1964) observed that this algorithm never needs \(w\) itself: because \(w\) starts at zero and only ever accumulates multiples of training inputs, it always lives in their span, and every quantity the algorithm touches is an inner product. Replacing that inner product by a kernel gives the kernel perceptron, one of the earliest instances of the kernel trick.

### The dual update and the kernel expansion {#dual-update}

What exactly must the learner store, and what does an update cost? A line of bookkeeping answers both. Work in the feature space of a kernel \(K\) with map \(\varphi\), and let the weight vector be \(w=\sum_{j}\alpha_j y_j\,\varphi(x_j)\) for nonnegative integer counts \(\alpha_j\). The counts start at zero, so \(w_0=0\). The prediction on any point is then a pure kernel expansion,

$$f(x)=\langle w,\varphi(x)\rangle=\sum_{j}\alpha_j y_j\,K(x_j,x),$$

and the primal update \(w\leftarrow w+y_t\varphi(x_t)\), triggered when \(\operatorname{sgn} f(x_t)\ne y_t\), becomes the trivially simple dual update

$$\alpha_t\ \leftarrow\ \alpha_t+1.$$

Each mistake on example \(t\) increments that example's own coefficient by one and adds nothing anywhere else. The count \(\alpha_j\) is literally the number of times the algorithm has blundered on \(x_j\), and an example that is never a mistake keeps \(\alpha_j=0\) and drops out of the expansion entirely. This is where the sparsity and the streaming friendliness come from at once: the hypothesis is supported only on the mistaken examples, so the learner remembers only those, and each round costs one kernel evaluation against each remembered point. Compare this with the batch [[ch:solving-the-svm|dual solver]], which weighs every example against every other.

:::: {.algorithm #algo-10-1}
[Algorithm (kernel perceptron)]{.box-title}

::: algo-io
[Input]{.algo-lab} stream \((x_1,y_1),(x_2,y_2),\dots\) with \(y_t\in\{-1,+1\}\); kernel \(K\).

[Output]{.algo-lab} dual counts \(\alpha\) and the hypothesis \(f(x)=\sum_j\alpha_j y_j K(x_j,x)\).
:::

1.  Initialize \(\alpha\leftarrow 0\) and the support set \(\mathrm{SV}\leftarrow\varnothing\).
2.  For each incoming example \((x_t,y_t)\):
3.  compute the prediction \(f(x_t)=\sum_{j\in\mathrm{SV}}\alpha_j y_j K(x_j,x_t)\);
4.  if \(\operatorname{sgn} f(x_t)\ne y_t\) (treating \(\operatorname{sgn} 0=0\) as an error), set \(\alpha_t\leftarrow\alpha_t+1\) and add \(t\) to \(\mathrm{SV}\);
5.  otherwise leave \(\alpha\) unchanged and discard \(x_t\).
::::

Run as a batch procedure, the algorithm sweeps the training set repeatedly, updating on each mistake, and stops once a full sweep passes with no mistake at all, at which point the current \(\alpha\) classifies every training point correctly. The following worked example runs exactly this loop on a set that no straight line through the origin can separate, and watches the quadratic kernel resolve it.

:::::: {.example #example-10-1}
[Example (kernel perceptron on a non-separable set)]{.box-title}

::::: wex
:::: wex-setup
Four points on the line, \(x=-2,-1,+1,+2\), with the outer pair labelled \(+1\) and the inner pair \(-1\). No bias-free line \(\operatorname{sgn}(wx)\) can separate outer from inner. Use the quadratic kernel \(K(x,z)=(1+xz)^2\), whose feature map contains \(x^2\); there the outer points sit at \(x^2=4\) and the inner at \(x^2=1\), on opposite sides of a threshold. The Gram matrix is

$$K=\begin{pmatrix}25&9&1&9\\ 9&4&0&1\\ 1&0&4&9\\ 9&1&9&25\end{pmatrix}.$$

Process the four points in the fixed order shown, pass after pass, with \(\alpha=0\) initially.
::::

1.  [Sweep pass 1.]{.wex-op} With \(\alpha=0\) every prediction is \(f=0\), and \(\operatorname{sgn} 0=0\) counts as a mistake, so as the pass proceeds the predictions read \(f=[\,{+}0,{+}9,{+}1,{-}1\,]\); all four signs disagree with the labels, so every coefficient is incremented: \(\alpha=(1,1,1,1)\), \(4\) mistakes.
2.  [Sweep pass 2.]{.wex-op} Now \(f=[\,{+}24,{+}6,{+}6,{+}14\,]\). The two positives are already correct, but both inner points still read positive when they should be negative, so \(\alpha_2\) and \(\alpha_3\) each rise: \(\alpha=(1,2,2,1)\), \(2\) mistakes.
3.  [Sweep pass 3.]{.wex-op} Predictions \(f=[\,{+}14,{+}2,{+}2,{+}4\,]\); the inner points are still on the wrong side, so again \(\alpha=(1,3,3,1)\), \(2\) mistakes.
4.  [Sweep pass 4.]{.wex-op} Predictions \(f=[\,{+}4,{-}2,{-}2,{+}4\,]\): every sign now matches its label. A clean pass, no update, and the algorithm halts with \(\alpha=(1,3,3,1)\).

**Reading.** The perceptron made \(8\) updates in all and converged after four sweeps to \(\alpha=(1,3,3,1)\), a hypothesis that classifies all four points correctly even though the raw inputs are not linearly separable. The kernel did the lifting; the update rule never changed. Notice the coefficients are larger on the two inner points, which straddle the decision region and cost the learner more mistakes.
:::::

**Verification artifact.** checks/example-ch-online-example-10-1.json records the example source hash and verification scope.
::::::

## Novikoff's mistake bound {#novikoff}

The kernel perceptron never explicitly tries to make few mistakes, yet on a separable stream the number it can make is sharply limited. The classical guarantee is due to Novikoff (1962), and it is a statement purely about the margin: the harder the data are to separate, the smaller the margin, and the more mistakes the perceptron is entitled to make, with the count scaling as the inverse square of the margin. The proof is a small marvel, squeezing the number of updates between a lower and an upper bound on the length of the running weight vector.

:::: {.theorem #thm-10-1}
[Theorem (Novikoff, 1962)]{.box-title}

Let the examples \((x_1,y_1),(x_2,y_2),\dots\) all satisfy \(\|\varphi(x_t)\|\le R\), so that their feature images lie in a ball of radius \(R\) about the origin. Suppose there is a unit weight vector \(w^\ast\), \(\|w^\ast\|=1\), and a margin \(\gamma\gt 0\) with \(y_t\langle w^\ast,\varphi(x_t)\rangle\ge\gamma\) for every \(t\). Then the kernel perceptron, started from \(w_0=0\), makes at most

$$\left(\frac{R}{\gamma}\right)^2$$

updates before it ceases to err on the stream.

**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.
**Proof status.** Proved immediately below.
::::

:::::: {.proof}
[Proof]{.box-title}

Index the mistakes \(t=1,2,\dots\) and let \(w_t\) be the weight vector after the \(t\)-th update. Each update is triggered by a misclassified example, which we call \((x_{(t)},y_{(t)})\), and performs \(w_t=w_{t-1}+y_{(t)}\varphi(x_{(t)})\), which in dual terms is the increment \(\alpha_{(t)}\leftarrow\alpha_{(t)}+1\).

**Lower bound: the projection onto \(w^\ast\) grows linearly.** Project each update onto the margin vector:

$$\langle w_t,w^\ast\rangle=\langle w_{t-1},w^\ast\rangle+y_{(t)}\langle\varphi(x_{(t)}),w^\ast\rangle\ \ge\ \langle w_{t-1},w^\ast\rangle+\gamma,$$

using the margin hypothesis for the last inequality. Since \(\langle w_0,w^\ast\rangle=0\), induction gives \(\langle w_t,w^\ast\rangle\ge t\gamma\).

**Upper bound: the squared length grows at most linearly.** Expand

$$\|w_t\|^2=\|w_{t-1}\|^2+2y_{(t)}\langle w_{t-1},\varphi(x_{(t)})\rangle+\|\varphi(x_{(t)})\|^2.$$

The example was a mistake, meaning \(y_{(t)}\langle w_{t-1},\varphi(x_{(t)})\rangle\le 0\), so that middle term is not positive; and \(\|\varphi(x_{(t)})\|^2\le R^2\). Hence \(\|w_t\|^2\le\|w_{t-1}\|^2+R^2\), and by induction from \(w_0=0\), \(\|w_t\|^2\le tR^2\).

**Squeeze.** Combine the two bounds through Cauchy-Schwarz, \(\langle w_t,w^\ast\rangle\le\|w_t\|\,\|w^\ast\|=\|w_t\|\):

$$t\gamma\ \le\ \langle w_t,w^\ast\rangle\ \le\ \|w_t\|\ \le\ \sqrt{t}\,R.$$

Dividing by \(\sqrt{t}\) gives \(\sqrt{t}\,\gamma\le R\), that is \(t\le R^2/\gamma^2\). The number of updates can therefore never exceed \((R/\gamma)^2\). [\(\square\)]{.qed}
::::::

Two features of this argument deserve emphasis. It is entirely dimension-free: neither \(R\) nor \(\gamma\) refers to the dimension of the feature space, so the bound holds verbatim in the infinite-dimensional space of a Gaussian kernel. And it makes no probabilistic assumption at all, holding for any ordering of any separable stream. The margin \(\gamma\) here is exactly the geometric margin of the hard-margin support vector machine with no bias, which ties the perceptron's mistake count directly to the object the batch SVM maximizes. Shawe-Taylor and Cristianini (2004) build on precisely this to convert the update count into a generalization bound, since a hypothesis reconstructed from few mistakes is compressible.

:::::: {.example #example-10-2}
[Example (evaluating Novikoff's bound)]{.box-title}

::::: wex
:::: wex-setup
Six points in \(\mathbb{R}^2\), separable through the origin by the vertical axis \(x_1=0\): the three with \(x_1\gt 0\) are positive, the three with \(x_1\lt 0\) negative.

$$\begin{array}{c|cc|c}
& x_1 & x_2 & y\\\hline
& 1 & 3 & +1\\
& 1 & -3 & +1\\
& 5 & 0 & +1\\
& -1 & 3 & -1\\
& -1 & -3 & -1\\
& -5 & 0 & -1
\end{array}$$

Two points sit at \(x_1=\pm 1\), close to the boundary; two sit far out at \((\pm 5,0)\). Use the linear kernel.
::::

1.  [Measure the radius.]{.wex-op} The farthest points from the origin are \((\pm5,0)\), so \(R=\sqrt{25}=5\) and \(R^2=25\).
2.  [Solve for the margin.]{.wex-op} The hard-margin, no-bias solution minimizes \(\|w\|^2\) subject to \(y_i\langle w,x_i\rangle\ge 1\); it comes out at \(w^\ast\propto(1,0)\), scaled so the closest points \(x_1=\pm1\) sit at functional margin \(1\). The geometric margin is \(\gamma=1/\|w\|=1\), so \(\gamma^2=1\).
3.  [Form the bound.]{.wex-op} Novikoff promises at most \(R^2/\gamma^2=25/1=25\) updates.
4.  [Run the perceptron.]{.wex-op} Sweeping the six points in order with the linear kernel, the algorithm updates only on the two boundary points and then passes cleanly: \(2\) updates in total, with final counts nonzero only at \((1,3)\) and \((-1,3)\).

**Reading.** The guarantee, \(25\) mistakes, is honored with room to spare by the actual count of \(2\). The gap is instructive: the two far-flung points at \((\pm5,0)\) inflate \(R\) and hence the bound, yet the perceptron never errs on them, because they lie deep inside their half-spaces. Novikoff's theorem bounds the worst case over all orderings, not the typical run; it is an upper bound, and a loose one whenever the geometry is benign.
:::::

**Verification artifact.** checks/example-ch-online-example-10-2.json records the example source hash and verification scope.
::::::

### Voted and averaged perceptrons {#voted-averaged}

The plain perceptron keeps only its final weight vector, and on noisy or non-separable data that last vector may reflect whatever example happened to arrive most recently. Freund and Schapire (1999) proposed a simple and effective fix that costs nothing extra to train. Record every intermediate hypothesis \(w_1,w_2,\dots\) together with its survival time, the number of consecutive examples it classified correctly before the next update. At prediction time, let each stored hypothesis vote, weighted by its survival time, and take the sign of the weighted vote: this is the voted perceptron. A cheaper approximation replaces the vote by the survival-weighted average weight vector, the averaged perceptron, which predicts with a single \(\bar w=\sum_t c_t w_t/\sum_t c_t\). Both remain pure kernel expansions, since each \(w_t\) is one, and both markedly improve robustness on data the perceptron cannot perfectly separate, while inheriting the mistake-bound analysis of the underlying algorithm.

## The kernel adatron {#kernel-adatron}

The perceptron finds some separator; it does not seek the best one. The kernel adatron of Frieß, Cristianini, and Campbell (1998) closes that gap while keeping the online, one-coordinate-at-a-time character. Its idea is to run gradient ascent directly on the support vector dual objective, coordinate by coordinate, so that the online process converges not merely to a separator but to the maximum-margin separator itself.

Recall the hard-margin dual (with no bias, so no equality constraint survives),

$$W(\alpha)=\sum_{i=1}^{\ell}\alpha_i-\frac12\sum_{i,j=1}^{\ell}\alpha_i\alpha_j y_i y_j K(x_i,x_j),\qquad \alpha_i\ge 0.$$

Its partial derivative in one coordinate is strikingly clean:

$$\frac{\partial W}{\partial\alpha_i}=1-y_i\sum_{j=1}^{\ell}\alpha_j y_j K(x_j,x_i)=1-y_i f(x_i),$$

where \(f(x_i)=\sum_j\alpha_j y_j K(x_j,x_i)\) is the current prediction. A gradient-ascent step in coordinate \(i\) is therefore

$$\alpha_i\ \leftarrow\ \alpha_i+\eta\big(1-y_i f(x_i)\big).$$

This has an appealing reading. When the example is classified with functional margin below one, \(y_i f(x_i)\lt 1\), the derivative is positive and the coefficient grows, pulling the boundary toward the example; when it sits comfortably beyond the margin, \(y_i f(x_i)\gt 1\), the coefficient shrinks. At a fixed point every in-play example has \(y_i f(x_i)=1\), which is exactly the support vector condition. The perceptron's mistake-driven increment \(\alpha_i\leftarrow\alpha_i+1\) is the crude ancestor of this rule (Schölkopf and Smola 2002); the adatron replaces the binary \"did we err\" by the graded \"by how much is the margin violated.\"

The one subtlety is the constraint. The unconstrained step may drive \(\alpha_i\) negative, or, in the soft-margin case, above the box ceiling \(C\). Because the feasible set is the box \(0\le\alpha_i\le C\), the correct move is projected gradient ascent: take the gradient step, then project back onto the box by clipping,

$$\alpha_i\ \leftarrow\ \min\!\big(C,\ \max(0,\ \alpha_i+\eta(1-y_i f(x_i)))\big).$$

For the hard-margin problem \(C=\infty\) and the ceiling is inactive, leaving the single floor \(\alpha_i\ge 0\). Projected coordinate ascent on a concave quadratic over a box converges, and its limit is the constrained maximizer, so the adatron converges to the max-margin dual solution (Frieß, Cristianini, and Campbell 1998). This is the same box constraint that the [[ch:support-vector-machines|soft-margin SVM]] imposes, arrived at here through the projection rather than through a quadratic-program solver.

:::: {.algorithm #algo-10-2}
[Algorithm (kernel adatron)]{.box-title}

::: algo-io
[Input]{.algo-lab} examples \((x_1,y_1),\dots,(x_\ell,y_\ell)\); kernel \(K\); learning rate \(\eta\gt 0\); box ceiling \(C\) (use \(C=\infty\) for hard margin).

[Output]{.algo-lab} dual variables \(\alpha\) approximating the max-margin solution, and \(f(x)=\sum_j\alpha_j y_j K(x_j,x)\).
:::

1.  Initialize \(\alpha\leftarrow 0\).
2.  Repeat until every \(|1-y_i f(x_i)|\) with \(0\lt\alpha_i\lt C\) is below a tolerance \(\tau\):
3.  pick a coordinate \(i\) (cyclically or by largest KKT violation) and evaluate \(f(x_i)=\sum_j\alpha_j y_j K(x_j,x_i)\);
4.  form the gradient \(g_i=1-y_i f(x_i)\) and step \(\alpha_i\leftarrow\alpha_i+\eta\,g_i\);
5.  project onto the box: \(\alpha_i\leftarrow\min(C,\max(0,\alpha_i))\).
::::

:::::: {.example #example-10-3}
[Example (kernel adatron reaching the max-margin dual)]{.box-title}

::::: wex
:::: wex-setup
Four points, linear kernel, no bias, hard margin.

$$\begin{array}{c|cc|c}
& x_1 & x_2 & y\\\hline
p_1 & 1 & 2 & +1\\
p_2 & 4 & 4 & +1\\
p_3 & -2 & -1 & -1\\
p_4 & -4 & -4 & -1
\end{array}\qquad
K=\begin{pmatrix}5&12&-4&-12\\ 12&32&-12&-32\\ -4&-12&5&12\\ -12&-32&12&32\end{pmatrix}.$$

Take \(\eta=0.05\), \(\alpha=0\). The reference max-margin dual, from the quadratic program, is \(\alpha^\ast=(\tfrac19,0,\tfrac19,0)\) with margin \(\gamma=1/\|w\|=2.1213\) and optimum \(W^\ast=\tfrac19\).
::::

1.  [Sweep the coordinates.]{.wex-op} One full pass of the projected update over \(i=1,\dots,4\) already lifts the objective from \(0\) to \(W(\alpha)=0.0591\), with \(\alpha\approx(0.050,0.020,0.028,0)\); the two outer points \(p_2,p_4\) get clipped back toward zero as the closer points take over.
2.  [Watch the objective climb.]{.wex-op} The dual value increases monotonically: after \(2\) sweeps \(W=0.0906\), after \(5\) sweeps \(W=0.1101\), after \(20\) sweeps \(W=0.11109\). Projected coordinate ascent never lowers a concave objective, so each sweep is an improvement.
3.  [Converge.]{.wex-op} By \(100\) sweeps \(W=0.111111\) to six places and \(\alpha=(0.1112,0,0.1111,0)\); by \(1000\) sweeps \(\alpha=(\tfrac19,0,\tfrac19,0)\) exactly to the printed precision.
4.  [Read off the machine.]{.wex-op} The converged weights give \(w=(\tfrac13,\tfrac13)\), hence margin \(\gamma=1/\|w\|=2.1213\), matching the support vector machine's own margin.

**Reading.** The adatron, driven only by local gradient steps and a clip to the box, lands on \(\alpha^\ast=(\tfrac19,0,\tfrac19,0)\): the two margin points \(p_1,p_3\) become support vectors, the two interior points \(p_2,p_4\) are switched off. The dual value climbs to \(W^\ast=\tfrac19\) and the induced margin equals the batch SVM's exactly. The online process and the batch quadratic program reach the same optimum; only the route differs.
:::::

**Verification artifact.** checks/example-ch-online-example-10-3.json records the example source hash and verification scope.
::::::

## Online support vector regression {#online-svr}

Nothing about the online recipe is special to classification. The same stochastic-gradient view that produced the adatron produces an online regressor once we swap the loss. Kivinen, Smola, and Williamson (2004) formulate this directly in the [[ch:solving-the-svm|reproducing kernel Hilbert space]]: minimize the regularized risk \(\tfrac\lambda2\|f\|_{\mathcal H}^2+\mathbb{E}\,c(x,y,f(x))\) by stochastic gradient descent, using at round \(t\) the single-example estimate \(\tfrac\lambda2\|f\|_{\mathcal H}^2+c(x_t,y_t,f(x_t))\). The gradient of \(\|f\|_{\mathcal H}^2\) is \(2f\), and the gradient of the loss term, by the reproducing property, is \(c'(x_t,y_t,f(x_t))\,K(x_t,\cdot)\). The update is therefore

$$f_{t+1}=(1-\eta\lambda)\,f_t-\eta\,c'\big(x_t,y_t,f_t(x_t)\big)\,K(x_t,\cdot),$$

a two-part move: shrink the whole current expansion by the factor \(1-\eta\lambda\), then append one new kernel term centered on the fresh example, weighted by the loss derivative.

For regression with the \(\varepsilon\)-insensitive loss \(c=\max(0,|y-f(x)|-\varepsilon)\), the derivative is \(-\operatorname{sgn}(y_t-f_t(x_t))\) when the residual exceeds \(\varepsilon\) and zero inside the tube. So a new support vector is created, with coefficient \(\pm\eta\), exactly when the prediction lands more than \(\varepsilon\) from the target; examples the current \(f\) already fits to within \(\varepsilon\) leave the expansion untouched. This reproduces the sparsity of batch [[ch:solving-the-svm|support vector regression]], now generated one example at a time: the tube decides membership. Setting \(\varepsilon=0\) recovers an online least-absolute-deviation regressor, and swapping the hinge loss back in recovers the regularized kernel perceptron, so a single update template covers classification, regression, and novelty detection by changing only the loss derivative \(c'\).

The shrink factor \(1-\eta\lambda\) is the quiet workhorse. It is the regularization acting online, and it means the coefficient attached to an example decays geometrically with every subsequent round. An example seen long ago contributes \((1-\eta\lambda)^{k}\) times its original weight after \(k\) further rounds, so its influence fades. This built-in forgetting is what makes the method track a drifting target, and, as we now see, it is also the lever for controlling memory.

## NORMA and passive-aggressive updates {#norma-passive-aggressive}

The shrink-and-append rule above is the **NORMA** template of Kivinen, Smola, and Williamson (2004). Its decisive feature is regularized stochastic gradient: the learning rate and \(\lambda\) jointly control both the newest coefficient and the decay of every older one. For a convex loss and a bounded kernel, standard stochastic-approximation guarantees require a step schedule whose sum diverges while the sum of squared steps converges, or a carefully stated constant-step tracking objective. A constant step on a drifting stream is useful, but it is not the same theorem as convergence to a fixed population minimizer.

Passive-aggressive learning starts from a different local question: what is the smallest RKHS change that makes the current example satisfy a unit-margin constraint? For classification it solves

$$
\min_{f,\xi\ge0}\;\frac12\lVert f-f_t\rVert_{\mathcal H}^2+C\xi
\quad\text{subject to}\quad y_t f(x_t)\ge1-\xi.
$$

The solution is

$$
f_{t+1}=f_t+\tau_t y_tK(x_t,\cdot),\qquad
\tau_t=\min\!\left(C,\frac{\max(0,1-y_tf_t(x_t))}{K(x_t,x_t)}\right).
$$

It is passive when the constraint already holds and otherwise makes the smallest capped correction. The regression version replaces hinge loss by the \(\varepsilon\)-insensitive violation and uses the residual sign. Unlike NORMA, the displayed update has no global shrinkage, so regularization comes from the conservative projection, the cap \(C\), and any separate budget policy [@crammer2006pa].

Two implementation details matter. First, normalize the kernel or guard against tiny \(K(x_t,x_t)\), which otherwise creates an enormous step. Second, store a global decay multiplier for NORMA instead of rescaling every coefficient on every round; periodically fold that multiplier into the coefficients to avoid underflow. Both methods cost one kernel evaluation per retained support vector, so their time and memory remain governed by the budget problem.

## The budget problem {#budget}

There is a catch that the mistake bound hides. Novikoff's theorem caps the number of updates only when the stream is separable with a positive margin. On a noisy or non-separable stream, or one whose target drifts, the perceptron and the online SVR keep erring, and every error appends a term to the kernel expansion. The support set grows without bound, so the memory footprint and the per-round prediction cost both climb linearly with time. One can show that under nonzero minimal risk the number of support vectors indeed grows linearly with the number of examples seen (Schölkopf and Smola 2002). An online learner that must run indefinitely cannot afford an unbounded model; this is the budget problem, and it is the central practical obstacle to deploying online kernel methods.

The left panel below turns that asymptotic warning into a systems quantity: prediction cost follows the number of retained kernel terms. A hard budget flattens that curve, but the right panel shows the price. Even the cheapest maintenance rule must decide which coefficients are too small to keep, and every deletion perturbs the current hypothesis.

<figure class="viz" data-figure="online-budget" data-alt="The left panel shows an unbudgeted support set growing stepwise to nearly fifty terms while a hard budget keeps twelve. The right panel shows eight signed kernel coefficients, with small-magnitude terms marked for removal and larger terms retained."><figcaption>Budget maintenance converts unbounded support growth into fixed prediction cost, but it is an approximation step: removal is cheap because it drops small coefficients, whereas projection or merging spends more computation to preserve the RKHS function more faithfully.</figcaption></figure>

Two broad families of remedy exist, and the discussion here is deliberately at the level of strategy rather than specific update formulas. The first is forgetting through decay. The regularized update already multiplies every coefficient by \(1-\eta\lambda\) each round, so old coefficients shrink geometrically; once a coefficient falls below a threshold its term contributes negligibly and can be pruned. Truncating the expansion to a fixed time horizon, keeping only the most recent terms whose decayed weight is still material, bounds the memory at the cost of a controlled approximation, and the learning rate trades the size of the model against how far into the past the learner remembers (Kivinen, Smola, and Williamson 2004). This is well suited to non-stationary streams, where forgetting the distant past is a feature.

The second family fixes a hard budget \(B\) on the number of support vectors and maintains it actively. When a new example would push the support set past \(B\), the learner must make room, and three moves are standard. Removal discards the least useful current support vector, for instance the one of smallest coefficient or smallest contribution to the margin. Projection is subtler: rather than deleting the chosen term outright, it redistributes that term's contribution onto the remaining support vectors by projecting it onto their span in feature space, so the hypothesis changes as little as possible in RKHS norm. Merging replaces two nearby support vectors by a single synthetic one that approximates their combined effect. All three keep the model size at \(B\) forever, differing in how much accuracy they sacrifice to do so; projection is the most faithful and the most expensive, removal the cheapest and the crudest. The unifying principle is the one that has organized this whole chapter: because the hypothesis is a kernel expansion, budget maintenance is geometry in feature space, choosing which directions to keep and how to account for the ones we drop, and it connects directly to the low-rank and sparse-approximation techniques used for [[ch:large-scale-kernels|large-scale kernel machines]].

## Operational view {#summary}

Online kernel learning trades the single batch optimization for a stream of cheap, local updates, storing the hypothesis as a dual kernel expansion and never forming the full Gram matrix. The kernel perceptron makes this vivid: its update is the one-line increment \(\alpha_t\leftarrow\alpha_t+1\) on a mistake, its prediction is a kernel expansion over the mistaken examples, and Novikoff's two-sided argument bounds its total mistakes by \((R/\gamma)^2\), tying the mistake count to the very margin the [[ch:support-vector-machines|support vector machine]] maximizes and to the [[ch:learning-theory|generalization theory]] of margins. The kernel adatron upgrades the crude mistake increment to projected gradient ascent on the support vector dual, converging to the exact max-margin solution while respecting the box constraint through a clip. The same stochastic-gradient template, with the loss derivative swapped, gives online support vector regression and novelty detection, all sharing the shrink-and-append update whose decay factor performs regularization online. The price of streaming is the budget problem, the unbounded growth of the support set on hard streams, and controlling it, by forgetting through decay or by maintaining a hard budget through removal, projection, or merging, is what turns these compact updates into systems that can run forever. The next chapters put the same dual machinery to work on [[ch:ranking-and-ordinal-regression|ranking]] and on the approximations that make [[ch:large-scale-kernels|large-scale kernels]] tractable.

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

Novikoff's bound vanishes as a guarantee once separability, a positive margin, or bounded feature norm fails; a drifting stream can therefore grow the expansion indefinitely. Every budget rule changes the function, so report not only memory but also the RKHS or prediction perturbation caused by removal, projection, or merging. Evaluate online methods prequentially, before updating on each observation, and log per-round latency, support-set size, drift handling, and kernel normalization. A final batch score conceals both adaptation delay and transient failures.

## Summary and further reading {#summary-and-further-reading}

Online kernel methods replace a global Gram solve with local expansion updates. The perceptron admits a finite mistake guarantee only under bounded feature norm and positive separable margin; the adatron and regularized loss updates broaden the optimization view but do not remove support growth. A deployable learner therefore needs an explicit memory budget, maintenance rule, per-round cost, and drift policy, together with the approximation error those choices introduce. The lineage runs from the perceptron [@rosenblatt1958] and its mistake analysis [@novikoff1962] to the kernelized construction [@aizerman1964].

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} Show that the kernel perceptron's dual update follows from its primal update. Starting from \(w=\sum_j\alpha_j y_j\varphi(x_j)\) with \(w_0=0\), verify that the primal move \(w\leftarrow w+y_t\varphi(x_t)\) on a mistake is identical to \(\alpha_t\leftarrow\alpha_t+1\), and that the prediction \(\langle w,\varphi(x)\rangle\) equals \(\sum_j\alpha_j y_j K(x_j,x)\). Explain why an example never misclassified keeps \(\alpha_j=0\).
2.  [computation]{.ex-tag} Repeat the worked kernel-perceptron example with the homogeneous quadratic kernel \(K(x,z)=(xz)^2\) instead of \((1+xz)^2\), on the same four points \(x=-2,-1,1,2\) with outer label \(+1\) and inner \(-1\). Write out the \(4\times4\) Gram matrix and run the mistake-driven sweeps by hand until a clean pass. Does the algorithm still converge, and to what \(\alpha\)? Compare the number of updates with the \((1+xz)^2\) run.
3.  [proof]{.ex-tag} In Novikoff's proof, identify precisely where each of the two hypotheses is used: the margin condition \(y_t\langle w^\ast,\varphi(x_t)\rangle\ge\gamma\) and the radius condition \(\|\varphi(x_t)\|\le R\). Then show that the middle-term inequality \(y_{(t)}\langle w_{t-1},\varphi(x_{(t)})\rangle\le 0\) is exactly the statement that the update was triggered by a mistake, and explain why the bound would fail if updates were made on correctly classified points.
    Hint

    ::: hint-body
    The lower bound uses only the margin condition; the upper bound uses only the mistake condition and the radius bound. A correctly classified example has \(y_{(t)}\langle w_{t-1},\varphi(x_{(t)})\rangle\gt 0\), which would make the squared-length recursion grow faster than \(tR^2\).
    :::
4.  [computation]{.ex-tag} Evaluate Novikoff's bound on the two-point set \(x_1=(1,0)\), \(y_1=+1\) and \(x_2=(-1,0)\), \(y_2=-1\), with the linear kernel and no bias. Compute \(R\), the max-margin \(\gamma\), and the bound \(R^2/\gamma^2\), then run the perceptron and count the actual updates. Now scale both points by a factor \(a\gt 0\); show that \(R\), \(\gamma\), and the bound all respond so that \(R^2/\gamma^2\) is unchanged, and explain why the mistake count cannot depend on the overall scale.
5.  [proof]{.ex-tag} Derive the adatron gradient. Starting from \(W(\alpha)=\sum_i\alpha_i-\tfrac12\sum_{i,j}\alpha_i\alpha_j y_i y_j K(x_i,x_j)\), compute \(\partial W/\partial\alpha_i\) and confirm it equals \(1-y_i f(x_i)\) with \(f(x_i)=\sum_j\alpha_j y_j K(x_j,x_i)\). Then argue that at any interior fixed point of the projected update (one with \(0\lt\alpha_i\lt C\)) the support vector condition \(y_i f(x_i)=1\) holds.
    Hint

    ::: hint-body
    Differentiate term by term; the quadratic contributes \(-y_i\sum_j\alpha_j y_j K(x_i,x_j)\). At an interior fixed point the clip is inactive, so the gradient must vanish, giving \(1-y_i f(x_i)=0\).
    :::
6.  [proof]{.ex-tag} Show that the projected adatron update never decreases \(W\) along a single coordinate for a small enough step. Treating \(W\) as a function of \(\alpha_i\) with the other coordinates fixed, note it is a concave quadratic \(W(\alpha_i)=\alpha_i(1-y_i \tilde f)-\tfrac12 K_{ii}\alpha_i^2+\text{const}\), where \(\tilde f\) collects the other terms. Find the unconstrained maximizer in \(\alpha_i\), and explain why clipping it to \([0,C]\) still yields the constrained maximizer along that coordinate.
    Hint

    ::: hint-body
    The one-variable maximizer is \(\alpha_i^\star=\alpha_i+(1-y_i f(x_i))/K_{ii}\). A concave function on an interval attains its constrained maximum either at the unconstrained optimum or at the nearest endpoint, which is exactly what the clip returns; the choice \(\eta=1/K_{ii}\) makes one adatron step land on it.
    :::
7.  [computation]{.ex-tag} Consider the online SVR update \(f_{t+1}=(1-\eta\lambda)f_t-\eta c'(x_t,y_t,f_t(x_t))K(x_t,\cdot)\) with the \(\varepsilon\)-insensitive loss. Suppose at rounds \(1,2,3\) the residuals \(y_t-f_t(x_t)\) are \(+2,\,0,\,-2\) and \(\varepsilon=1\), \(\eta=0.5\), \(\lambda=0.1\). Track which rounds create a new support vector and write the coefficient attached to the round-1 term after rounds \(2\) and \(3\), given the geometric decay factor \(1-\eta\lambda\).
8.  [challenge]{.ex-tag} The budget problem asks us to remove one support vector from an expansion \(f=\sum_{j=1}^{B+1}\beta_j K(x_j,\cdot)\) with least damage. Formalize the projection strategy: to drop index \(r\), choose new coefficients \(\{\beta_j'\}_{j\ne r}\) minimizing \(\big\|\sum_{j\ne r}\beta_j' K(x_j,\cdot)-f\big\|_{\mathcal H}^2\). Write this as a linear least-squares problem in the remaining coefficients, identify the normal equations in terms of the Gram matrix of the kept points, and explain why removal (setting \(\beta_j'=\beta_j\) for \(j\ne r\) and dropping \(\beta_r\)) is the special case that ignores the coupling.
    Hint

    ::: hint-body
    Expand the squared RKHS norm using \(\langle K(x_i,\cdot),K(x_j,\cdot)\rangle_{\mathcal H}=K(x_i,x_j)\). The minimizer solves \(K_{\text{keep}}\,\beta'=K_{\text{keep,all}}\,\beta\), a projection of the full expansion onto the span of the retained kernel functions; plain removal keeps the old coefficients and so pays the full norm of the dropped term.
    :::
:::
