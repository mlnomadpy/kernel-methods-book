---
id: ch-strings2
slug: efficient-string-and-tree-kernels
title: Efficient String and Tree Kernels
part: VII · Designing Kernels for Data
order: 22
tier: advanced
prerequisites:
  - string-kernels
objectives:
  - >-
    Explain the central definitions and claims in Efficient String and Tree
    Kernels.
  - Apply the chapter's principal methods and interpret their outputs.
  - >-
    State the assumptions behind formal results and connect them to earlier
    chapters.
review_status: draft
reviewers:
  technical: null
  pedagogical: null
  specialist: null
provenance: provenance/ch-strings2.yml
verification_date: null
bibliography:
  - watkins2000
  - lodhi2002
  - shawe2004
  - leslie2002
  - leslie2004
  - vishwanathan2003
  - collins2002
  - haussler1999
---
# Efficient String and Tree Kernels

<p class="lead">A gap-weighted string kernel lives in a feature space with one coordinate for every pattern the alphabet can spell: \(|\Sigma|^p\) of them for patterns of length \(p\), and infinitely many for the all-subsequences kernel. Writing even one such feature vector down is out of the question, yet a classifier needs the inner product between two of them, and needs it fast enough to fill an entire kernel matrix. The escape is a single structural fact: a pattern shared by two strings either avoids the last character of one or ends on it, and this split, applied to every prefix, telescopes the exponential sum into a small table over pairs of prefixes. This chapter turns that observation into dynamic programs for the all-subsequences, fixed-length, and gap-weighted kernels of [[ch:string-kernels|the companion chapter]], into a trie that makes the spectrum and mismatch kernels almost linear in the input length, and, applied bottom up from the leaves, into kernels between trees. We give each recurrence with its derivation, a worked table, and its cost, then place them all inside Haussler's convolution-kernel framework, the common ancestor of string, tree, and graph kernels alike.</p>

## From feature counts to dynamic programming {#from-counts-to-dp}

What makes a kernel over exponentially many patterns computable at all is one recursion on prefixes, and the whole chapter is variations on it. Setting it up needs a little notation. Fix a finite alphabet \(\Sigma\). For a string \(s\) we write \(|s|\) for its length, \(s(i{:}j)\) for the substring \(s_i\cdots s_j\), and \(s(\mathbf{i})\) for the subsequence picked out by an index tuple \(\mathbf{i}=(i_1\lt\cdots\lt i_k)\); the span of that subsequence is \(l(\mathbf{i})=i_k-i_1+1\), the number of characters of \(s\) it reaches across. The feature maps of the previous chapter are all of the form \"count, or weight, the occurrences of a pattern \(u\)\", and the kernel is the inner product

$$\kappa(s,t)=\sum_{u}\varphi_u(s)\,\varphi_u(t),$$

a sum over a pattern set that is \(|\Sigma|^p\) for length-\(p\) patterns and infinite for the all-subsequences kernel. The reason any of this is tractable is that the sum telescopes: a common pattern in \(s\) and \(t\) either avoids the last character of \(s\) or ends on it, and this two-case split, applied to every prefix, replaces the exponential sum by a table with one cell per pair of prefixes. Dynamic programming is the systematic filling of that table, and the rest of the chapter is a catalogue of the recurrences it uses. Following Watkins (2000) and Lodhi et al. (2002), we start with the most permissive feature set, all subsequences at once, because its recurrence is the cleanest instance of the telescoping idea.

## The all-subsequences kernel {#all-subsequences}

The richest subsequence feature map indexes coordinates by *every* finite string and counts occurrences as a gapped subsequence.

:::: {.definition #def-22-1}
[Definition (all-subsequences kernel)]{.box-title}

The feature space is indexed by \(u\in\Sigma^\ast\), the set of all finite strings including the empty string \(\varepsilon\), with embedding

$$\varphi_u(s)=\big|\{\mathbf{i}:u=s(\mathbf{i})\}\big|,$$

the number of ways \(u\) occurs as a subsequence of \(s\). The kernel is \(\kappa(s,t)=\sum_{u\in\Sigma^\ast}\varphi_u(s)\varphi_u(t)\), the number of common subsequences of \(s\) and \(t\), counted with multiplicity and including \(\varepsilon\).
::::

To evaluate it, consider extending \(s\) by one symbol \(a\) to form \(sa\). A common subsequence of \(sa\) and \(t\) either does not use the appended \(a\), in which case it is already a common subsequence of \(s\) and \(t\), or it ends in \(a\) matched to some occurrence of \(a\) in \(t\), say at position \(k\), and the part before it is a common subsequence of \(s\) and the prefix \(t(1{:}k-1)\). Summing over all occurrences of \(a\) in \(t\) gives the recurrence

$$\kappa(\varepsilon,t)=1,\qquad \kappa(sa,t)=\kappa(s,t)+\!\!\sum_{k:\,t_k=a}\!\!\kappa\big(s,\,t(1{:}k-1)\big).$$

Writing \(D[i][j]=\kappa(s(1{:}i),t(1{:}j))\) turns this into a table filled row by row. The inner sum over matching positions can be precomputed into a running array so that each row costs \(O(|t|)\), giving the algorithm of Shawe-Taylor and Cristianini (2004).

:::: {.algorithm #algo-22-1}
[Algorithm (all-subsequences kernel)]{.box-title}

::: algo-io
[Input]{.algo-lab} strings \(s\) of length \(n\), \(t\) of length \(m\).

[Output]{.algo-lab} \(\kappa(s,t)=\)number of common subsequences \(=D[n][m]\).
:::

1.  Initialise \(D[0][j]=1\) for \(j=0,\dots,m\) and \(D[i][0]=1\) for \(i=0,\dots,n\) (only \(\varepsilon\) matches an empty prefix).
2.  For each \(i=1,\dots,n\): scan \(t\) once to build \(P[k]=\sum_{k'\le k,\ t_{k'}=s_i}D[i-1][k'-1]\), the cumulative contribution of matches of \(s_i\).
3.  For each \(k=1,\dots,m\): set \(D[i][k]=D[i-1][k]+P[k]\).
4.  Return \(D[n][m]\).
::::

Because the array \(P\) collapses the inner sum, each of the \(n\) rows costs \(O(m)\), so the whole kernel is computed in \(O(|s|\,|t|)\) time and \(O(|t|)\) working memory, an exponential saving over enumerating subsequences. The following table makes the recurrence concrete.

:::::: {.example #example-22-1}
[Example (all-subsequences kernel on two short strings)]{.box-title}

::::: wex
::: wex-setup
Take \(s=\)\"gatta\" and \(t=\)\"cata\". Cells where \(s_i=t_j\) are shaded; these are the positions that trigger the inner sum. Base row and column are all \(1\).
:::

::: tablewrap
  \(D\)   \(\varepsilon\)   c   a   t   a
  ------------------------------------- ------------------------------------- --- --- --- ----
  \(\varepsilon\)   1                                     1   1   1   1
  g                                     1                                     1   1   1   1
  a                                     1                                     1   2   2   3
  t                                     1                                     1   2   4   5
  t                                     1                                     1   2   6   7
  a                                     1                                     1   3   7   14
:::

1.  [Fill the base.]{.wex-op} The empty subsequence is common to every pair of prefixes, so the \(\varepsilon\) row and column are \(1\). The row for \"g\" stays \(1\), since \"g\" never appears in \(t\) and contributes no inner-sum term.
2.  [Trigger a match.]{.wex-op} At row \"a\", column \"a\" (\(j=2\)), the recurrence gives \(D=D_{\text{above}}+D[i-1][1]=1+1=2\); at the second \"a\" (\(j=4\)) both earlier occurrences contribute, \(1+1+1=3\).
3.  [Read the intermediate value.]{.wex-op} After four rows, \(D[4][4]=7\), so \(\kappa(\text{"gatt"},\text{"cata"})=7\): the seven common subsequences are \(\varepsilon\), \"a\", \"t\", \"t\", \"at\", \"at\", \"att\" counted with multiplicity.
4.  [Append the last character.]{.wex-op} Extending \(s\) to \"gatta\" adds row \"a\", and the appended \"a\" matched against the two \"a\"s of \(t\) lifts the corner to \(D[5][4]=7+1+6=14\).

**Reading.** The kernel value \(\kappa(\text{"gatta"},\text{"cata"})=14\) is read straight off the bottom-right cell, and the whole table cost \(5\times4\) constant-time updates, matching the \(O(|s|\,|t|)\) bound.
:::::

**Verification artifact.** checks/example-ch-strings2-example-22-1.json records the example source hash and verification scope.
::::::

## The fixed-length subsequences kernel {#fixed-length}

The all-subsequences kernel mixes patterns of every length, and long ones can swamp short ones. Restricting to subsequences of a fixed length \(p\) gives a more controllable feature map, at the cost of an extra recursion. The feature map is \(\varphi_u(s)=|\{\mathbf{i}:u=s(\mathbf{i})\}|\) for \(u\in\Sigma^p\), and repeating the last-character split now has to track the remaining length as well, since matching the final symbol consumes one of the \(p\) positions. This produces a recursion over both prefixes and lengths (Shawe-Taylor and Cristianini 2004).

:::: {.algorithm #algo-22-2}
[Algorithm (fixed-length subsequences kernel)]{.box-title}

::: algo-io
[Input]{.algo-lab} strings \(s,t\), target length \(p\).

[Output]{.algo-lab} \(\kappa_p(s,t)\), the inner product of the length-\(p\) subsequence counts.
:::

1.  Base cases: \(\kappa_0(s,t)=1\) for all \(s,t\), and \(\kappa_q(s,\varepsilon)=0\) for \(q\gt0\).
2.  For each level \(q=1,\dots,p\) and each pair of prefixes, apply

$$\kappa_q(sa,t)=\kappa_q(s,t)+\!\!\sum_{k:\,t_k=a}\!\!\kappa_{q-1}\big(s,\,t(1{:}k-1)\big),$$

    storing the previous level's table \(\kappa_{q-1}\) as it goes.
3.  Return \(\kappa_p(s,t)\).
::::

The double loop over lengths and prefixes evaluates \(p\) tables of size \(|s|\times|t|\), so the kernel costs \(O(p\,|s|\,|t|)\). A weighted blend \(\sum_{l=1}^{p}a_l\,\kappa_l(s,t)\) with weights \(a_l\ge0\) is obtained at no extra asymptotic cost by accumulating the intermediate levels, since every \(\kappa_l\) is computed on the way to \(\kappa_p\). This fixed-length recursion is the scaffold on which the gap-weighted kernel is built next, the only change being that positions now carry a decay weight.

## The gap-weighted subsequences kernel {#gap-weighted}

The kernel most often meant by \"the string kernel\", after Lodhi et al. (2002), weights each occurrence of a length-\(p\) subsequence by \(\lambda^{l(\mathbf{i})}\), where \(0\lt\lambda\le1\) is a decay factor and \(l(\mathbf{i})\) is the span. Tightly packed occurrences (few gaps) are rewarded, spread-out ones are penalised, and as \(\lambda\to0\) the kernel degenerates to the \(p\)-spectrum kernel of contiguous substrings. The feature map is

$$\varphi_u(s)=\sum_{\mathbf{i}:\,u=s(\mathbf{i})}\lambda^{l(\mathbf{i})},\qquad u\in\Sigma^p.$$

A direct evaluation via the fixed-length recursion would repeatedly recompute gap weights and cost \(O(p\,|s|^2|t|^2)\). The efficient route, following the previous chapter's suffix idea, introduces an auxiliary *suffix kernel* that only counts subsequences whose final matched symbol is the last character of each string.

:::: {.definition #def-22-2}
[Definition (gap-weighted suffix kernel)]{.box-title}

For length \(p\), the suffix kernel \(\kappa^S_p(s,t)\) sums \(\lambda^{l(\mathbf{i})+l(\mathbf{j})}\) over pairs of length-\(p\) index tuples with \(s(\mathbf{i})=t(\mathbf{j})\) whose last indices are \(|s|\) and \(|t|\) respectively. Then the full kernel recovers every occurrence by letting the endpoints range over all prefixes,

$$\kappa_p(s,t)=\sum_{i=1}^{|s|}\sum_{j=1}^{|t|}\kappa^S_p\big(s(1{:}i),\,t(1{:}j)\big).$$
::::

The suffix kernel is nonzero only when the two strings end in the same symbol. Splitting the pairs of subsequences by where their penultimate symbols fall leads, for \(a=b\), to

$$\kappa^S_p(sa,tb)=\lambda^2\!\!\sum_{i=1}^{|s|}\sum_{j=1}^{|t|}\lambda^{\,|s|-i+|t|-j}\,\kappa^S_{p-1}\big(s(1{:}i),t(1{:}j)\big),$$

and \(0\) when \(a\ne b\). The nested geometric sum is exactly what makes a naive pass quadratic in each length. The trick of Lodhi et al. (2002) is to precompute it as an intermediate table

$$\mathrm{DP}_p(k,l)=\sum_{i=1}^{k}\sum_{j=1}^{l}\lambda^{\,k-i+l-j}\,\kappa^S_{p-1}\big(s(1{:}i),t(1{:}j)\big),$$

which satisfies a two-dimensional inclusion-exclusion recurrence that adds one row and column at a time. Working it out, the overlap of the shifted sums is corrected by a single subtracted term.

:::: {.algorithm #algo-22-3}
[Algorithm (gap-weighted subsequences kernel, dynamic program)]{.box-title}

::: algo-io
[Input]{.algo-lab} strings \(s,t\), length \(p\), decay \(\lambda\in(0,1]\).

[Output]{.algo-lab} \(\kappa_p(s,t)\), the gap-weighted subsequence kernel.
:::

1.  Set \(\kappa^S_1(s(1{:}i),t(1{:}j))=[\,s_i=t_j\,]\,\lambda^2\) for all \(i,j\), the level-1 suffix table.
2.  For each level \(q=2,\dots,p\), fill the auxiliary table by

$$\mathrm{DP}_q(k,l)=\kappa^S_{q-1}(k,l)+\lambda\,\mathrm{DP}_q(k-1,l)+\lambda\,\mathrm{DP}_q(k,l-1)-\lambda^2\,\mathrm{DP}_q(k-1,l-1),$$

    with \(\mathrm{DP}_q(0,l)=\mathrm{DP}_q(k,0)=0\).
3.  Recover the next suffix table: \(\kappa^S_q(k,l)=[\,s_k=t_l\,]\,\lambda^2\,\mathrm{DP}_q(k-1,l-1)\).
4.  Return \(\kappa_p(s,t)=\sum_{k,l}\kappa^S_p(k,l)\).
::::

<figure class="viz" data-widget="dp-fill">

<figcaption>The table is the chapter's exact recursion filling in evaluation order for any two words you type: matched letters seed \(\mathrm{DPS}\) entries in accent, and each cell combines its left, upper, and diagonal neighbors with the stated \(\lambda\) weights. Setting cat against car with \(\lambda=0.5\) and \(p=2\) reproduces the worked example: \(K_2=\lambda^4=0.0625\), normalized \(0.4444\).</figcaption>
</figure>

Each level fills one \(|s|\times|t|\) table with constant work per cell, so the kernel costs \(O(p\,|s|\,|t|)\) time, the headline result of the section. The subtracted term \(-\lambda^2\mathrm{DP}_q(k-1,l-1)\) is pure inclusion-exclusion: the two shifted tables \(\lambda\,\mathrm{DP}_q(k-1,l)\) and \(\lambda\,\mathrm{DP}_q(k,l-1)\) both include the block up to \((k-1,l-1)\), so it is counted twice and must be removed once. We fill the tables by hand for a two-letter difference.

:::::: {.example #example-22-2}
[Example (gap-weighted DP table for \"cat\" and \"car\", \(p=2\))]{.box-title}

::::: wex
::: wex-setup
Length \(p=2\), decay \(\lambda\). The level-1 suffix table has \(\lambda^2\) at the two matches \((\text{c},\text{c})\) and \((\text{a},\text{a})\) and \(0\) elsewhere. The auxiliary table \(\mathrm{DP}_2(k,l)\) below is built from it; shaded cells are those where \(s_k=t_l\).
:::

::: tablewrap
  \(\mathrm{DP}_2\)   c                                      a                                      r
  -------------------------------------- -------------------------------------- -------------------------------------- --------------------------------------
  c                                      \(\lambda^2\)   \(\lambda^3\)   \(\lambda^4\)
  a                                      \(\lambda^3\)   \(\lambda^2+\lambda^4\)   \(\lambda^3+\lambda^5\)
  t                                      \(\lambda^4\)   \(\lambda^3+\lambda^5\)   \(\lambda^4+\lambda^6\)
:::

1.  [Seed the corner.]{.wex-op} \(\mathrm{DP}_2(1,1)=\kappa^S_1(1,1)=\lambda^2\), since \"c\" matches \"c\"; the first row and column then propagate it by the \(\lambda\)-shift, giving \(\lambda^3,\lambda^4\) and \(\lambda^3,\lambda^4\).
2.  [Apply inclusion-exclusion.]{.wex-op} At \((\text{a},\text{a})\), \(\mathrm{DP}_2=\lambda^2+\lambda\!\cdot\!\lambda^3+\lambda\!\cdot\!\lambda^3-\lambda^2\!\cdot\!\lambda^2=\lambda^2+\lambda^4\); the doubled block \(\lambda^4\) is removed exactly once.
3.  [Extract the suffix kernel.]{.wex-op} Only \((\text{a},\text{a})\) is a match with a nonzero predecessor: \(\kappa^S_2(2,2)=\lambda^2\,\mathrm{DP}_2(1,1)=\lambda^4\). Every other matched cell has \(\mathrm{DP}_2(k-1,l-1)=0\).
4.  [Sum the suffix table.]{.wex-op} \(\kappa_2(\text{"cat"},\text{"car"})=\sum_{k,l}\kappa^S_2(k,l)=\lambda^4\): the lone shared length-2 subsequence is \"ca\", spanning two characters in each word.

**Reading.** The self-kernels are \(\kappa_2(\text{"cat"},\text{"cat"})=\kappa_2(\text{"car"},\text{"car"})=2\lambda^4+\lambda^6\), so the normalised kernel is \(\hat\kappa=\lambda^4/(2\lambda^4+\lambda^6)=(2+\lambda^2)^{-1}\). At \(\lambda=\tfrac12\) this is \(0.0625/0.140625=0.4444\), independent of any long-range structure the two three-letter words do not have.
:::::

**Verification artifact.** checks/example-ch-strings2-example-22-2.json records the example source hash and verification scope.
::::::

### Variants: character weightings, gap counts, and soft matching {#gap-variants}

The DP recurrence is a template, and small edits to its lines produce a family of related kernels without changing the \(O(p\,|s|\,|t|)\) cost (Shawe-Taylor and Cristianini 2004). A *character-weighting* kernel lets each skipped or matched symbol \(u\) carry its own decay \(\lambda_u\), replacing the uniform \(\lambda\) by symbol-dependent factors in the recurrence

$$\mathrm{DP}_p(k,l)=\kappa^S_{p-1}(k,l)+\lambda_{t_l}\mathrm{DP}_p(k,l-1)+\lambda_{s_k}\mathrm{DP}_p(k-1,l)-\lambda_{s_k}\lambda_{t_l}\mathrm{DP}_p(k-1,l-1),$$

useful when some symbols, such as low-information residues, should count for less. A *gap-number* weighting charges only for the gaps rather than the whole span, obtained by setting \(\lambda=1\) on the matched positions and adjusting the correction term. Most importantly, a *soft-matching* or substitution kernel replaces the hard test \([s_k=t_l]\) by an entry \(A_{s_k t_l}\) of a symbol similarity matrix \(A\), so that near-synonymous symbols (mutating base pairs in DNA, confusable characters) contribute partial matches. For this to define a valid kernel \(A\) must itself be positive semidefinite, since it is exactly the Gram matrix of the single-character strings at \(p=1\); with that assumption the suffix recurrence becomes

$$\kappa^S_p(sa,tb)=\lambda^2 A_{ab}\!\!\sum_{i,j}\lambda^{\,|s|-i+|t|-j}\kappa^S_{p-1}\big(s(1{:}i),t(1{:}j)\big),$$

and the same dynamic program runs unchanged. Numeric sequences fall in scope too: with a soft-matching \(A\) that compares reals, the gap-weighted kernel applies to time series and not just to symbolic text.

## Beyond dynamic programming: trie-based kernels {#tries}

Dynamic programming pays \(O(|s|\,|t|)\) per kernel evaluation, which is expensive when one string is compared against a whole database. For the spectrum and mismatch kernels, whose features are contiguous substrings, a different data structure wins: the trie, a \"retrieval tree\" whose root-to-node paths spell out strings. A complete trie of depth \(p\) has one node per string of length up to \(p\), and the idea is to push the length-\(p\) substrings of \(s\) and of \(t\) down the tree together, so that two substrings meet at a leaf exactly when they are equal. Only paths that both strings actually populate are ever created, and the cost is charged to those paths rather than to the product of the lengths (Leslie et al. 2002, Vishwanathan and Smola 2003).

:::: {.algorithm #algo-22-4}
[Algorithm (trie-based \(p\)-spectrum kernel)]{.box-title}

::: algo-io
[Input]{.algo-lab} strings \(s,t\), substring length \(p\).

[Output]{.algo-lab} \(\kappa_p(s,t)\), the \(p\)-spectrum kernel, in the global accumulator Kern.
:::

1.  Attach to the root the lists \(L_s(\varepsilon)=\{(s(i{:}i+p-1),0)\}\) and \(L_t(\varepsilon)\) of all length-\(p\) substrings of each string, each tagged with a depth index \(0\); set Kern \(=0\) and call processnode\((\varepsilon)\).
2.  At a node \(v\) of depth \(=p\): add \(|L_s(v)|\cdot|L_t(v)|\) to Kern (all substrings reaching this leaf are equal).
3.  Otherwise, if both lists are nonempty, extend each substring by its next symbol \(u_{i+1}\), moving \((u,i)\) into the child list \(L_s(v\,u_{i+1})\), and likewise for \(t\).
4.  Recurse into processnode\((va)\) for each symbol \(a\in\Sigma\), then discard the subtree.
::::

Each of the \(|s|-p+1\) substrings of \(s\) descends one level per recursion step and is touched \(O(p)\) times, and the leaf products accumulate the shared-substring counts. Evaluating a full row of the kernel matrix of \(s\) against strings \(t^{(1)},\dots,t^{(\ell)}\) costs

$$O\Big(p\,\ell\,\textstyle\sum_{i=1}^{\ell}|t^{(i)}|\Big),$$

because the trie for \(s\) is built once and reused. Since only \(p\,|\Sigma|\) nodes are ever held in memory at a time (a path plus the children being expanded), the method is far lighter than the \(|s|\,|t|\) dynamic-programming table, and as a by-product it yields every lower-order spectrum along the way, so a blended kernel \(\sum_i a_i\kappa_i\) comes for free by indexing Kern by depth (Vishwanathan and Smola 2003).

The same trie carries the *mismatch* kernel of Leslie et al. (2002, 2004). Now each substring descends not only along its own symbols but also along symbols that differ, provided the running number of mismatches, stored as a third component of each list entry, does not exceed a budget \(m\). A length-\(p\) substring can therefore end up in as many as \(\binom{p}{m}(|\Sigma|-1)^m\) leaf lists, one for each allowed error pattern, and the leaf products count \((p,m)\)-neighbouring substrings as matches. Weighting the mismatches by a cost matrix and thresholding the total cost gives a further generalisation. A *restricted gap-weighted* kernel lives on the same trie: allowing up to \(m\) gaps, each of the \(|s|-p-m+1\) substrings spawns \(\binom{p+m-1}{m-1}\) leaf entries, and the leaf computation reweights them by their recorded gap counts, at overall cost

$$O\big((|s|+|t|)\,(p+m)\,\tbinom{p+m}{m}\big).$$

When \(m\) is small this beats the full dynamic program and approximates it well for small \(\lambda\); when \(m\) grows large the dynamic program is again preferable, so the two computational styles are complementary rather than competing.

## Kernels for trees {#tree-kernels}

The dynamic-programming idea is not tied to the linear order of a string. Any object built by combining simpler objects of the same type, above all a tree, admits a kernel evaluated by a recursion that proceeds bottom up from the leaves. Trees arise from parsing natural language, from biological taxonomy, and from XML and program syntax, and a kernel between them lets the same classifiers operate on parse forests (Collins and Duffy 2002). We take a tree to be a directed acyclic graph in which every node but the root has in-degree one; edges point away from the root, \(d^+(v)\) is the out-degree of \(v\), and \(\tau(v)\) is the *complete subtree* rooted at \(v\), all of \(v\)'s descendants. Two notions of \"subtree\" give two kernels.

::: {.definition #def-22-3}
[Definition (co-rooted and general subtrees)]{.box-title}

A *co-rooted subtree* of \(T\) is obtained by deleting, from some internal nodes, all the complete subtrees hanging off their children; it keeps the root of \(T\) and, whenever it keeps a node, keeps all that node's siblings. A *general subtree* is any co-rooted subtree of some complete subtree \(\tau(v)\). A tree is *proper* if it has at least one edge.
:::

Before the kernel, a simpler count fixes the recursive style. The number \(N(T)\) of proper co-rooted subtrees satisfies \(N(\tau(v))=0\) at a leaf, because there are no proper subtrees, and at an internal node each child \(v_i\) offers either one of its \(N(\tau(v_i))\) co-rooted subtrees or the option of being left as a leaf, so

$$N(T)=\prod_{i=1}^{d^+(r(T))}\big(N(\tau(\mathrm{ch}_i(r(T))))+1\big),$$

with the \"\(+1\)\" for the leaf option. This same product, with a matched pair of trees, is the co-rooted subtree kernel.

::: {.definition #def-22-4}
[Definition (co-rooted subtree kernel)]{.box-title}

The feature space is indexed by all proper trees with \(\varphi^r_S(T)=1\) if \(S\) is a co-rooted subtree of \(T\) and \(0\) otherwise, so \(\kappa_r(T_1,T_2)=\sum_S\varphi^r_S(T_1)\varphi^r_S(T_2)\) counts the co-rooted subtrees common to \(T_1\) and \(T_2\).
:::

If the two roots have different out-degrees, or either tree is a single node, no proper co-rooted subtree is shared and \(\kappa_r=0\). Otherwise a common co-rooted subtree is any independent choice, at each of the \(d^+\) matched children, of a shared co-rooted subtree of that child or the leaf option, giving

$$\kappa_r(T_1,T_2)=\prod_{i=1}^{d^+(r(T_1))}\Big(\kappa_r\big(\tau(\mathrm{ch}_i(r(T_1))),\tau(\mathrm{ch}_i(r(T_2)))\big)+1\Big).$$

The recursion visits each node at most once, so \(\kappa_r\) costs \(O(\min(|T_1|,|T_2|))\); a mismatch in degree or, for labelled trees, in label prunes the subtree at once and only speeds it up.

:::::: {.example #example-22-3}
[Example (co-rooted and all-subtree counts on two tiny trees)]{.box-title}

::::: wex
::: wex-setup
Let \(S\) be a root with two children, a leaf and a \"cherry\" (a node with two leaf children), so \(|S|=5\). Let \(T\) be a root with two leaf children, \(|T|=3\). Both roots have out-degree \(2\).
:::

::: tablewrap
  node pair                                                                      degrees                                \(\kappa_r\)
  ------------------------------------------------------------------------------ -------------------------------------- --------------------------------------
  leaf vs leaf                                                                   \(0,0\)   0
  cherry vs \(T\)                                 \(2,2\)   1
  \(S\) vs \(T\)   \(2,2\)   1
:::

1.  [Match the roots.]{.wex-op} Both roots have out-degree \(2\), so \(\kappa_r(S,T)=(\kappa_r(\text{leaf},\text{leaf})+1)(\kappa_r(\text{cherry},\text{leaf})+1)\).
2.  [Evaluate the children.]{.wex-op} The first factor is \((0+1)=1\); the second pairs the cherry (degree \(2\)) against a leaf (degree \(0\)), degrees differ, so \(\kappa_r=0\) and the factor is \((0+1)=1\).
3.  [Take the product.]{.wex-op} \(\kappa_r(S,T)=1\cdot1=1\): the single shared co-rooted subtree is the root with two leaf children.
4.  [Count self-similarities.]{.wex-op} \(\kappa_r(S,S)=N(S)=(0+1)(N(\text{cherry})+1)=(1)(1+1)=2\), while \(\kappa_r(T,T)=N(T)=1\).

**Reading.** Summing the co-rooted kernel over all node pairs gives the all-subtree kernel \(\kappa(S,T)=\kappa_r(S,T)+\kappa_r(\text{cherry},T)=1+1=2\): the cherry pattern is shared once as the whole of \(T\) matched to \(S\)'s root and once as \(T\) matched to \(S\)'s internal cherry.
:::::

**Verification artifact.** checks/example-ch-strings2-example-22-3.json records the example source hash and verification scope.
::::::

The all-subtree kernel promotes the feature set from co-rooted subtrees to all subtrees, and its value can be assembled from the co-rooted kernel by the identity \(\kappa(T_1,T_2)=\sum_{v_1\in T_1,\,v_2\in T_2}\kappa_r(\tau(v_1),\tau(v_2))\), since any subtree is co-rooted in the complete subtree at some node. A direct recursion is cheaper than that double sum. Partitioning the subtrees into those co-rooted with both roots and those sitting inside a child, with an inclusion-exclusion correction for the subtrees counted in two children, gives the dynamic program of Shawe-Taylor and Cristianini (2004).

:::: {.algorithm #algo-22-5}
[Algorithm (all-subtree kernel)]{.box-title}

::: algo-io
[Input]{.algo-lab} unlabelled trees \(T_1,T_2\), nodes ordered so each parent follows its children.

[Output]{.algo-lab} \(\kappa(T_1,T_2)=\mathrm{DP}(n_1,n_2)\), the all-subtree kernel.
:::

1.  Fill the co-rooted table: \(\mathrm{DP}_r(i_1,i_2)=0\) if the out-degrees differ or either node is a leaf, else \(\prod_{k}\big(\mathrm{DP}_r(\mathrm{ch}_k(i_1),\mathrm{ch}_k(i_2))+1\big)\).
2.  Fill the all-subtree table: at internal matched pairs set \(\mathrm{DP}(i_1,i_2)=\mathrm{DP}_r(i_1,i_2)\), then add \(\sum_{j_1}\mathrm{DP}(\mathrm{ch}_{j_1}(i_1),i_2)\) and \(\sum_{j_2}\mathrm{DP}(i_1,\mathrm{ch}_{j_2}(i_2))\), and subtract \(\sum_{j_1,j_2}\mathrm{DP}(\mathrm{ch}_{j_1}(i_1),\mathrm{ch}_{j_2}(i_2))\).
3.  Return \(\mathrm{DP}(n_1,n_2)\).
::::

The two tables are indexed by pairs of nodes and each cell does work proportional to the product of the out-degrees, so the kernel costs \(O\big(|T_1|\,|T_2|\,d_{\max}^2\big)\) with \(d_{\max}\) the largest out-degree. Labelled trees are handled by adding a label-equality test to the degree test, which for sparse labellings can be exploited to build small per-label tables and sum them, avoiding the full quadratic table.

## The unifying view: convolution kernels {#convolution}

Every kernel in this chapter has the same shape: decompose each object into parts, compare the parts with a base kernel, and sum over the allowed decompositions. Haussler (1999) abstracted this into the *convolution* or \(R\)-kernel. A decomposition structure \(R\) relates an object \(x\) to the tuples \(((x_1,\kappa_1),\dots,(x_d,\kappa_d))\) of parts, each carrying its own kernel, into which \(x\) can be split; the associated kernel is

$$\kappa_R(x,z)=\sum_{\bar x\in R^{-1}(x)}\ \sum_{\bar z\in R^{-1}(z)}\ [\,T(\bar x)=T(\bar z)\,]\prod_{i=1}^{|T(\bar x)|}\kappa_i(x_i,z_i),$$

where the type match \([\,T(\bar x)=T(\bar z)\,]\) ensures only decompositions of the same shape are compared. This is a genuine kernel, an inner product in a suitable feature space (Shawe-Taylor and Cristianini 2004), and it specialises to everything above. Taking the parts to be co-rooted subtrees recovers the co-rooted subtree kernel and taking them to be all subtrees recovers the all-subtree kernel; taking the parts to be the length-\(p\) subsequences with weight \(\lambda^{l(\mathbf{i})}\kappa_0\) recovers the gap-weighted kernel; taking them to be distinct attribute subsets recovers the ANOVA kernel of [[ch:kernels-for-text|the text-kernel chapter]]; and indexing the sub-kernels by the edges of a graph along paths recovers the walk-based [[ch:graph-kernels|graph kernels]]. The framework thus stitches the string and tree kernels here to the ANOVA and graph kernels elsewhere, and to the marginalization view of [[ch:generative-and-marginalization-kernels|generative kernels]], all as one family of sums of products over substructures; the open question it leaves is when such a sum, though exponentially large, admits an efficient dynamic program, which is exactly what the earlier sections answer case by case.

## Summary {#summary}

The feature spaces of string and tree kernels are astronomically large, but their inner products are cheap because a common pattern either avoids or ends on the last symbol, which telescopes the exponential sum into a table over prefixes. The all-subsequences kernel costs \(O(|s|\,|t|)\), the fixed-length and gap-weighted kernels \(O(p\,|s|\,|t|)\) through a suffix recursion with an inclusion-exclusion auxiliary table, and small edits to that table yield character-weighted, gap-counting, and soft-matching variants at the same cost. For contiguous-substring features the trie beats dynamic programming, evaluating the spectrum and mismatch kernels in time governed by the populated paths rather than the product of the lengths. Trees inherit the same recursive style bottom up: the co-rooted subtree kernel is a product over matched children in \(O(\min(|T_1|,|T_2|))\), and the all-subtree kernel a dynamic program in \(O(|T_1|\,|T_2|\,d_{\max}^2)\). Haussler's convolution kernels reveal all of these, together with the ANOVA and graph kernels, as one construction: sum a product of part comparisons over the decompositions of a structured object. The following chapters push the same recursive computation onto [[ch:graph-kernels|graphs]] and onto kernels read from [[ch:generative-and-marginalization-kernels|generative models]].

::: {.exercises}
## Common mistakes and practical implications {#common-mistakes-and-practical-implications}

For **Efficient String and Tree Kernels**, do not apply a displayed formula without checking its domain, statistical assumptions, and numerical conditioning. Avoid selecting kernels or hyperparameters on test data, and do not interpret an optimization residual as a generalization guarantee. When the method is computational, report preprocessing, kernel parameters, regularization, solver tolerance, condition diagnostics, runtime, and a non-kernel baseline. When the result is theoretical, distinguish sufficient conditions from necessary ones and finite-sample claims from asymptotic statements.

## Summary and further reading {#summary-and-further-reading}

This chapter established explain the central definitions and claims in Efficient String and Tree Kernels; Apply the chapter's principal methods and interpret their outputs; State the assumptions behind formal results and connect them to earlier chapters. Revisit the assumptions attached to each formal result before transferring it to a new setting. For primary and extended treatments, consult [@watkins2000], [@lodhi2002], [@shawe2004].

## Exercises {#exercises}

1.  [warm-up]{.ex-tag} Using the all-subsequences recurrence \(\kappa(sa,t)=\kappa(s,t)+\sum_{k:\,t_k=a}\kappa(s,t(1{:}k-1))\), fill the table for \(s=\)\"bar\" and \(t=\)\"bat\" by hand and read off \(\kappa(\text{"bar"},\text{"bat"})\). List the common subsequences explicitly and check that their count, empty subsequence included, matches the corner cell.
2.  [computation]{.ex-tag} For the gap-weighted kernel of length \(p=2\), fill the auxiliary table \(\mathrm{DP}_2\) for \(s=\)\"bat\" and \(t=\)\"bar\" and show that \(\kappa_2(\text{"bat"},\text{"bar"})=\lambda^4\), the shared subsequence being \"ba\". Then compute the normalised kernel and confirm it equals \((2+\lambda^2)^{-1}\), the same value the text finds for \"cat\" and \"car\".
    Hint

    ::: hint-body
    The level-1 suffix table has \(\lambda^2\) at \((\text{b},\text{b})\) and \((\text{a},\text{a})\). Only the match at \((\text{a},\text{a})\) has a nonzero predecessor \(\mathrm{DP}_2(1,1)=\lambda^2\), so \(\kappa^S_2=\lambda^2\cdot\lambda^2=\lambda^4\); the self-kernel is \(2\lambda^4+\lambda^6\).
    :::
3.  [computation]{.ex-tag} A complete trie of depth \(p\) over an alphabet of size \(|\Sigma|\) is used for the \(p\)-spectrum kernel. Explain why at most \(p\,|\Sigma|\) nodes need be held in memory at any moment, and why the leaf products \(|L_s(v)|\cdot|L_t(v)|\) sum to the correct spectrum inner product. How does the cost of evaluating one string against \(\ell\) database strings compare with \(\ell\) independent dynamic-programming evaluations?
4.  [proof]{.ex-tag} Derive the inclusion-exclusion recurrence for \(\mathrm{DP}_p(k,l)\) from its definition \(\mathrm{DP}_p(k,l)=\sum_{i\le k,\,j\le l}\lambda^{k-i+l-j}\kappa^S_{p-1}(s(1{:}i),t(1{:}j))\). Show that the two shifted tables \(\lambda\,\mathrm{DP}_p(k-1,l)\) and \(\lambda\,\mathrm{DP}_p(k,l-1)\) overlap on the block indexed up to \((k-1,l-1)\), so that the term \(-\lambda^2\mathrm{DP}_p(k-1,l-1)\) corrects the double count, leaving the fresh row-and-column contribution \(\kappa^S_{p-1}(k,l)\).
    Hint

    ::: hint-body
    Write \(\mathrm{DP}_p(k,l)\) as the sum over the last row \(i=k\), the last column \(j=l\), and the interior block \(i\lt k,\,j\lt l\). The interior block is \(\lambda^2\mathrm{DP}_p(k-1,l-1)\); the last row and column, minus their shared corner, give the two \(\lambda\)-shifted terms.
    :::
5.  [proof]{.ex-tag} Prove the co-rooted recursion \(\kappa_r(T_1,T_2)=\prod_i(\kappa_r(\tau(\mathrm{ch}_i(r(T_1))),\tau(\mathrm{ch}_i(r(T_2))))+1)\) when the two roots share out-degree \(d^+\). Argue that a co-rooted subtree common to both trees is exactly an independent choice at each matched child of a common co-rooted subtree of that child or the leaf option, and that this is why the empty (improper) choice contributes the \(+1\).
    Hint

    ::: hint-body
    The feature \(\varphi^r_S\) factors over the children: \(S\) is a co-rooted subtree of \(T\) iff each child slot of \(S\) is either a leaf or a co-rooted subtree of the matching child of \(T\). Summing the product of features over all such \(S\) turns into a product of sums, one per child.
    :::
6.  [computation]{.ex-tag} For the trees \(S\) (root with a leaf and a cherry) and \(T\) (a cherry) of the worked example, verify by direct enumeration that the all-subtree kernel \(\kappa(S,T)=2\) using the identity \(\kappa(T_1,T_2)=\sum_{v_1,v_2}\kappa_r(\tau(v_1),\tau(v_2))\). List every node pair whose co-rooted kernel is nonzero and confirm the sum. Then compute \(\kappa(S,S)\).
    Hint

    ::: hint-body
    Only pairs of proper complete subtrees with equal root degrees contribute. In \(S\) the proper complete subtrees are \(S\) itself and the cherry; in \(T\) only \(T\). Both pairs give \(\kappa_r=1\), so \(\kappa(S,T)=2\); for \(\kappa(S,S)\) enumerate all pairs of \(S\)'s proper complete subtrees.
    :::
7.  [exploration]{.ex-tag} The gap-weighted kernel interpolates between two limits as \(\lambda\) varies. Argue that as \(\lambda\to0\) the length-\(p\) gap-weighted kernel tends to the \(p\)-spectrum kernel of contiguous substrings, since a subsequence with any gap carries a strictly higher power of \(\lambda\) and vanishes faster. What does the kernel tend to, up to weighting, as \(\lambda\to1\)? Relate your answer to the fixed-length subsequences kernel of the text.
8.  [challenge]{.ex-tag} Cast the gap-weighted \(k\)-subsequences kernel as a Haussler convolution kernel: exhibit the decomposition structure \(R\) whose parts are the pairs \((u,\lambda^{l(\mathbf{i})}\kappa_0)\) over index tuples \(\mathbf{i}\in I_k\) with \(u=s(\mathbf{i})\), where \(\kappa_0\) returns \(1\) on identical substrings and \(0\) otherwise. Verify that the associated \(R\)-kernel \(\kappa_R(s,t)\) reproduces \(\sum_u\varphi_u(s)\varphi_u(t)\) with \(\varphi_u(s)=\sum_{\mathbf{i}:u=s(\mathbf{i})}\lambda^{l(\mathbf{i})}\), and explain why the type-match indicator in the \(R\)-kernel is what enforces \(u=u'\) in the product.
    Hint

    ::: hint-body
    Each decomposition of \(s\) selects one occurrence \(\mathbf{i}\) and emits the single part \((s(\mathbf{i}),\lambda^{l(\mathbf{i})}\kappa_0)\). The \(R\)-kernel sums \(\lambda^{l(\mathbf{i})}\lambda^{l(\mathbf{j})}\kappa_0(s(\mathbf{i}),t(\mathbf{j}))\) over all pairs, and \(\kappa_0\) is nonzero only when the substrings coincide, giving \(\sum_u\varphi_u(s)\varphi_u(t)\).
    :::
:::
