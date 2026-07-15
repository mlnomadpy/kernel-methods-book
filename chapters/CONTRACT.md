# Chapter author contract

Each chapter is one HTML **body fragment** at `chapters/src/<chNN>.body.html`.
The build wraps it in the template (sidebar, KaTeX, navigation); write only the
content. Source material: the matching `extract/<chNN>.txt` (raw slide text,
`--- slide N ---` markers).

## The job

Turn the slides into a real book chapter: connected pedagogical prose with
full explanations, not slide bullets. Motivate every object before defining
it, state results precisely, give the proofs the slides give (cleaned up and
completed where the slide compresses), work the examples, and keep every
attribution the slides make (Aronszajn 1950, Rahimi and Recht 2007, ...).
Cover **all** the material in your extract; do not skip subsections. Where a
slide is a picture you cannot see, reconstruct the idea it illustrates from
its caption/context in one or two sentences; never invent results.

Chapters authored from the two textbooks (Schölkopf-Smola 2002,
Shawe-Taylor-Cristianini 2004) rather than slides use the matching source
slices under `extract/books/slices/` as their ground truth; the same rules
apply. The depth bar is `chapters/src/ch08.body.html` (the deepest chapter):
match its density of worked examples, proofs, and motivation.

## Format

- Start with exactly one `<h1>Chapter title</h1>`, then a `<p class="lead">`
  paragraph saying what the chapter does and why it matters.
- Sections `<h2>`, subsections `<h3>`, with `id` attributes (kebab-case).
- Math: KaTeX. Inline `\( ... \)`, display `$$ ... $$`. Inside math use
  `\lt` and `\gt` instead of raw `<` `>` (HTML). Blackboard/script letters,
  `\mathcal{H}`, `\langle \cdot,\cdot\rangle_{\mathcal H}` as in the course.
- Boxed environments (the only classes the CSS knows):
  `<div class="box def"><span class="box-title">Definition (p.d. kernel)</span> ... </div>`
  kinds: `def`, `thm`, `lem`, `prop`, `cor`, `ex`, `rmk`, `proof`, `algo`.
  The build numbers def/thm/lem/prop/cor on one counter (Theorem 4.2 style),
  `ex` and `algo` on their own counters, and turns the `(annotation)` after the
  kind word into an italic attribution. Write the title as
  `<span class="box-title">Theorem (Aronszajn, 1950)</span>`; the build strips
  the kind word and keeps the parenthetical.
  A proof is just `<div class="box proof"><span class="box-title">Proof</span> ...
  <span class="qed">\(\square\)</span></div>`; the build makes it collapsible
  automatically. Proof bodies must contain no nested `<div>`.

- **Algorithm boxes** for procedures (the source's pseudocode):
  ```
  <div class="box algo"><span class="box-title">Algorithm (SMO, one step)</span>
  <div class="algo-io">
    <p class="algo-in"><span class="algo-lab">Input</span> Gram matrix \(K\), labels \(y\), penalty \(C\).</p>
    <p class="algo-out"><span class="algo-lab">Output</span> dual variables \(\alpha\).</p>
  </div>
  <ol class="algo-steps">
    <li>Select a violating pair \((i,j)\).</li>
    <li class="algo-loop">Repeat until no KKT violation exceeds \(\tau\).</li>
  </ol>
  </div>
  ```
  The procedure lives in the box; justify it in the surrounding prose, not inside.

- **Worked numeric examples**: an `ex` box whose body is a `<div class="wex">`
  with three zones, `wex-setup` (the givens as numbers), `<ol class="wex-steps">`
  (each `<li>` opens with a bold `<span class="wex-op">verb phrase.</span>`), and
  a closing `<p class="wex-take"><strong>Reading.</strong> ...</p>`. Keep them
  small: Gram matrices 3x3 to 5x5, DP tables <= 6x6, SVM duals <= 4 points. Render
  string-kernel DP tables as `<table class="dp">` (cells that match get `class="hit"`).
  **Every displayed number must be real, computed, and checkable.** For each worked
  example write `checks/<src>-exM.py`: a short numpy script that reproduces the
  setup and prints every quantity the example displays. These scripts are pure
  linear algebra (numpy solve/eig, hand DP loops), not ML training, so they run
  locally. Never type a number into a worked example that a check script did not
  print.

- Tables for comparisons; `<pre><code>` for pseudocode that is not an `algo` box.
- Real generated plots only, as `<figure class="fig"><img ...><figcaption>...`;
  no decorative images, no stock diagrams. (Real-data figures are produced by a
  Kaggle-run script in a later pass; do not fabricate a plot.) External links only
  as paper references in plain text.
- No em dashes in prose; use commas, colons, or restructure.

## Voice

Textbook register: precise, warm, motivated. Every section opens with why
this object is needed, not with its definition. A reader who knows linear
algebra and probability should be able to follow the whole chapter without
the slides.
