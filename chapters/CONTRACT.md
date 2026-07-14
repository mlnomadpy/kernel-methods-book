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

## Format

- Start with exactly one `<h1>Chapter title</h1>`, then a `<p class="lead">`
  paragraph saying what the chapter does and why it matters.
- Sections `<h2>`, subsections `<h3>`, with `id` attributes (kebab-case).
- Math: KaTeX. Inline `\( ... \)`, display `$$ ... $$`. Inside math use
  `\lt` and `\gt` instead of raw `<` `>` (HTML). Blackboard/script letters,
  `\mathcal{H}`, `\langle \cdot,\cdot\rangle_{\mathcal H}` as in the course.
- Boxed environments (the only classes the CSS knows):
  `<div class="box def"><span class="box-title">Definition (p.d. kernel)</span> ... </div>`
  kinds: `def`, `thm`, `lem`, `prop`, `cor`, `ex`, `rmk`, `proof`.
  A proof ends with `<span class="qed">\(\square\)</span>`.
  Long proofs (over ~a screen) go inside
  `<details class="proof-details"><summary>Proof</summary><div class="box proof">...</div></details>`.
- Tables for comparisons; `<pre><code>` for pseudocode.
- No images, no external links except paper references as plain text.
- No em dashes in prose; use commas, colons, or restructure.

## Voice

Textbook register: precise, warm, motivated. Every section opens with why
this object is needed, not with its definition. A reader who knows linear
algebra and probability should be able to follow the whole chapter without
the slides.
