# Canonical chapter authoring contract

Each chapter is one Pandoc Markdown file at `manuscript/chapters/<src>.md`.
Markdown is canonical: do not create a parallel HTML authoring source. The
Astro site, PDF, EPUB, search data, and indexes are derived from these files.

## Required frontmatter

Every file declares `id`, `slug`, `title`, `part`, `order`, `tier`,
`prerequisites`, three to six `objectives`, `review_status`, `reviewers`,
`provenance`, `verification_date`, and `bibliography`. Values must agree with
`book.yml`; bibliography keys must exist in `bibliography.bib`.

## Structure and links

- Start with one `#` title and one raw `<p class="lead">...</p>` motivation.
- Give every `##` and `###` heading an explicit kebab-case anchor.
- Link chapters with `[[ch:slug]]` or `[[ch:slug|custom text]]`.
- Cite sources with Pandoc citations such as `[@aronszajn1950]`. Narrative
  author-year text is retained in migrated chapters until its review pass.
- Use `\(...\)` for inline math and `$$...$$` for display math. Avoid raw angle
  brackets inside math and do not use em dashes in prose.

## Semantic containers

Use Pandoc fenced divs. Formal results and numbered examples/algorithms require
stable explicit ids; never renumber an existing id when moving content.

```markdown
::: {.theorem #thm-aronszajn}
[Theorem (Aronszajn, 1950)]{.box-title}

State assumptions and conclusion, then identify whether the proof is complete,
sketched, cited, heuristic, or omitted.
:::

::: {.proof}
[Proof]{.box-title}

Proof body. [\(\square\)]{.qed}
:::
```

Supported types are `definition`, `theorem`, `lemma`, `proposition`,
`corollary`, `algorithm`, `example`, `remark`, `proof`, `hint`, `exercise`, and
`solution`. Algorithms state inputs, outputs, stopping rules, computational
complexity, and conditioning concerns. A numerical example declares its check
script in metadata and every displayed value must be reproduced by that script.

## Exercises and review

End every chapter with `## Exercises {#exercises}` inside an `.exercises`
container. Labels are exactly `warm-up`, `computation`, `proof`, `exploration`,
`challenge`, or `synthesis`. Every exercise receives an answer or rubric in the
public solutions companion.

Chapter status advances through `draft`, `technical review`, `pedagogical
review`, `copy edit`, and `verified`. Do not fill reviewer names, approval dates,
source locators, or permissions speculatively. The corresponding files under
`reviews/` and `provenance/` are release records, not placeholders to bypass.

Run `npm run check` before proposing a change. The check suite blocks broken
anchors, routes, references, metadata, numerical fixtures, web links, migration
drift, and high/critical production advisories.
