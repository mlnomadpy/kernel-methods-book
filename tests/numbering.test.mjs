import test from "node:test";
import assert from "node:assert/strict";
import {
  expandBookObjectReferences,
  mergeBookObjectReferences,
  numberBookObjects,
} from "../src/lib/numbering.js";

test("numbers equations, figures, and tables within a chapter", () => {
  const result = numberBookObjects(`
<h2>Objects</h2>
$$a=b$$<p>{#eq-balance}</p>
<figure class="viz" data-figure="geometry"><div></div><figcaption>A lift.</figcaption></figure>
<table><thead><tr><th>A</th></tr></thead><tbody><tr><td>B</td></tr></tbody></table><p>{#tbl-values}</p>
`, { chapterLabel: "7", chapterSlug: "objects", chapterTitle: "Objects" });

  assert.deepEqual(result.counts, { equations: 1, figures: 1, tables: 1, listings: 0 });
  assert.match(result.body, /id="eq-balance"/);
  assert.match(result.body, />\(7\.1\)<\/a>/);
  assert.match(result.body, />Figure 7\.1\.<\/a>/);
  assert.match(result.body, />Table 7\.1\.<\/a>/);
});

test("semantic references resolve across chapters with stable display numbers", () => {
  const first = numberBookObjects(
    "$$x=y$$<p>{#eq-system}</p>",
    { chapterLabel: "2", chapterSlug: "first", chapterTitle: "First" },
  );
  const second = numberBookObjects(
    "See [[eq:eq-system]].",
    { chapterLabel: "3", chapterSlug: "second", chapterTitle: "Second" },
  );
  const refs = mergeBookObjectReferences([first.refs, second.refs]);
  const body = expandBookObjectReferences(second.body, refs, "second");

  assert.match(body, /href="first\.html#eq-system">Equation 2\.1<\/a>/);
  assert.throws(
    () => expandBookObjectReferences("[[eq:eq-missing]]", refs, "second"),
    /unknown eq reference 'eq-missing'/,
  );
});

test("does not number TeX displays shown inside code", () => {
  const result = numberBookObjects(
    "<pre><code>$$not-book-math$$</code></pre>$$book-math$$",
    { chapterLabel: "1", chapterSlug: "intro", chapterTitle: "Intro" },
  );
  assert.equal(result.counts.equations, 1);
  assert.equal(result.counts.listings, 0);
  assert.match(result.body, /<code>\$\$not-book-math\$\$<\/code>/);
});

test("presents code as a numbered listing with caption and copy control", () => {
  const result = numberBookObjects(
    '<h2>Stable solve</h2><div class="sourceCode" id="cb1"><pre class="sourceCode python"><code>solve(K, y)</code></pre></div><p>{#lst-solve caption="Solve the Gram system stably"}</p>',
    { chapterLabel: "4", chapterSlug: "ridge", chapterTitle: "Ridge" },
  );
  assert.equal(result.counts.listings, 1);
  assert.match(result.body, /<figure class="code-listing" id="lst-solve"/);
  assert.match(result.body, />Listing 4\.1\.<\/a> Solve the Gram system stably/);
  assert.match(result.body, /<span class="listing-language">Python<\/span>/);
  assert.match(result.body, /class="listing-copy"/);
});
