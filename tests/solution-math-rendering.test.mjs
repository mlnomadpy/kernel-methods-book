import assert from "node:assert/strict";
import test from "node:test";

import { renderMath } from "../src/lib/book.js";
import { renderCanonicalMarkdown } from "../src/lib/manuscript.js";

test("solution-style dollar math is server-rendered by KaTeX", () => {
  const markdown = "The matrix is $K=\\begin{pmatrix}1&2\\\\2&5\\end{pmatrix}$ and $K\\succeq0$.";
  const html = renderMath(renderCanonicalMarkdown(markdown));
  assert.match(html, /class="katex"/);
  assert.doesNotMatch(html, /\$K=/);
  assert.doesNotMatch(html, /Could not convert/);
});
