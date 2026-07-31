import fs from "node:fs";
import matter from "gray-matter";
import test from "node:test";
import assert from "node:assert/strict";
import { renderCanonicalMarkdown } from "../src/lib/manuscript.js";

test("kernel quiz table preserves inline-math placeholders as five columns", () => {
  const chapter = matter(fs.readFileSync("manuscript/chapters/ch01.md", "utf8")).content;
  const section = chapter.match(
    /^## Worked examples: a kernel quiz \{#a-kernel-quiz\}([\s\S]*?)(?=^### Series,)/m,
  )?.[0];

  assert.ok(section, "kernel quiz section is present");
  const html = renderCanonicalMarkdown(section);
  assert.doesNotMatch(html, /KBMATH(?:INLINE|BLOCK)/);
  assert.doesNotMatch(html, /data-kb-math/);
  assert.doesNotMatch(html, /&lt;\/span&gt;/);

  const rows = [...html.matchAll(/<tr\b[\s\S]*?<\/tr>/g)].map((match) => match[0]);
  assert.equal(rows.length, 14, "one header row and thirteen quiz rows");
  for (const row of rows) {
    assert.equal(
      [...row.matchAll(/<(?:th|td)\b/g)].length,
      5,
      "every quiz row has exactly five cells",
    );
  }
});
