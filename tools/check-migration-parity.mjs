import fs from "node:fs";
import matter from "gray-matter";
import { parse as parseYaml } from "yaml";
import { renderCanonicalMarkdown } from "../src/lib/manuscript.js";

const baseline = JSON.parse(fs.readFileSync("migration-baseline.json", "utf8"));
const book = parseYaml(fs.readFileSync("book.yml", "utf8"));
const approved = fs.existsSync("migration-approved-changes.yml")
  ? parseYaml(fs.readFileSync("migration-approved-changes.yml", "utf8"))
  : {};
const chapters = book.parts.flatMap((part) => part.chapters);
const errors = [];

function metric(body, markdown) {
  return {
    h1: body.match(/<h1>([\s\S]*?)<\/h1>/)?.[1].replace(/<[^>]+>/g, "") || "",
    headings: [...body.matchAll(/<h([23]) id="([^"]+)"/g)].map((m) => `${m[1]}:${m[2]}`),
    statements: (body.match(/<div class="box (?:def|thm|lem|prop|cor)"/g) || []).length,
    examples: (body.match(/<div class="box ex"/g) || []).length,
    algorithms: (body.match(/<div class="box algo"/g) || []).length,
    proofs: (body.match(/<div class="box proof"/g) || []).length,
    exercises: (body.match(/class="ex-tag"/g) || []).length,
    widgets: (body.match(/data-widget=/g) || []).length,
    chrefs: (markdown.match(/\[\[ch:/g) || []).length,
  };
}

const order = chapters.map((chapter) => chapter.src);
const legacyOrder = order.filter((src) => baseline.chapterOrder.includes(src));
if (
  JSON.stringify(legacyOrder) !== JSON.stringify(baseline.chapterOrder) &&
  !approved._meta?.allow_order_change
) {
  errors.push("legacy chapter order changed");
}
for (const chapter of chapters) {
  const parsed = matter(fs.readFileSync(`manuscript/chapters/${chapter.src}.md`, "utf8"));
  const actual = metric(renderCanonicalMarkdown(parsed.content), parsed.content);
  const expected = baseline.chapters[chapter.src];
  if (!expected) continue; // reviewed additions have no legacy HTML baseline
  const approvedMetrics = new Set([
    ...(approved._all?.metrics || []),
    ...(approved[chapter.src]?.metrics || []),
  ]);
  for (const key of Object.keys(expected)) {
    if (JSON.stringify(actual[key]) !== JSON.stringify(expected[key])) {
      if (!approvedMetrics.has(key)) {
        errors.push(`${chapter.src}: ${key} changed (${JSON.stringify(expected[key])} -> ${JSON.stringify(actual[key])})`);
      }
    }
  }
}

if (errors.length) {
  errors.forEach((error) => console.error(`ERROR ${error}`));
  process.exitCode = 1;
} else {
  console.log(`Migration parity passed for ${baseline.chapterOrder.length} legacy chapters; ${chapters.length - baseline.chapterOrder.length} canonical additions detected.`);
}
