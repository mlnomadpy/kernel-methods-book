#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import { parse as parseYaml } from "yaml";

const root = path.resolve(import.meta.dirname, "..");
const chapterDir = path.join(root, "manuscript", "chapters");
const provenanceDir = path.join(root, "provenance");
const output = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(root, "checks", "provenance-coverage-audit.json");
const enforce = process.argv.includes("--enforce");

const chapters = [];
for (const filename of fs.readdirSync(chapterDir).filter((name) => name.endsWith(".md"))) {
  const id = filename.slice(0, -3);
  const manuscript = fs.readFileSync(path.join(chapterDir, filename), "utf8");
  const citationText = manuscript.replace(/```[\s\S]*?```/g, "");
  const cited = [...new Set(
    [...citationText.matchAll(/@([A-Za-z0-9_:.+-]+)/g)].map((match) => match[1]),
  )].sort();
  const provenanceFile = path.join(provenanceDir, `${id}.yml`);
  const provenance = parseYaml(fs.readFileSync(provenanceFile, "utf8"));
  const locators = (provenance.sections ?? []).flatMap((section) =>
    (section.locators ?? []).map((locator) => ({
      section: section.id,
      ...locator,
    })));

  const missing = [];
  for (const source of cited) {
    const matches = locators.filter((locator) => locator?.source === source);
    const exact = matches.filter((locator) => {
      const value = String(locator?.source_locator ?? "").trim();
      return value.length > 0 && !/\bpending\b/i.test(value);
    });
    if (!exact.length) {
      missing.push({
        source,
        reason: matches.length ? "locator-null-or-pending" : "source-not-mapped",
      });
    }
  }
  chapters.push({
    id,
    cited: cited.length,
    citedWithExactLocator: cited.length - missing.length,
    missing,
  });
}

chapters.sort((left, right) => left.id.localeCompare(right.id));
const summary = {
  chapters: chapters.length,
  citedSources: chapters.reduce((sum, chapter) => sum + chapter.cited, 0),
  citedSourcesWithExactLocator: chapters.reduce(
    (sum, chapter) => sum + chapter.citedWithExactLocator,
    0,
  ),
  missingCitedSourceLocators: chapters.reduce(
    (sum, chapter) => sum + chapter.missing.length,
    0,
  ),
  chaptersWithMissingCitedSourceLocators: chapters.filter(
    (chapter) => chapter.missing.length,
  ).length,
};

fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, `${JSON.stringify({ summary, chapters }, null, 2)}\n`);
console.log(JSON.stringify(summary, null, 2));
console.log(`Wrote ${path.relative(root, output)}`);

if (enforce && summary.missingCitedSourceLocators > 0) {
  for (const chapter of chapters.filter((entry) => entry.missing.length)) {
    console.error(
      `ERROR ${chapter.id}: ${chapter.missing.map((entry) => entry.source).join(", ")}`,
    );
  }
  process.exitCode = 1;
}
