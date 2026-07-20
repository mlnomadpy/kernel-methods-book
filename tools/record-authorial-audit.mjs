#!/usr/bin/env node
/** Record the commissioned AI-assisted audit without impersonating reviewers. */
import fs from "node:fs";
import { parse as parseYaml, stringify as stringifyYaml } from "yaml";
import { parse as parseBookYaml } from "yaml";

const book = parseBookYaml(fs.readFileSync("book.yml", "utf8"));
const chapters = book.parts.flatMap((part) => part.chapters);
for (const chapter of chapters) {
  const file = "reviews/" + chapter.src + ".yml";
  const review = parseYaml(fs.readFileSync(file, "utf8"));
  review.authorial_audit = {
    commissioned_by: {
      name: "Taha Bouhsine",
      role: "author",
      website: "https://www.tahabouhsine.com",
      affiliation: "AzettaAI",
    },
    performed_by: "OpenAI Codex",
    completed_on: "2026-07-19",
    scope: [
      "formal-result metadata presence",
      "chapter-template completeness",
      "citation-key integrity",
      "cross-reference integrity",
      "numerical and notebook regression suites",
    ],
    status: "ai-assisted-authorial-audit",
    satisfies_independent_technical_review: false,
    satisfies_independent_pedagogical_review: false,
  };
  fs.writeFileSync(file, stringifyYaml(review, { lineWidth: 100 }));
}
console.log("Recorded an AI-assisted authorial audit for " + chapters.length + " chapters.");
