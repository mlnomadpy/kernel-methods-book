#!/usr/bin/env node
/** Populate a clearly labelled answer/rubric draft for every exercise. */
import fs from "node:fs";
import matter from "gray-matter";
import { parse as parseYaml, stringify as stringifyYaml } from "yaml";

const book = parseYaml(fs.readFileSync("book.yml", "utf8"));
const chapters = book.parts.flatMap((part) => part.chapters);
let createdFiles = 0;
let createdAnswers = 0;

function compact(text) {
  return text
    .replace(/\s+/g, " ")
    .replace(/\[\[ch:[^\]]+\]\]/g, "the referenced chapter")
    .trim();
}

function draft(kind, prompt, chapter) {
  const task = compact(prompt);
  const context =
    "The response must use the definitions and notation of " + chapter.title +
    " and explicitly state every assumption it invokes. ";
  if (kind === "computation") {
    return (
      "Draft solution method. " + context +
      "For the task “" + task + "”, write the relevant Gram matrix, operator, " +
      "objective, or statistic before substituting values; show intermediate " +
      "quantities; then report the requested value with precision and units. " +
      "Finish with an independent check such as symmetry, positive semidefiniteness, " +
      "a residual, a limiting case, or direct substitution. Exact arithmetic and " +
      "the executable fixture remain subject to independent verification."
    );
  }
  if (kind === "proof" || kind === "challenge") {
    return (
      "Draft proof plan. " + context +
      "For “" + task + "”, restate the claim with its quantifiers, expand the " +
      "defining identity, and identify the chapter theorem or inequality that " +
      "controls the key step. Prove both any existence and uniqueness claims, " +
      "treat boundary or zero-eigenvalue cases separately, and conclude with the " +
      "exact statement requested. A challenge solution must also explain why the " +
      "hypotheses cannot simply be dropped. This draft requires line-by-line " +
      "independent mathematical verification."
    );
  }
  if (kind === "exploration" || kind === "synthesis") {
    return (
      "Assessment rubric. " + context +
      "For “" + task + "”, full credit requires: a reproducible setup and data " +
      "split; a justified kernel/model and baseline; diagnostics for conditioning " +
      "and optimization; an evaluation tied to the stated question; and a " +
      "limitations section addressing sensitivity, failure cases, and uncertainty. " +
      "Award one point for each element, with the final point requiring enough " +
      "detail for another reader to reproduce the result."
    );
  }
  return (
    "Draft concise answer. " + context +
    "For “" + task + "”, begin from the relevant definition, substitute the " +
    "objects named in the exercise, and verify each defining condition in turn. " +
    "State the resulting conclusion in one sentence and include a counterexample " +
    "or limiting case when the claim is not unconditional. This answer outline " +
    "requires independent technical verification before release."
  );
}

for (const chapter of chapters) {
  const manuscript = matter(
    fs.readFileSync("manuscript/chapters/" + chapter.src + ".md", "utf8"),
  );
  const exerciseSection = manuscript.content.match(/^## Exercises \{#exercises\}([\s\S]*)$/m)?.[1] || "";
  const matches = [...exerciseSection.matchAll(
    /^(\d+)\.\s+\[([a-z-]+)\]\{\.ex-tag\}\s+([\s\S]*?)(?=^\d+\.\s+\[[a-z-]+\]\{\.ex-tag\}|\s*$)/gm,
  )];
  if (!matches.length) continue;
  const file = "solutions/" + chapter.src + ".yml";
  const existing = fs.existsSync(file)
    ? parseYaml(fs.readFileSync(file, "utf8"))
    : {
        chapter: chapter.src,
        review_status: "draft",
        reviewer: null,
        prepared_for: {
          author: "Taha Bouhsine",
          website: "https://www.tahabouhsine.com",
          affiliation: "AzettaAI",
        },
        exercises: [],
      };
  if (!fs.existsSync(file)) createdFiles += 1;
  existing.exercises ||= [];
  for (const match of matches) {
    const number = Number(match[1]);
    if (existing.exercises.some((entry) => entry.number === number)) continue;
    existing.exercises.push({
      number,
      kind: match[2],
      status: "draft-outline",
      prompt: compact(match[3]),
      answer: draft(match[2], match[3], chapter),
    });
    createdAnswers += 1;
  }
  existing.exercises.sort((a, b) => a.number - b.number);
  fs.writeFileSync(file, stringifyYaml(existing, { lineWidth: 100 }));
}

console.log(
  "Created " + createdFiles + " solution files and " +
  createdAnswers + " answer/rubric drafts.",
);
