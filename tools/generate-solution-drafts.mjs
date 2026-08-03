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
  const fileExists = fs.existsSync(file);
  const original = fileExists ? fs.readFileSync(file, "utf8") : "";
  const existing = fileExists
    ? parseYaml(original)
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
  if (!fileExists) createdFiles += 1;
  existing.exercises ||= [];
  const metadataUpdates = [];
  let addedEntry = false;
  for (const match of matches) {
    const number = Number(match[1]);
    const current = existing.exercises.find((entry) => entry.number === number);
    if (current) {
      // The manuscript is canonical for exercise wording. Depth edits often
      // sharpen a prompt after its answer has been authored; keep that answer
      // and review status intact while synchronizing the metadata checked by
      // CI. This makes the generator safe to rerun after editorial revisions.
      const prompt = compact(match[3]);
      if (current.kind !== match[2] || current.prompt !== prompt) {
        metadataUpdates.push({ number, kind: match[2], prompt });
      }
      continue;
    }
    existing.exercises.push({
      number,
      kind: match[2],
      status: "draft-outline",
      prompt: compact(match[3]),
      answer: draft(match[2], match[3], chapter),
    });
    addedEntry = true;
    createdAnswers += 1;
  }
  existing.exercises.sort((a, b) => a.number - b.number);
  if (!fileExists || addedEntry) {
    fs.writeFileSync(file, stringifyYaml(existing, { lineWidth: 100 }));
  } else if (metadataUpdates.length) {
    // Preserve hand-authored answer formatting. Re-serializing an entire YAML
    // file to change one prompt creates hundreds of meaningless line-wrap
    // changes, so patch only the canonical kind/prompt fields in place.
    let updated = original;
    for (const { number, kind, prompt } of metadataUpdates) {
      const entry = new RegExp(
        `(^  - number: ${number}\\n    kind: )[^\\n]+(\\n    status: [^\\n]+\\n)    prompt: [\\s\\S]*?(?=\\n    answer:)`,
        "m",
      );
      if (!entry.test(updated)) throw new Error(`${chapter.src}: cannot patch exercise ${number}`);
      updated = updated.replace(entry, `$1${kind}$2    prompt: ${JSON.stringify(prompt)}`);
    }
    fs.writeFileSync(file, updated);
  }
}

console.log(
  "Created " + createdFiles + " solution files and " +
  createdAnswers + " answer/rubric drafts.",
);
