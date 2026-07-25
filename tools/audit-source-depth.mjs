#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";

const root = path.resolve(import.meta.dirname, "..");
const chapterDir = path.join(root, "manuscript", "chapters");
const outputPath = process.argv[2]
  ? path.resolve(process.argv[2])
  : path.join(root, "checks", "source-depth-audit.json");

const book = fs.readFileSync(path.join(root, "book.yml"), "utf8");
const orderedIds = [...book.matchAll(/^\s+- src: ([\w-]+)$/gm)]
  .map((match) => match[1])
  .filter((id) => fs.existsSync(path.join(chapterDir, `${id}.md`)));

function frontmatter(text) {
  const match = text.match(/^---\n([\s\S]*?)\n---\n/);
  return match?.[1] ?? "";
}

function listField(yaml, name) {
  const lines = yaml.split("\n");
  const start = lines.findIndex((line) => line === `${name}:`);
  if (start < 0) return [];
  const values = [];
  for (let index = start + 1; index < lines.length; index += 1) {
    const line = lines[index];
    const item = line.match(/^\s{2}-\s+(.+?)\s*$/);
    if (item) {
      values.push(item[1].replace(/^['"]|['"]$/g, ""));
      continue;
    }
    if (/^\S/.test(line)) break;
  }
  return values;
}

function scalarField(yaml, name) {
  const match = yaml.match(new RegExp(`^${name}:\\s*['"]?(.+?)['"]?\\s*$`, "m"));
  return match?.[1] ?? null;
}

function proseWords(text) {
  return text
    .replace(/^---[\s\S]*?^---$/m, "")
    .replace(/```[\s\S]*?```/g, "")
    .replace(/\$\$[\s\S]*?\$\$/g, "")
    .replace(/\\\([\s\S]*?\\\)/g, "")
    .replace(/<[^>]+>/g, " ")
    .replace(/\{[^}\n]*\}/g, " ")
    .match(/[A-Za-z][A-Za-z'-]*/g)?.length ?? 0;
}

function uniqueMatches(text, pattern, group = 1) {
  return [...new Set([...text.matchAll(pattern)].map((match) => match[group]))].sort();
}

const chapters = orderedIds.map((id, index) => {
  const file = path.join(chapterDir, `${id}.md`);
  const text = fs.readFileSync(file, "utf8");
  const yaml = frontmatter(text);
  const bibliography = listField(yaml, "bibliography");
  const citations = uniqueMatches(text, /@([A-Za-z0-9_:.+-]+)/g);
  const formalResults = (text.match(/:{3,}\s*\{\.(?:theorem|proposition|lemma|corollary)\b/g) ?? []).length;
  const proofs = (text.match(/:{3,}\s*\{\.proof\b/g) ?? []).length;
  const examples = (text.match(/:{3,}\s*\{\.example\b/g) ?? []).length;
  const sections = (text.match(/^##+\s+/gm) ?? []).length;
  const words = proseWords(text);
  const undeclaredCitations = citations.filter((key) => !bibliography.includes(key));
  const uncitedBibliography = bibliography.filter((key) => !citations.includes(key));
  const wordsPerSection = Math.round(words / Math.max(sections, 1));
  const evidenceUnits = proofs + examples;
  const flags = [];

  if (wordsPerSection < 300 && sections >= 7) flags.push("compressed");
  if (bibliography.length < 5 && sections >= 8) flags.push("thin-bibliography");
  if (formalResults >= 3 && proofs / formalResults < 0.34) flags.push("thin-proof-chain");
  if (examples === 0) flags.push("no-worked-example");
  if (uncitedBibliography.length) flags.push("frontmatter-only-sources");
  if (undeclaredCitations.length) flags.push("undeclared-citations");
  if (evidenceUnits < 2 && sections >= 8) flags.push("thin-evidence");

  return {
    order: index,
    id,
    title: scalarField(yaml, "title"),
    tier: scalarField(yaml, "tier"),
    words,
    sections,
    wordsPerSection,
    bibliography: bibliography.length,
    citations: citations.length,
    formalResults,
    proofs,
    examples,
    uncitedBibliography,
    undeclaredCitations,
    flags,
  };
});

const summary = {
  chapters: chapters.length,
  flagged: chapters.filter((chapter) => chapter.flags.length).length,
  compressed: chapters.filter((chapter) => chapter.flags.includes("compressed")).length,
  thinBibliography: chapters.filter((chapter) => chapter.flags.includes("thin-bibliography")).length,
  thinProofChain: chapters.filter((chapter) => chapter.flags.includes("thin-proof-chain")).length,
  noWorkedExample: chapters.filter((chapter) => chapter.flags.includes("no-worked-example")).length,
  frontmatterOnlySources: chapters.filter((chapter) =>
    chapter.flags.includes("frontmatter-only-sources")).length,
  undeclaredCitations: chapters.filter((chapter) =>
    chapter.flags.includes("undeclared-citations")).length,
};

fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, `${JSON.stringify({ summary, chapters }, null, 2)}\n`);

console.log(`Source-depth audit: ${chapters.length} chapters`);
console.log(JSON.stringify(summary, null, 2));
console.log(`Wrote ${path.relative(root, outputPath)}`);
