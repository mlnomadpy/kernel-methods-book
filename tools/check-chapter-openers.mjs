#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { parse as parseYaml } from "yaml";

const root = process.cwd();
const pdfPath = path.resolve(
  root,
  process.argv[2] || "release/kernels-the-geometry-of-learning.pdf",
);

if (!fs.existsSync(pdfPath)) {
  console.error(`ERROR missing publication PDF: ${pdfPath}`);
  process.exit(1);
}

function normalize(value) {
  return value
    .normalize("NFKD")
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim()
    .toLowerCase();
}

const book = parseYaml(fs.readFileSync(path.join(root, "book.yml"), "utf8"));
const chapters = book.parts.flatMap((part) => part.chapters);
const pdfText = execFileSync("pdftotext", ["-layout", pdfPath, "-"], {
  cwd: root,
  encoding: "utf8",
  maxBuffer: 64 * 1024 * 1024,
});
const pages = pdfText.split("\f").map((page) => ({
  text: normalize(page),
  lines: page.split(/\r?\n/).map(normalize).filter(Boolean),
}));
const openerPages = new Map();
const errors = [];

for (const [index, chapter] of chapters.entries()) {
  const label = chapter.src === "ch-prelim" ? "reference chapter" : `chapter ${index}`;
  const normalizedLabel = normalize(label);
  const normalizedTitle = normalize(chapter.title);
  const matches = [];

  for (let pageIndex = 0; pageIndex < pages.length; pageIndex += 1) {
    // The opener label is a standalone line. Requiring line equality prevents
    // prose references such as “see Chapter 12” from masquerading as openers.
    if (pages[pageIndex].lines.includes(normalizedLabel) && pages[pageIndex].text.includes(normalizedTitle)) {
      matches.push(pageIndex + 1);
    }
  }

  if (matches.length !== 1) {
    errors.push(`${chapter.src}: expected one page containing "${label}" and "${chapter.title}", found ${matches.length}${matches.length ? ` (${matches.join(", ")})` : ""}`);
    continue;
  }

  const page = matches[0];
  if (openerPages.has(page)) {
    errors.push(`${chapter.src}: opener page ${page} is already assigned to ${openerPages.get(page)}`);
  } else {
    openerPages.set(page, chapter.src);
  }
}

if (errors.length) {
  for (const error of errors) console.error(`ERROR ${error}`);
  process.exit(1);
}

console.log(`Chapter opener PDF audit passed: ${chapters.length} chapter numbers and titles are co-located exactly once.`);
