#!/usr/bin/env node
/**
 * Require explicit incoming and outgoing chapter links for chapters that opt
 * into `narrative_link_policy: exact`.
 */
import fs from "node:fs";
import matter from "gray-matter";
import { parse as parseYaml } from "yaml";

const book = parseYaml(fs.readFileSync("book.yml", "utf8"));
const chapters = book.parts.flatMap((part) => part.chapters);
const validTargets = new Set(chapters.map((chapter) => chapter.slug));
const errors = [];
let enforced = 0;

for (const chapter of chapters) {
  const sourceFile = `manuscript/chapters/${chapter.src}.md`;
  const parsed = matter(fs.readFileSync(sourceFile, "utf8"));
  if (parsed.data.narrative_link_policy !== "exact") continue;
  enforced += 1;

  const links = [
    ...parsed.content.matchAll(/\[\[ch:([^|\]]+)\|[^\]]+\]\]/g),
  ].map((match) => ({ target: match[1], offset: match.index }));

  if (!links.length) {
    errors.push(`${chapter.src}: exact narrative policy requires chapter links`);
    continue;
  }
  for (const link of links) {
    if (!validTargets.has(link.target)) {
      errors.push(`${chapter.src}: unresolved chapter target ${link.target}`);
    }
  }

  const firstQuarter = parsed.content.length * 0.25;
  const finalQuarter = parsed.content.length * 0.75;
  if (!links.some((link) => link.offset <= firstQuarter)) {
    errors.push(`${chapter.src}: no explicit inherited-result link in the first quarter`);
  }
  if (!links.some((link) => link.offset >= finalQuarter)) {
    errors.push(`${chapter.src}: no explicit handoff link in the final quarter`);
  }
}

console.log(`Narrative links: ${enforced} chapter(s) under the exact-link gate.`);
if (errors.length) {
  for (const error of errors) console.error(`ERROR ${error}`);
  process.exitCode = 1;
}
