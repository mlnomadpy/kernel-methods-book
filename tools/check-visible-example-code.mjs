#!/usr/bin/env node
/**
 * Enforce visible code for executable worked examples.
 *
 * Chapters opt into the migration-safe release gate with
 * `example_code_policy: visible-for-executable` in front matter. Passing
 * `--audit-all` also reports executable examples in chapters that have not
 * opted in yet, without weakening the gate for chapters that have.
 */
import fs from "node:fs";
import matter from "gray-matter";
import { parse as parseYaml } from "yaml";

const auditAll = process.argv.includes("--audit-all");
const book = parseYaml(fs.readFileSync("book.yml", "utf8"));
const chapters = book.parts.flatMap((part) => part.chapters);
const errors = [];
const pending = [];
let executableExamples = 0;
let visibleExecutableExamples = 0;
let enforcedChapters = 0;

function exampleBlocks(source) {
  const lines = source.split("\n");
  const blocks = [];
  for (let index = 0; index < lines.length; index += 1) {
    const opening = lines[index].match(/^(:{3,}) \{\.example\b([^}]*)\}/);
    if (!opening) continue;
    const fence = opening[1];
    const id = lines[index].match(/#([A-Za-z][A-Za-z0-9_.:-]*)/)?.[1];
    let end = index + 1;
    while (end < lines.length && lines[end].trim() !== fence) end += 1;
    blocks.push({
      id,
      line: index + 1,
      malformed: !id || end >= lines.length,
      body: lines.slice(index + 1, end).join("\n"),
    });
    index = end;
  }
  return blocks;
}

for (const chapter of chapters) {
  const sourceFile = `manuscript/chapters/${chapter.src}.md`;
  const parsed = matter(fs.readFileSync(sourceFile, "utf8"));
  const enforced = parsed.data.example_code_policy === "visible-for-executable";
  if (enforced) enforcedChapters += 1;

  for (const block of exampleBlocks(parsed.content)) {
    if (block.malformed) {
      errors.push(`${chapter.src}: malformed example near line ${block.line}`);
      continue;
    }
    const artifactFile =
      `checks/example-${chapter.src}-${block.id.replace(/[^A-Za-z0-9_.-]/g, "-")}.json`;
    if (!fs.existsSync(artifactFile)) continue;
    const artifact = JSON.parse(fs.readFileSync(artifactFile, "utf8"));
    if (artifact.verification_status !== "executable-reference") continue;

    executableExamples += 1;
    const hasVisibleCode = /^```[A-Za-z0-9_+-]+\s*$/m.test(block.body);
    if (hasVisibleCode) {
      visibleExecutableExamples += 1;
    } else if (enforced) {
      errors.push(
        `${chapter.src}#${block.id}: executable example has no visible code fence`,
      );
    } else if (auditAll) {
      pending.push(`${chapter.src}#${block.id}`);
    }
  }
}

console.log(
  `Visible example code: ${visibleExecutableExamples}/${executableExamples} executable examples; ` +
  `${enforcedChapters} chapter(s) under the release gate.`,
);
if (auditAll && pending.length) {
  console.log(`Migration backlog (${pending.length}):`);
  for (const item of pending) console.log(`- ${item}`);
}
if (errors.length) {
  for (const error of errors) console.error(`ERROR ${error}`);
  process.exitCode = 1;
}
