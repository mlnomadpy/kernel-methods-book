#!/usr/bin/env node
/**
 * Refresh only source hashes and numeric-literal inventories.
 *
 * Verification status, semantic check bindings, scopes, and review metadata
 * are intentionally preserved. This makes the operation safe after an
 * editorial change without re-inferring evidence from example ordinals.
 */
import crypto from "node:crypto";
import fs from "node:fs";
import matter from "gray-matter";
import { parse as parseYaml } from "yaml";

const book = parseYaml(fs.readFileSync("book.yml", "utf8"));
const chapters = book.parts.flatMap((part) => part.chapters);
let refreshed = 0;

function digest(text) {
  return crypto.createHash("sha256").update(text).digest("hex");
}

function numericLiterals(text) {
  return [...text.matchAll(/(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?/gi)]
    .map((match) => match[0])
    .filter((value, position, values) => values.indexOf(value) === position)
    .slice(0, 100);
}

for (const chapter of chapters) {
  const sourceFile = `manuscript/chapters/${chapter.src}.md`;
  const lines = matter(fs.readFileSync(sourceFile, "utf8")).content.split("\n");
  for (let index = 0; index < lines.length; index += 1) {
    const opening = lines[index].match(/^(:{3,}) \{\.example\b([^}]*)\}/);
    if (!opening) continue;
    const fence = opening[1];
    const id = lines[index].match(/#([A-Za-z][A-Za-z0-9_.:-]*)/)?.[1];
    let end = index + 1;
    while (end < lines.length && lines[end].trim() !== fence) end += 1;
    if (!id || end >= lines.length) {
      throw new Error(`${chapter.src}: malformed example near line ${index + 1}`);
    }

    const clean = lines.slice(index, end + 1);
    while (clean.length > 1 && clean[clean.length - 2] === "") {
      clean.splice(clean.length - 2, 1);
    }
    const artifactFile =
      `checks/example-${chapter.src}-${id.replace(/[^A-Za-z0-9_.-]/g, "-")}.json`;
    const artifact = JSON.parse(fs.readFileSync(artifactFile, "utf8"));
    artifact.source_sha256 = digest(clean.join("\n"));
    artifact.numeric_literals = numericLiterals(clean.slice(1, -1).join("\n"));
    fs.writeFileSync(artifactFile, `${JSON.stringify(artifact, null, 2)}\n`);
    refreshed += 1;
    index = end;
  }
}

console.log(`Refreshed metadata for ${refreshed} example artifacts.`);
