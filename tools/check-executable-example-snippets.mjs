#!/usr/bin/env node
/**
 * Execute every visible Python fence in an executable example as an
 * independent reader would: from the repository root, without a private
 * PYTHONPATH. A wrapper that only works from checks/ therefore fails here.
 */
import fs from "node:fs";
import { spawnSync } from "node:child_process";
import matter from "gray-matter";
import { parse as parseYaml } from "yaml";

const book = parseYaml(fs.readFileSync("book.yml", "utf8"));
const chapters = book.parts.flatMap((part) => part.chapters);
const failures = [];
let snippets = 0;

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
    blocks.push({ id, body: lines.slice(index + 1, end).join("\n") });
    index = end;
  }
  return blocks;
}

for (const chapter of chapters) {
  const parsed = matter(
    fs.readFileSync(`manuscript/chapters/${chapter.src}.md`, "utf8"),
  );
  for (const block of exampleBlocks(parsed.content)) {
    const artifactFile =
      `checks/example-${chapter.src}-${block.id.replace(/[^A-Za-z0-9_.-]/g, "-")}.json`;
    if (!fs.existsSync(artifactFile)) continue;
    const artifact = JSON.parse(fs.readFileSync(artifactFile, "utf8"));
    if (artifact.verification_status !== "executable-reference") continue;

    const python = [
      ...block.body.matchAll(/^```python\s*$\n([\s\S]*?)^```\s*$/gm),
    ].map((match) => match[1]);
    if (!python.length) {
      failures.push(`${chapter.src}#${block.id}: no Python fence`);
      continue;
    }

    for (const [position, code] of python.entries()) {
      snippets += 1;
      const env = { ...process.env };
      delete env.PYTHONPATH;
      const result = spawnSync("python3", ["-c", code], {
        cwd: process.cwd(),
        env,
        encoding: "utf8",
        timeout: 30_000,
        maxBuffer: 4 * 1024 * 1024,
      });
      if (result.error || result.status !== 0) {
        const detail = result.error?.message ||
          result.stderr?.trim().split("\n").at(-1) ||
          `exit ${result.status}`;
        failures.push(
          `${chapter.src}#${block.id}: Python fence ${position + 1} failed: ${detail}`,
        );
      }
    }
  }
}

console.log(`Self-contained executable snippets: ${snippets - failures.length}/${snippets}.`);
for (const failure of failures) console.error(`ERROR ${failure}`);
if (failures.length) process.exitCode = 1;
