#!/usr/bin/env node
import crypto from "node:crypto";
import fs from "node:fs";
import matter from "gray-matter";
import { parse as parseYaml } from "yaml";

const release = process.env.RELEASE_VERIFIED === "1";
const book = parseYaml(fs.readFileSync("book.yml", "utf8"));
const chapters = book.parts.flatMap((part) => part.chapters);
const errors = [];
let total = 0;
let executable = 0;
let pendingNumerical = 0;

function digest(text) {
  return crypto.createHash("sha256").update(text).digest("hex");
}

for (const chapter of chapters) {
  const sourceFile = "manuscript/chapters/" + chapter.src + ".md";
  const source = matter(fs.readFileSync(sourceFile, "utf8")).content;
  const lines = source.split("\n");
  for (let index = 0; index < lines.length; index++) {
    const opening = lines[index].match(/^(:{3,}) \{\.example\b([^}]*)\}/);
    if (!opening) continue;
    total += 1;
    const fence = opening[1];
    const id = lines[index].match(/#([A-Za-z][A-Za-z0-9_.:-]*)/)?.[1];
    let end = index + 1;
    while (end < lines.length && lines[end].trim() !== fence) end += 1;
    if (!id || end >= lines.length) {
      errors.push(chapter.src + ": malformed example near line " + (index + 1));
      continue;
    }
    const block = lines.slice(index, end + 1);
    const reference = block.join("\n").match(/checks\/(example-[A-Za-z0-9_.-]+\.json)/);
    if (!reference || !fs.existsSync("checks/" + reference[1])) {
      errors.push(chapter.src + "#" + id + ": missing example artifact");
      continue;
    }
    const artifact = JSON.parse(fs.readFileSync("checks/" + reference[1], "utf8"));
    const clean = block.filter((line) => !line.startsWith("**Verification artifact.**"));
    while (clean.length > 1 && clean[clean.length - 2] === "") clean.splice(clean.length - 2, 1);
    if (artifact.chapter !== chapter.src || artifact.anchor !== id) {
      errors.push(chapter.src + "#" + id + ": artifact identity mismatch");
    }
    if (artifact.source_sha256 !== digest(clean.join("\n"))) {
      errors.push(chapter.src + "#" + id + ": artifact source hash is stale");
    }
    if (artifact.verification_status === "executable-reference") executable += 1;
    if (artifact.verification_status === "executable-reference") {
      if (!artifact.executable_checks?.length) {
        errors.push(chapter.src + "#" + id + ": executable artifact has no check");
      }
      for (const check of artifact.executable_checks || []) {
        if (!fs.existsSync(check)) errors.push(chapter.src + "#" + id + ": missing " + check);
      }
    }
    if (artifact.verification_status === "literal-audit") pendingNumerical += 1;
    if (release && artifact.verification_status === "literal-audit") {
      errors.push(chapter.src + "#" + id + ": numerical example lacks an executable check");
    }
    index = end;
  }
}

console.log(
  "Example artifacts: " + total + "/" + total +
  "; executable: " + executable +
  "; pending executable numerical checks: " + pendingNumerical + ".",
);
if (errors.length) {
  for (const error of errors) console.error("ERROR " + error);
  process.exitCode = 1;
}
