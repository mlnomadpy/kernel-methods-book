#!/usr/bin/env node
/** Declare a stable verification artifact for every example container. */
import crypto from "node:crypto";
import fs from "node:fs";
import matter from "gray-matter";
import { parse as parseYaml } from "yaml";

const book = parseYaml(fs.readFileSync("book.yml", "utf8"));
const chapters = book.parts.flatMap((part) => part.chapters);
const overridesPath = "checks/example-verification-overrides.json";
const overrides = fs.existsSync(overridesPath)
  ? JSON.parse(fs.readFileSync(overridesPath, "utf8"))
  : {};
let total = 0;
let executable = 0;
let literal = 0;
let conceptual = 0;

function digest(text) {
  return crypto.createHash("sha256").update(text).digest("hex");
}

for (const chapter of chapters) {
  const sourceFile = "manuscript/chapters/" + chapter.src + ".md";
  const parsed = matter(fs.readFileSync(sourceFile, "utf8"));
  const lines = parsed.content.split("\n");
  const output = [];
  let ordinal = 0;

  for (let index = 0; index < lines.length; index++) {
    let line = lines[index];
    const opening = line.match(/^(:{3,}) \{\.example\b([^}]*)\}/);
    if (!opening) {
      output.push(line);
      continue;
    }
    ordinal += 1;
    const fence = opening[1];
    let id = line.match(/#([A-Za-z][A-Za-z0-9_.:-]*)/)?.[1];
    if (!id) {
      id = "example-" + chapter.src + "-" + ordinal;
      line = line.replace(/\}$/, " #" + id + "}");
    }
    let end = index + 1;
    while (end < lines.length && lines[end].trim() !== fence) end += 1;
    if (end >= lines.length) throw new Error(sourceFile + ": unclosed example near line " + (index + 1));
    const block = [line, ...lines.slice(index + 1, end + 1)];
    const cleanBlock = block.filter((item) => !item.startsWith("**Verification artifact.**"));
    while (cleanBlock.length > 1 && cleanBlock[cleanBlock.length - 2] === "") {
      cleanBlock.splice(cleanBlock.length - 2, 1);
    }
    const body = cleanBlock.slice(1, -1).join("\n");
    const filename = "example-" + chapter.src + "-" + id.replace(/[^A-Za-z0-9_.-]/g, "-") + ".json";
    const previousArtifact = fs.existsSync("checks/" + filename)
      ? JSON.parse(fs.readFileSync("checks/" + filename, "utf8"))
      : null;
    const existingChecks = [...body.matchAll(/checks\/([A-Za-z0-9_.-]+\.(?:py|json))/g)]
      .map((match) => "checks/" + match[1])
      .filter((value, pos, all) => all.indexOf(value) === pos);
    for (const check of previousArtifact?.executable_checks || []) {
      if (!existingChecks.includes(check)) existingChecks.push(check);
    }
    const conventionalCheck = "checks/" + chapter.src + "-ex" + ordinal + ".py";
    if (fs.existsSync(conventionalCheck) && !existingChecks.includes(conventionalCheck)) {
      existingChecks.push(conventionalCheck);
    }
    const numericLiterals = [...body.matchAll(/(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?/gi)]
      .map((match) => match[0])
      .filter((value, pos, all) => all.indexOf(value) === pos)
      .slice(0, 100);
    const override = overrides[chapter.src + "#" + id];
    if (override?.checks) {
      for (const check of override.checks) {
        if (!existingChecks.includes(check)) existingChecks.push(check);
      }
    }
    const status = override?.status || (existingChecks.some((file) => file.endsWith(".py"))
      ? "executable-reference"
      : previousArtifact?.verification_status === "conceptual-audit"
        ? "conceptual-audit"
      : numericLiterals.length
        ? "literal-audit"
        : "conceptual-audit");
    if (status === "executable-reference") executable += 1;
    else if (status === "literal-audit") literal += 1;
    else conceptual += 1;

    const artifact = {
      schema_version: 1,
      chapter: chapter.src,
      anchor: id,
      source_file: sourceFile,
      source_sha256: digest(cleanBlock.join("\n")),
      verification_status: status,
      executable_checks: existingChecks.filter((file) => file.endsWith(".py")),
      numeric_literals: numericLiterals,
      scope:
        status === "executable-reference"
          ? "The referenced executable check validates the numerical construction; this manifest binds it to the canonical example."
          : status === "literal-audit"
            ? "The manifest verifies source identity and enumerates displayed numeric literals; independent executable validation is still required."
            : override?.rationale || "The example is conceptual and contains no detected numeric literals; the manifest verifies source identity only.",
      prepared_for: {
        author: "Taha Bouhsine",
        website: "https://www.tahabouhsine.com",
        affiliation: "AzettaAI",
      },
      independent_review_required: true,
    };
    fs.writeFileSync("checks/" + filename, JSON.stringify(artifact, null, 2) + "\n");
    output.push(...cleanBlock);
    index = end;
    total += 1;
  }
  fs.writeFileSync(sourceFile, matter.stringify(output.join("\n"), parsed.data));
}

console.log(
  "Example artifacts: " + total +
  " total; " + executable + " executable; " +
  literal + " literal-audit; " + conceptual + " conceptual.",
);
