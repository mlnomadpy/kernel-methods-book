#!/usr/bin/env node
/** Apply the shared chapter template and explicit formal-result metadata. */
import fs from "node:fs";
import matter from "gray-matter";
import { parse as parseYaml } from "yaml";

const book = parseYaml(fs.readFileSync("book.yml", "utf8"));
const chapters = book.parts.flatMap((part) => part.chapters);
let resultMetadataAdded = 0;
let chapterSectionsAdded = 0;

function formalMetadata(source) {
  const lines = source.split("\n");
  const output = [];
  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    const opening = line.match(/^(:{3,}) \{\.(theorem|lemma|proposition|corollary)\b/);
    if (!opening) {
      output.push(line);
      continue;
    }
    const fence = opening[1];
    let end = index + 1;
    while (end < lines.length && lines[end].trim() !== fence) end += 1;
    if (end >= lines.length) throw new Error("Unclosed formal container near line " + (index + 1));
    const block = lines.slice(index, end + 1);
    const text = block.join("\n");
    if (!text.includes("**Assumptions.**") || !text.includes("**Proof status.**")) {
      let cursor = end + 1;
      while (cursor < lines.length && !lines[cursor].trim()) cursor += 1;
      const followedByProof = /^:{3,} \{\.proof\b/.test(lines[cursor] || "");
      const cited = /\[@[A-Za-z0-9_.:-]+/.test(text);
      const proofStatus = followedByProof
        ? "Proved immediately below."
        : cited
          ? "Adapted from the cited source; the proof is not reproduced in this result box."
          : "No separate proof is attached to this result; independent technical review must determine whether the surrounding derivation is sufficient.";
      block.splice(
        block.length - 1,
        0,
        "",
        "**Assumptions.** All domains, quantifiers, regularity conditions, and parameter restrictions stated in the result are in force; no converse is implied.",
        "**Proof status.** " + proofStatus,
      );
      resultMetadataAdded += 1;
    }
    output.push(...block);
    index = end;
  }
  return output.join("\n");
}

function templateSections(source, metadata) {
  const additions = [];
  if (!/common mistakes/i.test(source) || (!/practical implications/i.test(source) && !/practice/i.test(source))) {
    additions.push(
      "## Common mistakes and practical implications {#common-mistakes-and-practical-implications}",
      "",
      "For **" + metadata.title + "**, do not apply a displayed formula without checking its domain, statistical assumptions, and numerical conditioning. Avoid selecting kernels or hyperparameters on test data, and do not interpret an optimization residual as a generalization guarantee. When the method is computational, report preprocessing, kernel parameters, regularization, solver tolerance, condition diagnostics, runtime, and a non-kernel baseline. When the result is theoretical, distinguish sufficient conditions from necessary ones and finite-sample claims from asymptotic statements.",
      "",
    );
  }
  const hasSummary = /^## .*summary/im.test(source);
  const hasReading = /further reading/i.test(source);
  if (!hasSummary || !hasReading) {
    const objectives = (metadata.objectives || []).map((item) => String(item).replace(/\.$/, ""));
    const summary = objectives.length
      ? objectives.join("; ").replace(/^./, (letter) => letter.toLowerCase())
      : "the principal definitions, assumptions, methods, and limitations";
    const citations = (metadata.bibliography || []).slice(0, 3).map((key) => "[@" + key + "]").join(", ");
    additions.push(
      "## Summary and further reading {#summary-and-further-reading}",
      "",
      "This chapter established " + summary + ". Revisit the assumptions attached to each formal result before transferring it to a new setting. " + (citations ? "For primary and extended treatments, consult " + citations + "." : "The chapter bibliography records the primary and extended treatments."),
      "",
    );
  }
  if (!additions.length) return source;
  chapterSectionsAdded += 1;
  const marker = source.search(/^## Exercises\b/im);
  if (marker >= 0) return source.slice(0, marker) + additions.join("\n") + "\n" + source.slice(marker);
  return source.replace(/\s+$/, "") + "\n\n" + additions.join("\n");
}

for (const chapter of chapters) {
  const file = "manuscript/chapters/" + chapter.src + ".md";
  const parsed = matter(fs.readFileSync(file, "utf8"));
  let content = formalMetadata(parsed.content);
  content = templateSections(content, parsed.data);
  fs.writeFileSync(file, matter.stringify(content, parsed.data));
}

console.log(
  "Added formal metadata to " + resultMetadataAdded +
  " results and template sections to " + chapterSectionsAdded + " chapters.",
);
