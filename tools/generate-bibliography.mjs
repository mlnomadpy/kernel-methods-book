#!/usr/bin/env node
/** Generate the website bibliography JSON from the canonical constrained BibTeX. */
import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const input = fs.readFileSync(path.join(root, "bibliography.bib"), "utf8");
const output = {};
const entryRe = /@(?:misc|article|book|inproceedings)\s*\{\s*([^,\s]+)\s*,([\s\S]*?)\n\}/gi;
let match;
while ((match = entryRe.exec(input))) {
  const [, key, block] = match;
  if (output[key]) throw new Error(`Duplicate BibTeX key: ${key}`);
  const fields = {};
  const fieldRe = /^\s*([A-Za-z]+)\s*=\s*\{([^{}]*)\}\s*,?\s*$/gm;
  let field;
  while ((field = fieldRe.exec(block))) fields[field[1].toLowerCase()] = field[2];
  if (!fields.author || !fields.title || !fields.year) {
    throw new Error(`BibTeX entry ${key} must have author, title, and year fields`);
  }
  output[key] = {
    authors: fields.author,
    year: /^\d+$/.test(fields.year) ? Number(fields.year) : fields.year,
    title: fields.title,
    venue: fields.journal || fields.booktitle || fields.publisher || fields.howpublished || "",
    ...(fields.url ? { url: fields.url } : {}),
  };
}
if (!Object.keys(output).length) throw new Error("No BibTeX entries parsed.");
const target = path.join(root, "bibliography.web.json");
const generated = JSON.stringify(output, null, 2) + "\n";
if (process.argv.includes("--check")) {
  if (!fs.existsSync(target) || fs.readFileSync(target, "utf8") !== generated) {
    throw new Error("bibliography.web.json is stale; run npm run bibliography:generate");
  }
  console.log(`Bibliography artifact is current (${Object.keys(output).length} entries).`);
} else {
  fs.writeFileSync(target, generated);
  console.log(`Generated bibliography.web.json with ${Object.keys(output).length} entries.`);
}
