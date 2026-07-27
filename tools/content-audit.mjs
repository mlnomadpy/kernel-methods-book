import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { parse as parseYaml } from "yaml";

const root = process.cwd();
const read = (file) => fs.readFileSync(path.join(root, file), "utf8");
const book = parseYaml(read("book.yml"));
const bibliography = JSON.parse(read("bibliography.web.json"));
const retainedBibliography = fs.existsSync(path.join(root, "bibliography-retained.yml"))
  ? parseYaml(read("bibliography-retained.yml")).retained || {}
  : {};
const glossary = JSON.parse(read("glossary.json"));
const chapters = book.parts.flatMap((part) => part.chapters.map((chapter) => ({ ...chapter, part: part.part })));
const slugs = new Set(chapters.map((chapter) => chapter.slug));
const errors = [];
const warnings = [];
const missingLinks = [];
const referenced = new Set();
const allowedExercises = new Set(["warm-up", "computation", "proof", "exploration", "challenge", "synthesis"]);

for (const [order, chapter] of chapters.entries()) {
  const sourceFile = `manuscript/chapters/${chapter.src}.md`;
  if (!fs.existsSync(path.join(root, sourceFile))) {
    errors.push(`${chapter.src}: missing ${sourceFile}`);
    continue;
  }
  const parsed = matter(read(sourceFile));
  // Structural and citation checks apply to manuscript prose, not to literal
  // code. Python comments beginning with "# " are not chapter headings, and
  // decorators such as @jax.jit are not bibliography citations.
  const source = parsed.content.replace(/^```[\s\S]*?^```/gm, "");
  const meta = parsed.data;
  const required = ["id", "slug", "title", "part", "order", "tier", "prerequisites", "objectives", "review_status", "reviewers", "provenance", "verification_date", "bibliography"];
  for (const key of required) if (!(key in meta)) errors.push(`${chapter.src}: missing frontmatter field ${key}`);
  if (meta.id !== chapter.src || meta.slug !== chapter.slug || meta.title !== chapter.title || meta.part !== chapter.part || meta.order !== order) {
    errors.push(`${chapter.src}: frontmatter does not match canonical book.yml`);
  }
  if (!Array.isArray(meta.objectives) || meta.objectives.length < 3 || meta.objectives.length > 6) {
    errors.push(`${chapter.src}: expected three to six learning objectives`);
  }
  const h1s = [...source.matchAll(/^# (.+)$/gm)];
  if (h1s.length !== 1) errors.push(`${chapter.src}: expected exactly one h1, found ${h1s.length}`);
  if (h1s[0]?.[1].trim() !== chapter.title) errors.push(`${chapter.src}: h1 differs from book.yml title`);
  if ((source.match(/<p class="lead">/g) || []).length !== 1) errors.push(`${chapter.src}: expected exactly one lead paragraph`);
  if (!/data-(?:widget|figure)="[a-z0-9-]+"/.test(source)) {
    errors.push(`${chapter.src}: chapter lacks a purposeful visualization`);
  }

  const ids = [
    ...[...source.matchAll(/\{#([a-z0-9-]+)[^}]*\}/g)].map((match) => match[1]),
    ...[...source.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]),
  ];
  const duplicateIds = [...new Set(ids.filter((id, index) => ids.indexOf(id) !== index))];
  if (duplicateIds.length) errors.push(`${chapter.src}: duplicate ids ${duplicateIds.join(", ")}`);
  for (const heading of source.matchAll(/^#{2,3} (.+)$/gm)) {
    if (!/\{#[a-z0-9-]+\}\s*$/.test(heading[1])) errors.push(`${chapter.src}: heading lacks an explicit anchor: ${heading[1]}`);
  }
  for (const opening of source.matchAll(/^:{3,} \{\.(definition|theorem|lemma|proposition|corollary|algorithm|example)([^}]*)\}$/gm)) {
    if (!/#[-a-z0-9]+/.test(opening[2]) && opening[1] !== "example") {
      errors.push(`${chapter.src}: ${opening[1]} container lacks an explicit anchor`);
    }
  }
  for (const match of source.matchAll(/\[\[ch:([a-z0-9-]+)/g)) {
    if (!slugs.has(match[1])) errors.push(`${chapter.src}: unknown chapter reference ${match[1]}`);
  }
  for (const match of source.matchAll(
    /<figure class="viz" data-figure="([a-z0-9-]+)"([^>]*)>([\s\S]*?)<\/figure>/g,
  )) {
    const [, figure, attrs, inner] = match;
    if (!/\bdata-alt="[^"]+"/.test(attrs)) {
      errors.push(`${chapter.src}: static figure ${figure} lacks data-alt`);
    }
    if (!/<figcaption[^>]*>[\s\S]*?<\/figcaption>/i.test(inner)) {
      errors.push(`${chapter.src}: static figure ${figure} lacks figcaption`);
    }
    for (const [target, file] of [
      ["web", `public/figures/${figure}.svg`],
      ["PDF", `publication/figures/${figure}.pdf`],
    ]) {
      if (!fs.existsSync(path.join(root, file))) {
        errors.push(`${chapter.src}: static figure ${figure} lacks ${target} asset ${file}`);
      }
    }
  }
  if (/—/.test(source)) errors.push(`${chapter.src}: em dash violates the authoring contract`);
  for (const match of source.matchAll(/\$\$([\s\S]*?)\$\$|\\\(([\s\S]*?)\\\)/g)) {
    if (/[<>]/.test(match[1] || match[2] || "")) errors.push(`${chapter.src}: raw angle bracket inside math`);
  }
  for (const match of source.matchAll(/\[([a-z-]+)\]\{\.ex-tag\}/g)) {
    if (!allowedExercises.has(match[1])) errors.push(`${chapter.src}: unsupported exercise label ${match[1]}`);
  }
  const keys = Array.isArray(meta.bibliography) ? meta.bibliography : [];
  for (const key of keys) {
    referenced.add(key);
    if (!bibliography[key]) errors.push(`${chapter.src}: missing bibliography key ${key}`);
  }
  for (const match of source.matchAll(/@([a-zA-Z0-9:_-]+)/g)) {
    if (!keys.includes(match[1])) errors.push(`${chapter.src}: citation @${match[1]} is not declared in frontmatter`);
  }
  if (chapter.src !== "ch-prelim" && !/^## Exercises \{#exercises\}$/m.test(source)) {
    errors.push(`${chapter.src}: missing explicitly anchored exercises section`);
  }
}

for (const collection of [glossary.symbols, glossary.terms]) {
  for (const entry of collection) if (!slugs.has(entry.slug)) errors.push(`glossary: '${entry.term || entry.sym}' points to unknown slug '${entry.slug}'`);
}
for (const key of Object.keys(bibliography)) {
  if (!referenced.has(key) && !retainedBibliography[key]) warnings.push(`unused bibliography key: ${key}`);
}
for (const key of Object.keys(retainedBibliography)) {
  if (!bibliography[key]) errors.push(`retained bibliography key is missing: ${key}`);
  if (!retainedBibliography[key]?.reason) errors.push(`retained bibliography key lacks a reason: ${key}`);
}
for (const [key, entry] of Object.entries(bibliography)) if (!entry.url) missingLinks.push(key);

console.log(`Audited ${chapters.length} canonical chapters and ${Object.keys(bibliography).length} bibliography entries.`);
console.log(`Intentionally retained unused bibliography entries: ${Object.keys(retainedBibliography).length}.`);
for (const warning of warnings) console.warn(`WARN ${warning}`);
if (missingLinks.length) console.warn(`WARN ${missingLinks.length} bibliography entries have no DOI/URL; see the bibliography metadata backlog.`);
if (errors.length) {
  for (const error of errors) console.error(`ERROR ${error}`);
  process.exitCode = 1;
} else {
  console.log(`Content audit passed with ${warnings.length + (missingLinks.length ? 1 : 0)} non-blocking metadata warnings.`);
}
