import fs from "node:fs";
import matter from "gray-matter";
import { parse as parseYaml } from "yaml";

const book = parseYaml(fs.readFileSync("book.yml", "utf8"));
const chapters = book.parts.flatMap((part) => part.chapters.map((chapter) => ({ ...chapter, part: part.part })));
fs.mkdirSync("provenance", { recursive: true });
fs.mkdirSync("reviews", { recursive: true });

const quote = (value) => JSON.stringify(String(value));
for (const chapter of chapters) {
  const parsed = matter(fs.readFileSync(`manuscript/chapters/${chapter.src}.md`, "utf8"));
  const refs = parsed.data.bibliography || [];
  const sections = [...parsed.content.matchAll(/^## (.*?) \{#([^}]+)\}$/gm)]
    .map((match) => ({ id: match[2], title: match[1].replace(/[*_`]/g, "").trim() }));

  const provenanceFile = `provenance/${chapter.src}.yml`;
  if (!fs.existsSync(provenanceFile)) {
    const lines = [
      `chapter: ${quote(chapter.src)}`,
      `slug: ${quote(chapter.slug)}`,
      `title: ${quote(chapter.title)}`,
      `status: needs-verification`,
      `private_source_sha256: null`,
      `references: [${refs.map(quote).join(", ")}]`,
      `sections:`,
    ];
    for (const section of sections) lines.push(
      `  - id: ${quote(section.id)}`,
      `    title: ${quote(section.title)}`,
      `    contribution: adapted-and-expanded`,
      `    sources: []`,
      `    locators: []`,
      `    verified_by: null`,
      `    verified_on: null`,
      `    permission: citation-only`,
    );
    fs.writeFileSync(provenanceFile, `${lines.join("\n")}\n`);
  }

  const reviewFile = `reviews/${chapter.src}.yml`;
  if (!fs.existsSync(reviewFile)) {
    fs.writeFileSync(reviewFile, [
      `chapter: ${quote(chapter.src)}`,
      `status: draft`,
      `technical_review:`,
      `  reviewer: null`,
      `  approved_on: null`,
      `  findings_resolved: false`,
      `pedagogical_review:`,
      `  reviewer: null`,
      `  approved_on: null`,
      `  findings_resolved: false`,
      `specialist_review_required: ${["ch-causal", "ch-ksd", "ch-signature", "ch-ot", "ch-modern", "ch14", "ch-operator", "ch-manifold", "ch-inverse", "ch-dkl", "ch-splines", "ch-spatial", "ch-reliability", "ch-dynamics", "ch-scientific", "ch-rkbs"].includes(chapter.src)}`,
      `specialist_review:`,
      `  reviewer: null`,
      `  approved_on: null`,
      `copy_edit:`,
      `  editor: null`,
      `  completed_on: null`,
      `waivers: []`,
      ``,
    ].join("\n"));
  }
}

console.log(`Manifest coverage: ${chapters.length} provenance files and ${chapters.length} review files.`);
