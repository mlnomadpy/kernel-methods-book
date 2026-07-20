#!/usr/bin/env node
/** Build an honest section-to-bibliography map without fabricating source locators. */
import fs from "node:fs";
import matter from "gray-matter";
import { parse as parseYaml, stringify as stringifyYaml } from "yaml";

const book = parseYaml(fs.readFileSync("book.yml", "utf8"));
const chapters = book.parts.flatMap((part) => part.chapters);
let mappedSections = 0;
let citationMappedSections = 0;

function unique(values) {
  return [...new Set(values)];
}

function citations(text) {
  return unique(
    [...text.matchAll(/\[@([^\]]+)\]/g)].flatMap((match) =>
      match[1]
        .split(";")
        .map((item) => item.trim().replace(/^@/, "").split(/[\s,]/)[0])
        .filter(Boolean),
    ),
  );
}

function manuscriptSections(content) {
  const matches = [...content.matchAll(/^##\s+(.+?)\s+\{#([^}]+)\}\s*$/gm)];
  return matches.map((match, index) => {
    const start = match.index + match[0].length;
    const end = index + 1 < matches.length ? matches[index + 1].index : content.length;
    return { id: match[2], title: match[1], body: content.slice(start, end) };
  });
}

for (const chapter of chapters) {
  const manuscript = matter(fs.readFileSync(`manuscript/chapters/${chapter.src}.md`, "utf8"));
  const manifestPath = `provenance/${chapter.src}.yml`;
  const manifest = parseYaml(fs.readFileSync(manifestPath, "utf8"));
  const sections = manuscriptSections(manuscript.content);
  const bodies = new Map(sections.map((section) => [section.id, section.body]));
  const chapterSources = unique(manuscript.data.bibliography || manifest.references || []);

  // Keep the manifest structurally synchronized with the canonical manuscript.
  // Existing editorial judgments are preserved; newly introduced sections receive
  // explicit pending metadata instead of silently disappearing from provenance.
  const existingSections = new Map((manifest.sections || []).map((section) => [section.id, section]));
  manifest.sections = sections.map(({ id, title }) => existingSections.get(id) || ({
    id,
    title,
    contribution: "newly synthesized",
    permission: "original prose; cited ideas remain attributed to their primary sources",
  }));

  manifest.status = "author-mapped-pending-independent-verification";
  manifest.mapping_method = "section citations, with chapter-bibliography fallback where a section has no inline citation";
  manifest.mapping_author = {
    name: "Taha Bouhsine",
    website: "https://www.tahabouhsine.com",
    affiliation: "AzettaAI",
  };
  manifest.mapping_date = "2026-07-20";

  for (const section of manifest.sections) {
    const inline = citations(bodies.get(section.id) || "");
    const sources = inline.length ? inline : chapterSources;
    section.sources = sources;
    section.locators = sources.map((key) => ({
      source: key,
      manuscript_locator: `${chapter.src}.md#${section.id}`,
      source_locator: null,
      status: "exact-page-section-or-theorem-locator-pending-independent-verification",
    }));
    section.verified_by = null;
    section.verified_on = null;
    mappedSections += 1;
    if (inline.length) citationMappedSections += 1;
  }

  fs.writeFileSync(manifestPath, stringifyYaml(manifest, { lineWidth: 110 }));
}

console.log(
  `Mapped ${mappedSections} provenance sections; ${citationMappedSections} use inline citations and ` +
  `${mappedSections - citationMappedSections} use the declared chapter bibliography pending exact-locator review.`,
);
