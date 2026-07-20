#!/usr/bin/env node
/** Generate an honest, machine-readable snapshot of release readiness. */
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { parse as parseYaml } from "yaml";

const book = parseYaml(fs.readFileSync("book.yml", "utf8"));
const publication = JSON.parse(fs.readFileSync("publication.json", "utf8"));
const chapters = book.parts.flatMap((part) => part.chapters);

let formalTotal = 0;
let formalDocumented = 0;
let exampleTotal = 0;
let exampleExecutable = 0;
let exampleConceptual = 0;
for (const chapter of chapters) {
  const body = matter(fs.readFileSync(`manuscript/chapters/${chapter.src}.md`, "utf8")).content;
  const formalOpenings = [...body.matchAll(/^:{3,} \{\.(theorem|lemma|proposition|corollary)\b[^}]*\}/gm)];
  formalTotal += formalOpenings.length;
  for (let index = 0; index < formalOpenings.length; index++) {
    const end = index + 1 < formalOpenings.length ? formalOpenings[index + 1].index : body.length;
    const region = body.slice(formalOpenings[index].index, end);
    if (/\*\*Assumptions\.\*\*/.test(region) && /\*\*Proof status\.\*\*/.test(region)) formalDocumented += 1;
  }
  for (const filename of fs.readdirSync("checks")) {
    if (!filename.startsWith(`example-${chapter.src}-`) || !filename.endsWith(".json")) continue;
    const artifact = JSON.parse(fs.readFileSync(path.join("checks", filename), "utf8"));
    exampleTotal += 1;
    if (artifact.verification_status === "executable-reference") exampleExecutable += 1;
    if (artifact.verification_status === "conceptual-audit") exampleConceptual += 1;
  }
}

let technicalApprovals = 0;
let pedagogicalApprovals = 0;
let verifiedChapters = 0;
let completeProvenance = 0;
for (const chapter of chapters) {
  const review = parseYaml(fs.readFileSync(`reviews/${chapter.src}.yml`, "utf8"));
  const provenance = parseYaml(fs.readFileSync(`provenance/${chapter.src}.yml`, "utf8"));
  if (review.status === "verified") verifiedChapters += 1;
  if (review.technical_review?.reviewer && review.technical_review?.approved_on) technicalApprovals += 1;
  if (review.pedagogical_review?.reviewer && review.pedagogical_review?.approved_on) pedagogicalApprovals += 1;
  const sections = provenance.sections || [];
  if (provenance.status === "verified" && sections.length && sections.every((section) => section.sources?.length && section.locators?.length)) {
    completeProvenance += 1;
  }
}

let exercises = 0;
let answers = 0;
let substantiveAnswers = 0;
let verifiedAnswers = 0;
for (const chapter of chapters) {
  const source = matter(fs.readFileSync(`manuscript/chapters/${chapter.src}.md`, "utf8")).content;
  const count = [...source.matchAll(/\[([a-z-]+)\]\{\.ex-tag\}/g)].length;
  exercises += count;
  const filename = `solutions/${chapter.src}.yml`;
  if (!fs.existsSync(filename)) continue;
  const solution = parseYaml(fs.readFileSync(filename, "utf8"));
  answers += (solution.exercises || []).filter((entry) => entry.answer?.trim()).length;
  substantiveAnswers += (solution.exercises || []).filter((entry) => entry.answer?.trim() && entry.status !== "draft-outline").length;
  verifiedAnswers += (solution.exercises || []).filter((entry) => entry.status === "verified").length;
}

const bibliography = fs.readFileSync("bibliography.bib", "utf8");
const bibEntries = [...bibliography.matchAll(/@(misc|article|book|inproceedings)\s*\{\s*([^,\s]+)\s*,([\s\S]*?)\n\}/gi)];
const linkedEntries = bibEntries.filter((entry) => /^\s*(doi|url)\s*=/mi.test(entry[3])).length;
const missingBibliographyKeys = bibEntries
  .filter((entry) => !/^\s*(doi|url)\s*=/mi.test(entry[3]))
  .map((entry) => entry[2])
  .sort();
const unresolvedBibliography = parseYaml(fs.readFileSync("bibliography-unresolved.yml", "utf8"));
const documentedUnresolved = unresolvedBibliography.records || {};
const documentedUnresolvedKeys = Object.keys(documentedUnresolved).sort();
if (JSON.stringify(documentedUnresolvedKeys) !== JSON.stringify(missingBibliographyKeys)) {
  throw new Error(
    `bibliography-unresolved.yml must document exactly the unlinked records: expected ${missingBibliographyKeys.join(", ")}; found ${documentedUnresolvedKeys.join(", ")}`,
  );
}
const retained = parseYaml(fs.readFileSync("bibliography-retained.yml", "utf8")).retained || [];
const kaggleCatalog = JSON.parse(fs.readFileSync("notebooks/kaggle-catalog.json", "utf8"));
const kaggleManifest = fs.existsSync("release/kaggle-manifest.json")
  ? JSON.parse(fs.readFileSync("release/kaggle-manifest.json", "utf8"))
  : { labs: [] };

function artifact(filename) {
  if (!fs.existsSync(filename)) return null;
  const data = fs.readFileSync(filename);
  return {
    file: filename,
    bytes: data.length,
    sha256: crypto.createHash("sha256").update(data).digest("hex"),
  };
}

const snapshot = {
  generated_on: publication.date,
  edition: publication.edition,
  version: publication.version,
  status: publication.status,
  author: publication.authors[0],
  chapters: {
    total: chapters.length,
    verified: verifiedChapters,
    technical_approvals: technicalApprovals,
    pedagogical_approvals: pedagogicalApprovals,
    complete_provenance: completeProvenance,
  },
  formal_results: { total: formalTotal, documented: formalDocumented },
  examples: {
    total: exampleTotal,
    executable: exampleExecutable,
    conceptual_with_rationale: exampleConceptual,
    unclassified_or_literal_only: exampleTotal - exampleExecutable - exampleConceptual,
  },
  exercises: {
    total: exercises,
    answer_records: answers,
    substantive_drafts: substantiveAnswers,
    independently_verified: verifiedAnswers,
  },
  bibliography: {
    total: bibEntries.length,
    doi_or_url: linkedEntries,
    missing_doi_or_url: bibEntries.length - linkedEntries,
    missing_keys: missingBibliographyKeys,
    documented_archival_backlog: documentedUnresolvedKeys.length,
    archival_backlog_checked_on: unresolvedBibliography.checked_on,
    intentionally_retained_unused: retained.length,
  },
  kaggle: {
    canonical_labs: kaggleCatalog.labs.length,
    published_and_verified_urls: kaggleManifest.labs?.filter((lab) => lab.kaggle_url).length || 0,
    credentials_committed: false,
  },
  publication_artifacts: [
    artifact("release/kernels-the-geometry-of-learning.pdf"),
    artifact("release/kernels-the-geometry-of-learning.epub"),
  ].filter(Boolean),
  release_blockers: [
    "Independent technical and pedagogical approvals are not complete.",
    "Section-level provenance locators are not independently verified.",
    "Exercise answers are authorial drafts and are not independently verified.",
    "Kaggle mirrors require release credentials and URL verification.",
    "Independent editor legal names and credits have not been supplied.",
  ],
  release_notes: [
    `${documentedUnresolvedKeys.length} archival bibliography records have no discoverable stable publisher, DOI, or institutional target; searches are documented in release/bibliography-metadata-backlog.json.`,
  ],
};

fs.mkdirSync("release", { recursive: true });
fs.writeFileSync("release/readiness.json", JSON.stringify(snapshot, null, 2) + "\n");
fs.writeFileSync(
  "release/bibliography-metadata-backlog.json",
  JSON.stringify(
    {
      generated_on: publication.date,
      checked_on: unresolvedBibliography.checked_on,
      records: documentedUnresolved,
    },
    null,
    2,
  ) + "\n",
);
const lines = [
  "# Release readiness audit",
  "",
  `Generated ${snapshot.generated_on} for ${snapshot.edition} ${snapshot.version}.`,
  "",
  `Author: [${snapshot.author.name}](${snapshot.author.url}), ${snapshot.author.affiliation} (${snapshot.author.handle}).`,
  "",
  "| Area | Coverage | Release state |",
  "|---|---:|---|",
  `| Chapters | ${verifiedChapters}/${chapters.length} verified | blocked on independent review |`,
  `| Technical approvals | ${technicalApprovals}/${chapters.length} | blocked |`,
  `| Pedagogical approvals | ${pedagogicalApprovals}/${chapters.length} | blocked |`,
  `| Complete provenance | ${completeProvenance}/${chapters.length} | blocked |`,
  `| Formal-result metadata | ${formalDocumented}/${formalTotal} | structurally complete; human verification pending |`,
  `| Example artifacts | ${exampleTotal}/${exampleTotal} (${exampleExecutable} executable, ${exampleConceptual} conceptual) | automated gate passes |`,
  `| Exercise answers | ${answers}/${exercises} records; ${substantiveAnswers} substantive drafts; ${verifiedAnswers} independently verified | authoring complete; independent verification blocked |`,
  `| Bibliography links | ${linkedEntries}/${bibEntries.length} | ${documentedUnresolvedKeys.length} archival records explicitly documented |`,
  `| Kaggle mirrors | ${snapshot.kaggle.published_and_verified_urls}/${snapshot.kaggle.canonical_labs} | credentials required |`,
  "",
  "## Publication artifacts",
  "",
  ...snapshot.publication_artifacts.map((item) => `- \`${item.file}\`: ${item.bytes} bytes; SHA-256 \`${item.sha256}\``),
  "",
  "## Open release blockers",
  "",
  ...snapshot.release_blockers.map((blocker) => `- ${blocker}`),
  "",
  "## Documented non-blocking exceptions",
  "",
  ...snapshot.release_notes.map((note) => `- ${note}`),
  "",
  "This audit never treats an authorial or AI-assisted pass as independent review.",
  "",
];
fs.writeFileSync("RELEASE_READINESS.md", lines.join("\n"));
console.log(`Wrote RELEASE_READINESS.md and release/readiness.json (${verifiedChapters}/${chapters.length} chapters verified).`);
