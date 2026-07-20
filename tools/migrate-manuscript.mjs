#!/usr/bin/env node
/**
 * One-time, deterministic migration from the legacy HTML/JSON edition to the
 * canonical Markdown/YAML/BibTeX edition.  Re-running it is intentionally
 * refused unless --force is supplied, because Markdown is canonical after the
 * parity gate has passed.
 */
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import matter from "gray-matter";
import { stringify as yamlStringify } from "yaml";

const root = process.cwd();
const force = process.argv.includes("--force");
const bookPath = path.join(root, "book.json");
const bibPath = path.join(root, "bibliography.json");
const outDir = path.join(root, "manuscript", "chapters");

if (!fs.existsSync(bookPath) || !fs.existsSync(bibPath)) {
  throw new Error("Legacy book.json and bibliography.json are required for the one-time migration.");
}
if (fs.existsSync(outDir) && !force) {
  throw new Error("manuscript/chapters already exists; pass --force only when deliberately re-running migration.");
}

const book = JSON.parse(fs.readFileSync(bookPath, "utf8"));
const bibliography = JSON.parse(fs.readFileSync(bibPath, "utf8"));
const paths = JSON.parse(fs.readFileSync(path.join(root, "reading-paths.json"), "utf8"));
const core = new Set(paths.graduate.chapters);
const practitioner = new Set(paths.practitioner.chapters);
const flat = book.parts.flatMap((part) => part.chapters.map((ch) => ({ ...ch, part: part.part })));

fs.mkdirSync(outDir, { recursive: true });

function protect(source) {
  const saved = [];
  const put = (kind, value, block = false) => {
    const id = `${kind}-${saved.length}`;
    saved.push({ id, value });
    return block
      ? `<div data-kb-placeholder="${id}"></div>`
      : `<span data-kb-placeholder="${id}"></span>`;
  };
  source = source.replace(/\$\$[\s\S]*?\$\$/g, (m) => put("math-block", m, true));
  source = source.replace(/\\\([\s\S]*?\\\)/g, (m) => put("math-inline", m));
  source = source.replace(/\[\[ch:[^\]]+\]\]/g, (m) => put("chref", m));
  source = source.replace(/<p class="lead">[\s\S]*?<\/p>/g, (m) => put("lead", m, true));
  return { source, saved };
}

function restore(source, saved) {
  // Restore outer protected regions before placeholders nested inside them.
  for (const { id, value } of [...saved].reverse()) {
    const escaped = id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    source = source.replace(new RegExp(`<span data-kb-placeholder="${escaped}"></span>`, "g"), () => value);
    source = source.replace(new RegExp(`\\[\\]\\{kb-placeholder="${escaped}"\\}`, "g"), () => value);
    source = source.replace(new RegExp(`<div data-kb-placeholder="${escaped}"></div>`, "g"), () => value);
    source = source.replace(
      new RegExp(`^[ \\t]*:{3,} \\{kb-placeholder="${escaped}"\\}\\n(?:[ \\t]*\\n)?[ \\t]*:{3,}$`, "gm"),
      () => value,
    );
  }
  return source;
}

const kindName = {
  def: "definition",
  thm: "theorem",
  lem: "lemma",
  prop: "proposition",
  cor: "corollary",
  ex: "example",
  algo: "algorithm",
  rmk: "remark",
  proof: "proof",
};

function semanticContainers(markdown, chapterLabel) {
  let statements = 0;
  let examples = 0;
  let algorithms = 0;
  return markdown.replace(/^(:{3,}) \{\.box \.(def|thm|lem|prop|cor|ex|algo|rmk|proof)\}\n(?=\[([^\n]*)\]\{\.box-title\})/gm,
    (open, fence, kind, title) => {
      let id = "";
      if (["def", "thm", "lem", "prop", "cor"].includes(kind)) {
        statements += 1;
        id = ` #${kind}-${chapterLabel.toLowerCase()}-${statements}`;
      } else if (kind === "algo") {
        algorithms += 1;
        id = ` #algo-${chapterLabel.toLowerCase()}-${algorithms}`;
      } else if (kind === "ex" && !/^Exercise/i.test(title)) {
        examples += 1;
        id = ` #example-${chapterLabel.toLowerCase()}-${examples}`;
      }
      return `${fence} {.${kindName[kind]}${id}}\n`;
    })
    .replace(/^(:{3,}) \{\.box \.(def|thm|lem|prop|cor|ex|algo|rmk|proof)\}$/gm,
      (open, fence, kind) => `${fence} {.${kindName[kind]}}`)
    .replace(/^(:{3,}) \{\.section \.exercises\}$/gm, "$1 {.exercises}")
    .replace(/^(:{3,}) hint-body$/gm, "$1 {.hint-body}")
    .replace(/^(\[[^\n]*\]\{\.box-title\})[ \t]+/gm, "$1\n\n");
}

function explicitHeadingAnchors(markdown, html) {
  const expected = [...html.matchAll(/<h([23]) id="([^"]+)"/g)].map((match) => ({
    level: match[1], id: match[2],
  }));
  let index = 0;
  return markdown.replace(/^(#{2,3}) ([^\n]+)$/gm, (line, marks, title) => {
    const current = expected[index++];
    if (!current || current.level !== String(marks.length)) return line;
    const existing = title.match(/\s+\{#([^}]+)\}\s*$/)?.[1];
    if (existing && existing !== current.id) {
      throw new Error(`Heading anchor changed: expected ${current.id}, received ${existing}`);
    }
    return existing ? line : `${line} {#${current.id}}`;
  });
}

function metric(body) {
  return {
    h1: body.match(/<h1>([\s\S]*?)<\/h1>/)?.[1].replace(/<[^>]+>/g, "") || "",
    headings: [...body.matchAll(/<h([23]) id="([^"]+)"/g)].map((m) => `${m[1]}:${m[2]}`),
    statements: (body.match(/<div class="box (?:def|thm|lem|prop|cor)">/g) || []).length,
    examples: (body.match(/<div class="box ex">/g) || []).length,
    algorithms: (body.match(/<div class="box algo">/g) || []).length,
    proofs: (body.match(/<div class="box proof">/g) || []).length,
    exercises: (body.match(/class="ex-tag"/g) || []).length,
    widgets: (body.match(/data-widget=/g) || []).length,
    chrefs: (body.match(/\[\[ch:/g) || []).length,
  };
}

const baseline = { version: 1, chapterOrder: flat.map((ch) => ch.src), chapters: {} };

for (const [index, ch] of flat.entries()) {
  const html = fs.readFileSync(path.join(root, "chapters", "src", `${ch.src}.body.html`), "utf8");
  const refsPath = path.join(root, "chapters", "refs", `${ch.src}.json`);
  const refs = fs.existsSync(refsPath) ? JSON.parse(fs.readFileSync(refsPath, "utf8")) : [];
  baseline.chapters[ch.src] = metric(html);
  const { source, saved } = protect(html);
  let markdown = execFileSync("pandoc", [
    "-f", "html", "-t", "markdown+fenced_divs+bracketed_spans+raw_html", "--wrap=none",
  ], { input: source, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 });
  markdown = restore(markdown, saved);
  markdown = semanticContainers(markdown, ch.src === "ch-prelim" ? "P" : String(index));
  markdown = explicitHeadingAnchors(markdown, html);
  markdown = markdown.replace(/\\'/g, "'");
  const exerciseLabels = { concept: "warm-up", core: "computation", easy: "warm-up", mechanical: "warm-up", medium: "computation", hard: "challenge" };
  markdown = markdown.replace(/\[([a-z-]+)\]\{\.ex-tag\}/g,
    (whole, label) => `[${exerciseLabels[label] || label}]{.ex-tag}`);

  const tier = core.has(ch.slug) ? "core" : practitioner.has(ch.slug) ? "practitioner" : "advanced";
  const previous = index > 0 ? [flat[index - 1].slug] : [];
  const data = {
    id: ch.src,
    slug: ch.slug,
    title: ch.title,
    part: ch.part,
    order: index,
    tier,
    prerequisites: previous,
    objectives: [
      `Explain the central definitions and claims in ${ch.title}.`,
      "Apply the chapter's principal methods and interpret their outputs.",
      "State the assumptions behind formal results and connect them to earlier chapters.",
    ],
    review_status: "draft",
    reviewers: { technical: null, pedagogical: null, specialist: null },
    provenance: `provenance/${ch.src}.yml`,
    verification_date: null,
    bibliography: refs,
  };
  fs.writeFileSync(path.join(outDir, `${ch.src}.md`), matter.stringify(markdown.trim() + "\n", data));
}

fs.writeFileSync(path.join(root, "book.yml"), yamlStringify(book, { lineWidth: 0 }));
fs.writeFileSync(path.join(root, "migration-baseline.json"), JSON.stringify(baseline, null, 2) + "\n");

const bib = Object.entries(bibliography).map(([key, entry]) => {
  const fields = [
    ["author", entry.authors], ["year", entry.year], ["title", entry.title],
    ["howpublished", entry.venue], ["url", entry.url],
  ].filter(([, value]) => value !== undefined && value !== null && value !== "");
  return `@misc{${key},\n${fields.map(([name, value]) => `  ${name} = {${value}}`).join(",\n")}\n}`;
});
fs.writeFileSync(path.join(root, "bibliography.bib"), bib.join("\n\n") + "\n");
console.log(`Migrated ${flat.length} chapters and ${bib.length} bibliography entries.`);
