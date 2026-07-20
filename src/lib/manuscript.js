import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import matter from "gray-matter";
import { parse as parseYaml } from "yaml";

const ROOT = path.resolve(process.cwd());

export function readYaml(relativePath) {
  return parseYaml(fs.readFileSync(path.join(ROOT, relativePath), "utf8"));
}

export function readCanonicalChapter(src) {
  const file = path.join(ROOT, "manuscript", "chapters", `${src}.md`);
  const parsed = matter(fs.readFileSync(file, "utf8"));
  return { metadata: parsed.data, markdown: parsed.content };
}

function protectMath(markdown) {
  const values = [];
  const block = (value) => {
    const id = `KBMATHBLOCK${values.length}`;
    values.push({ id, value, block: true });
    return `<div data-kb-math="${id}"></div>`;
  };
  const inline = (value) => {
    const id = `KBMATHINLINE${values.length}`;
    values.push({ id, value, block: false });
    return `<span data-kb-math="${id}"></span>`;
  };
  markdown = markdown.replace(/\$\$[\s\S]*?\$\$/g, block);
  markdown = markdown.replace(/\\\([\s\S]*?\\\)/g, inline);
  // Solution records use conventional single-dollar inline math. Protect it
  // from Pandoc too, then normalize to the delimiters consumed by renderMath.
  markdown = markdown.replace(
    /(?<!\\)(?<!\$)\$([^$\n]+?)\$(?!\$)/g,
    (_, source) => inline(`\\(${source}\\)`),
  );
  return { markdown, values };
}

function restoreMath(html, values) {
  for (const { id, value, block } of values) {
    const escaped = id.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const tag = block ? "div" : "span";
    html = html.replace(
      new RegExp(`<${tag} data-kb-math="${escaped}">\\s*</${tag}>`, "g"),
      () => value,
    );
  }
  return html;
}

const BOX_CLASSES = {
  definition: "def",
  theorem: "thm",
  lemma: "lem",
  proposition: "prop",
  corollary: "cor",
  example: "ex",
  algorithm: "algo",
  remark: "rmk",
  proof: "proof",
};

function normalizeContainers(html) {
  // Pandoc promotes a fenced div that begins with a heading into a section and
  // puts the anchor on the section.  The existing public route contract puts
  // it on the heading, so move it back without duplicating the id.
  html = html.replace(
    /<section id="([^"]+)" class="([^"]+)">\s*<h([23])>/g,
    (_, id, classes, level) => `<section class="${classes}">\n<h${level} id="${id}">`,
  );
  html = html.replace(/<div([^>]*)class="([^"]+)"([^>]*)>/g, (whole, before, classes, after) => {
    const list = classes.split(/\s+/);
    const semantic = list.find((name) => BOX_CLASSES[name]);
    if (!semantic) return whole;
    const rest = list.filter((name) => name !== semantic);
    const normalized = ["box", BOX_CLASSES[semantic], ...rest].join(" ");
    const attrs = `${before}${after}`;
    const id = attrs.match(/\sid="([^"]+)"/)?.[1];
    const cleaned = attrs.replace(/\s*id="[^"]+"/, "").trim();
    return `<div class="${normalized}"${id ? ` id="${id}"` : ""}${cleaned ? ` ${cleaned}` : ""}>`;
  });
  html = html.replace(/<p>\s*(<span class="box-title">[\s\S]*?<\/span>)\s*<\/p>/g, "$1");
  // The legacy pipeline injects a chapter label into a plain first-level heading.
  html = html.replace(/<h1(?:\s+[^>]*)?>/, "<h1>");
  html = html.replace(/<h1>([\s\S]*?)<\/h1>/, (whole, title) => `<h1>${title.replace(/’/g, "'")}</h1>`);
  return html;
}

/** Render canonical Pandoc Markdown while preserving the site's KaTeX delimiters. */
export function renderCanonicalMarkdown(markdown) {
  const protectedMath = protectMath(markdown);
  let html = execFileSync("pandoc", [
    "-f", "markdown+fenced_divs+bracketed_spans+raw_html",
    "-t", "html5",
    "--wrap=none",
  ], {
    input: protectedMath.markdown,
    encoding: "utf8",
    maxBuffer: 32 * 1024 * 1024,
  });
  html = restoreMath(html, protectedMath.values);
  return normalizeContainers(html);
}
