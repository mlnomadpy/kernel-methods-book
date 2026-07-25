/**
 * The book pipeline: reads book.yml, canonical Markdown chapters, the bibliography,
 * and the glossary, and produces fully processed chapter HTML plus the search
 * index. This is the content layer; the page chrome lives in the Astro layout.
 *
 * Transforms applied to each chapter body, in order:
 *   1. [[ch:slug]] cross-reference tokens -> numbered chapter links
 *   2. statement boxes (definition/theorem/lemma/proposition/corollary and
 *      examples) -> numbered, self-linkable cards with a structured header
 *   3. proof boxes -> collapsible <details> with an end-of-proof mark
 *   4. in-prose "Author (Year)" mentions -> hover-card bibliography links
 */
import fs from "node:fs";
import path from "node:path";
import katex from "katex";
import { readCanonicalChapter, readYaml, renderCanonicalMarkdown } from "./manuscript.js";

const ROOT = path.resolve(process.cwd());

function readJSON(p) {
  return JSON.parse(fs.readFileSync(path.join(ROOT, p), "utf8"));
}

export const BOOK = readYaml("book.yml");
export const BIB = readJSON("bibliography.web.json");
export const GLOSSARY = readJSON("glossary.json");
export const FIGURES = readJSON("publication/figures/registry.json").figures;

export function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** Render manuscript math at build time; retain client KaTeX only for captions
 * created dynamically by interactive figures. */
export function renderMath(body) {
  const protectedCode = [];
  body = body.replace(/<(pre|code)\b[\s\S]*?<\/\1>/g, (value) => {
    const token = `\u0000CODE${protectedCode.length}\u0000`;
    protectedCode.push([token, value]);
    return token;
  });
  const render = (source, displayMode) => katex.renderToString(source.trim(), {
    displayMode,
    throwOnError: false,
    strict: "warn",
    output: "htmlAndMathml",
  });
  body = body.replace(/\$\$([\s\S]*?)\$\$/g, (_, source) => render(source, true));
  body = body.replace(/\\\(([\s\S]*?)\\\)/g, (_, source) => render(source, false));
  for (const [token, value] of protectedCode) body = body.split(token).join(value);
  return body;
}

/** Make horizontally scrollable publication regions reachable without a mouse.
 * Focusability is applied at build time so the fallback remains accessible when
 * JavaScript is unavailable. */
export function decorateScrollableRegions(body) {
  body = body.replace(
    /<span class="katex-display">/g,
    '<span class="katex-display" role="region" tabindex="0" aria-label="Scrollable mathematical expression">',
  );
  body = body.replace(
    /<pre(?![^>]*\btabindex=)([^>]*)>/g,
    '<pre$1 tabindex="0" aria-label="Scrollable code or formula">',
  );
  body = body.replace(
    /<div class="tablewrap"(?![^>]*\btabindex=)([^>]*)>/g,
    '<div class="tablewrap"$1 role="region" tabindex="0" aria-label="Scrollable table">',
  );
  return body;
}

/** Give exercises stable destinations and keep answers on a separate,
 * spoiler-safe companion page. */
export function decorateExercises(body, slug) {
  let number = 0;
  body = body.replace(/<li>(\s*(?:<p>)?\s*<span class="ex-tag[^>]*>)/g, (_, tag) => {
    number += 1;
    return `<li id="exercise-${number}">${tag}`;
  });
  if (number) {
    body = body.replace(
      '<h2 id="exercises">Exercises</h2>',
      `<h2 id="exercises">Exercises</h2><p class="solution-link"><a href="solutions.html#${escapeHtml(slug)}-exercise-1">Open the separate answers and rubrics companion</a>. Draft answers remain visibly marked until review.</p>`,
    );
  }
  return body;
}

const WIDGET_ALTS = {
  "bo-loop": "Gaussian-process posterior and acquisition function across Bayesian-optimization iterations.",
  "cme-explore": "Conditional mean embedding changes as the conditioning input and regularization vary.",
  "dp-fill": "Dynamic-programming table used to compute a string-kernel recurrence.",
  "feature-lift": "Input points lifted into a feature space where a linear separator becomes available.",
  "gram-heatmap": "Kernel Gram matrix heatmap whose off-diagonal similarities change with bandwidth.",
  "heat-graph": "Graph heat diffusion across vertices as diffusion time changes.",
  "herding-greedy": "Kernel-herding points selected sequentially to reduce integration error.",
  "kernel-lab": "Kernel ridge-regression fit for editable observations and kernel hyperparameters.",
  "mmd-twosample": "Two sample clouds and their empirical maximum mean discrepancy.",
  "permutation-null": "Permutation null distribution and observed statistic for a kernel hypothesis test.",
  "rff-converge": "Random Fourier-feature kernel approximation converging toward the exact kernel.",
  "sig-draw": "A drawn path and the low-order components of its path signature.",
  "sinkhorn-plan": "Entropically regularized optimal-transport plan between two discrete measures.",
  "spectrum-surgery": "Kernel eigenvalues before and after a spectral stabilization transform.",
  "svgd-flow": "Particles transported toward a target density by the SVGD velocity field.",
  "svm-margin": "Soft-margin SVM decision boundary, margins, support vectors, and KKT residual.",
  "wl-refine": "Weisfeiler-Lehman vertex labels updated across refinement rounds.",
};

export function decorateWidgets(body) {
  let index = 0;
  return body.replace(/<figure class="viz" data-widget="([^"]+)"([^>]*)>([\s\S]*?)<\/figure>/g,
    (whole, kind, attrs, inner) => {
      index += 1;
      const id = `viz-fallback-${kind}-${index}`;
      const label = WIDGET_ALTS[kind] || `Interactive ${kind.replace(/-/g, " ")} visualization.`;
      const inlineCaption = inner.match(/<figcaption[^>]*>([\s\S]*?)<\/figcaption>/i)?.[1]?.trim();
      const record = FIGURES[kind];
      if (!record || record.mode !== "interactive") throw new Error(`Widget ${kind} is absent from the figure registry`);
      const svgPath = path.join(ROOT, record.web);
      const fallback = fs.existsSync(svgPath)
        ? `<img class="viz-plate" src="${record.web.replace(/^public\//, "")}" alt="${escapeHtml(label)}" loading="lazy" decoding="async" />`
        : (() => {
            const points = [8, 19, 13, 31, 22, 37, 28]
              .map((y, i) => `${8 + i * 14},${42 - y * 0.7}`)
              .join(" ");
            return `<svg viewBox="0 0 100 48" aria-hidden="true" focusable="false"><path d="M7 4v38h89"/>` +
              `<polyline points="${points}"/><circle cx="22" cy="29" r="2"/><circle cx="64" cy="20" r="2"/></svg>`;
          })();
      return `<figure class="viz" data-widget="${kind}"${attrs} aria-describedby="${id}">` +
        `<div class="viz-static">${fallback}</div>` +
        `<figcaption class="viz-fallback" id="${id}">${inlineCaption || escapeHtml(label)} ` +
        `<span class="viz-mode">Interactive controls enhance this static plate.</span></figcaption></figure>`;
    });
}

/** Render a deterministic Python-generated plate without requiring JavaScript.
 *
 * Canonical manuscript syntax:
 *   <figure class="viz" data-figure="power-function" data-alt="...">
 *     <figcaption>Interpretive caption.</figcaption>
 *   </figure>
 *
 * The matching SVG is generated into public/figures by tools/figures; the PDF
 * build resolves the same id to publication/figures/<id>.pdf.
 */
export function decorateStaticFigures(body) {
  return body.replace(
    /<figure class="viz" data-figure="([a-z0-9-]+)"([^>]*)>([\s\S]*?)<\/figure>/g,
    (whole, kind, attrs, inner) => {
      const alt = attrs.match(/\bdata-alt="([^"]+)"/)?.[1];
      const caption = inner.match(/<figcaption[^>]*>([\s\S]*?)<\/figcaption>/i)?.[1]?.trim();
      if (!alt || !caption) {
        throw new Error(`Static figure ${kind} requires data-alt and figcaption`);
      }
      const record = FIGURES[kind];
      if (!record || record.mode !== "static") throw new Error(`Static figure ${kind} is absent from the figure registry`);
      const svgPath = path.join(ROOT, record.web);
      if (!fs.existsSync(svgPath)) {
        throw new Error(`Static figure ${kind} is missing public/figures/${kind}.svg`);
      }
      return `<figure class="viz viz--static" data-figure="${kind}">` +
        `<div class="viz-static"><img class="viz-plate" src="${record.web.replace(/^public\//, "")}" ` +
        `alt="${escapeHtml(alt)}" loading="lazy" decoding="async" /></div>` +
        `<figcaption>${caption}</figcaption></figure>`;
    },
  );
}

export function chaptersFlat() {
  const out = [];
  for (const part of BOOK.parts) {
    for (const ch of part.chapters) out.push({ ...ch, part: part.part });
  }
  return out;
}

/** slug -> {label, title}. Preliminaries is unnumbered; the rest count 1..N. */
export function chapterMap() {
  const m = {};
  let i = 0;
  for (const ch of chaptersFlat()) {
    const label = ch.src === "ch-prelim" ? "the Preliminaries" : `Chapter ${i}`;
    m[ch.slug] = { label, title: ch.title };
    i++;
  }
  return m;
}

/** Replace [[ch:slug]] / [[ch:slug|text]] with a numbered chapter link. */
export function expandChrefs(body, cmap) {
  return body.replace(
    /\[\[ch:([a-z0-9-]+)(?:\|([^\]]+))?\]\]/g,
    (m, slug, custom) => {
      const e = cmap[slug];
      if (!e) return m; // leave unknown tokens visible for debugging
      const text = custom || e.label;
      return `<a class="chref" href="${slug}.html" title="${escapeHtml(e.title)}">${escapeHtml(text)}</a>`;
    },
  );
}

// ---- statement boxes ------------------------------------------------------

const STMT_KINDS = {
  def: "Definition",
  thm: "Theorem",
  lem: "Lemma",
  prop: "Proposition",
  cor: "Corollary",
};

/** Strip the leading kind word and outer parens from a box title, keeping the
 *  attribution/topic: "Theorem (Aronszajn, 1950)" -> "Aronszajn, 1950". */
function boxAnnotation(title, kindName) {
  let ann = title.trim();
  ann = ann.replace(new RegExp(`^${kindName}s?\\b[:.]?\\s*`, "i"), "").trim();
  // keep the outer parens: "(Aronszajn, 1950)" is the conventional form and
  // lets the citation pass link the attribution to the bibliography
  if (ann && !ann.startsWith("(")) ann = `(${ann})`;
  return ann;
}

function boxHead(kindLabel, id, ann) {
  const link = id
    ? `<a class="box-kind" href="#${id}">${kindLabel}</a>`
    : `<span class="box-kind">${kindLabel}</span>`;
  const annHtml = ann ? `<span class="box-ann">${ann}</span>` : "";
  return `<div class="box-head">${link}${annHtml}</div>`;
}

/**
 * Number the statement boxes within one chapter and give each a structured,
 * self-linkable header. Definitions/theorems/lemmas/propositions/corollaries
 * share one counter (Theorem 4.2 style); examples count separately; remarks
 * keep their free-form titles unnumbered.
 */
export function decorateBoxes(body, chLabel) {
  let nStmt = 0;
  let nEx = 0;
  let nAlgo = 0;
  body = body.replace(
    /<div class="box (thm|lem|prop|cor|def)"(?: id="([^"]+)")?>\s*<span class="box-title">([\s\S]*?)<\/span>/g,
    (m, kind, sourceId, title) => {
      nStmt++;
      const kindName = STMT_KINDS[kind];
      const num = `${chLabel}.${nStmt}`;
      const id = sourceId || `${kind}-${String(chLabel).toLowerCase()}-${nStmt}`;
      const ann = boxAnnotation(title, kindName);
      return `<div class="box ${kind}" id="${id}">${boxHead(`${kindName} ${num}`, id, ann)}`;
    },
  );
  body = body.replace(
    /<div class="box ex"(?: id="([^"]+)")?>\s*<span class="box-title">([\s\S]*?)<\/span>/g,
    (m, sourceId, title) => {
      const t = title.trim();
      if (/^Exercise/i.test(t)) {
        // a couple of inline exercises reuse the example styling; keep their title
        return `<div class="box ex">${boxHead(escapeHtml(t), null, "")}`;
      }
      nEx++;
      const num = `${chLabel}.${nEx}`;
      const id = sourceId || `example-${String(chLabel).toLowerCase()}-${nEx}`;
      const ann = boxAnnotation(t, "Example");
      return `<div class="box ex" id="${id}">${boxHead(`Example ${num}`, id, ann)}`;
    },
  );
  body = body.replace(
    /<div class="box algo"(?: id="([^"]+)")?>\s*<span class="box-title">([\s\S]*?)<\/span>/g,
    (m, sourceId, title) => {
      nAlgo++;
      const num = `${chLabel}.${nAlgo}`;
      const id = sourceId || `algo-${String(chLabel).toLowerCase()}-${nAlgo}`;
      const ann = boxAnnotation(title.trim(), "Algorithm");
      return `<div class="box algo" id="${id}">${boxHead(`Algorithm ${num}`, id, ann)}`;
    },
  );
  body = body.replace(
    /<div class="box rmk">\s*<span class="box-title">([\s\S]*?)<\/span>/g,
    (m, title) => `<div class="box rmk">${boxHead(title.trim(), null, "")}`,
  );
  return body;
}

/**
 * Convert proof boxes into collapsible <details>. Proof bodies contain no
 * nested <div> (verified across the corpus), so the first </div> closes the
 * box. Proofs without an explicit QED marker get one appended.
 */
export function transformProofs(body) {
  return body.replace(
    /<div class="box proof">\s*(?:<span class="box-title">([\s\S]*?)<\/span>)?([\s\S]*?)<\/div>/g,
    (m, title = "Proof", content) => {
      const inner = content.includes('class="qed"')
        ? content
        : `${content.replace(/\s+$/, "")} <span class="qed">∎</span>`;
      return (
        `<details class="proof"><summary><span class="proof-label">${title.trim()}</span>` +
        `<span class="proof-toggle" aria-hidden="true"></span></summary>` +
        `<div class="proof-body">${inner}</div></details>`
      );
    },
  );
}

// ---- citations ------------------------------------------------------------

/** Parse BibTeX's two common author forms, preserving surname particles and
 *  non-ASCII letters. */
const SURNAME_PARTICLES = new Set([
  "de", "del", "della", "van", "von", "der", "den", "la", "le", "di",
  "dos", "du", "ten", "ter", "st",
]);
function surnames(authors) {
  const commaForm = [];
  const commaPattern = /((?:(?:de|del|della|van|von|der|den|la|le|di|dos|du|ten|ter|st)\s+)*\p{Lu}[\p{L}'’-]+),\s+(?=\p{Lu})/gu;
  let commaMatch;
  while ((commaMatch = commaPattern.exec(authors))) commaForm.push(commaMatch[1]);
  if (commaForm.length) return commaForm;
  return authors
    .split(/\s+(and|&)\s+/u)
    .filter((part) => part !== "and" && part !== "&")
    .map((raw) => {
      const name = raw.trim();
      if (!name) return "";
      if (name.includes(",")) return name.slice(0, name.indexOf(",")).trim();
      const words = name.split(/\s+/u);
      let start = words.length - 1;
      while (
        start > 0 &&
        SURNAME_PARTICLES.has(words[start - 1].replace(/[.'’]/gu, "").toLowerCase())
      ) {
        start -= 1;
      }
      return words.slice(start).join(" ");
    })
    .filter(Boolean);
}

function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// accented letters in a bib surname should also match their ASCII form in prose
// ("Schölkopf" links a prose "Scholkopf"; "Matérn" links "Matern").
const ASCII_FOLD = {
  à: "a", á: "a", â: "a", ä: "a", ã: "a", å: "a", ç: "c", è: "e", é: "e",
  ê: "e", ë: "e", ì: "i", í: "i", î: "i", ï: "i", ı: "i", ñ: "n", ò: "o",
  ó: "o", ô: "o", ö: "o", õ: "o", ø: "o", ù: "u", ú: "u", û: "u", ü: "u",
  ý: "y", ÿ: "y", š: "s", ž: "z", č: "c", ć: "c", ř: "r", ł: "l", ð: "d",
};
function foldSurname(escaped) {
  return escaped.replace(/[À-ÿıšžčćřłĀ-ſ]/gu, (ch) => {
    const ascii = ASCII_FOLD[ch.toLowerCase()];
    if (!ascii) return ch;
    const a = ch === ch.toLowerCase() ? ascii : ascii.toUpperCase();
    return a === ch ? ch : `[${ch}${a}]`;
  });
}

/** Regexes matching the in-prose citation forms for a work, most-specific first.
 *  Covers narrative "Author (Year)" including multi-year "(2020, 2021)", and
 *  parenthetical "(Author, Year)" including grouped "(A, y1; B, y2)" lists and
 *  "(see ... Author Year)". Patterns use lookaround so the link wraps only the
 *  citation text, never the surrounding punctuation. */
function citePatterns(names, year) {
  const y = String(year);
  const esc = names.map((n) => foldSurname(escapeRe(n)));
  const poss = "(?:['’]s)?"; // allow a possessive "Neal's (1996)"
  // author-name tokens, most specific first
  const toks = [];
  if (esc.length >= 3) {
    const mid = esc.slice(0, -1).join(",?\\s+");
    toks.push(`${mid},?\\s+and\\s+${esc[esc.length - 1]}`);
  }
  if (esc.length >= 2) toks.push(`${esc[0]}\\s+and\\s+${esc[1]}`);
  if (esc.length) {
    toks.push(`${esc[0]}\\s+et\\s+al\\.?`);
    toks.push(esc[0]);
  }
  // the target year sitting anywhere in a comma-separated year list
  const yList = `(?:\\d{4}[a-z]?,\\s*)*${y}[a-z]?(?:,\\s*\\d{4}[a-z]?)*`;
  const pats = [];
  for (const t of toks) {
    // narrative: Author (Year) / Author's (Year) / Author (2020, 2021)
    pats.push(`${t}${poss}\\s+\\(${yList}\\)`);
    // parenthetical, incl. grouped lists: preceded by ( or ; , followed by ) ; ,
    pats.push(`(?<=[(;]\\s{0,2})${t},?\\s+${y}[a-z]?(?=\\s*[);,])`);
    // "(see ... in Author Year)": author mid-paren, year closes/separates
    pats.push(`(?<=[\\s(])${t}\\s+${y}[a-z]?(?=\\s*[);,])`);
  }
  return pats;
}

function refTooltip(r) {
  const bits = [];
  if (r.authors) bits.push(r.authors);
  if (r.year) bits.push(`(${r.year})`);
  if (r.title) bits.push(`${r.title}.`);
  if (r.venue) bits.push(`${r.venue}.`);
  return bits.join(" ");
}

/** Link in-prose 'Author (Year)' mentions of this chapter's cited works.
 *  Placeholder-based so links never nest. Returns [body, count]. */
export function linkCitations(body, keys, bib) {
  const stash = new Map();
  const ordered = [...keys].sort(
    (a, b) =>
      surnames(bib[b]?.authors || "").length -
      surnames(bib[a]?.authors || "").length,
  );
  for (const k of ordered) {
    const r = bib[k];
    if (!r) continue;
    const surn = surnames(r.authors || "");
    if (!surn.length || !r.year) continue;
    const tip = escapeHtml(refTooltip(r));
    for (const pat of citePatterns(surn, r.year)) {
      body = body.replace(new RegExp(pat, "g"), (match) => {
        const tok = `\x00C${stash.size}\x00`;
        stash.set(
          tok,
          `<a class="cite" href="bibliography.html#${k}" data-ref="${tip}">${match}</a>`,
        );
        return tok;
      });
    }
  }
  for (const [tok, link] of stash) body = body.split(tok).join(link);
  return [body, stash.size];
}

/** Resolve canonical Pandoc citation markers. The bibliography order remains
 * declared in frontmatter; an undeclared or unknown key stays visibly marked
 * so the content audit can reject it. */
export function expandCitationKeys(body, declaredKeys, bib) {
  const allowed = new Set(declaredKeys);
  let count = 0;
  body = body.replace(
    /<span class="citation"\s+data-cites="([^"]+)">[\s\S]*?<\/span>/g,
    (whole, packed) => {
      const keys = packed.trim().split(/\s+/);
      if (keys.some((key) => !allowed.has(key) || !bib[key])) return whole;
      count += keys.length;
      const labels = keys.map((key) => {
        const ref = bib[key];
        const names = surnames(ref.authors || "");
        const author = names.length > 2 ? `${names[0]} et al.` : names.join(" and ");
        return `<a class="cite" href="bibliography.html#${escapeHtml(key)}" data-ref="${escapeHtml(refTooltip(ref))}">${escapeHtml(author)}, ${escapeHtml(ref.year)}</a>`;
      });
      return `[${labels.join("; ")}]`;
    },
  );
  return [body, count];
}

/** One formatted bibliography entry, anchored at #key. */
export function fmtRef(key, r, withId = true) {
  const parts = [];
  if (r.authors) parts.push(`<span class="bib-authors">${escapeHtml(r.authors)}</span>`);
  if (r.year) parts.push(`<span class="bib-year">(${escapeHtml(String(r.year))})</span>`);
  if (r.title) parts.push(`<span class="bib-title">${escapeHtml(r.title)}</span>.`);
  if (r.venue) parts.push(`<span class="bib-venue">${escapeHtml(r.venue)}.</span>`);
  if (r.url) {
    const u = escapeHtml(r.url);
    parts.push(`<a href="${u}">${u}</a>`);
  }
  const id = withId ? ` id="${escapeHtml(key)}"` : "";
  return `<li${id}>${parts.join(" ")}</li>`;
}

function chapterRefsHtml(keys) {
  const rows = keys.map((k) =>
    BIB[k]
      ? fmtRef(k, BIB[k], false)
      : `<li>${escapeHtml(k)} (missing from bibliography.bib)</li>`,
  );
  if (!rows.length) return "";
  return (
    '<section class="chapter-refs"><h2 id="references">References</h2>' +
    `<ul class="bib-list">${rows.join("\n")}</ul>` +
    '<p class="hint">Full details, with every work cited across the book, are in the ' +
    '<a href="bibliography.html">bibliography</a>.</p></section>'
  );
}

// ---- navigation and search ------------------------------------------------

/** The chapter's top-level (h2) sections, for the sidebar drill-down:
 *  [{id, label}]. Kept to h2 so the active chapter stays compact in the TOC. */
export function chapterSections(body) {
  return [...body.matchAll(/<h2\s+id="([^"]+)"[^>]*>([\s\S]*?)<\/h2>/g)].map((m) => ({
    id: m[1],
    label: m[2]
      .replace(/<[^>]+>/g, "")
      .replace(/\\\(/g, "")
      .replace(/\\\)/g, "")
      .trim(),
  }));
}

/** The chapter's marginal map, set like a monograph's running apparatus. */
export function onpageNav(body) {
  const heads = [...body.matchAll(/<h([23])\s+id="([^"]+)"[^>]*>([\s\S]*?)<\/h[23]>/g)];
  if (heads.length < 2) return "";
  const rows = ['<div class="onpage-title">Chapter map</div>'];
  for (const [, lvl, hid, txt] of heads) {
    const label = txt
      .replace(/<[^>]+>/g, "")
      .replace(/\\\(/g, "")
      .replace(/\\\)/g, "")
      .trim();
    rows.push(`<a class="${lvl === "3" ? "l3" : "l2"}" href="#${hid}">${escapeHtml(label)}</a>`);
  }
  return `<aside class="onpage">${rows.join("\n")}</aside>`;
}

function plain(t, limit = 230) {
  t = t.replace(/\$\$[\s\S]*?\$\$/g, " ");
  t = t.replace(/\\\([\s\S]*?\\\)/g, " ");
  t = t.replace(/<[^>]+>/g, " ");
  t = t
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&hellip;/g, "…")
    .replace(/&middot;/g, "·")
    .replace(/&nbsp;/g, " ");
  return t.replace(/\s+/g, " ").trim().slice(0, limit);
}

// ---- the build ------------------------------------------------------------

let cache = null;

/** Process every chapter once; pages import the result. */
export function buildBook() {
  if (cache) return cache;
  const cmap = chapterMap();
  const chs = chaptersFlat();
  const searchIndex = [];
  const chapters = [];
  let ncites = 0;

  chs.forEach((ch, i) => {
    const canonical = readCanonicalChapter(ch.src);
    if (canonical.metadata.slug !== ch.slug || canonical.metadata.title !== ch.title) {
      throw new Error(`Canonical metadata mismatch for ${ch.src}`);
    }
    let body = renderCanonicalMarkdown(canonical.markdown);
    body = decorateWidgets(body);
    body = decorateStaticFigures(body);
    const isPrelim = ch.src === "ch-prelim";
    const chLabel = isPrelim ? "P" : String(i);

    body = expandChrefs(body, cmap);
    body = decorateBoxes(body, chLabel);
    body = transformProofs(body);
    // per-tag class on exercise pills so CSS can color them
    body = body.replace(
      /class="ex-tag">([a-z-]+)</g,
      (m, tag) => `class="ex-tag t-${tag}">${tag}<`,
    );
    body = decorateExercises(body, ch.slug);
    const chkeys = canonical.metadata.bibliography || [];
    const [linked, proseCitationCount] = linkCitations(body, chkeys, BIB);
    const [keyed, keyedCitationCount] = expandCitationKeys(linked, chkeys, BIB);
    body = keyed;
    ncites += proseCitationCount + keyedCitationCount;

    const onpage = onpageNav(body);
    const sections = chapterSections(body);

    // search index: the chapter lead, then one entry per h2/h3 section
    const num = isPrelim ? "Preliminaries" : i;
    const lead = body.match(/<p class="lead">([\s\S]*?)<\/p>/);
    searchIndex.push({
      h: `${num}. ${ch.title}`,
      c: ch.part,
      u: `${ch.slug}.html`,
      x: lead ? plain(lead[1]) : "",
    });
    const heads = [...body.matchAll(/<h([23])\s+id="([^"]+)"[^>]*>([\s\S]*?)<\/h[23]>/g)];
    heads.forEach((m, k) => {
      const end = k + 1 < heads.length ? heads[k + 1].index : body.length;
      searchIndex.push({
        h: plain(m[3], 90),
        c: `${num}. ${ch.title}`,
        u: `${ch.slug}.html#${m[2]}`,
        x: plain(body.slice(m.index + m[0].length, end)),
      });
    });

    // chapter number header injected into the first h1
    const chno = isPrelim
      ? "Preliminaries"
      : `Chapter ${i} &middot; ${escapeHtml(ch.part)}`;
    body = body.replace("<h1>", `<h1><span class="chno">${chno}</span>`);
    body += chapterRefsHtml(chkeys);
    body = decorateScrollableRegions(renderMath(body));

    const widgets = [...new Set(
      [...body.matchAll(/data-widget="([a-z0-9-]+)"/g)].map((match) => match[1]),
    )];
    chapters.push({ ...ch, i, isPrelim, body, onpage, sections, widgets });
  });

  // prev/next cards
  const label = (k) =>
    chs[k].src === "ch-prelim"
      ? "Preliminaries"
      : `${k}. ${escapeHtml(chs[k].title)}`;
  chapters.forEach((ch, i) => {
    const prevHref = i > 0 ? `${chs[i - 1].slug}.html` : "index.html";
    const nextHref = i < chs.length - 1 ? `${chs[i + 1].slug}.html` : "";
    const nav = ['<nav class="chnav" aria-label="Chapter navigation">'];
    nav.push(
      i > 0
        ? `<a class="nav-card prev" href="${prevHref}"><span class="dir">&larr; Previous</span><span class="nav-title">${label(i - 1)}</span></a>`
        : '<a class="nav-card prev" href="index.html"><span class="dir">&larr; Previous</span><span class="nav-title">Contents</span></a>',
    );
    nav.push(
      nextHref
        ? `<a class="nav-card next" href="${nextHref}"><span class="dir">Next &rarr;</span><span class="nav-title">${label(i + 1)}</span></a>`
        : '<a class="nav-card next" href="bibliography.html"><span class="dir">Next &rarr;</span><span class="nav-title">Bibliography</span></a>',
    );
    nav.push("</nav>");
    ch.body += nav.join("\n");
    ch.prevHref = prevHref;
    ch.nextHref = nextHref || "bibliography.html";
  });

  // bibliography, alphabetical by author then year
  const bibEntries = Object.entries(BIB).sort(([, a], [, b]) => {
    const ka = `${String(a.authors || "zzz").toLowerCase()}|${a.year || ""}`;
    const kb = `${String(b.authors || "zzz").toLowerCase()}|${b.year || ""}`;
    return ka < kb ? -1 : ka > kb ? 1 : 0;
  });

  searchIndex.push({
    h: "Notation and Glossary",
    c: "End matter",
    u: "glossary.html",
    x: "Symbols and terms used across the book, cross-linked to the chapters that introduce them.",
  });
  searchIndex.push({
    h: "Dependency Map",
    c: "End matter",
    u: "dependency-map.html",
    x: "The logical scaffolding of the book: which theorem, definition, and algorithm each result depends on, as an interactive graph and a full cross-linked listing.",
  });
  searchIndex.push({
    h: "Projects and Competitions",
    c: "End matter",
    u: "projects.html",
    x: "Build-it capstones and a final project, each with a dataset, a metric, and a measured baseline, and each ready to run as a Kaggle community competition.",
  });
  searchIndex.push({
    h: "Bibliography",
    c: "End matter",
    u: "bibliography.html",
    x: `Every work cited across the book, ${bibEntries.length} entries.`,
  });

  cache = { chapters, searchIndex, bibEntries, ncites, cmap };
  return cache;
}

/**
 * Every numbered statement in the book (definitions, theorems, lemmas,
 * propositions, corollaries, algorithms), with the same ids the build gives
 * their boxes. This is the node set of the dependency map; the curated edges
 * in depmap/edges/*.json reference these nodes by `chapterSlug#id`.
 */
export function statementIndex() {
  const chs = chaptersFlat();
  const nodes = [];
  const KIND_LABEL = { ...STMT_KINDS, algo: "Algorithm" };
  chs.forEach((ch, i) => {
    const isPrelim = ch.src === "ch-prelim";
    const chLabel = isPrelim ? "P" : String(i);
    const body = renderCanonicalMarkdown(readCanonicalChapter(ch.src).markdown);
    let nStmt = 0;
    let nAlgo = 0;
    const re =
      /<div class="box (thm|lem|prop|cor|def|algo)"(?: id="([^"]+)")?>\s*<span class="box-title">([\s\S]*?)<\/span>/g;
    let m;
    while ((m = re.exec(body))) {
      const kind = m[1];
      let num, id;
      if (kind === "algo") {
        nAlgo++;
        num = `${chLabel}.${nAlgo}`;
        id = m[2] || `algo-${chLabel.toLowerCase()}-${nAlgo}`;
      } else {
        nStmt++;
        num = `${chLabel}.${nStmt}`;
        id = m[2] || `${kind}-${chLabel.toLowerCase()}-${nStmt}`;
      }
      const title = m[3]
        .replace(/<[^>]+>/g, "")
        .replace(/\\\(/g, "")
        .replace(/\\\)/g, "")
        .replace(/\s+/g, " ")
        .trim();
      nodes.push({
        key: `${ch.slug}#${id}`,
        id,
        kind,
        kindLabel: `${KIND_LABEL[kind]} ${num}`,
        chapter: ch.slug,
        chapterTitle: ch.title,
        chapterNum: isPrelim ? "P" : i,
        src: ch.src,
        title,
      });
    }
  });
  return nodes;
}

/** The curated dependency edges, merged from depmap/edges/*.json. */
export function dependencyEdges() {
  const dir = path.join(ROOT, "depmap", "edges");
  if (!fs.existsSync(dir)) return [];
  const edges = [];
  const seen = new Set();
  for (const f of fs.readdirSync(dir).sort()) {
    if (!f.endsWith(".json")) continue;
    for (const e of JSON.parse(fs.readFileSync(path.join(dir, f), "utf8"))) {
      const sig = `${e.from}->${e.to}`;
      if (seen.has(sig) || e.from === e.to) continue;
      seen.add(sig);
      edges.push({ from: e.from, to: e.to, note: e.note || "" });
    }
  }
  return edges;
}

/** Sidebar table of contents (shared by every page). Each part is a
 *  collapsible <details>; the part holding the current chapter is open, and the
 *  active chapter expands to a nested list of its sections (passed in from the
 *  page). `sections` is [{id,label}] (the active chapter's h2 headings). */
export function tocHtml(currentSlug = null, sections = []) {
  const partHtml = (label, inner, open) =>
    `<details class="toc-part"${open ? " open" : ""}><summary>${escapeHtml(label)}</summary>${inner}</details>`;
  const rows = [
    `<div class="booktitle"><a href="index.html">${escapeHtml(BOOK.title)}</a></div>`,
    `<div class="bookmeta">${escapeHtml(BOOK.subtitle)}</div>`,
  ];
  rows.push(partHtml(
    "Front matter",
    `<a href="preface.html"${currentSlug === "preface" ? ' class="here"' : ""}>Dedication &amp; Preface</a>`,
    currentSlug === "preface",
  ));
  let n = 0;

  for (const part of BOOK.parts) {
    const links = [];
    let hasCurrent = false;
    for (const ch of part.chapters) {
      const here = ch.slug === currentSlug;
      if (here) hasCurrent = true;
      const marker = ch.src === "ch-prelim" ? "&middot;" : String(++n);
      links.push(
        `<a href="${ch.slug}.html"${here ? ' class="here"' : ""}>${marker}. ${escapeHtml(ch.title)}</a>`,
      );
      if (here && sections.length) {
        const secs = sections
          .map((s) => `<a class="sec" href="#${s.id}">${escapeHtml(s.label)}</a>`)
          .join("");
        links.push(`<div class="toc-sections">${secs}</div>`);
      }
    }
    rows.push(partHtml(part.part, links.join("\n"), hasCurrent));
  }

  const endLinks = [
    `<a href="reading-paths.html"${currentSlug === "reading-paths" ? ' class="here"' : ""}>Reading Paths</a>`,
    `<a href="glossary.html"${currentSlug === "glossary" ? ' class="here"' : ""}>Notation &amp; Glossary</a>`,
    `<a href="dependency-map.html"${currentSlug === "dependency-map" ? ' class="here"' : ""}>Dependency Map</a>`,
    `<a href="indexes.html"${currentSlug === "indexes" ? ' class="here"' : ""}>Indexes</a>`,
    `<a href="solutions.html"${currentSlug === "solutions" ? ' class="here"' : ""}>Answers &amp; Rubrics</a>`,
    `<a href="projects.html"${currentSlug === "projects" ? ' class="here"' : ""}>Projects &amp; Competitions</a>`,
    `<a href="bibliography.html"${currentSlug === "bibliography" ? ' class="here"' : ""}>Bibliography</a>`,
    `<a href="about.html"${currentSlug === "about" ? ' class="here"' : ""}>About This Edition</a>`,
  ];
  const endOpen = ["reading-paths", "glossary", "dependency-map", "indexes", "solutions", "projects", "bibliography", "about"].includes(currentSlug);
  rows.push(partHtml("End matter", endLinks.join("\n"), endOpen));
  return rows.join("\n");
}
