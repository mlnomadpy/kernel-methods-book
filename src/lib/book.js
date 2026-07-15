/**
 * The book pipeline: reads book.json, chapter body fragments, the bibliography,
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

const ROOT = path.resolve(process.cwd());

function readJSON(p) {
  return JSON.parse(fs.readFileSync(path.join(ROOT, p), "utf8"));
}

export const BOOK = readJSON("book.json");
export const BIB = readJSON("bibliography.json");
export const GLOSSARY = readJSON("glossary.json");

export function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
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
  body = body.replace(
    /<div class="box (thm|lem|prop|cor|def)">\s*<span class="box-title">([\s\S]*?)<\/span>/g,
    (m, kind, title) => {
      nStmt++;
      const kindName = STMT_KINDS[kind];
      const num = `${chLabel}.${nStmt}`;
      const id = `${kind}-${String(chLabel).toLowerCase()}-${nStmt}`;
      const ann = boxAnnotation(title, kindName);
      return `<div class="box ${kind}" id="${id}">${boxHead(`${kindName} ${num}`, id, ann)}`;
    },
  );
  body = body.replace(
    /<div class="box ex">\s*<span class="box-title">([\s\S]*?)<\/span>/g,
    (m, title) => {
      const t = title.trim();
      if (/^Exercise/i.test(t)) {
        // a couple of inline exercises reuse the example styling; keep their title
        return `<div class="box ex">${boxHead(escapeHtml(t), null, "")}`;
      }
      nEx++;
      const num = `${chLabel}.${nEx}`;
      const id = `example-${String(chLabel).toLowerCase()}-${nEx}`;
      const ann = boxAnnotation(t, "Example");
      return `<div class="box ex" id="${id}">${boxHead(`Example ${num}`, id, ann)}`;
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

/** 'Schölkopf, B. and Smola, A. J.' -> ['Schölkopf', 'Smola'] */
function surnames(authors) {
  const out = [];
  const re = /([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]+),\s+(?:[A-Z]\.[- ]?)+/g;
  let m;
  while ((m = re.exec(authors))) out.push(m[1]);
  return out;
}

function escapeRe(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Regexes matching the in-prose citation forms for a work, longest first. */
function citePatterns(names, year) {
  const y = String(year);
  const esc = names.map(escapeRe);
  const pats = [];
  if (esc.length >= 3) {
    const mid = esc.slice(0, -1).join(",?\\s+");
    pats.push(`${mid},?\\s+and\\s+${esc[esc.length - 1]}\\s+\\(${y}\\)`);
    pats.push(`\\(\\s*${mid},?\\s+and\\s+${esc[esc.length - 1]},?\\s+${y}\\s*\\)`);
  }
  if (esc.length >= 2) {
    pats.push(`${esc[0]}\\s+and\\s+${esc[1]}\\s+\\(${y}\\)`);
    pats.push(`\\(\\s*${esc[0]}\\s+and\\s+${esc[1]},?\\s+${y}\\s*\\)`);
  }
  if (esc.length) {
    pats.push(`${esc[0]}\\s+et\\s+al\\.?\\s*\\(${y}\\)`);
    pats.push(`\\(\\s*${esc[0]}\\s+et\\s+al\\.?,?\\s+${y}\\s*\\)`);
    pats.push(`${esc[0]}\\s+\\(${y}\\)`);
    pats.push(`\\(\\s*${esc[0]},?\\s+${y}\\s*\\)`);
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

function chapterRefsHtml(src) {
  const p = path.join(ROOT, "chapters", "refs", `${src}.json`);
  if (!fs.existsSync(p)) return "";
  const keys = JSON.parse(fs.readFileSync(p, "utf8"));
  const rows = keys.map((k) =>
    BIB[k]
      ? fmtRef(k, BIB[k], false)
      : `<li>${escapeHtml(k)} (missing from bibliography.json)</li>`,
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

/** The 'On this page' rail from the chapter's h2/h3 headings. */
export function onpageNav(body) {
  const heads = [...body.matchAll(/<h([23])\s+id="([^"]+)"[^>]*>([\s\S]*?)<\/h[23]>/g)];
  if (heads.length < 2) return "";
  const rows = ['<div class="onpage-title">On this page</div>'];
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
    const srcPath = path.join(ROOT, "chapters", "src", `${ch.src}.body.html`);
    let body = fs.readFileSync(srcPath, "utf8");
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
    const keyfile = path.join(ROOT, "chapters", "refs", `${ch.src}.json`);
    const chkeys = fs.existsSync(keyfile)
      ? JSON.parse(fs.readFileSync(keyfile, "utf8"))
      : [];
    const [linked, nc] = linkCitations(body, chkeys, BIB);
    body = linked;
    ncites += nc;

    const onpage = onpageNav(body);

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
    body += chapterRefsHtml(ch.src);

    chapters.push({ ...ch, i, isPrelim, body, onpage });
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
    h: "Bibliography",
    c: "End matter",
    u: "bibliography.html",
    x: `Every work cited across the book, ${bibEntries.length} entries.`,
  });

  cache = { chapters, searchIndex, bibEntries, ncites, cmap };
  return cache;
}

/** Sidebar table of contents (shared by every page). */
export function tocHtml(currentSlug = null) {
  const rows = [
    `<div class="booktitle"><a href="index.html">${escapeHtml(BOOK.title)}</a></div>`,
    `<div class="bookmeta">${escapeHtml(BOOK.subtitle)}</div>`,
  ];
  let n = 0;
  for (const part of BOOK.parts) {
    rows.push(`<div class="part">${escapeHtml(part.part)}</div>`);
    for (const ch of part.chapters) {
      n++;
      const cls = ch.slug === currentSlug ? ' class="here"' : "";
      rows.push(`<a href="${ch.slug}.html"${cls}>${n}. ${escapeHtml(ch.title)}</a>`);
    }
  }
  rows.push('<div class="part">End matter</div>');
  rows.push(
    `<a href="glossary.html"${currentSlug === "glossary" ? ' class="here"' : ""}>Notation &amp; Glossary</a>`,
  );
  rows.push(
    `<a href="bibliography.html"${currentSlug === "bibliography" ? ' class="here"' : ""}>Bibliography</a>`,
  );
  return rows.join("\n");
}
