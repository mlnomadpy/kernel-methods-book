#!/usr/bin/env node
/** Add exact-title links from official JMLR and PMLR proceedings indexes. */
import fs from "node:fs";

const write = process.argv.includes("--write");
const bibPath = "bibliography.bib";
const source = fs.readFileSync(bibPath, "utf8");
const entryRe = /@(misc|article|book|inproceedings)\s*\{\s*([^,\s]+)\s*,([\s\S]*?)\n\}/gi;
const fieldRe = /^\s*([A-Za-z]+)\s*=\s*\{([^{}]*)\}\s*,?\s*$/gm;

function fields(block) {
  const result = {};
  let match;
  fieldRe.lastIndex = 0;
  while ((match = fieldRe.exec(block))) result[match[1].toLowerCase()] = match[2];
  return result;
}

function decodeHtml(text) {
  return text
    .replace(/<[^>]+>/g, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&quot;/gi, '"')
    .replace(/&#39;|&apos;/gi, "'")
    .replace(/&#(\d+);/g, (_, value) => String.fromCodePoint(Number(value)))
    .replace(/&#x([0-9a-f]+);/gi, (_, value) => String.fromCodePoint(Number.parseInt(value, 16)));
}

function normalized(text) {
  return decodeHtml(String(text || ""))
    .normalize("NFKD")
    .replace(/\\[A-Za-z]+/g, " ")
    .replace(/[{}]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

async function get(url) {
  const response = await fetch(url, {
    headers: { "User-Agent": "KernelMethodsBook/1.1 (+https://www.tahabouhsine.com)" },
    signal: AbortSignal.timeout(20000),
  });
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response.text();
}

const entries = [];
let entryMatch;
while ((entryMatch = entryRe.exec(source))) {
  const data = fields(entryMatch[3]);
  if (!data.url && !data.doi) {
    entries.push({
      key: entryMatch[2],
      title: data.title || "",
      venue: data.journal || data.booktitle || data.howpublished || "",
    });
  }
}

const official = new Map();
const jmlrVolumes = [...new Set(entries.flatMap((entry) => {
  const match = entry.venue.match(/Journal of Machine Learning Research\s+(\d+)/i);
  return match ? [Number(match[1])] : [];
}))].sort((a, b) => a - b);

for (const volume of jmlrVolumes) {
  try {
    const html = await get(`https://www.jmlr.org/papers/v${volume}/`);
    const paperRe = /<dl\b[^>]*>([\s\S]*?)<\/dl>/gi;
    let paper;
    while ((paper = paperRe.exec(html))) {
      const title = paper[1].match(/<dt>([\s\S]*?)(?:<\/dt>|\r?\n)/i)?.[1];
      const hrefMatch = paper[1].match(
        /<a\s+href=(?:"([^"]+\.html)"|'([^']+\.html)'|([^\s>]+\.html))>\[?abs\]?<\/a>/i,
      );
      if (!title || !hrefMatch) continue;
      const href = hrefMatch[1] || hrefMatch[2] || hrefMatch[3];
      official.set(
        normalized(title),
        new URL(href, `https://www.jmlr.org/papers/v${volume}/`).href.replace(/^http:/, "https:"),
      );
    }
  } catch (error) {
    console.error(`WARN JMLR volume ${volume}: ${error.message}`);
  }
}

// Explicit volumes plus volumes for venue records that predate standardized
// PMLR metadata in this bibliography.
const pmlrVolumes = new Set([5, 9, 22, 28, 31, 32, 37, 38, 48, 51, 70, 84, 89, 97, 119, 130, 139]);
for (const entry of entries) {
  for (const match of entry.venue.matchAll(/PMLR\s+(\d+)/gi)) pmlrVolumes.add(Number(match[1]));
}

for (const volume of [...pmlrVolumes].sort((a, b) => a - b)) {
  try {
    const html = await get(`https://proceedings.mlr.press/v${volume}/`);
    const paperRe = /<div class="paper">[\s\S]*?<p class="title">([\s\S]*?)<\/p>[\s\S]*?<p class="links">[\s\S]*?<a href="([^"]+\.html)">abs<\/a>/gi;
    let paper;
    while ((paper = paperRe.exec(html))) official.set(normalized(paper[1]), paper[2].replace(/^http:/, "https:"));
  } catch (error) {
    console.error(`WARN PMLR volume ${volume}: ${error.message}`);
  }
}

const matches = new Map();
for (const entry of entries) {
  const url = official.get(normalized(entry.title));
  if (url) matches.set(entry.key, url);
}

let enriched = 0;
const updated = source.replace(entryRe, (whole, type, key, block) => {
  const data = fields(block);
  const url = matches.get(key);
  if (data.url || data.doi || !url) return whole;
  enriched += 1;
  const trimmed = block.replace(/\s+$/, "").replace(/,?$/, ",");
  return `@${type}{${key},${trimmed}\n  url = {${url}}\n}`;
});

console.log(
  `Official venue indexes: ${official.size} titles loaded; ${enriched} exact unresolved titles matched.`,
);
if (write) {
  fs.writeFileSync(bibPath, updated);
  console.log("Updated bibliography.bib");
} else {
  for (const [key, url] of matches) console.log(`${key}\t${url}`);
  console.log("Dry run only; pass --write to update bibliography.bib");
}
