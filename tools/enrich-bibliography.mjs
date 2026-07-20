#!/usr/bin/env node
/**
 * Conservatively enrich missing BibTeX links from OpenAlex.
 *
 * Results are cached under .context so an interrupted run is resumable.
 * A candidate is accepted only when its normalized title is an exact match, or
 * when token overlap is very high and the publication year is compatible.
 */
import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const bibPath = path.join(root, "bibliography.bib");
const cachePath = path.join(root, ".context", "bibliography-enrichment-cache.json");
const write = process.argv.includes("--write");
const localOnly = process.argv.includes("--local-only");
const useDblp = process.argv.includes("--dblp");
const useCrossref = process.argv.includes("--crossref");
const source = fs.readFileSync(bibPath, "utf8");
const entryRe = /@(misc|article|book|inproceedings)\s*\{\s*([^,\s]+)\s*,([\s\S]*?)\n\}/gi;
const fieldRe = /^\s*([A-Za-z]+)\s*=\s*\{([^{}]*)\}\s*,?\s*$/gm;
const cache = fs.existsSync(cachePath)
  ? JSON.parse(fs.readFileSync(cachePath, "utf8"))
  : {};

function fields(block) {
  const out = {};
  let match;
  fieldRe.lastIndex = 0;
  while ((match = fieldRe.exec(block))) out[match[1].toLowerCase()] = match[2];
  return out;
}

function normalized(text) {
  return String(text || "")
    .normalize("NFKD")
    .replace(/\\[A-Za-z]+/g, " ")
    .replace(/[{}]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function similarity(left, right) {
  const a = new Set(normalized(left).split(/\s+/).filter(Boolean));
  const b = new Set(normalized(right).split(/\s+/).filter(Boolean));
  if (!a.size || !b.size) return 0;
  let overlap = 0;
  for (const token of a) if (b.has(token)) overlap += 1;
  return (2 * overlap) / (a.size + b.size);
}

async function lookup(entry) {
  if (cache[entry.key]) return cache[entry.key];
  const params = new URLSearchParams({
    search: entry.title,
    "per-page": "3",
    select: "id,doi,title,publication_year,primary_location",
  });
  let response;
  for (let attempt = 0; attempt < 5; attempt++) {
    response = await fetch("https://api.openalex.org/works?" + params, {
      headers: {
        "User-Agent": "KernelMethodsBook/1.1 (+https://www.tahabouhsine.com)",
      },
      signal: AbortSignal.timeout(10000),
    });
    if (response.status !== 429) break;
    await new Promise((resolve) => setTimeout(resolve, 2000 * (attempt + 1)));
  }
  if (!response.ok) throw new Error("OpenAlex HTTP " + response.status);
  const items = (await response.json()).results || [];
  const ranked = items
    .map((item) => ({
      item,
      score: similarity(entry.title, item.title || ""),
      year: item.publication_year ? Number(item.publication_year) : null,
    }))
    .sort((a, b) => b.score - a.score);
  const best = ranked[0];
  const year = Number(entry.year);
  const yearCompatible =
    !best?.year || !Number.isFinite(year) || Math.abs(best.year - year) <= 1;
  const exact = best && normalized(entry.title) === normalized(best.item.title);
  const accepted = Boolean(best && yearCompatible && (exact || best.score >= 0.9));
  cache[entry.key] = accepted
    ? {
        accepted: true,
        score: best.score,
        matched_title: best.item.title || "",
        matched_year: best.year,
        doi: best.item.doi
          ? best.item.doi.replace(/^https?:\/\/doi\.org\//i, "")
          : null,
        url:
          best.item.doi ||
          best.item.primary_location?.landing_page_url ||
          null,
      }
    : {
        accepted: false,
        score: best?.score || 0,
        matched_title: best?.item?.title || null,
        matched_year: best?.year || null,
      };
  fs.mkdirSync(path.dirname(cachePath), { recursive: true });
  fs.writeFileSync(cachePath, JSON.stringify(cache, null, 2) + "\n");
  return cache[entry.key];
}

async function lookupDblp(entry) {
  const params = new URLSearchParams({
    q: entry.title,
    h: "3",
    format: "json",
  });
  let response;
  for (let attempt = 0; attempt < 5; attempt++) {
    response = await fetch("https://dblp.org/search/publ/api?" + params, {
      headers: {
        "User-Agent": "KernelMethodsBook/1.1 (+https://www.tahabouhsine.com)",
      },
      signal: AbortSignal.timeout(10000),
    });
    if (response.status !== 429) break;
    await new Promise((resolve) => setTimeout(resolve, 2000 * (attempt + 1)));
  }
  if (!response.ok) throw new Error("DBLP HTTP " + response.status);
  let hits = (await response.json()).result?.hits?.hit || [];
  if (!Array.isArray(hits)) hits = [hits];
  const ranked = hits
    .map((hit) => ({
      item: hit.info || {},
      score: similarity(entry.title, hit.info?.title || ""),
      year: Number(hit.info?.year) || null,
    }))
    .sort((a, b) => b.score - a.score);
  const best = ranked[0];
  const year = Number(entry.year);
  const yearCompatible =
    !best?.year || !Number.isFinite(year) || Math.abs(best.year - year) <= 1;
  const exact = best && normalized(entry.title) === normalized(best.item.title);
  let electronic = best?.item?.ee || null;
  if (Array.isArray(electronic)) electronic = electronic[0] || null;
  const accepted = Boolean(
    best && electronic && yearCompatible && (exact || best.score >= 0.9),
  );
  cache[entry.key] = accepted
    ? {
        accepted: true,
        score: best.score,
        matched_title: best.item.title,
        matched_year: best.year,
        doi: /^https?:\/\/doi\.org\//i.test(electronic)
          ? electronic.replace(/^https?:\/\/doi\.org\//i, "")
          : null,
        url: electronic,
        source: "dblp-electronic-edition",
      }
    : {
        accepted: false,
        score: best?.score || 0,
        matched_title: best?.item?.title || null,
        matched_year: best?.year || null,
        source: "dblp",
      };
  fs.mkdirSync(path.dirname(cachePath), { recursive: true });
  fs.writeFileSync(cachePath, JSON.stringify(cache, null, 2) + "\n");
  return cache[entry.key];
}

async function lookupCrossref(entry) {
  const params = new URLSearchParams({
    // Title search is materially more reliable for classic papers whose exact
    // titles were later reprinted in books; year-aware ranking below selects
    // the original record when Crossref returns both editions.
    "query.title": entry.title,
    rows: "10",
    select: "DOI,title,published,published-print,published-online,issued,URL",
  });
  let response;
  for (let attempt = 0; attempt < 5; attempt++) {
    response = await fetch("https://api.crossref.org/works?" + params, {
      headers: {
        "User-Agent": "KernelMethodsBook/1.1 (+https://www.tahabouhsine.com)",
      },
      signal: AbortSignal.timeout(10000),
    });
    if (response.status !== 429) break;
    const retryAfter = Number(response.headers.get("retry-after")) || 2 * (attempt + 1);
    await new Promise((resolve) => setTimeout(resolve, retryAfter * 1000));
  }
  if (!response.ok) throw new Error("Crossref HTTP " + response.status);
  const items = (await response.json()).message?.items || [];
  const queryYear = Number(entry.year);
  const ranked = items
    .map((item) => {
      const date = item["published-print"] || item["published-online"] || item.published || item.issued;
      const title = Array.isArray(item.title) ? item.title[0] : item.title || "";
      return {
        item,
        title,
        score: similarity(entry.title, title),
        year: Number(date?.["date-parts"]?.[0]?.[0]) || null,
      };
    })
    .map((candidate) => ({
      ...candidate,
      exact: normalized(entry.title) === normalized(candidate.title),
      year_delta:
        candidate.year && Number.isFinite(queryYear)
          ? Math.abs(candidate.year - queryYear)
          : 0,
    }))
    .sort((a, b) =>
      Number(b.year_delta <= 1) - Number(a.year_delta <= 1) ||
      Number(b.exact) - Number(a.exact) ||
      b.score - a.score ||
      a.year_delta - b.year_delta,
    );
  const best = ranked[0];
  const yearCompatible =
    !best?.year || !Number.isFinite(queryYear) || Math.abs(best.year - queryYear) <= 1;
  const exact = best?.exact;
  const accepted = Boolean(
    best && best.item.DOI && yearCompatible && (exact || best.score >= 0.9),
  );
  cache[entry.key] = accepted
    ? {
        accepted: true,
        score: best.score,
        matched_title: best.title,
        matched_year: best.year,
        doi: best.item.DOI,
        url: "https://doi.org/" + best.item.DOI,
        source: "crossref",
      }
    : {
        accepted: false,
        score: best?.score || 0,
        matched_title: best?.title || null,
        matched_year: best?.year || null,
        source: "crossref",
      };
  fs.mkdirSync(path.dirname(cachePath), { recursive: true });
  fs.writeFileSync(cachePath, JSON.stringify(cache, null, 2) + "\n");
  return cache[entry.key];
}

const entries = [];
let match;
while ((match = entryRe.exec(source))) {
  const data = fields(match[3]);
  if (!data.url && !data.doi) {
    entries.push({
      key: match[2],
      title: data.title,
      year: data.year,
      venue: data.howpublished || data.journal || data.booktitle || "",
    });
  }
}

for (const entry of entries) {
  const arxiv = entry.venue.match(/arXiv\s*:?\s*(\d{4}\.\d{4,5})(?:v\d+)?/i)?.[1];
  if (arxiv) {
    cache[entry.key] = {
      accepted: true,
      score: 1,
      matched_title: entry.title,
      matched_year: Number(entry.year),
      doi: null,
      url: "https://arxiv.org/abs/" + arxiv,
      source: "explicit-arxiv-identifier",
    };
  }
}
fs.mkdirSync(path.dirname(cachePath), { recursive: true });
fs.writeFileSync(cachePath, JSON.stringify(cache, null, 2) + "\n");

const queryEntries = localOnly
  ? []
  : entries.filter((entry) => !cache[entry.key]?.accepted);
for (let index = 0; index < queryEntries.length; index += 2) {
  const batch = queryEntries.slice(index, index + 2);
  await Promise.all(
    batch.map(async (entry) => {
      try {
        if (useCrossref) await lookupCrossref(entry);
        else if (useDblp) await lookupDblp(entry);
        else await lookup(entry);
      } catch (error) {
        console.error("WARN", entry.key, error.message);
      }
    }),
  );
  await new Promise((resolve) => setTimeout(resolve, useDblp || useCrossref ? 600 : 250));
  if ((index + batch.length) % 50 === 0 || index + batch.length === queryEntries.length) {
    console.log(
      (useCrossref ? "Crossref" : useDblp ? "DBLP" : "OpenAlex") + " " +
      (index + batch.length) + "/" + queryEntries.length,
    );
  }
}

let enriched = 0;
const updated = source.replace(entryRe, (whole, type, key, block) => {
  const data = fields(block);
  const found = cache[key];
  if (data.url || data.doi || !found?.accepted || !found.url) return whole;
  enriched += 1;
  const additions = [];
  if (found.doi) additions.push("  doi = {" + found.doi + "}");
  additions.push("  url = {" + found.url + "}");
  const trimmed = block.replace(/\s+$/, "").replace(/,?$/, ",");
  return "@" + type + "{" + key + "," + trimmed + "\n" + additions.join(",\n") + "\n}";
});

const rejected = entries.filter((entry) => !cache[entry.key]?.accepted).length;
const acceptedWithoutUrl = entries.filter(
  (entry) => cache[entry.key]?.accepted && !cache[entry.key]?.url,
).length;
console.log(
  "Accepted links: " + enriched +
  "; unresolved: " + rejected +
  "; accepted records without URL: " + acceptedWithoutUrl,
);
if (write) {
  fs.writeFileSync(bibPath, updated);
  console.log("Updated bibliography.bib");
} else {
  console.log("Dry run only; pass --write to update bibliography.bib");
}
