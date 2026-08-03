#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { parse as parseYaml } from "yaml";
import { renderCanonicalMarkdown } from "../src/lib/manuscript.js";
import {
  expandBookObjectReferences,
  mergeBookObjectReferences,
  numberBookObjects,
} from "../src/lib/numbering.js";

const root = process.cwd();
const book = parseYaml(fs.readFileSync(path.join(root, "book.yml"), "utf8"));
const chapters = book.parts.flatMap((part) => part.chapters);
const errors = [];
const rendered = [];

for (const [index, chapter] of chapters.entries()) {
  const file = path.join(root, "manuscript", "chapters", `${chapter.src}.md`);
  const source = matter(fs.readFileSync(file, "utf8")).content;
  const prose = source.replace(/^(`{3,}|~{3,})[^\n]*\n[\s\S]*?^\1\s*$/gm, "");
  if (/\\tag\{/.test(prose)) errors.push(`${chapter.src}: manual \\tag equation number`);
  for (const match of prose.matchAll(
    /\b(equation|figure|table|listing|code|identity|bound|display|objective|expression)\s+(above|below)\b/gi,
  )) errors.push(`${chapter.src}: fragile positional reference '${match[0]}'`);

  const html = renderCanonicalMarkdown(source);
  try {
    rendered.push({
      chapter,
      numbered: numberBookObjects(html, {
        chapterLabel: chapter.src === "ch-prelim" ? "P" : String(index),
        chapterSlug: chapter.slug,
        chapterTitle: chapter.title,
      }),
    });
  } catch (error) {
    errors.push(`${chapter.src}: ${error.message}`);
  }
}

let refs;
try {
  refs = mergeBookObjectReferences(rendered.map(({ numbered }) => numbered.refs));
} catch (error) {
  errors.push(error.message);
}

if (refs) {
  for (const { chapter, numbered } of rendered) {
    try {
      const expanded = expandBookObjectReferences(numbered.body, refs, chapter.slug);
      if (/\[\[(?:eq|fig|tbl|lst):/.test(expanded)) {
        errors.push(`${chapter.src}: unresolved semantic object reference`);
      }
      const equations = (expanded.match(/class="numbered-equation"/g) || []).length;
      const figures = (expanded.match(/<figure\b[^>]*\bclass="[^"]*\bviz\b/g) || []).length;
      // Tables embedded inside a figure are explanatory parts of that figure,
      // not independent book objects; numbered tables always carry a tbl-* id.
      const tables = (expanded.match(/<table\b[^>]*\bid="tbl-/g) || []).length;
      const listings = (expanded.match(/<figure\b[^>]*\bclass="code-listing"/g) || []).length;
      if (equations !== numbered.counts.equations) errors.push(`${chapter.src}: equation numbering count drift`);
      if (figures !== numbered.counts.figures) errors.push(`${chapter.src}: figure numbering count drift`);
      if (tables !== numbered.counts.tables) errors.push(`${chapter.src}: table numbering count drift`);
      if (listings !== numbered.counts.listings) errors.push(`${chapter.src}: listing numbering count drift`);
    } catch (error) {
      errors.push(`${chapter.src}: ${error.message}`);
    }
  }
}

if (errors.length) {
  for (const error of errors) console.error(`ERROR ${error}`);
  process.exitCode = 1;
} else {
  const totals = rendered.reduce((sum, { numbered }) => ({
    equations: sum.equations + numbered.counts.equations,
    figures: sum.figures + numbered.counts.figures,
    tables: sum.tables + numbered.counts.tables,
    listings: sum.listings + numbered.counts.listings,
  }), { equations: 0, figures: 0, tables: 0, listings: 0 });
  console.log(`Object numbering passed: ${totals.equations} equations, ${totals.figures} figures, ${totals.tables} tables, ${totals.listings} listings.`);
}
