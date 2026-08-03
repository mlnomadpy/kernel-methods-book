#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { parse as parseYaml } from "yaml";

const root = process.cwd();
const texArg = process.argv[2] || ".build/publication/book.tex";
const texPath = path.resolve(root, texArg);
const logIndex = process.argv.indexOf("--log");
const logPath = logIndex >= 0 ? path.resolve(root, process.argv[logIndex + 1]) : null;
const errors = [];

if (!fs.existsSync(texPath)) {
  console.error(`ERROR missing generated TeX: ${texPath}`);
  process.exit(1);
}

const book = parseYaml(fs.readFileSync(path.join(root, "book.yml"), "utf8"));
const chapters = book.parts.flatMap((part) => part.chapters);
const expectedPartArt = book.parts.map((part) => {
  const numeral = part.part.split("·")[0].trim();
  return part.chapters[0]?.src === "ch-prelim" ? "reference" : `part-${numeral.toLowerCase()}`;
});
const partArtPath = path.join(root, "publication", "part-art.tex");
if (!fs.existsSync(partArtPath)) errors.push("missing publication/part-art.tex");
else {
  const partArt = fs.readFileSync(partArtPath, "utf8");
  const declaredPartArt = [...partArt.matchAll(/\\kbpartDeclare\{([^}]+)\}/g)].map((match) => match[1]);
  const missingPartArt = expectedPartArt.filter((id) => !declaredPartArt.includes(id));
  const duplicatePartArt = declaredPartArt.filter((id, index) => declaredPartArt.indexOf(id) !== index);
  const orphanPartArt = declaredPartArt.filter((id) => !expectedPartArt.includes(id));
  if (missingPartArt.length) errors.push(`parts without individual art: ${missingPartArt.join(", ")}`);
  if (duplicatePartArt.length) errors.push(`duplicate part-art ids: ${[...new Set(duplicatePartArt)].join(", ")}`);
  if (orphanPartArt.length) errors.push(`part art without a book.yml part: ${orphanPartArt.join(", ")}`);
}
const chapterArtPath = path.join(root, "publication", "chapter-art.tex");
if (!fs.existsSync(chapterArtPath)) errors.push("missing publication/chapter-art.tex");
else {
  const chapterArt = fs.readFileSync(chapterArtPath, "utf8");
  const declaredArt = [...chapterArt.matchAll(/\\kbartDeclare\{([^}]+)\}/g)].map((match) => match[1]);
  const duplicateArt = [...new Set(declaredArt.filter((id, index) => declaredArt.indexOf(id) !== index))];
  const missingArt = chapters.map((chapter) => chapter.src).filter((src) => !declaredArt.includes(src));
  const orphanArt = declaredArt.filter((src) => !chapters.some((chapter) => chapter.src === src));
  if (duplicateArt.length) errors.push(`duplicate chapter-art ids: ${duplicateArt.join(", ")}`);
  if (missingArt.length) errors.push(`chapters without individual art: ${missingArt.join(", ")}`);
  if (orphanArt.length) errors.push(`chapter art without a book.yml chapter: ${orphanArt.join(", ")}`);
  if (!/\\clip \(0,0\) rectangle \(11\.25,1\.78\)/.test(chapterArt)) {
    errors.push("chapter art is missing its hard clipping boundary");
  }
}
let expectedDisplays = 0;
for (const chapter of chapters) {
  const file = path.join(root, "manuscript", "chapters", `${chapter.src}.md`);
  const source = matter(fs.readFileSync(file, "utf8")).content.replace(
    /^(`{3,}|~{3,})[^\n]*\n[\s\S]*?^\1\s*$/gm,
    "",
  );
  expectedDisplays += (source.match(/\$\$[\s\S]*?\$\$/g) || []).length;
}

const tex = fs.readFileSync(texPath, "utf8");
for (const artId of expectedPartArt) {
  const partSelections = tex.match(new RegExp(`\\\\kbpart(?:reference)?\\{[^\\n]+\\}\\{[^\\n]+\\}\\{${artId}\\}`, "g")) || [];
  if (partSelections.length !== 1) errors.push(`${artId}: expected one part-art selection in generated TeX, found ${partSelections.length}`);
}
for (const chapter of chapters) {
  const artSelections = tex.match(new RegExp(`\\\\kbchapterartset\\{${chapter.src}\\}`, "g")) || [];
  if (artSelections.length !== 1) {
    errors.push(`${chapter.src}: expected one chapter-art selection in generated TeX, found ${artSelections.length}`);
  }
}
const begins = (tex.match(/\\begin\{equation\}/g) || []).length;
const ends = (tex.match(/\\end\{equation\}/g) || []).length;
const equationLabels = (tex.match(/\\label\{eq-[a-z0-9-]+\}/g) || []).length;
if (begins !== expectedDisplays) errors.push(`display parity failed: manuscript ${expectedDisplays}, TeX ${begins}`);
if (ends !== begins) errors.push(`unbalanced equation environments: ${begins} begin, ${ends} end`);
if (equationLabels !== begins) errors.push(`equation label parity failed: ${begins} equations, ${equationLabels} labels`);

for (const [name, pattern] of [
  ["browser/KaTeX markup", /katex|<math\b|<span\b/i],
  ["raw display delimiters", /\$\$/],
  ["manual equation tags", /\\tag\{/],
  ["unexpanded semantic references", /\[\[(?:eq|fig|tbl|lst):/],
]) if (pattern.test(tex)) errors.push(`${name} leaked into generated TeX`);

if (logPath) {
  if (!fs.existsSync(logPath)) errors.push(`missing LuaLaTeX log: ${logPath}`);
  else {
    const log = fs.readFileSync(logPath, "utf8");
    if (/Missing character: There is no/.test(log)) errors.push("LuaLaTeX reported missing glyphs");
    if (/LaTeX Warning: Reference .* undefined|LaTeX Warning: There were undefined references/.test(log)) {
      errors.push("LuaLaTeX reported unresolved references");
    }
  }
}

if (errors.length) {
  for (const error of errors) console.error(`ERROR ${error}`);
  process.exit(1);
}
console.log(`Publication TeX passed: ${begins} native equation environments, ${expectedPartArt.length} individual part plates, ${chapters.length} clipped chapter designs in one safe composition${logPath ? ", no missing glyphs or references" : ""}.`);
