import fs from "node:fs";
import matter from "gray-matter";
import { parse as parseYaml } from "yaml";

const release = process.env.RELEASE_VERIFIED === "1";
const book = parseYaml(fs.readFileSync("book.yml", "utf8"));
const chapters = book.parts.flatMap((part) => part.chapters);
const gaps = [];
let formal = 0, formalDocumented = 0, examples = 0, linkedExamples = 0;

for (const chapter of chapters) {
  const source = matter(fs.readFileSync(`manuscript/chapters/${chapter.src}.md`, "utf8")).content;
  const chapterGaps = [];
  if (!/^## .*summary/im.test(source)) chapterGaps.push("summary");
  if (!/further reading/i.test(source)) chapterGaps.push("further reading");
  if (!/common mistakes/i.test(source)) chapterGaps.push("common mistakes");
  if (!/practical implications/i.test(source) && !/practice/i.test(source)) chapterGaps.push("practical implications");

  const openings = [...source.matchAll(/^:{3,} \{\.(theorem|lemma|proposition|corollary)\b[^}]*\}/gm)];
  formal += openings.length;
  for (let i = 0; i < openings.length; i++) {
    const end = i + 1 < openings.length ? openings[i + 1].index : source.length;
    const region = source.slice(openings[i].index, end);
    if (/\*\*Assumptions\.\*\*/.test(region) && /\*\*Proof status\.\*\*/.test(region)) formalDocumented += 1;
    else chapterGaps.push(`formal metadata near line ${source.slice(0, openings[i].index).split("\n").length}`);
  }

  const exampleOpenings = [...source.matchAll(/^:{3,} \{\.example\b[^}]*\}/gm)];
  examples += exampleOpenings.length;
  for (const opening of exampleOpenings) {
    const id = opening[0].match(/#([A-Za-z][A-Za-z0-9_.:-]*)/)?.[1];
    if (!id) continue;
    const filename =
      `checks/example-${chapter.src}-${id.replace(/[^A-Za-z0-9_.-]/g, "-")}.json`;
    if (fs.existsSync(filename)) linkedExamples += 1;
  }
  if (chapterGaps.length) gaps.push({ chapter: chapter.src, items: [...new Set(chapterGaps)] });
}

console.log(`Editorial template gaps: ${gaps.length}/${chapters.length} chapters.`);
console.log(`Formal result metadata: ${formalDocumented}/${formal}; examples with verification artifacts: ${linkedExamples}/${examples}.`);
if (release && gaps.length) {
  for (const gap of gaps) console.error(`ERROR ${gap.chapter}: ${gap.items.join(", ")}`);
  process.exitCode = 1;
} else if (gaps.length) {
  console.warn(`WARN editorial revision remains open; RELEASE_VERIFIED=1 prints and rejects every gap.`);
}
