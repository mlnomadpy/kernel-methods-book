import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const chapterDir = path.join(root, "manuscript", "chapters");
const entries = {};

for (const file of fs.readdirSync(chapterDir).filter((name) => name.endsWith(".md")).sort()) {
  const chapter = file.replace(/\.md$/, "");
  const text = fs.readFileSync(path.join(chapterDir, file), "utf8");
  for (const match of text.matchAll(/data-(widget|figure)="([a-z0-9-]+)"/g)) {
    const [, rawMode, id] = match;
    const mode = rawMode === "widget" ? "interactive" : "static";
    const entry = entries[id] ||= {
      id,
      mode,
      source: `tools/figures/${id.replaceAll("-", "_")}.py`,
      web: `public/figures/${id}.svg`,
      print: `publication/figures/${id}.pdf`,
      chapters: [],
    };
    if (entry.mode !== mode) throw new Error(`${id} is embedded as both ${entry.mode} and ${mode}`);
    if (!entry.chapters.includes(chapter)) entry.chapters.push(chapter);
  }
}

const target = path.join(root, "publication", "figures", "registry.json");
fs.writeFileSync(target, `${JSON.stringify({ version: 1, figures: entries }, null, 2)}\n`);
console.log(`Wrote ${path.relative(root, target)} with ${Object.keys(entries).length} figures.`);
