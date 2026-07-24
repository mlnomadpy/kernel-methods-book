import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const chapterDir = path.join(root, "manuscript", "chapters");
const sourceDir = path.join(root, "tools", "figures");
const svgDir = path.join(root, "public", "figures");
const pdfDir = path.join(root, "publication", "figures");
const captions = JSON.parse(fs.readFileSync(path.join(pdfDir, "captions.json"), "utf8"));
const registry = JSON.parse(fs.readFileSync(path.join(pdfDir, "registry.json"), "utf8")).figures;
const errors = [];

const sources = fs.readdirSync(sourceDir)
  .filter((name) => name.endsWith(".py") && !["_style.py", "build_figures.py"].includes(name))
  .map((name) => name.replace(/\.py$/, "").replaceAll("_", "-"));

const used = new Set();
for (const name of fs.readdirSync(chapterDir).filter((item) => item.endsWith(".md"))) {
  const text = fs.readFileSync(path.join(chapterDir, name), "utf8");
  for (const match of text.matchAll(/data-(?:widget|figure)="([a-z0-9-]+)"/g)) used.add(match[1]);
}

for (const id of new Set([...sources, ...used])) {
  const record = registry[id];
  if (!record) {
    errors.push(`${id}: missing from publication/figures/registry.json`);
    continue;
  }
  const source = path.join(root, record.source);
  const svg = path.join(root, record.web);
  const pdf = path.join(root, record.print);
  if (!fs.existsSync(source)) errors.push(`${id}: missing generator ${path.relative(root, source)}`);
  if (!fs.existsSync(svg)) errors.push(`${id}: missing web/EPUB SVG`);
  if (!fs.existsSync(pdf)) errors.push(`${id}: missing print PDF`);
  if (!captions[id] && used.has(id) && !fs.existsSync(source)) {
    errors.push(`${id}: no publication caption`);
  }
  if (fs.existsSync(svg)) {
    const text = fs.readFileSync(svg, "utf8");
    if (!/<svg\b[^>]*\bwidth="[^"]+"[^>]*\bheight="[^"]+"/s.test(text)) {
      errors.push(`${id}: SVG needs intrinsic width and height`);
    }
    if (/(?:^|[^\w])(?:nan|inf)(?:[^\w]|$)/i.test(text)) errors.push(`${id}: SVG contains NaN or Inf`);
  }
  if (fs.existsSync(pdf)) {
    const bytes = fs.readFileSync(pdf);
    if (bytes.length < 100 || bytes.subarray(0, 5).toString() !== "%PDF-") {
      errors.push(`${id}: malformed PDF`);
    }
  }
}

for (const id of Object.keys(registry)) {
  if (!used.has(id)) errors.push(`${id}: registry entry is not embedded by any chapter`);
}

for (const id of Object.keys(captions)) {
  if (!used.has(id)) errors.push(`${id}: caption exists but no chapter embeds the figure`);
}

if (errors.length) {
  console.error(`Figure integrity failed (${errors.length}):\n${errors.map((item) => `- ${item}`).join("\n")}`);
  process.exitCode = 1;
} else {
  console.log(`Figure integrity passed: ${used.size} embedded IDs, ${sources.length} deterministic generators.`);
}
