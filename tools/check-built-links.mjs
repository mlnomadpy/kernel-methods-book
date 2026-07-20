import fs from "node:fs";
import path from "node:path";

const pages = fs.readdirSync("dist").filter((file) => file.endsWith(".html"));
const ids = new Map(pages.map((page) => {
  const html = fs.readFileSync(path.join("dist", page), "utf8");
  return [page, new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]))];
}));
const errors = [];

for (const page of pages) {
  const html = fs.readFileSync(path.join("dist", page), "utf8");
  for (const match of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
    const url = match[1];
    if (/^(?:https?:|mailto:|data:)/.test(url)) continue;
    const [targetPart, fragment] = url.split("#");
    const target = targetPart || page;
    const file = path.join("dist", target);
    if (targetPart && !fs.existsSync(file)) errors.push(`${page}: missing target ${url}`);
    else if (fragment && !ids.get(target)?.has(fragment)) errors.push(`${page}: missing anchor ${url}`);
  }
}

if (errors.length) {
  errors.forEach((error) => console.error(error));
  process.exitCode = 1;
} else {
  console.log(`Internal link check passed across ${pages.length} pages.`);
}
