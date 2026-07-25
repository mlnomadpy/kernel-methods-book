import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const buildDir = path.join(root, ".build", "publication", "cover");
const releaseDir = path.join(root, "release");
fs.mkdirSync(buildDir, { recursive: true });
fs.mkdirSync(releaseDir, { recursive: true });

const texCache = path.join(root, ".context", "texlive");
fs.mkdirSync(texCache, { recursive: true });
execFileSync("latexmk", [
  "-lualatex",
  ...(process.env.PDF_DEBUG ? [] : ["-silent"]),
  "-interaction=nonstopmode",
  "-halt-on-error",
  "-file-line-error",
  `-outdir=${buildDir}`,
  path.join(root, "publication", "cover", "cover.tex"),
], {
  cwd: root,
  stdio: "inherit",
  env: { ...process.env, TEXMFVAR: texCache, TEXMFCACHE: texCache },
});
fs.copyFileSync(
  path.join(buildDir, "cover.pdf"),
  path.join(releaseDir, "kernels-the-geometry-of-learning-cover-print.pdf"),
);
console.log("Built release/kernels-the-geometry-of-learning-cover-print.pdf.");
