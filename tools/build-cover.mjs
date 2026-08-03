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
// EPUB readers need a raster/vector image rather than a print-cover PDF. Keep
// the TikZ spread as the only design source and crop its front panel at 144 dpi
// (2 pixels per PostScript point) into the EPUB asset. The crop excludes both
// bleed strips, the back cover, and the provisional spine.
execFileSync("pdftocairo", [
  "-png", "-singlefile", "-r", "144",
  "-x", "1552", "-y", "18", "-W", "1224", "-H", "1584",
  path.join(buildDir, "cover.pdf"),
  path.join(root, "publication", "cover"),
], { cwd: root, stdio: "inherit" });
console.log("Built release/kernels-the-geometry-of-learning-cover-print.pdf.");
