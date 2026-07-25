import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

const root = process.cwd();
const chapterDir = path.join(root, "manuscript", "chapters");
const sourceDir = path.join(root, "tools", "figures");
const svgDir = path.join(root, "public", "figures");
const pdfDir = path.join(root, "publication", "figures");
const captions = JSON.parse(fs.readFileSync(path.join(pdfDir, "captions.json"), "utf8"));
const registry = JSON.parse(fs.readFileSync(path.join(pdfDir, "registry.json"), "utf8")).figures;
const artifactManifestPath = path.join(pdfDir, "artifacts.json");
const artifacts = fs.existsSync(artifactManifestPath)
  ? JSON.parse(fs.readFileSync(artifactManifestPath, "utf8")).figures
  : {};
const errors = [];
const sha256 = (file) => crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const stylePath = path.join(sourceDir, "_style.py");
const pythonStyleHash = sha256(stylePath);
const tikzStyleHash = crypto.createHash("sha256")
  .update(fs.readFileSync(path.join(sourceDir, "_tikz.py")))
  .update(fs.readFileSync(path.join(sourceDir, "tikz-style.tex")))
  .digest("hex");
const sourceHash = (file) => {
  const digest = crypto.createHash("sha256").update(fs.readFileSync(file));
  const companion = file.replace(/\.py$/, ".tex");
  if (fs.existsSync(companion)) digest.update(fs.readFileSync(companion));
  return digest.digest("hex");
};
const styleText = fs.readFileSync(stylePath, "utf8");
if (!/jax\.config\.update\(\s*["']jax_enable_x64["']\s*,\s*True\s*\)/.test(styleText)) {
  errors.push("_style.py: JAX x64 mode is not enabled centrally");
}
if (/default_rng|np\.random/.test(styleText)) {
  errors.push("_style.py: randomness must come from jax.random, not NumPy");
}
for (const [pattern, message] of [
  [/axes\.prop_cycle/, "semantic colour/stroke cycle is missing"],
  [/"svg\.fonttype"\s*:\s*"none"/, "SVG text must remain selectable"],
  [/_normalize_figure_typography/, "legacy figures are not normalized at export"],
]) {
  if (!pattern.test(styleText)) errors.push(`_style.py: ${message}`);
}

const sources = fs.readdirSync(sourceDir)
  .filter((name) => name.endsWith(".py") && !["_style.py", "_tikz.py", "build_figures.py"].includes(name))
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
  if (!Array.isArray(record.chapters) || record.chapters.length !== 1) {
    errors.push(
      `${id}: every figure must have exactly one owning chapter; later chapters must cross-reference it`,
    );
  }
  const source = path.join(root, record.source);
  const expectedStyleHash = fs.existsSync(source.replace(/\.py$/, ".tex"))
    ? tikzStyleHash
    : pythonStyleHash;
  const svg = path.join(root, record.web);
  const pdf = path.join(root, record.print);
  if (!record.source.endsWith(".py")) errors.push(`${id}: generator must be Python`);
  if (!fs.existsSync(source)) errors.push(`${id}: missing generator ${path.relative(root, source)}`);
  if (fs.existsSync(source)) {
    const sourceText = fs.readFileSync(source, "utf8");
    if (!/(?:^|\n)(?:import jax|from jax)/.test(sourceText)) {
      errors.push(`${id}: generator does not import JAX`);
    }
    if (!/(?:jax\.numpy|from jax import numpy)/.test(sourceText)) {
      errors.push(`${id}: mathematical arrays must use jax.numpy`);
    }
    if (/(?:^|[^\w])(?:np|numpy)\.random|default_rng/m.test(sourceText)) {
      errors.push(`${id}: NumPy randomness is forbidden; use jax.random or S.rng`);
    }
    if (/(?:^|[^\w])(?:np|numpy)\.linalg\.(?:inv|solve|cholesky|eig|eigh|eigvalsh|svd|lstsq|norm)/m.test(sourceText)) {
      errors.push(`${id}: numerical linear algebra must use jax.numpy.linalg`);
    }
    if (/\.linalg\.inv\(/.test(sourceText)) {
      errors.push(`${id}: explicit matrix inverse is forbidden in figure generators`);
    }
    if (/#[0-9a-fA-F]{6}\b/.test(sourceText)) {
      errors.push(`${id}: raw hex colors bypass the plot engine's semantic palette`);
    }
    if (/plt\.style\.use|seaborn\.set|sns\.set_(?:theme|style|context)/.test(sourceText)) {
      errors.push(`${id}: local style configuration bypasses the book plot engine`);
    }
    if (/(?:cmap\s*=\s*|get_cmap\()["'](?:jet|rainbow|turbo|hsv)["']/.test(sourceText)) {
      errors.push(`${id}: rainbow colormaps are forbidden in mathematical plates`);
    }
    if (/plt\.title\(/.test(sourceText)) {
      errors.push(`${id}: global titles belong in the manuscript caption`);
    }
  }
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
  const artifact = artifacts[id];
  if (!artifact) {
    errors.push(`${id}: missing freshness record in publication/figures/artifacts.json`);
  } else if (fs.existsSync(source) && fs.existsSync(svg) && fs.existsSync(pdf)) {
    if (artifact.source !== record.source) errors.push(`${id}: freshness source path disagrees with registry`);
    if (artifact.source_sha256 !== sourceHash(source)) errors.push(`${id}: generator changed without regeneration`);
    if (artifact.style_sha256 !== expectedStyleHash) errors.push(`${id}: shared style changed without regeneration`);
    if (artifact.svg_sha256 !== sha256(svg)) errors.push(`${id}: SVG is stale or hand-edited`);
    if (artifact.pdf_sha256 !== sha256(pdf)) errors.push(`${id}: PDF is stale or hand-edited`);
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
