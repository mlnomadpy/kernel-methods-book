/* Validate the curated dependency edges against the generated node set.
 * Usage: node depmap/validate.mjs
 * Exit 0 if clean, 1 if any hard error. */
import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const nodes = JSON.parse(fs.readFileSync(path.join(ROOT, "depmap/nodes.json"), "utf8"));
const byKey = new Map(nodes.map((n) => [n.key, n]));
const chapterNum = new Map(nodes.map((n) => [n.chapter, n.chapterNum]));

const edgeDir = path.join(ROOT, "depmap/edges");
const files = fs.existsSync(edgeDir)
  ? fs.readdirSync(edgeDir).filter((f) => f.endsWith(".json")).sort()
  : [];

const errors = [];
const warnings = [];
const edges = [];
const seen = new Set();

for (const f of files) {
  const src = f.replace(/\.json$/, "");
  let arr;
  try {
    arr = JSON.parse(fs.readFileSync(path.join(edgeDir, f), "utf8"));
  } catch (e) {
    errors.push(`${f}: invalid JSON (${e.message})`);
    continue;
  }
  if (!Array.isArray(arr)) {
    errors.push(`${f}: not a JSON array`);
    continue;
  }
  for (const e of arr) {
    if (!e || typeof e.from !== "string" || typeof e.to !== "string") {
      errors.push(`${f}: edge missing from/to: ${JSON.stringify(e)}`);
      continue;
    }
    if (!byKey.has(e.from)) errors.push(`${f}: unknown 'from' node ${e.from}`);
    if (!byKey.has(e.to)) errors.push(`${f}: unknown 'to' node ${e.to}`);
    if (e.from === e.to) errors.push(`${f}: self-loop on ${e.from}`);
    // the edge file for a chapter should own edges whose source is in that chapter
    const fromNode = byKey.get(e.from);
    if (fromNode && fromNode.src !== src)
      warnings.push(`${f}: edge from ${e.from} belongs in ${fromNode.src}.json`);
    const sig = `${e.from}->${e.to}`;
    if (seen.has(sig)) {
      warnings.push(`${f}: duplicate edge ${sig}`);
      continue;
    }
    seen.add(sig);
    if (byKey.has(e.from) && byKey.has(e.to)) edges.push(e);
  }
}

// forward-reference check: a dependency should point backward in reading order.
// same-chapter edges are allowed either direction (a proof may cite a later
// corollary rarely), but cross-chapter forward edges are flagged.
for (const e of edges) {
  const a = byKey.get(e.from), b = byKey.get(e.to);
  if (a.chapter === b.chapter) continue;
  const na = a.chapterNum === "P" ? -1 : a.chapterNum;
  const nb = b.chapterNum === "P" ? -1 : b.chapterNum;
  if (nb > na) warnings.push(`forward cross-chapter edge: ${e.from} (ch ${a.chapterNum}) uses ${e.to} (ch ${b.chapterNum})`);
}

// cycle detection (Tarjan-free: DFS with colors)
const adj = new Map();
for (const n of nodes) adj.set(n.key, []);
for (const e of edges) adj.get(e.from)?.push(e.to);
const color = new Map();
const cycles = [];
const stack = [];
function dfs(u) {
  color.set(u, 1);
  stack.push(u);
  for (const v of adj.get(u) || []) {
    if (color.get(v) === 1) {
      const i = stack.indexOf(v);
      cycles.push(stack.slice(i).concat(v));
    } else if (!color.get(v)) dfs(v);
  }
  stack.pop();
  color.set(u, 2);
}
for (const n of nodes) if (!color.get(n.key)) dfs(n.key);

// coverage: statements with no outgoing and no incoming edge
const hasOut = new Set(edges.map((e) => e.from));
const hasIn = new Set(edges.map((e) => e.to));
const isolated = nodes.filter((n) => !hasOut.has(n.key) && !hasIn.has(n.key));

console.log(`nodes: ${nodes.length}  edges: ${edges.length}  files: ${files.length}`);
console.log(`isolated statements (no edges): ${isolated.length}`);
if (cycles.length) {
  console.log(`\nCYCLES (${cycles.length}):`);
  for (const c of cycles.slice(0, 10)) console.log("  " + c.join(" -> "));
}
if (warnings.length) {
  console.log(`\nWARNINGS (${warnings.length}):`);
  for (const w of warnings.slice(0, 40)) console.log("  " + w);
}
if (errors.length) {
  console.log(`\nERRORS (${errors.length}):`);
  for (const e of errors) console.log("  " + e);
  process.exit(1);
}
console.log("\nOK: no hard errors.");
