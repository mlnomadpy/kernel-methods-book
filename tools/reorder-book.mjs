#!/usr/bin/env node
/**
 * Apply the post-audit pedagogical order and synchronize chapter frontmatter.
 *
 * This is intentionally data-driven: chapter titles and slugs remain canonical in
 * book.yml, while only part membership, sequence, and the corresponding frontmatter
 * fields change.
 */
import fs from "node:fs";
import matter from "gray-matter";
import { parse as parseYaml, stringify as yamlStringify } from "yaml";

const book = parseYaml(fs.readFileSync("book.yml", "utf8"));
const chapters = new Map(
  book.parts.flatMap((part) => part.chapters).map((chapter) => [chapter.src, chapter]),
);

const plan = [
  {
    part: "0 · Preliminaries",
    intro: "A compact mathematical reference for the objects used throughout the book. Read the dependency map first, then use this chapter as a clinic for linear algebra, probability, optimization, operators, geometry, and concentration.",
    ids: ["ch-prelim"],
  },
  {
    part: "I · The Language of Kernels",
    intro: "The story begins with similarity, but immediately asks what makes a similarity mathematically legal, what function space it creates, and why an infinite-dimensional problem can still collapse to finite linear algebra.",
    ids: ["ch00", "ch01", "ch02"],
  },
  {
    part: "II · Learning with a Fixed Kernel",
    intro: "With the geometry fixed, losses and constraints determine the estimator. Ridge regression, maximum-margin classification, tolerance-based regression, novelty detection, and ranking become variations on one regularized empirical-risk problem.",
    ids: ["ch03", "ch05", "ch-svr", "ch-oneclass", "ch-ranking", "ch-structured"],
  },
  {
    part: "III · Optimization and Scaling",
    intro: "A representer theorem gives a finite problem, not a cheap one. This part moves from exact quadratic optimization and decomposition methods to streaming updates, matrix-free solvers, low-rank structure, random features, and sketches.",
    ids: ["ch-solve", "ch-online", "ch13", "ch-randomized"],
  },
  {
    part: "IV · Generalization, Approximation, and Limits",
    intro: "The central theoretical question is not whether a method can fit, but how approximation, estimation, optimization, and numerical errors combine. Complexity, spectra, inverse regularization, interpolation geometry, universality, consistency, and modern interpolation theory answer different pieces of that question.",
    ids: ["ch04", "ch-vc", "ch07", "ch-inverse", "ch-approx", "ch-universal", "ch-lower", "ch-modern"],
  },
  {
    part: "V · Spectral Geometry and Unlabeled Structure",
    intro: "The centered Gram matrix is also a geometric instrument. Its eigenvectors expose nonlinear coordinates, clusters, shared views, discriminating directions, visual embeddings, and the graph geometry that lets unlabeled observations constrain a predictor.",
    ids: ["ch06", "ch-cluster", "ch-cca", "ch-discriminant", "ch-mds", "ch-manifold"],
  },
  {
    part: "VI · Designing Kernels",
    intro: "A kernel is a modeling decision. This part builds valid similarities from spectra, invariance, splines, spatial covariance, strings, text, graphs, latent models, paths, groups, operator-valued outputs, and even indefinite geometry, while tracking both validity and computational cost.",
    ids: ["ch08", "ch-invariance", "ch-splines", "ch-spatial", "ch09", "ch-strings2", "ch-text", "ch10", "ch-generative", "ch-signature", "ch-geom", "ch-operator", "ch-krein"],
  },
  {
    part: "VII · Distributions as Objects",
    intro: "The input to a kernel need not be a point. Mean embeddings, hypothesis tests, transport geometry, and quadrature turn probability distributions into objects that can be compared, tested, moved, and integrated.",
    ids: ["ch11", "ch-testing", "ch-ot", "ch-quad"],
  },
  {
    part: "VIII · Conditional, Stein, and Causal Inference",
    intro: "Population identities become statistical procedures only after regularization and identification assumptions are made explicit. Conditional embeddings, Stein discrepancies, causal operators, and distribution regression develop that transition.",
    ids: ["ch-cme", "ch-ksd", "ch-causal", "ch-distreg"],
  },
  {
    part: "IX · Gaussian Processes and Sequential Decisions",
    intro: "An RKHS norm and a Gaussian covariance share a kernel but support different claims. This part develops posterior computation and approximation, then uses model-based uncertainty to decide what should be measured next.",
    ids: ["ch-gp", "ch-bo"],
  },
  {
    part: "X · Dynamics and Scientific Learning",
    intro: "Static prediction gives way to evolving systems, differential constraints, inverse problems, and learned operators. Stability, discretization, rollout error, and validation of physical quantities become first-class requirements.",
    ids: ["ch-dynamics", "ch-scientific"],
  },
  {
    part: "XI · Learning the Representation",
    intro: "The kernel itself can be selected, combined, parameterized, or replaced by a richer non-Hilbert geometry. Multiple-kernel learning, infinite-width limits, deep kernel learning, variation spaces, and current feature-learning research mark the boundary between fixed geometry and learned representation.",
    ids: ["ch12", "ch14", "ch-dkl", "ch-rkbs", "ch-frontier"],
  },
  {
    part: "XII · Reliable Practice",
    intro: "The final test is whether the method survives contact with changing data, software, scientific constraints, and consequential decisions. This part turns diagnostics, reproducibility, calibration, shift detection, influence, and domain validation into an end-to-end workflow.",
    ids: ["ch-reliability", "ch-apps", "ch-accountable", "ch-highstakes"],
  },
];

const planned = plan.flatMap((part) => part.ids);
const missing = planned.filter((id) => !chapters.has(id));
const omitted = [...chapters.keys()].filter((id) => !planned.includes(id));
const duplicates = planned.filter((id, index) => planned.indexOf(id) !== index);
if (missing.length || omitted.length || duplicates.length) {
  throw new Error(JSON.stringify({ missing, omitted, duplicates }));
}

book.parts = plan.map(({ part, intro, ids }) => ({
  part,
  intro,
  chapters: ids.map((id) => chapters.get(id)),
}));
fs.writeFileSync("book.yml", yamlStringify(book, { lineWidth: 0 }));

let order = 0;
for (const group of plan) {
  for (const id of group.ids) {
    const file = `manuscript/chapters/${id}.md`;
    const parsed = matter(fs.readFileSync(file, "utf8"));
    parsed.data.part = group.part;
    parsed.data.order = order++;
    fs.writeFileSync(file, matter.stringify(parsed.content, parsed.data));
  }
}

console.log(`Reordered ${order} chapters into ${plan.length} narrative parts.`);
