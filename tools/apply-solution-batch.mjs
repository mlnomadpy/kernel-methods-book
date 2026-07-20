#!/usr/bin/env node
/** Apply an explicitly authored answer batch to canonical solution manifests. */
import fs from "node:fs";
import { parse, stringify } from "yaml";

const filename = process.argv[2];
if (!filename || !fs.existsSync(filename)) {
  throw new Error("Usage: node tools/apply-solution-batch.mjs <batch.json>");
}
const batch = JSON.parse(fs.readFileSync(filename, "utf8"));
let updated = 0;
for (const [chapter, answers] of Object.entries(batch)) {
  const solutionFile = `solutions/${chapter}.yml`;
  if (!fs.existsSync(solutionFile)) throw new Error(`Missing ${solutionFile}`);
  const document = parse(fs.readFileSync(solutionFile, "utf8"));
  for (const [numberText, answer] of Object.entries(answers)) {
    const number = Number(numberText);
    const exercise = (document.exercises || []).find((entry) => entry.number === number);
    if (!exercise) throw new Error(`${chapter}: missing exercise ${number}`);
    if (typeof answer !== "string" || answer.trim().length < 80) {
      throw new Error(`${chapter} exercise ${number}: answer is too short`);
    }
    exercise.answer = answer.trim();
    exercise.status = "draft";
    updated += 1;
  }
  fs.writeFileSync(solutionFile, stringify(document, { lineWidth: 100 }));
}
console.log(`Applied ${updated} substantive draft answers.`);
