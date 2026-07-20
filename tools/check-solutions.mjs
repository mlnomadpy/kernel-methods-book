import fs from "node:fs";
import matter from "gray-matter";
import { parse as parseYaml } from "yaml";

const book = parseYaml(fs.readFileSync("book.yml", "utf8"));
const chapters = book.parts.flatMap((part) => part.chapters);
const release = process.env.RELEASE_VERIFIED === "1";
const errors = [];
let total = 0, covered = 0, authored = 0, verified = 0;
const answerLocations = new Map();
const boilerplate = [
  /Draft solution method/i,
  /Draft proof plan/i,
  /Draft concise answer/i,
  /answer outline requires/i,
  /This draft requires line-by-line/i,
  /Assessment rubric\. The response must/i,
];

function compactPrompt(text) {
  return text
    .replace(/\s+/g, " ")
    .replace(/\[\[ch:[^\]]+\]\]/g, "the referenced chapter")
    .trim();
}

for (const chapter of chapters) {
  const parsed = matter(fs.readFileSync(`manuscript/chapters/${chapter.src}.md`, "utf8"));
  const exerciseSection = parsed.content.match(/^## Exercises \{#exercises\}([\s\S]*)$/m)?.[1] || "";
  const manuscriptExercises = [...exerciseSection.matchAll(
    /^(\d+)\.\s+\[([a-z-]+)\]\{\.ex-tag\}\s+([\s\S]*?)(?=^\d+\.\s+\[[a-z-]+\]\{\.ex-tag\}|\s*$)/gm,
  )];
  const kinds = manuscriptExercises.map((match) => match[2]);
  total += kinds.length;
  const file = `solutions/${chapter.src}.yml`;
  if (!fs.existsSync(file)) {
    if (release && kinds.length) errors.push(`${chapter.src}: missing solution file for ${kinds.length} exercises`);
    continue;
  }
  const data = parseYaml(fs.readFileSync(file, "utf8"));
  if (data.prepared_for?.author !== "Taha Bouhsine") errors.push(`${chapter.src}: missing author attribution`);
  if (data.prepared_for?.website !== "https://www.tahabouhsine.com") errors.push(`${chapter.src}: missing author website`);
  if (data.prepared_for?.affiliation !== "AzettaAI") errors.push(`${chapter.src}: missing AzettaAI affiliation`);
  const entries = data.exercises || [];
  for (let i = 0; i < kinds.length; i++) {
    const entry = entries.find((item) => item.number === i + 1);
    if (!entry) { errors.push(`${chapter.src}: missing exercise ${i + 1}`); continue; }
    if (entry.kind !== kinds[i]) errors.push(`${chapter.src}: exercise ${i + 1} kind mismatch`);
    const expectedPrompt = compactPrompt(manuscriptExercises[i][3]);
    if (entry.prompt !== expectedPrompt) errors.push(`${chapter.src}: exercise ${i + 1} prompt is out of sync`);
    const answer = entry.answer?.trim() || "";
    if (answer.length < 80) errors.push(`${chapter.src}: exercise ${i + 1} answer/rubric is incomplete`);
    if (entry.status === "draft-outline") errors.push(`${chapter.src}: exercise ${i + 1} still has outline status`);
    for (const pattern of boilerplate) {
      if (pattern.test(answer)) errors.push(`${chapter.src}: exercise ${i + 1} contains generated-outline boilerplate`);
    }
    if (answer) {
      const prior = answerLocations.get(answer);
      if (prior) errors.push(`${chapter.src}: exercise ${i + 1} duplicates the answer for ${prior}`);
      else answerLocations.set(answer, `${chapter.src} exercise ${i + 1}`);
    }
    covered += 1;
    if (entry.status !== "draft-outline") authored += 1;
    if (entry.status === "verified") verified += 1;
    if (release && entry.status !== "verified") errors.push(`${chapter.src}: exercise ${i + 1} is not verified`);
  }
  if (release && (data.review_status !== "verified" || !data.reviewer)) errors.push(`${chapter.src}: solution review is incomplete`);
}

console.log(`Solution files: ${covered}/${total}; substantive drafts: ${authored}/${total}; independently verified: ${verified}/${total}.`);
if (errors.length) {
  errors.forEach((error) => console.error(`ERROR ${error}`));
  process.exitCode = 1;
} else if (!release && covered < total) {
  console.warn(`WARN ${total - covered} exercises remain in the answer-authoring backlog.`);
}
