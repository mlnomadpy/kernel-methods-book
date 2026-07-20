import fs from "node:fs";

import { parse as parseYaml } from "yaml";

const book = parseYaml(fs.readFileSync("book.yml", "utf8"));
const chapters = book.parts.flatMap((part) => part.chapters);
const release = process.env.RELEASE_VERIFIED === "1";
const errors = [];
let pendingProvenance = 0, pendingReviews = 0;

for (const chapter of chapters) {
  const provenanceFile = `provenance/${chapter.src}.yml`;
  const reviewFile = `reviews/${chapter.src}.yml`;
  if (!fs.existsSync(provenanceFile)) errors.push(`${chapter.src}: missing provenance manifest`);
  if (!fs.existsSync(reviewFile)) errors.push(`${chapter.src}: missing review manifest`);
  if (!fs.existsSync(provenanceFile) || !fs.existsSync(reviewFile)) continue;

  const provenance = fs.readFileSync(provenanceFile, "utf8");
  const review = fs.readFileSync(reviewFile, "utf8");
  if (!/^status: verified$/m.test(provenance) || /source_locator: null|verified_by: null|verified_on: null/.test(provenance)) pendingProvenance++;
  if (!/^status: verified$/m.test(review)) pendingReviews++;

  if (release) {
    if (!/^status: verified$/m.test(review)) errors.push(`${chapter.src}: review status is not verified`);
    if (/reviewer: null|approved_on: null|findings_resolved: false/.test(review)) errors.push(`${chapter.src}: incomplete required review approval`);
    if (!/^status: verified$/m.test(provenance) || /sources: \[\]|locators: \[\]|source_locator: null|verified_by: null|verified_on: null/.test(provenance)) {
      errors.push(`${chapter.src}: provenance is incomplete`);
    }
    if (/specialist_review_required: true/.test(review) && /specialist_review:\n  reviewer: null/.test(review)) {
      errors.push(`${chapter.src}: specialist approval is required`);
    }
  }
}

if (errors.length) {
  errors.forEach((error) => console.error(`ERROR ${error}`));
  process.exitCode = 1;
} else {
  console.log(`Manifest coverage passed for ${chapters.length} chapters; pending provenance: ${pendingProvenance}, pending review: ${pendingReviews}.`);
}
