import test from "node:test";
import assert from "node:assert/strict";
import { buildBook } from "../src/lib/book.js";

test("numbered equations remain inside proofs without closing them early", () => {
  const { chapters } = buildBook();
  for (const chapter of chapters) {
    assert.doesNotMatch(
      chapter.body,
      /<\/details><a class="equation-number"/,
      `${chapter.slug}: equation number escaped its proof`,
    );
  }
});
