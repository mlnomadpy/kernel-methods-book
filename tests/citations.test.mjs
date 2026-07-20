import test from "node:test";
import assert from "node:assert/strict";

import { expandCitationKeys } from "../src/lib/book.js";

test("Pandoc citations retain surnames when BibTeX stores full given names", () => {
  const html =
    '<span class="citation" data-cites="micchelli2005vv">placeholder</span>';
  const bib = {
    micchelli2005vv: {
      authors: "Micchelli, Charles A. and Pontil, Massimiliano",
      year: 2005,
      title: "On Learning Vector-Valued Functions",
    },
  };
  const [rendered, count] = expandCitationKeys(html, ["micchelli2005vv"], bib);

  assert.equal(count, 1);
  assert.match(rendered, />Micchelli and Pontil, 2005<\/a>/);
});

test("Pandoc citations preserve surname particles in natural-name form", () => {
  const html = '<span class="citation" data-cites="example">placeholder</span>';
  const bib = {
    example: {
      authors: "Ulrike von Luxburg and Laurens van der Maaten",
      year: 2007,
      title: "Example",
    },
  };
  const [rendered] = expandCitationKeys(html, ["example"], bib);

  assert.match(rendered, />von Luxburg and van der Maaten, 2007<\/a>/);
});

test("Pandoc citations parse the legacy comma-separated author convention", () => {
  const html = '<span class="citation" data-cites="example">placeholder</span>';
  const bib = {
    example: {
      authors: "Abbe, E., Boix-Adsera, E., and Misiakiewicz, T.",
      year: 2022,
      title: "Example",
    },
  };
  const [rendered] = expandCitationKeys(html, ["example"], bib);

  assert.match(rendered, />Abbe et al\., 2022<\/a>/);
});
