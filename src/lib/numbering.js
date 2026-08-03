/** Number and cross-reference the book's mathematical display objects.
 *
 * Canonical authoring syntax:
 *
 *   $$ ... $$
 *   {#eq-representer}
 *
 *   [[eq:eq-representer]], [[fig:power-function]], [[tbl:kernel-quiz]],
 *   [[lst:jax-krr-solve]]
 *
 * Equation labels are optional: every display is numbered, while an explicit
 * label gives prose a stable target. Figures use their data-figure/data-widget
 * key. Tables may place `{#tbl-name}` immediately after the Markdown table.
 */

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function stripHtml(value) {
  return String(value)
    .replace(/<[^>]+>/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, " ")
    .trim();
}

function protect(body, pattern, prefix) {
  const values = [];
  const text = body.replace(pattern, (value) => {
    const token = `\u0000${prefix}${values.length}\u0000`;
    values.push([token, value]);
    return token;
  });
  return {
    text,
    restore(value) {
      for (const [token, original] of values) value = value.split(token).join(original);
      return value;
    },
  };
}

function nearestSectionTitle(body, offset, chapterTitle) {
  const prefix = body.slice(0, offset);
  const headings = [...prefix.matchAll(/<h[23]\b[^>]*>([\s\S]*?)<\/h[23]>/g)];
  return stripHtml(headings.at(-1)?.[1] || chapterTitle || "Reference table");
}

function listingLanguage(inner) {
  const language = inner.match(/<pre\b[^>]*class="[^"]*\bsourceCode\s+([a-z0-9-]+)/i)?.[1]
    || inner.match(/<code\b[^>]*class="[^"]*\bsourceCode\s+([a-z0-9-]+)/i)?.[1]
    || inner.match(/<pre\b[^>]*class="([a-z0-9-]+)"/i)?.[1]
    || "code";
  const labels = { py: "Python", python: "Python", sh: "Shell", bash: "Shell", json: "JSON", yaml: "YAML", yml: "YAML", text: "Output" };
  return labels[language] || language.replace(/(^|-)([a-z])/g, (_, space, letter) => `${space ? " " : ""}${letter.toUpperCase()}`);
}

function addReference(refs, aliases, record) {
  for (const alias of aliases.filter(Boolean)) {
    if (refs.has(alias)) {
      throw new Error(`Duplicate book-object reference label: ${alias}`);
    }
    refs.set(alias, record);
  }
}

/** Add chapter-scoped numbers and return the references introduced. */
export function numberBookObjects(body, { chapterLabel, chapterSlug, chapterTitle }) {
  const refs = new Map();

  // Code may legitimately show TeX delimiters. It is instructional text, not
  // a displayed equation in the surrounding book.
  const code = protect(body, /<(pre|code)\b[\s\S]*?<\/\1>/g, "KBCODE");
  body = code.text;

  let equationNumber = 0;
  body = body.replace(
    /\$\$([\s\S]*?)\$\$(?:\s*<p>\{#(eq-[a-z0-9-]+)\}<\/p>)?/g,
    (_, source, explicitId) => {
      equationNumber += 1;
      const number = `${chapterLabel}.${equationNumber}`;
      const id = explicitId || `eq-${chapterSlug}-${equationNumber}`;
      const record = { kind: "Equation", number, id, chapterSlug };
      addReference(refs, explicitId ? [explicitId] : [], record);
      return (
        `<div class="numbered-equation" id="${id}">` +
        `<div class="equation-math">$$${source}$$</div>` +
        `<a class="equation-number" href="#${id}" aria-label="Equation ${number}">(${number})</a>` +
        `</div>`
      );
    },
  );
  body = code.restore(body);

  // Code is a book object, not an anonymous grey rectangle. Every executable
  // block receives a chapter-scoped listing number and an informative caption;
  // authors can replace the section-based fallback with a stable semantic id
  // and caption using `{#lst-name caption="..."}` immediately after the fence.
  let listingNumber = 0;
  body = body.replace(
    /(?:<div\b([^>]*\bclass="[^"]*\bsourceCode\b[^"]*"[^>]*)>([\s\S]*?)<\/div>|<pre\b([^>]*\bclass="[^"]+"[^>]*)>([\s\S]*?)<\/pre>)(?:\s*<p>\{#(lst-[a-z0-9-]+)\s+caption=["“]([^"”]+)["”]\}<\/p>)?/g,
    (whole, divAttrs, divInner, preAttrs, preInner, explicitId, explicitCaption, offset) => {
      listingNumber += 1;
      const number = `${chapterLabel}.${listingNumber}`;
      const id = explicitId || `lst-${chapterSlug}-${listingNumber}`;
      const caption = explicitCaption || `${nearestSectionTitle(body, offset, chapterTitle)}: executable check`;
      const inner = divAttrs
        ? `<div${divAttrs}>${divInner}</div>`
        : `<div class="sourceCode"><pre${preAttrs}>${preInner}</pre></div>`;
      const language = listingLanguage(inner);
      const record = { kind: "Listing", number, id, chapterSlug };
      addReference(refs, explicitId ? [explicitId] : [], record);
      const captionId = `${id}-caption`;
      return (
        `<figure class="code-listing" id="${id}" aria-labelledby="${captionId}">` +
        `<figcaption class="listing-caption" id="${captionId}">` +
        `<span class="listing-caption-text"><a class="object-label" href="#${id}">Listing ${number}.</a> ${escapeHtml(caption)}</span>` +
        `<span class="listing-tools"><span class="listing-language">${escapeHtml(language)}</span>` +
        `<button class="listing-copy" type="button" aria-label="Copy Listing ${number}">Copy</button></span>` +
        `</figcaption>${inner}</figure>`
      );
    },
  );

  // Protect figures while tables are numbered: a figure may itself contain a
  // small explanatory table, which belongs to the figure rather than the table
  // sequence.
  let figureNumber = 0;
  const figureValues = [];
  body = body.replace(
    /<figure\b([^>]*\bclass="[^"]*\bviz\b[^"]*"[^>]*)>([\s\S]*?)<\/figure>/g,
    (whole, attrs, inner) => {
      figureNumber += 1;
      const key = attrs.match(/\bdata-(?:figure|widget)="([a-z0-9-]+)"/)?.[1];
      if (!key) throw new Error(`${chapterSlug}: numbered figure has no data-figure or data-widget key`);
      const number = `${chapterLabel}.${figureNumber}`;
      const id = attrs.match(/\bid="([^"]+)"/)?.[1] || `fig-${key}`;
      const record = { kind: "Figure", number, id, chapterSlug };
      addReference(refs, [key, id], record);
      const numberedAttrs = /\bid="/.test(attrs) ? attrs : ` id="${id}"${attrs}`;
      const caption = inner.replace(
        /<figcaption([^>]*)>([\s\S]*?)<\/figcaption>/i,
        (_, capAttrs, content) =>
          `<figcaption${capAttrs}><a class="object-label" href="#${id}">Figure ${number}.</a> ${content}</figcaption>`,
      );
      if (caption === inner) throw new Error(`${chapterSlug}#${id}: numbered figure has no figcaption`);
      const value = `<figure${numberedAttrs}>${caption}</figure>`;
      const token = `\u0000KBFIGURE${figureValues.length}\u0000`;
      figureValues.push([token, value]);
      return token;
    },
  );

  let tableNumber = 0;
  body = body.replace(
    /<table\b([^>]*)>([\s\S]*?)<\/table>(?:\s*<p>\{#(tbl-[a-z0-9-]+)\}<\/p>)?/g,
    (whole, attrs, inner, explicitId, offset) => {
      tableNumber += 1;
      const number = `${chapterLabel}.${tableNumber}`;
      const id = explicitId || `tbl-${chapterSlug}-${tableNumber}`;
      const record = { kind: "Table", number, id, chapterSlug };
      addReference(refs, explicitId ? [explicitId] : [], record);
      const numberedAttrs = /\bid="/.test(attrs) ? attrs : `${attrs} id="${id}"`;
      let content = inner;
      if (/<caption\b/i.test(content)) {
        content = content.replace(
          /<caption([^>]*)>([\s\S]*?)<\/caption>/i,
          (_, capAttrs, caption) =>
            `<caption${capAttrs}><a class="object-label" href="#${id}">Table ${number}.</a> ${caption}</caption>`,
        );
      } else {
        const title = nearestSectionTitle(body, offset, chapterTitle);
        content = `<caption><a class="object-label" href="#${id}">Table ${number}.</a> ${escapeHtml(title)}</caption>${content}`;
      }
      return `<table${numberedAttrs}>${content}</table>`;
    },
  );

  for (const [token, value] of figureValues) body = body.split(token).join(value);

  return {
    body,
    refs,
    counts: { equations: equationNumber, figures: figureNumber, tables: tableNumber, listings: listingNumber },
  };
}

/** Merge per-chapter maps while guarding global semantic-label uniqueness. */
export function mergeBookObjectReferences(chapterMaps) {
  const merged = new Map();
  for (const refs of chapterMaps) {
    for (const [label, record] of refs) {
      if (merged.has(label)) throw new Error(`Duplicate global book-object reference label: ${label}`);
      merged.set(label, record);
    }
  }
  return merged;
}

/** Expand semantic tokens only after every chapter has been indexed. */
export function expandBookObjectReferences(body, refs, currentSlug) {
  return body.replace(
    /\[\[(eq|fig|tbl|lst):([a-z0-9-]+)(?:\|([^\]]+))?\]\]/g,
    (whole, type, label, custom) => {
      const aliases = type === "fig" ? [label, `fig-${label}`] : [label];
      const record = aliases.map((key) => refs.get(key)).find(Boolean);
      if (!record) throw new Error(`${currentSlug}: unknown ${type} reference '${label}'`);
      const href = record.chapterSlug === currentSlug
        ? `#${record.id}`
        : `${record.chapterSlug}.html#${record.id}`;
      const text = custom || `${record.kind} ${record.number}`;
      return `<a class="object-ref ${type}-ref" href="${href}">${escapeHtml(text)}</a>`;
    },
  );
}
