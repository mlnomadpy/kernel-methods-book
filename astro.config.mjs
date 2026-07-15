import { defineConfig } from "astro/config";

// The book is deployed to GitHub Pages at /kernel-methods-book/.
// Every internal link in the content is relative (foo.html, assets/...),
// so `format: "file"` keeps the same flat URL scheme the book has always had.
export default defineConfig({
  site: "https://mlnomadpy.github.io",
  base: "/kernel-methods-book",
  trailingSlash: "never",
  build: { format: "file" },
});
