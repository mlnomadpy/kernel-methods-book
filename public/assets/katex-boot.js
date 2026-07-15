/* Render all math once KaTeX auto-render has loaded. All three scripts are
 * deferred, so they execute in document order: katex, auto-render, then this. */
renderMathInElement(document.body, {
  delimiters: [
    { left: "$$", right: "$$", display: true },
    { left: "\\(", right: "\\)", display: false },
  ],
  throwOnError: false,
});
