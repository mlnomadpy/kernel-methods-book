import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const widgetPages = [
  "bayesian-optimization-and-bandits.html", "conditional-mean-embeddings.html",
  "efficient-string-and-tree-kernels.html", "kernel-mean-embeddings.html",
  "kernels-and-rkhs.html", "geometric-and-equivariant-kernels.html",
  "kernel-quadrature-and-herding.html", "kernel-hypothesis-testing.html",
  "kernels-and-deep-learning.html", "signature-and-time-series-kernels.html",
  "optimal-transport-and-kernels.html", "indefinite-and-krein-kernels.html",
  "kernel-stein-discrepancy.html", "support-vector-machines.html", "graph-kernels.html",
];

for (const route of widgetPages) {
  test(`${route} exposes widget text, controls, and graphical semantics`, async ({ page }) => {
    await page.goto(`/${route}`);
    const figures = page.locator("figure.viz[data-widget]");
    await expect(figures.first()).toBeVisible();
    for (const figure of await figures.all()) {
      await expect(figure).toHaveAttribute("role", "group");
      const canvases = figure.locator("canvas");
      if (await canvases.count()) {
        await expect(canvases.first()).toHaveAttribute("role", "img");
        await expect(canvases.first()).toHaveAttribute("aria-describedby", /.+/);
      }
      await expect(figure.locator("figcaption:not([hidden])").first()).toBeVisible();
      await expect(figure.locator('[role="status"]').first()).toBeAttached();
    }
  });
}

for (const route of [
  "index.html", "introduction.html", "support-vector-machines.html", "glossary.html",
  "vector-and-operator-valued-kernels.html", "deep-kernel-learning.html",
  "indexes.html", "solutions.html",
]) {
  test(`${route} has no serious axe violations`, async ({ page }) => {
    await page.goto(`/${route}`);
    const results = await new AxeBuilder({ page }).analyze();
    const serious = results.violations.filter((item) => ["serious", "critical"].includes(item.impact));
    expect(serious, JSON.stringify(serious, null, 2)).toEqual([]);
  });
}

test("static widget alternatives survive without JavaScript", async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto("/support-vector-machines.html");
  await expect(page.locator(".viz-static").first()).toBeVisible();
  await expect(page.locator("figcaption.viz-fallback").first()).toContainText("Static alternative");
  await context.close();
});

test("navigation and controls are keyboard reachable", async ({ page }) => {
  await page.goto("/support-vector-machines.html");
  await page.locator(".skip").focus();
  await expect(page.locator(".skip")).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/#main$/);
  await page.locator("figure.viz input[type=range]").first().focus();
  const before = await page.locator("figure.viz input[type=range]").first().inputValue();
  await page.keyboard.press("ArrowRight");
  expect(await page.locator("figure.viz input[type=range]").first().inputValue()).not.toBe(before);
});
