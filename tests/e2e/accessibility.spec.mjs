import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const widgetPages = [
  "bayesian-optimization-and-bandits.html", "conditional-mean-embeddings.html",
  "efficient-string-and-tree-kernels.html", "kernel-mean-embeddings.html",
  "geometric-and-equivariant-kernels.html",
  "kernel-quadrature-and-herding.html", "kernel-hypothesis-testing.html",
  "signature-and-time-series-kernels.html",
  "optimal-transport-and-kernels.html", "indefinite-and-krein-kernels.html",
  "kernel-stein-discrepancy.html", "support-vector-machines.html",
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
      await expect(figure.locator('[role="status"]').first()).not.toBeEmpty();
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
  const plate = page.locator(".viz-static img.viz-plate").first();
  await expect(plate).toBeVisible();
  await expect(plate).toHaveAttribute("src", /figures\/svm-margin\.svg$/);
  await expect(plate).toHaveAttribute("alt", /decision boundary/i);
  await expect(page.locator("figcaption.viz-fallback").first()).toBeVisible();
  await context.close();
});

test("nonfigure chapters do not download the visualization runtime", async ({ page }) => {
  const requests = [];
  page.on("request", (request) => requests.push(request.url()));
  await page.goto("/preliminaries.html");
  expect(requests.filter((url) => /\/assets\/viz(?:-|\.js)/.test(url))).toEqual([]);
});

test("reduced motion leaves auto-running simulations paused", async ({ browser }) => {
  const context = await browser.newContext({ reducedMotion: "reduce" });
  const page = await context.newPage();
  await page.goto("/kernel-stein-discrepancy.html");
  const status = page.locator('figure[data-widget="svgd-flow"] [role="status"]').first();
  await expect(status).not.toBeEmpty();
  const before = await status.textContent();
  await page.waitForTimeout(300);
  expect(await status.textContent()).toBe(before);
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
