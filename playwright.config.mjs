import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  // Full chapters contain hundreds of rendered equations and semantic nodes.
  // Axe can exceed Playwright's 30 s default on a single-core hosted runner
  // even though the same audit completes quickly on a developer machine.
  timeout: 60_000,
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:4173",
    browserName: "chromium",
  },
  webServer: {
    command: "python3 -m http.server 4173 --directory dist",
    url: "http://127.0.0.1:4173/index.html",
    reuseExistingServer: true,
    stdout: "ignore",
    stderr: "ignore",
  },
});
