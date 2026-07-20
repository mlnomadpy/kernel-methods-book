import assert from "node:assert/strict";
import test from "node:test";

await import("../public/assets/smo.js");
const { solve } = globalThis.KernelBookSMO;

test("SMO recovers the analytic two-point hard-margin solution", () => {
  const fit = solve([[1, -1], [-1, 1]], [-1, 1], 10, { tol: 1e-8 });
  assert.ok(Math.abs(fit.alpha[0] - 0.5) < 1e-6);
  assert.ok(Math.abs(fit.alpha[1] - 0.5) < 1e-6);
  assert.ok(Math.abs(fit.equality) < 1e-10);
  assert.ok(fit.kkt < 1e-7);
  assert.ok(Math.abs(fit.objective - 0.5) < 1e-6);
});

test("SMO respects equality, bounds, and KKT conditions on a soft problem", () => {
  const x = [-2, -0.4, 0.2, 1.7];
  const y = [-1, -1, 1, 1];
  const K = x.map((a) => x.map((b) => Math.exp(-((a - b) ** 2) / 2)));
  const C = 1.5;
  const fit = solve(K, y, C);
  assert.ok(Math.abs(fit.equality) < 1e-8);
  assert.ok([...fit.alpha].every((a) => a >= -1e-10 && a <= C + 1e-10));
  assert.ok(fit.kkt < 5e-3);
});
