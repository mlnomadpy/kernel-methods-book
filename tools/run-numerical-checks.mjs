import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawn } from "node:child_process";

const scripts = fs.readdirSync("checks")
  .filter((file) => file.endsWith(".py") && file !== "pending_example_checks.py")
  .sort();
const concurrency = Math.max(
  1,
  Math.min(Number(process.env.NUMERICAL_CHECK_WORKERS || 4), os.availableParallelism?.() || 4),
);
const timeoutMs = Number(process.env.NUMERICAL_CHECK_TIMEOUT_MS || 15_000);
const started = performance.now();
let cursor = 0;

function tail(text, limit = 4_000) {
  return text.length <= limit ? text : `…${text.slice(-limit)}`;
}

function run(script) {
  return new Promise((resolve) => {
    const began = performance.now();
    const child = spawn("python3", [path.join("checks", script)], {
      env: { ...process.env, PYTHONHASHSEED: "0" },
      stdio: ["ignore", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
      setTimeout(() => child.kill("SIGKILL"), 1_000).unref();
    }, timeoutMs);
    child.stdout.on("data", (chunk) => { stdout = tail(stdout + chunk); });
    child.stderr.on("data", (chunk) => { stderr = tail(stderr + chunk); });
    child.on("error", (error) => {
      clearTimeout(timer);
      resolve({ script, ok: false, error: error.message, ms: performance.now() - began });
    });
    child.on("close", (code, signal) => {
      clearTimeout(timer);
      resolve({
        script,
        ok: code === 0 && !timedOut,
        code,
        signal,
        timedOut,
        stdout: stdout.trim(),
        stderr: stderr.trim(),
        ms: performance.now() - began,
      });
    });
  });
}

async function worker(results) {
  while (cursor < scripts.length) {
    const index = cursor++;
    results[index] = await run(scripts[index]);
  }
}

const results = new Array(scripts.length);
await Promise.all(Array.from({ length: Math.min(concurrency, scripts.length) }, () => worker(results)));

const failures = results.filter((result) => !result.ok);
for (const failure of failures) {
  const reason = failure.timedOut
    ? `timed out after ${timeoutMs} ms`
    : `exit ${failure.code ?? "spawn error"}${failure.signal ? ` (${failure.signal})` : ""}`;
  console.error(`\n${failure.script}: ${reason}`);
  if (failure.error) console.error(failure.error);
  if (failure.stderr) console.error(failure.stderr);
  if (failure.stdout) console.error(failure.stdout);
}

const slowest = [...results].sort((a, b) => b.ms - a.ms).slice(0, 5);
const elapsed = performance.now() - started;
console.log(
  `Numerical checks ${failures.length ? "failed" : "passed"}: ` +
  `${scripts.length - failures.length}/${scripts.length} in ${(elapsed / 1_000).toFixed(2)}s ` +
  `(${concurrency} workers).`,
);
console.log(`Slowest: ${slowest.map((r) => `${r.script} ${(r.ms / 1_000).toFixed(2)}s`).join(", ")}`);
if (failures.length) process.exitCode = 1;
