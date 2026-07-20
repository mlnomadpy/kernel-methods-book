import fs from "node:fs";
import { spawnSync } from "node:child_process";

const scripts = fs.readdirSync("checks").filter((file) => file.endsWith(".py")).sort();
const failures = [];
for (const script of scripts) {
  const result = spawnSync("python3", [`checks/${script}`], { encoding: "utf8" });
  if (result.status !== 0) failures.push({ script, stderr: result.stderr.trim() });
}
if (failures.length) {
  for (const failure of failures) console.error(`${failure.script}\n${failure.stderr}`);
  process.exitCode = 1;
} else {
  console.log(`Numerical checks passed: ${scripts.length}/${scripts.length}.`);
}
