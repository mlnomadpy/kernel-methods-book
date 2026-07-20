import fs from "node:fs";
import { statementIndex } from "../src/lib/book.js";

const nodes = statementIndex();
const output = `${JSON.stringify(nodes, null, 1)}\n`;
const file = "depmap/nodes.json";
if (process.argv.includes("--check")) {
  if (!fs.existsSync(file) || fs.readFileSync(file, "utf8") !== output) {
    console.error("ERROR depmap/nodes.json is stale; run npm run depmap:generate");
    process.exitCode = 1;
  } else {
    console.log("Generated dependency nodes are current.");
  }
} else {
  fs.writeFileSync(file, output);
  console.log(`Generated ${nodes.length} dependency nodes.`);
}
