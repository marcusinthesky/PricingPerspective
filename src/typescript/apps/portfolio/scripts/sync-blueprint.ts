import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { resolve } from "node:path";

const portfolioRoot = resolve(import.meta.dir, "..");
const repositoryRoot = resolve(portfolioRoot, "../../../..");
const source = resolve(repositoryRoot, "src/lean/blueprint/web");
const destination = resolve(portfolioRoot, "public/blueprint-doc");

if (!existsSync(resolve(source, "index.html"))) {
  throw new Error(
    `Generated LeanBlueprint web output is missing at ${source}. Run just lean::blueprint-web first.`,
  );
}

rmSync(destination, { force: true, recursive: true });
mkdirSync(resolve(destination, ".."), { recursive: true });
cpSync(source, destination, { recursive: true });

console.log(`Copied LeanBlueprint web output to ${destination}`);
