import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";

import { blogPosts } from "../src/content/data";

async function htmlFiles(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(
    entries.map(async (entry) => {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) return htmlFiles(path);
      return entry.name.endsWith(".html") ? [path] : [];
    }),
  );
  return nested.flat();
}

const files = await htmlFiles("out");
const seen = new Set<string>();
let articleCount = 0;
let blogPostCount = 0;
let audioCount = 0;
let breadcrumbCount = 0;

const documents = await Promise.all(
  files.map(async (file) => [file, await readFile(file, "utf8")] as const),
);

for (const [file, html] of documents) {
  const matches = html.matchAll(
    /<script[^>]+type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi,
  );
  for (const match of matches) {
    const parsed = JSON.parse(match[1]) as Record<string, unknown>;
    const nodes = Array.isArray(parsed["@graph"])
      ? (parsed["@graph"] as Array<Record<string, unknown>>)
      : [parsed];
    for (const node of nodes) {
      const type = node["@type"];
      if (typeof type === "string") seen.add(type);
      if (type === "ScholarlyArticle") {
        articleCount += 1;
        for (const field of [
          "name",
          "abstract",
          "datePublished",
          "dateModified",
          "author",
          "identifier",
        ]) {
          if (!(field in node)) throw new Error(`${file}: ScholarlyArticle is missing ${field}.`);
        }
      }
      if (type === "BlogPosting") {
        blogPostCount += 1;
        const audio = node.audio;
        if (
          !audio ||
          typeof audio !== "object" ||
          Array.isArray(audio) ||
          (audio as Record<string, unknown>)["@type"] !== "AudioObject"
        ) {
          throw new Error(`${file}: BlogPosting is missing its AudioObject.`);
        }
        audioCount += 1;
      }
      if (type === "BreadcrumbList") breadcrumbCount += 1;
    }
  }
}

const required = ["WebSite", "Organization", "CollectionPage", "VideoObject", "WebPage"];
for (const type of required) {
  if (!seen.has(type)) throw new Error(`Missing ${type} structured data.`);
}
if (articleCount !== 3)
  throw new Error(`Expected 3 ScholarlyArticle nodes; found ${articleCount}.`);
if (blogPostCount !== blogPosts.length)
  throw new Error(`Expected ${blogPosts.length} BlogPosting nodes; found ${blogPostCount}.`);
if (audioCount !== blogPosts.length)
  throw new Error(`Expected ${blogPosts.length} AudioObject nodes; found ${audioCount}.`);
const expectedBreadcrumbs = 3 + blogPosts.length;
if (breadcrumbCount !== expectedBreadcrumbs)
  throw new Error(
    `Expected ${expectedBreadcrumbs} BreadcrumbList nodes; found ${breadcrumbCount}.`,
  );

console.log(
  `Structured data audit passed: ${[...seen].toSorted().join(", ")}; ${articleCount} scholarly articles; ${blogPostCount} blog posts; ${audioCount} audio objects; ${breadcrumbCount} breadcrumbs.`,
);
