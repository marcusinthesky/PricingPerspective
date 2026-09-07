import { access, readdir, readFile, stat } from "node:fs/promises";
import { join, normalize, relative, resolve } from "node:path";

const root = resolve("out");
const basePath = (process.env.NEXT_PUBLIC_BASE_PATH ?? "").replace(/^\/+|\/+$/g, "");
const blogSlugs = (await readdir(resolve("src/content/blog"), { withFileTypes: true }))
  .filter((entry) => entry.isFile() && entry.name.endsWith(".mdx"))
  .map((entry) => entry.name.slice(0, -".mdx".length));
const required = [
  "index.html",
  "blog/index.html",
  "robots.txt",
  "sitemap.xml",
  "manifest.webmanifest",
  "blueprint-doc/index.html",
  "video/distributional-information-geometry.webm",
  "video/distributional-information-geometry.mp4",
  "video/poster.webp",
  "og/constellation-card.webp",
  ".nojekyll",
  ...blogSlugs.map((slug) => `blog/${slug}/index.html`),
  ...blogSlugs.flatMap((slug) => [`audio/${slug}.webm`, `audio/${slug}.vtt`]),
];

await Promise.all(required.map((item) => access(join(root, item))));

async function walk(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  return (
    await Promise.all(
      entries.map(async (entry) => {
        const path = join(directory, entry.name);
        return entry.isDirectory() ? walk(path) : [path];
      }),
    )
  ).flat();
}

const files = await walk(root);
const htmlFiles = files.filter((file) => {
  if (!file.endsWith(".html")) return false;
  const relativePath = relative(root, file).replaceAll("\\", "/");
  return !relativePath.startsWith("blueprint-doc/");
});
const pageFailures = await Promise.all(
  htmlFiles.map(async (htmlFile) => {
    const failures: string[] = [];
    const html = await readFile(htmlFile, "utf8");
    if (!html.includes("<title>")) failures.push(`${htmlFile}: missing title`);
    if (
      !html.includes("application/ld+json") &&
      !html.includes('name="robots" content="noindex"')
    ) {
      failures.push(`${htmlFile}: missing JSON-LD`);
    }
    if (!html.includes('name="description"')) failures.push(`${htmlFile}: missing description`);
    if (!html.includes('rel="canonical"')) failures.push(`${htmlFile}: missing canonical`);

    const assetFailures = await Promise.all(
      [...html.matchAll(/(?:href|src)=["'](\/[^"'#?]+)["']/g)].map(async (match) => {
        const pathname = match[1];
        let assetPath = decodeURIComponent(pathname).replace(/^\//, "");
        if (basePath && assetPath === basePath) assetPath = "";
        else if (basePath && assetPath.startsWith(`${basePath}/`)) {
          assetPath = assetPath.slice(basePath.length + 1);
        }
        let target = resolve(join(root, normalize(assetPath)));
        if (!target.startsWith(root)) return;
        try {
          const info = await stat(target);
          if (info.isDirectory()) target = join(target, "index.html");
          await access(target);
        } catch {
          if (!assetPath.includes(".")) {
            try {
              await access(join(target, "index.html"));
              return;
            } catch {
              // Recorded below.
            }
          }
          return `${htmlFile}: missing internal asset ${pathname}`;
        }
      }),
    );

    failures.push(...assetFailures.filter((failure): failure is string => failure !== undefined));
    return failures;
  }),
);

const internalFailures = pageFailures.flat();

if (internalFailures.length > 0) {
  console.error(internalFailures.join("\n"));
  process.exit(1);
}

console.log(
  `Static export verified: ${htmlFiles.length} HTML pages and ${files.length} total files.`,
);
