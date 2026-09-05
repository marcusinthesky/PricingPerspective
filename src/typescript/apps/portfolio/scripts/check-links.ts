import { papers, researchers } from "../src/content/data";

const urls = new Set<string>();
for (const paper of papers) for (const link of paper.links) urls.add(link.href);
for (const researcher of researchers)
  for (const profile of researcher.profiles) urls.add(profile.href);

const results = await Promise.all(
  [...urls].map(async (url) => {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 12_000);
    try {
      const response = await fetch(url, {
        redirect: "follow",
        signal: controller.signal,
        headers: { "User-Agent": "distributional-geometry-link-check/1.0" },
      });
      return {
        line: `${response.ok ? "✓" : "✗"} ${response.status} ${url}`,
        failure:
          !response.ok && response.status !== 403 && response.status !== 429
            ? { url, status: response.status }
            : undefined,
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : "unknown error";
      return {
        line: `✗ ${message} ${url}`,
        failure: { url, status: message },
      };
    } finally {
      clearTimeout(timeout);
    }
  }),
);

const failures = results.flatMap(({ failure }) => (failure ? [failure] : []));
for (const { line } of results) {
  if (line.startsWith("✓")) console.log(line);
  else console.error(line);
}

if (failures.length > 0) {
  console.error("Link audit failures:", failures);
  process.exit(1);
}
