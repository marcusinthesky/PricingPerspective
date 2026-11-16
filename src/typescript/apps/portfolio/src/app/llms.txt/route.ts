import { blogPosts, papers } from "@/content/data";
import { absoluteUrl } from "@/lib/site";

export const dynamic = "force-static";

export function GET() {
  const paperLinks = papers.map(
    (paper) =>
      `- [${paper.shortTitle}](${absoluteUrl(`/papers/${paper.slug}/`)}): ${paper.summary}`,
  );
  const blogLinks = blogPosts
    .toReversed()
    .map((post) => `- [${post.title}](${absoluteUrl(`/blog/${post.slug}/`)}): ${post.summary}`);
  const resources = ["citations.bib", "sitemap.xml"].map((path) => `- ${absoluteUrl(path)}`);
  return new Response(
    `# Pricing Perspective\n\n> Research by Marcus Gawronsky and Chun-Sung Huang on distribution-valued firm information and financial dependence.\n\n## Papers\n\n${paperLinks.join("\n")}\n\n## Blog\n\n${blogLinks.join("\n")}\n\n## Machine-readable resources\n\n${resources.join("\n")}\n`,
    { headers: { "Content-Type": "text/plain; charset=utf-8" } },
  );
}
