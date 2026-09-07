import type { MetadataRoute } from "next";

import { blogPosts, papers } from "@/content/data";
import { absoluteUrl } from "@/lib/site";

export const dynamic = "force-static";

export default function sitemap(): MetadataRoute.Sitemap {
  return [
    {
      url: absoluteUrl("/"),
      lastModified: new Date("2026-09-03"),
      changeFrequency: "monthly",
      priority: 1,
    },
    {
      url: absoluteUrl("/blueprint/"),
      lastModified: new Date("2026-09-04"),
      changeFrequency: "monthly",
      priority: 0.7,
    },
    {
      url: absoluteUrl("/media-kit/"),
      lastModified: new Date("2026-09-04"),
      changeFrequency: "monthly",
      priority: 0.7,
    },
    {
      url: absoluteUrl("/blog/"),
      lastModified: new Date("2026-09-04"),
      changeFrequency: "monthly",
      priority: 0.8,
    },
    ...blogPosts.map((post) => ({
      url: absoluteUrl(`/blog/${post.slug}/`),
      lastModified: new Date(post.dateModified),
      changeFrequency: "monthly" as const,
      priority: 0.7,
    })),
    ...papers.map((paper) => ({
      url: absoluteUrl(`/papers/${paper.slug}/`),
      lastModified: new Date(paper.dateModified),
      changeFrequency: "monthly" as const,
      priority: 0.8,
    })),
  ];
}
