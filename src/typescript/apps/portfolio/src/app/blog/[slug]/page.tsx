import type { Metadata } from "next";
import { notFound } from "next/navigation";
import type { ComponentType } from "react";

import { BlogShell } from "@/components/blog-shell";
import { JsonLd } from "@/components/json-ld";
import ReproducibilityContent from "@/content/blog/reproducibility-is-a-graph.mdx";
import AiFinanceContent from "@/content/blog/tidal-wave-of-ai-research-in-finance.mdx";
import EttaXVintagesContent from "@/content/blog/vintage-reproducible-ettax-models.mdx";
import { blogPosts } from "@/content/data";
import { blogPostJsonLd } from "@/lib/schema";
import { absoluteUrl } from "@/lib/site";

const contentBySlug: Record<string, ComponentType> = {
  "reproducibility-is-a-graph": ReproducibilityContent,
  "tidal-wave-of-ai-research-in-finance": AiFinanceContent,
  "vintage-reproducible-ettax-models": EttaXVintagesContent,
};

const discoveredSlugs = new Set(blogPosts.map((post) => post.slug));
const importedSlugs = new Set(Object.keys(contentBySlug));
const missingContent = [...discoveredSlugs].filter((slug) => !importedSlugs.has(slug));
const orphanedContent = [...importedSlugs].filter((slug) => !discoveredSlugs.has(slug));
if (missingContent.length > 0 || orphanedContent.length > 0) {
  throw new Error(
    `Blog content registry mismatch; missing imports: ${missingContent.join(", ") || "none"}; orphaned imports: ${orphanedContent.join(", ") || "none"}.`,
  );
}

type BlogPageProps = {
  params: Promise<{ slug: string }>;
};

export const dynamicParams = false;

export function generateStaticParams() {
  return blogPosts.map((post) => ({ slug: post.slug }));
}

export async function generateMetadata({ params }: BlogPageProps): Promise<Metadata> {
  const { slug } = await params;
  const post = blogPosts.find((item) => item.slug === slug);
  if (!post) return {};

  return {
    title: post.title,
    description: post.summary,
    alternates: { canonical: absoluteUrl(`/blog/${post.slug}/`) },
    authors: [{ name: "Marcus Gawronsky" }],
    keywords: post.keywords,
    openGraph: {
      type: "article",
      title: post.title,
      description: post.summary,
      url: absoluteUrl(`/blog/${post.slug}/`),
      publishedTime: post.datePublished,
      modifiedTime: post.dateModified,
      authors: ["Marcus Gawronsky"],
      images: [{ url: absoluteUrl("/og/constellation-card.webp"), width: 1200, height: 630 }],
    },
    twitter: {
      card: "summary",
      title: post.title,
      description: post.summary,
      images: [absoluteUrl("/og/constellation-card.webp")],
    },
  };
}

export default async function BlogPostPage({ params }: BlogPageProps) {
  const { slug } = await params;
  const post = blogPosts.find((item) => item.slug === slug);
  const Content = contentBySlug[slug];
  if (!post || !Content) notFound();

  return (
    <div className="site-shell blog-post-container">
      <JsonLd data={blogPostJsonLd(post)} />
      <BlogShell post={post}>
        <Content />
      </BlogShell>
    </div>
  );
}
