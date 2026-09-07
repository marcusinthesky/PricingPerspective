import type { Metadata } from "next";

import { BlogCard } from "@/components/blog-card";
import { JsonLd } from "@/components/json-ld";
import { SectionHeading } from "@/components/section-heading";
import { blogPosts } from "@/content/data";
import { blogCollectionJsonLd } from "@/lib/schema";
import { absoluteUrl, site } from "@/lib/site";

const description =
  "Research notes on AI in finance, financial econometrics, model design, and reproducible computational methods.";

export const dynamic = "force-static";

export const metadata: Metadata = {
  title: "Blog",
  description,
  alternates: { canonical: absoluteUrl("/blog/") },
  openGraph: {
    type: "website",
    url: absoluteUrl("/blog/"),
    title: "Blog",
    description,
    siteName: site.name,
    locale: site.locale,
  },
};

export default function BlogPage() {
  return (
    <>
      <JsonLd data={blogCollectionJsonLd()} />
      <section className="section section-ruled site-shell blog-index-page">
        <SectionHeading
          eyebrow="Research notes"
          title="Ideas in motion."
          description="Shorter essays on the methods, evidence, and research infrastructure around Pricing Perspective."
        />
        <div className="blog-grid">
          {blogPosts.map((post) => (
            <BlogCard key={post.slug} post={post} />
          ))}
        </div>
      </section>
    </>
  );
}
