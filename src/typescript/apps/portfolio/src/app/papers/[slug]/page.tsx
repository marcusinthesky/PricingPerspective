import type { Metadata } from "next";
import { notFound } from "next/navigation";
import type { ComponentType } from "react";

import { JsonLd } from "@/components/json-ld";
import { PaperShell } from "@/components/paper-shell";
import PortfolioRiskContent from "@/content/papers/portfolio-risk-bounds.mdx";
import SystematicCovarianceContent from "@/content/papers/systematic-covariance-envelopes.mdx";
import InteractionFieldsContent from "@/content/papers/wasserstein-barycentric-interaction-fields.mdx";
import { getPaper, papers } from "@/content/data";
import { paperJsonLd } from "@/lib/schema";
import { absoluteUrl } from "@/lib/site";

const contentBySlug: Record<string, ComponentType> = {
  "systematic-covariance-envelopes": SystematicCovarianceContent,
  "wasserstein-barycentric-interaction-fields": InteractionFieldsContent,
  "portfolio-risk-bounds": PortfolioRiskContent,
};

type PaperPageProps = {
  params: Promise<{ slug: string }>;
};

export const dynamicParams = false;

export function generateStaticParams() {
  return papers.map((paper) => ({ slug: paper.slug }));
}

export async function generateMetadata({ params }: PaperPageProps): Promise<Metadata> {
  const { slug } = await params;
  const paper = getPaper(slug);
  if (!paper) return {};

  return {
    title: paper.shortTitle,
    description: paper.summary,
    alternates: { canonical: absoluteUrl(`/papers/${paper.slug}/`) },
    authors: [{ name: "Marcus Gawronsky" }, { name: "Chun-Sung Huang" }],
    openGraph: {
      type: "article",
      title: paper.title,
      description: paper.summary,
      url: absoluteUrl(`/papers/${paper.slug}/`),
      publishedTime: paper.datePublished,
      modifiedTime: paper.dateModified,
      authors: ["Marcus Gawronsky", "Chun-Sung Huang"],
      images: [{ url: absoluteUrl("/og/constellation-card.webp"), width: 1200, height: 630 }],
    },
    twitter: {
      card: "summary_large_image",
      title: paper.title,
      description: paper.summary,
      images: [absoluteUrl("/og/constellation-card.webp")],
    },
    other: {
      citation_title: paper.title,
      citation_author: ["Marcus Gawronsky", "Chun-Sung Huang"],
      citation_publication_date: paper.datePublished,
      citation_online_date: paper.datePublished,
      citation_pdf_url: `https://arxiv.org/pdf/${paper.arxivId}`,
      citation_arxiv_id: paper.arxivId,
      citation_doi: paper.doi,
      citation_language: "en",
      "DC.type": "Text",
      "DC.title": paper.title,
      "DC.creator": ["Marcus Gawronsky", "Chun-Sung Huang"],
      "DC.date": paper.datePublished,
      "DC.date.modified": paper.dateModified,
      "DC.identifier": [`arXiv:${paper.arxivId}`, `doi:${paper.doi}`],
    },
  };
}

export default async function PaperPage({ params }: PaperPageProps) {
  const { slug } = await params;
  const paper = getPaper(slug);
  const Content = contentBySlug[slug];
  if (!paper || !Content) notFound();

  return (
    <div className="site-shell paper-shell-container">
      <JsonLd data={paperJsonLd(paper)} />
      <PaperShell paper={paper}>
        <Content />
      </PaperShell>
    </div>
  );
}
