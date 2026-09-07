import { blogPosts, papers, researchers } from "@/content/data";
import { absoluteUrl, site } from "@/lib/site";
import type { BlogPost, Paper, Researcher } from "@/lib/types";

export type JsonLdValue = Record<string, unknown> | Array<Record<string, unknown>>;

const organisationId = absoluteUrl("/#uct");

function personId(researcher: Researcher): string {
  return absoluteUrl(`/#${researcher.initials.toLowerCase()}`);
}

function personJsonLd(researcher: Researcher): Record<string, unknown> {
  return {
    "@type": "Person",
    "@id": personId(researcher),
    url: personId(researcher),
    name: researcher.name,
    affiliation: { "@id": organisationId },
    jobTitle: researcher.role,
    sameAs: researcher.profiles
      .filter((profile) => profile.kind === "profile")
      .map((profile) => profile.href),
  };
}

export function paperJsonLd(paper: Paper): Record<string, unknown> {
  const pageUrl = absoluteUrl(`/papers/${paper.slug}/`);
  const articleId = absoluteUrl(`/papers/${paper.slug}/#article`);

  return {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebPage",
        "@id": pageUrl,
        url: pageUrl,
        name: paper.title,
        description: paper.summary,
        datePublished: paper.datePublished,
        dateModified: paper.dateModified,
        inLanguage: site.language,
        isPartOf: { "@id": absoluteUrl("/#website") },
        mainEntity: { "@id": articleId },
        breadcrumb: { "@id": absoluteUrl(`/papers/${paper.slug}/#breadcrumb`) },
      },
      {
        "@type": "ScholarlyArticle",
        "@id": articleId,
        url: pageUrl,
        mainEntityOfPage: { "@id": pageUrl },
        name: paper.title,
        headline: paper.title,
        abstract: paper.abstract,
        description: paper.summary,
        datePublished: paper.datePublished,
        dateModified: paper.dateModified,
        inLanguage: site.language,
        isAccessibleForFree: true,
        author: researchers.map((researcher) => ({ "@id": personId(researcher) })),
        sameAs: paper.links.filter((link) => link.kind === "paper").map((link) => link.href),
        identifier: [
          {
            "@type": "PropertyValue",
            propertyID: "arXiv",
            value: paper.arxivId,
            url: `https://arxiv.org/abs/${paper.arxivId}`,
          },
          {
            "@type": "PropertyValue",
            propertyID: "DOI",
            value: paper.doi,
            url: `https://doi.org/${paper.doi}`,
          },
        ],
        encoding: {
          "@type": "MediaObject",
          contentUrl: `https://arxiv.org/pdf/${paper.arxivId}`,
          encodingFormat: "application/pdf",
        },
        keywords: paper.keywords,
        provider: {
          "@type": "Organization",
          name: "arXiv",
          url: "https://arxiv.org/",
        },
        isPartOf: { "@id": absoluteUrl("/#collection") },
        about: [
          { "@type": "Thing", name: "Quantitative finance" },
          { "@type": "Thing", name: "Financial econometrics" },
          { "@type": "Thing", name: "Optimal transport" },
        ],
        image: absoluteUrl("/og/constellation-card.webp"),
      },
      {
        "@type": "BreadcrumbList",
        "@id": absoluteUrl(`/papers/${paper.slug}/#breadcrumb`),
        itemListElement: [
          {
            "@type": "ListItem",
            position: 1,
            name: "Papers",
            item: absoluteUrl("/#papers"),
          },
          {
            "@type": "ListItem",
            position: 2,
            name: paper.shortTitle,
            item: pageUrl,
          },
        ],
      },
    ],
  };
}

export function blogCollectionJsonLd(): Record<string, unknown> {
  const postItems = blogPosts.map((post, index) => ({
    "@type": "ListItem",
    position: index + 1,
    item: {
      "@id": absoluteUrl(`/blog/${post.slug}/#article`),
      name: post.title,
      url: absoluteUrl(`/blog/${post.slug}/`),
    },
  }));

  return {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "CollectionPage",
        "@id": absoluteUrl("/blog/#collection"),
        url: absoluteUrl("/blog/"),
        name: "Pricing Perspective Blog",
        description:
          "Research notes on AI in finance, financial econometrics, model design, and reproducible computational methods.",
        inLanguage: site.language,
        isPartOf: { "@id": absoluteUrl("/#website") },
        mainEntity: {
          "@type": "ItemList",
          numberOfItems: blogPosts.length,
          itemListElement: postItems,
        },
      },
    ],
  };
}

export function blogPostJsonLd(post: BlogPost): Record<string, unknown> {
  const pageUrl = absoluteUrl(`/blog/${post.slug}/`);
  const articleId = absoluteUrl(`/blog/${post.slug}/#article`);

  return {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebPage",
        "@id": pageUrl,
        url: pageUrl,
        name: post.title,
        description: post.summary,
        datePublished: post.datePublished,
        dateModified: post.dateModified,
        inLanguage: site.language,
        isPartOf: { "@id": absoluteUrl("/#website") },
        mainEntity: { "@id": articleId },
        breadcrumb: { "@id": absoluteUrl(`/blog/${post.slug}/#breadcrumb`) },
      },
      {
        "@type": "BlogPosting",
        "@id": articleId,
        url: pageUrl,
        mainEntityOfPage: { "@id": pageUrl },
        name: post.title,
        headline: post.title,
        description: post.summary,
        datePublished: post.datePublished,
        dateModified: post.dateModified,
        inLanguage: site.language,
        isAccessibleForFree: true,
        author: { "@id": personId(researchers[0]) },
        articleSection: post.category,
        keywords: post.keywords,
        isPartOf: { "@id": absoluteUrl("/blog/#collection") },
        about: [
          { "@type": "Thing", name: "Quantitative finance" },
          { "@type": "Thing", name: "Financial econometrics" },
          { "@type": "Thing", name: "Artificial intelligence" },
        ],
        image: absoluteUrl("/og/constellation-card.webp"),
        audio: {
          "@type": "AudioObject",
          contentUrl: absoluteUrl(`/audio/${post.slug}.webm`),
          encodingFormat: "audio/webm",
          name: `${post.title} - computer-generated reading`,
          description: `A computer-generated audio reading of ${post.title}.`,
          inLanguage: site.language,
          isAccessibleForFree: true,
        },
      },
      {
        "@type": "BreadcrumbList",
        "@id": absoluteUrl(`/blog/${post.slug}/#breadcrumb`),
        itemListElement: [
          {
            "@type": "ListItem",
            position: 1,
            name: "Blog",
            item: absoluteUrl("/blog/"),
          },
          {
            "@type": "ListItem",
            position: 2,
            name: post.title,
            item: pageUrl,
          },
        ],
      },
    ],
  };
}

export function homeJsonLd(): Record<string, unknown> {
  const articleItems = papers.map((paper, index) => ({
    "@type": "ListItem",
    position: index + 1,
    item: {
      "@id": absoluteUrl(`/papers/${paper.slug}/#article`),
      name: paper.title,
      url: absoluteUrl(`/papers/${paper.slug}/`),
    },
  }));

  return {
    "@context": "https://schema.org",
    "@graph": [
      {
        "@type": "WebSite",
        "@id": absoluteUrl("/#website"),
        url: absoluteUrl("/"),
        name: site.name,
        description: site.description,
        inLanguage: site.language,
      },
      {
        "@type": "Organization",
        "@id": organisationId,
        name: "University of Cape Town",
        url: "https://www.uct.ac.za/",
      },
      {
        "@type": "CollectionPage",
        "@id": absoluteUrl("/#collection"),
        url: absoluteUrl("/"),
        name: site.name,
        description: site.description,
        inLanguage: site.language,
        isPartOf: { "@id": absoluteUrl("/#website") },
        mainEntity: {
          "@type": "ItemList",
          numberOfItems: papers.length,
          itemListElement: articleItems,
        },
      },
      ...researchers.map(personJsonLd),
      {
        "@type": "VideoObject",
        "@id": absoluteUrl("/#film"),
        name: "Pricing Perspective - a visual guide",
        description:
          "A five-minute, 40-second narrated animated guide to distributions, couplings, Wasserstein transport, barycentric reconstruction, and portfolio risk certificates.",
        thumbnailUrl: [absoluteUrl("/video/poster.avif"), absoluteUrl("/video/poster.webp")],
        uploadDate: "2026-09-03",
        duration: "PT5M40S",
        contentUrl: absoluteUrl("/video/distributional-information-geometry.mp4"),
        embedUrl: absoluteUrl("/#film"),
        encodingFormat: ["video/mp4", "video/webm"],
        width: 1280,
        height: 720,
        encoding: [
          {
            "@type": "MediaObject",
            contentUrl: absoluteUrl("/video/distributional-information-geometry.webm"),
            encodingFormat: "video/webm",
            width: 960,
            height: 540,
          },
          {
            "@type": "MediaObject",
            contentUrl: absoluteUrl("/video/distributional-information-geometry.mp4"),
            encodingFormat: "video/mp4",
            width: 1280,
            height: 720,
          },
        ],
        inLanguage: site.language,
        isFamilyFriendly: true,
        isAccessibleForFree: true,
        transcript: absoluteUrl("/video/transcript.txt"),
        potentialAction: {
          "@type": "WatchAction",
          target: absoluteUrl("/#film"),
        },
      },
    ],
  };
}
