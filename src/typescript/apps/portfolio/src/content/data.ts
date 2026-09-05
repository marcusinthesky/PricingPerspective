import { readdirSync, readFileSync } from "node:fs";
import { basename, join } from "node:path";

import matter from "gray-matter";

import type { BlogPost, Paper, Researcher } from "@/lib/types";
import { readingStats } from "@/lib/reading-time";

const blogContentDirectory = join(process.cwd(), "src/content/blog");

function requiredString(data: Record<string, unknown>, field: string, filename: string): string {
  const value = data[field];
  if (typeof value !== "string" || value.trim() === "") {
    throw new Error(`${filename}: frontmatter field '${field}' must be a non-empty string.`);
  }
  return value;
}

function requiredDate(data: Record<string, unknown>, field: string, filename: string): string {
  const value = requiredString(data, field, filename);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    throw new Error(`${filename}: frontmatter field '${field}' must use YYYY-MM-DD.`);
  }
  return value;
}

function requiredKeywords(data: Record<string, unknown>, filename: string): string[] {
  const value = data.keywords;
  if (
    !Array.isArray(value) ||
    value.length === 0 ||
    !value.every(
      (keyword): keyword is string => typeof keyword === "string" && keyword.trim() !== "",
    )
  ) {
    throw new Error(`${filename}: frontmatter field 'keywords' must be a non-empty string array.`);
  }
  return value;
}

function parseBlogPost(filename: string): BlogPost {
  const source = readFileSync(join(blogContentDirectory, filename), "utf8");
  const parsed = matter(source);
  const frontmatter = parsed.data as Record<string, unknown>;
  const slug = basename(filename, ".mdx");

  if ("slug" in frontmatter) {
    throw new Error(
      `${filename}: slug is derived from the filename and must not be repeated in frontmatter.`,
    );
  }
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) {
    throw new Error(`${filename}: filename must be a lowercase kebab-case blog slug.`);
  }

  const number = requiredString(frontmatter, "number", filename);
  if (!/^\d+$/.test(number)) {
    throw new Error(`${filename}: frontmatter field 'number' must contain only digits.`);
  }

  return {
    slug,
    number,
    category: requiredString(frontmatter, "category", filename),
    title: requiredString(frontmatter, "title", filename),
    summary: requiredString(frontmatter, "summary", filename),
    datePublished: requiredDate(frontmatter, "datePublished", filename),
    dateModified: requiredDate(frontmatter, "dateModified", filename),
    ...readingStats(parsed.content),
    keywords: requiredKeywords(frontmatter, filename),
  };
}

const blogFiles = readdirSync(blogContentDirectory, { withFileTypes: true })
  .filter((entry) => entry.isFile() && entry.name.endsWith(".mdx"))
  .map((entry) => entry.name)
  .toSorted();

export const blogPosts: BlogPost[] = blogFiles
  .map(parseBlogPost)
  .toSorted(
    (left, right) =>
      Number(left.number) - Number(right.number) || left.slug.localeCompare(right.slug),
  );

const blogNumbers = new Set(blogPosts.map((post) => post.number));
if (blogNumbers.size !== blogPosts.length) {
  throw new Error("Blog frontmatter field 'number' must be unique across all posts.");
}

export const papers: Paper[] = [
  {
    slug: "systematic-covariance-envelopes",
    number: "01",
    title:
      "Systematic Covariance Envelopes from Wasserstein Geometry: Evidence from Language-Model Representations",
    shortTitle: "Systematic Covariance Envelopes",
    summary:
      "Quadratic Wasserstein geometry turns distributional separation into sharp conditional bounds on attainable systematic covariance under explicit information-to-exposure restrictions.",
    abstract:
      "We study how distances between distributions of firm characteristics restrict systematic covariance. For fixed latent exposure laws, quadratic Wasserstein geometry yields sharp covariance endpoints over admissible couplings. Under a common randomized bi-Lipschitz characteristic-to-exposure map, bounded risk-coordinate slack, and a maintained return-covariance bridge, characteristic-side distance yields a conditional interval for the covariance ceiling.",
    datePublished: "2024-10-30",
    dateModified: "2026-08-30",
    year: 2024,
    arxivId: "2410.23447",
    arxivClass: "q-fin.CP",
    doi: "10.48550/arXiv.2410.23447",
    citationKey: "gawronsky2024systematic",
    keywords: ["Wasserstein geometry", "Systematic covariance", "Transport excess"],
    links: [
      {
        platform: "arXiv",
        href: "https://arxiv.org/abs/2410.23447",
        kind: "paper",
      },
      { platform: "Hugging Face", href: "https://huggingface.co/papers/2410.23447", kind: "paper" },
      {
        platform: "SSRN",
        href: "https://papers.ssrn.com/abstract=7402498",
        kind: "paper",
      },
      {
        platform: "ResearchGate",
        href: "https://www.researchgate.net/publication/413926753_Systematic_Covariance_Envelopes_from_Wasserstein_Geometry_Evidence_from_Language-Model_Representations",
        kind: "paper",
      },
    ],
  },
  {
    slug: "wasserstein-barycentric-interaction-fields",
    number: "02",
    title:
      "Wasserstein-Barycentric Interaction Fields for Spatial Factor Models: Evidence from Language-Model Representations",
    shortTitle: "Wasserstein-Barycentric Interaction Fields",
    summary:
      "Target-anchored transport and simplex reconstruction turn distributions of firm information into a directed, bandwidth-free peer field for spatial exposure adjustment.",
    abstract:
      "Spatial return models take the interaction matrix as given and leave feedback uninterpreted. We construct a bandwidth-free field from firms' language-model article-embedding distributions using target-anchored Wasserstein barycentric reconstruction. A quadratic exposure-adjustment problem maps feedback into a peer-misalignment penalty ratio.",
    datePublished: "2026-08-30",
    dateModified: "2026-08-30",
    year: 2026,
    arxivId: "2608.29669",
    arxivClass: "q-fin.ST",
    doi: "10.48550/arXiv.2608.29669",
    citationKey: "gawronsky2026barycentric",
    keywords: ["Optimal transport", "Spatial econometrics", "Barycentric reconstruction"],
    links: [
      {
        platform: "arXiv",
        href: "https://arxiv.org/abs/2608.29669",
        kind: "paper",
      },
      { platform: "Hugging Face", href: "https://huggingface.co/papers/2608.29669", kind: "paper" },
      {
        platform: "SSRN",
        href: "https://papers.ssrn.com/abstract=7402518",
        kind: "paper",
      },
      {
        platform: "ResearchGate",
        href: "https://www.researchgate.net/publication/413834526_Wasserstein-Barycentric_Interaction_Fields_for_Spatial_Factor_Models_Evidence_from_Language-Model_Representations",
        kind: "paper",
      },
    ],
  },
  {
    slug: "portfolio-risk-bounds",
    number: "03",
    title:
      "Portfolio Risk Bounds without Cross-Asset Return Covariances: Distributional Fields from Language-Model Representations",
    shortTitle: "Portfolio Risk Bounds",
    summary:
      "Multi-firm Wasserstein-2 dispersion yields a one-sided portfolio-risk certificate and an allocation rule that does not require cross-asset return covariances.",
    abstract:
      "Portfolio risk assessment ordinarily relies on reliable estimates of cross-asset return covariances, which are difficult to obtain in short, high-dimensional panels. We show that firm-level distribution-valued characteristics can instead provide one-sided certificates of portfolio risk. Under maintained links from characteristics to systematic exposures and from exposures to returns, multi-firm Wasserstein-2 dispersion yields a sharp upper bound on systematic portfolio variance.",
    datePublished: "2026-08-30",
    dateModified: "2026-08-30",
    year: 2026,
    arxivId: "2608.29692",
    arxivClass: "q-fin.ST",
    doi: "10.48550/arXiv.2608.29692",
    citationKey: "gawronsky2026portfolio",
    keywords: ["Portfolio risk", "Wasserstein-2", "Risk certificates"],
    links: [
      {
        platform: "arXiv",
        href: "https://arxiv.org/abs/2608.29692",
        kind: "paper",
      },
      { platform: "Hugging Face", href: "https://huggingface.co/papers/2608.29692", kind: "paper" },
      {
        platform: "SSRN",
        href: "https://papers.ssrn.com/abstract=7402481",
        kind: "paper",
      },
      {
        platform: "ResearchGate",
        href: "https://www.researchgate.net/publication/413833460_Portfolio_Risk_Bounds_without_Cross-Asset_Return_Covariances_Distributional_Fields_from_Language-Model_Representations",
        kind: "paper",
      },
    ],
  },
];

export const researchers: Researcher[] = [
  {
    name: "Marcus Gawronsky",
    initials: "MG",
    role: "Researcher",
    affiliation: "Department of Finance and Tax · University of Cape Town",
    description:
      "Research at the intersection of quantitative finance, distributional geometry, language-model representations, and mathematical foundations.",
    profiles: [
      {
        platform: "ResearchGate",
        href: "https://www.researchgate.net/profile/Marcus_Gawronsky",
        kind: "profile",
      },
      { platform: "ORCID", href: "https://orcid.org/0000-0003-0554-0850", kind: "profile" },
      { platform: "GitHub", href: "https://github.com/marcusinthesky", kind: "profile" },
      {
        platform: "Hugging Face",
        href: "https://huggingface.co/marcusinthesky",
        kind: "profile",
      },
      { platform: "Twitter", href: "https://twitter.com/marcusinthesky", kind: "profile" },
      {
        platform: "LinkedIn",
        href: "https://www.linkedin.com/in/marcussky",
        kind: "profile",
      },
      {
        platform: "Google Scholar",
        href: "https://scholar.google.com/citations?user=C_c7ZFEAAAAJ",
        kind: "profile",
      },
    ],
  },
  {
    name: "Chun-Sung Huang",
    initials: "CH",
    role: "Associate Professor",
    affiliation: "Department of Finance and Tax · University of Cape Town",
    description:
      "Research in quantitative finance, financial econometrics, computational finance, stochastic processes, and financial risk management.",
    profiles: [
      {
        platform: "ResearchGate",
        href: "https://www.researchgate.net/profile/Chun-sung-Huang",
        kind: "profile",
      },
      {
        platform: "Google Scholar",
        href: "https://scholar.google.com/citations?user=YLUgGVIAAAAJ&hl=en",
        kind: "profile",
      },
      {
        platform: "ORCID",
        href: "https://orcid.org/0000-0002-4758-3712",
        kind: "profile",
      },
    ],
  },
];

export function getPaper(slug: string): Paper | undefined {
  return papers.find((paper) => paper.slug === slug);
}
