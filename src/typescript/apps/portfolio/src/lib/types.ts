export type PaperPlatform = "Hugging Face" | "arXiv" | "SSRN" | "ResearchGate";

export type PaperLink = {
  platform: PaperPlatform;
  href: string;
  kind: "paper" | "search";
  note?: string;
};

export type Paper = {
  slug: string;
  number: string;
  title: string;
  shortTitle: string;
  summary: string;
  abstract: string;
  datePublished: string;
  dateModified: string;
  year: number;
  arxivId: string;
  arxivClass: string;
  doi: string;
  citationKey: string;
  keywords: string[];
  links: PaperLink[];
};

export type ProfilePlatform =
  | "LinkedIn"
  | "GitHub"
  | "Google Scholar"
  | "ORCID"
  | "ResearchGate"
  | "Hugging Face"
  | "Twitter";

export type ProfileLink = {
  platform: ProfilePlatform;
  href: string;
  kind: "profile";
};

export type Researcher = {
  name: string;
  initials: string;
  role: string;
  affiliation: string;
  description: string;
  profiles: ProfileLink[];
};

export type BlogPost = {
  slug: string;
  number: string;
  category: string;
  title: string;
  summary: string;
  datePublished: string;
  dateModified: string;
  wordCount: number;
  readingTimeMinutes: number;
  keywords: string[];
};
