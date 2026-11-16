const fallbackUrl = "https://marcusinthesky.github.io/PricingPerspective";
const configuredBasePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const basePath = configuredBasePath.replace(/\/$/, "");

export const site = {
  name: "Pricing Perspective",
  shortName: "Pricing Perspective",
  description:
    "Research on how distribution-valued firm information reveals the geometry of financial dependence.",
  url: (process.env.NEXT_PUBLIC_SITE_URL ?? fallbackUrl).replace(/\/$/, ""),
  basePath,
  locale: "en_ZA",
  language: "en",
  email: "gwrmar002@myuct.ac.za",
} as const;

export const researchVideos = [
  {
    id: "kK1yH6I-OU4",
    title: "Portfolio Risk Bounds without Asset Covariances",
    start: 0,
  },
  {
    id: "rMpOCYR5J8I",
    title: "Systematic covariance envelopes from Wasserstein geometry",
    start: 20,
  },
  {
    id: "ag6GZ5gl7H0",
    title: "Who are firms’ economic peers?",
    start: 0,
  },
  {
    id: "Sypt_B42xJw",
    title: "Information Games",
    start: 3,
  },
].map(({ id, title, start }) => ({
  id,
  title,
  embedUrl: `https://www.youtube-nocookie.com/embed/${id}${start ? `?start=${start}` : ""}`,
  href: `https://www.youtube.com/watch?v=${id}${start ? `&t=${start}s` : ""}`,
}));

export function sitePath(path = "/"): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  if (
    !site.basePath ||
    normalizedPath === site.basePath ||
    normalizedPath.startsWith(`${site.basePath}/`)
  ) {
    return normalizedPath;
  }
  return `${site.basePath}${normalizedPath}`;
}

export function absoluteUrl(path = "/"): string {
  return new URL(path.replace(/^\/+/, ""), `${site.url}/`).toString();
}
