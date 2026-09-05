const fallbackUrl = "https://marcusinthesky.github.io/pricing-perspective";
const configuredBasePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
const basePath = configuredBasePath.replace(/\/$/, "");

export const site = {
  name: "Pricing Perspective",
  shortName: "Pricing Perspective",
  description:
    "Three papers on how distribution-valued firm information can restrict covariance, construct interaction fields, and certify portfolio diversification.",
  url: (process.env.NEXT_PUBLIC_SITE_URL ?? fallbackUrl).replace(/\/$/, ""),
  basePath,
  locale: "en_ZA",
  language: "en",
  email: "gwrmar002@myuct.ac.za",
} as const;

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
