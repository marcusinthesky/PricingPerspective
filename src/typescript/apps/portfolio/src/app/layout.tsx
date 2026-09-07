import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";

import { SiteFooter } from "@/components/site-footer";
import { SiteHeader } from "@/components/site-header";
import { researchers } from "@/content/data";
import { absoluteUrl, site } from "@/lib/site";

import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL(site.url),
  title: {
    default: site.name,
    template: `%s · ${site.shortName}`,
  },
  description: site.description,
  applicationName: site.name,
  authors: researchers.map((researcher) => ({ name: researcher.name })),
  creator: researchers.map((researcher) => researcher.name).join(" and "),
  publisher: researchers.map((researcher) => researcher.name).join(" and "),
  category: "research",
  keywords: [
    "quantitative finance",
    "financial econometrics",
    "optimal transport",
    "Wasserstein geometry",
    "distribution-valued characteristics",
    "portfolio risk",
    "spatial factor models",
  ],
  alternates: {
    canonical: absoluteUrl("/"),
  },
  openGraph: {
    type: "website",
    url: absoluteUrl("/"),
    title: site.name,
    description: site.description,
    siteName: site.name,
    locale: site.locale,
    images: [
      {
        url: absoluteUrl("/og/constellation-card.webp"),
        width: 1200,
        height: 630,
        alt: "Pricing Perspective: three papers, one geometric language.",
        type: "image/webp",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: site.name,
    description: site.description,
    images: [absoluteUrl("/og/constellation-card.webp")],
  },
  icons: {
    icon: [{ url: absoluteUrl("/icons/favicon.svg"), type: "image/svg+xml" }],
    apple: [{ url: absoluteUrl("/icons/icon-180.png"), sizes: "180x180", type: "image/png" }],
  },
  manifest: absoluteUrl("/manifest.webmanifest"),
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-image-preview": "large",
      "max-snippet": -1,
      "max-video-preview": -1,
    },
  },
  other: {
    "theme-color": "#f7f7f2",
    "color-scheme": "light",
    citation_author: researchers.map((researcher) => researcher.name),
    "DC.type": "Collection",
    "DC.language": "en",
    "DC.title": site.name,
    "DC.identifier": absoluteUrl("/"),
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#f7f7f2",
  colorScheme: "light",
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <a className="skip-link" href="#main-content">
          Skip to content
        </a>
        <SiteHeader />
        <main id="main-content">{children}</main>
        <SiteFooter />
      </body>
    </html>
  );
}
