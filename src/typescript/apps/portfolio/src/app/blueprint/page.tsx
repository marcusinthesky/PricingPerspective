import type { Metadata } from "next";
import { ExternalLink } from "lucide-react";

import { JsonLd } from "@/components/json-ld";
import { SectionHeading } from "@/components/section-heading";
import { ButtonLink } from "@/components/ui/button";
import { absoluteUrl, site, sitePath } from "@/lib/site";

const description =
  "A generated Lean 4 proof-development companion for the Pricing Perspective research programme.";

export const dynamic = "force-static";

export const metadata: Metadata = {
  title: "Lean 4 Blueprint",
  description,
  alternates: {
    canonical: absoluteUrl("/blueprint/"),
  },
  openGraph: {
    type: "website",
    url: absoluteUrl("/blueprint/"),
    title: "Lean 4 Blueprint",
    description,
    siteName: site.name,
    locale: site.locale,
  },
};

export default function BlueprintPage() {
  const pageUrl = absoluteUrl("/blueprint/");

  return (
    <>
      <JsonLd
        data={{
          "@context": "https://schema.org",
          "@type": "WebPage",
          "@id": pageUrl,
          url: pageUrl,
          name: "Lean 4 Blueprint",
          description,
          inLanguage: site.language,
          isAccessibleForFree: true,
          isPartOf: { "@id": absoluteUrl("/#website") },
        }}
      />
      <section
        className="section section-ruled site-shell blueprint-page"
        aria-label="Lean 4 Blueprint"
      >
        <SectionHeading
          eyebrow="Formal companion"
          title="Lean 4 Blueprint"
          description="A machine-readable view of the proof-development layer behind the papers."
        />
        <div className="blueprint-intro">
          <p>
            Lean 4 and mathlib provide a checkable substrate for the programme&apos;s definitions,
            claims, and dependency structure. The generated document below keeps that formal layer
            close to the public research narrative.
          </p>
          <div className="blueprint-actions">
            <ButtonLink
              href={sitePath("/blueprint-doc/index.html")}
              target="_blank"
              rel="noreferrer"
              variant="outline"
            >
              Open standalone <ExternalLink aria-hidden="true" size={16} />
            </ButtonLink>
          </div>
        </div>
        <div className="blueprint-frame">
          <iframe
            title="Lean 4 Blueprint generated documentation"
            src={sitePath("/blueprint-doc/index.html")}
            loading="eager"
            sandbox="allow-scripts"
          />
        </div>
        <p className="blueprint-note">
          This is a proof-development companion, not a publication manuscript. The current generated
          view contains four formal chapters; claim mappings remain owned by the Lean blueprint.
        </p>
      </section>
    </>
  );
}
