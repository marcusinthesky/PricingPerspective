import type { Metadata } from "next";

import { JsonLd } from "@/components/json-ld";
import { SectionHeading } from "@/components/section-heading";
import { absoluteUrl, site } from "@/lib/site";

const description =
  "Privacy, copyright, and use information for the Pricing Perspective research website.";

export const dynamic = "force-static";

export const metadata: Metadata = {
  title: "Privacy & use",
  description,
  alternates: {
    canonical: absoluteUrl("/privacy/"),
  },
  openGraph: {
    type: "website",
    url: absoluteUrl("/privacy/"),
    title: "Privacy & use",
    description,
    siteName: site.name,
    locale: site.locale,
  },
};

export default function PrivacyPage() {
  const pageUrl = absoluteUrl("/privacy/");

  return (
    <>
      <JsonLd
        data={{
          "@context": "https://schema.org",
          "@type": "WebPage",
          "@id": pageUrl,
          url: pageUrl,
          name: "Privacy & use",
          description,
          inLanguage: site.language,
          isAccessibleForFree: true,
          isPartOf: { "@id": absoluteUrl("/#website") },
        }}
      />
      <section className="section section-ruled site-shell legal-page">
        <SectionHeading
          eyebrow="Privacy & use"
          title="A quiet site by design."
          description="How this research website handles privacy, copyright, and financial information."
        />

        <div className="mdx-prose legal-prose">
          <h3>Privacy</h3>
          <p>
            Pricing Perspective is a static research website. We do not run our own analytics,
            advertising, or mailing list.
          </p>
          <p>
            The site is hosted with GitHub Pages. GitHub logs visitor IP addresses for security
            purposes; see its{" "}
            <a href="https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages">
              Pages data-collection notice
            </a>{" "}
            and{" "}
            <a href="https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement">
              privacy statement
            </a>
            .
          </p>
          <p>
            Embedded videos are served by YouTube. Loading or playing a video contacts YouTube,
            which may collect data or use cookies under{" "}
            <a href="https://policies.google.com/privacy">Google&apos;s privacy policy</a>.
          </p>
          <p>
            If you email us, your address and message are received by our email provider and used to
            respond to your enquiry. For privacy requests concerning the University of Cape Town,
            contact the Information Officer at <a href="mailto:popia@uct.ac.za">popia@uct.ac.za</a>{" "}
            or consult UCT&apos;s{" "}
            <a href="https://uct.ac.za/protection-personal-information-act-popia">
              POPIA privacy notices
            </a>
            .
          </p>

          <h3>Copyright</h3>
          <p>
            Unless a page or asset states otherwise, copyright in the original research
            presentation, text, artwork, audio, and video published here is held by the University
            of Cape Town. Third-party materials, linked works, logos, and open-source software
            remain subject to their own rights and licences. Please cite the linked papers and
            obtain permission before reusing protected material.
          </p>

          <h3>Research and financial information</h3>
          <p>
            This website presents research and educational material. It is not personal financial
            advice, an investment recommendation, or an offer of a financial product or service.
          </p>

          <h3>Generated audio</h3>
          <p>
            Blog posts include computer-generated audio readings, which may contain pronunciation or
            audio errors. The written posts and linked research remain the authoritative sources.
          </p>
        </div>
      </section>
    </>
  );
}
