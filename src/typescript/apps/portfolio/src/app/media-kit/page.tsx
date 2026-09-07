import type { Metadata } from "next";
import { ArrowUpRight, Mail } from "lucide-react";

import { JsonLd } from "@/components/json-ld";
import { SectionHeading } from "@/components/section-heading";
import { ButtonLink } from "@/components/ui/button";
import { absoluteUrl, site, sitePath } from "@/lib/site";

const description =
  "Press materials and partnership formats for work on distribution-valued information, transport geometry, and financial risk.";
const contactHref = `mailto:${site.email}?subject=Pricing Perspective enquiry`;

export const dynamic = "force-static";

export const metadata: Metadata = {
  title: "Media & partnerships",
  description,
  alternates: {
    canonical: absoluteUrl("/media-kit/"),
  },
  openGraph: {
    type: "website",
    url: absoluteUrl("/media-kit/"),
    title: "Media & partnerships",
    description,
    siteName: site.name,
    locale: site.locale,
  },
};

const collaborationFormats = [
  {
    number: "01",
    title: "Research collaboration",
    description:
      "Develop theory, empirical studies, or open research infrastructure around richer representations of firms and markets.",
  },
  {
    number: "02",
    title: "Evaluation and benchmarking",
    description:
      "Test whether language-model representations add useful structure to financial dependence, interaction, or risk analysis.",
  },
  {
    number: "03",
    title: "Talks and technical writing",
    description:
      "Make the ideas legible to technical, investment, policy, and product audiences through talks or editorial collaborations.",
  },
] as const;

export default function MediaKitPage() {
  const pageUrl = absoluteUrl("/media-kit/");

  return (
    <>
      <JsonLd
        data={{
          "@context": "https://schema.org",
          "@type": "WebPage",
          "@id": pageUrl,
          url: pageUrl,
          name: "Media & partnerships",
          description,
          inLanguage: site.language,
          isAccessibleForFree: true,
          isPartOf: { "@id": absoluteUrl("/#website") },
          about: [
            { "@type": "Thing", name: "Quantitative finance" },
            { "@type": "Thing", name: "Financial risk" },
            { "@type": "Thing", name: "Optimal transport" },
          ],
        }}
      />
      <section className="section section-ruled site-shell media-kit-page">
        <SectionHeading
          eyebrow="Media & partnerships"
          title="Bring the work into the room."
          description="A concise guide to the research, the people behind it, and the collaborations we are ready to explore."
        />

        <div className="media-kit-intro">
          <div className="media-kit-lede">
            <p>
              Pricing Perspective develops methods for turning rich firm information into
              interpretable measures of dependence, interaction, and portfolio risk.
            </p>
            <p>
              The work sits between mathematical finance, machine learning, and applied risk
              analysis. We are interested in conversations that make those connections useful in
              practice while keeping the underlying claims precise.
            </p>
          </div>
          <aside className="media-kit-contact">
            <p className="eyebrow">Start a conversation</p>
            <h2>Research, media, or partnership enquiry.</h2>
            <ButtonLink href={contactHref}>
              <Mail aria-hidden="true" size={16} />
              Email the team
            </ButtonLink>
            <a className="media-kit-email" href={contactHref}>
              {site.email}
            </a>
          </aside>
        </div>

        <div className="media-kit-section">
          <SectionHeading
            eyebrow="For media and events"
            title="The short version."
            description="Use these materials to orient a reader, producer, host, or editor to the programme."
          />
          <div className="media-kit-resource-grid">
            <article className="media-kit-resource">
              <span className="media-kit-resource-index">01</span>
              <h3>Research overview</h3>
              <p>
                Three papers move from pairwise covariance bounds, to spatial interaction fields, to
                portfolio-level risk certificates.
              </p>
              <a href={sitePath("/#papers")}>
                Read the papers <ArrowUpRight aria-hidden="true" size={15} />
              </a>
            </article>
            <article className="media-kit-resource">
              <span className="media-kit-resource-index">02</span>
              <h3>Visual companion</h3>
              <p>
                A narrated animation of the programme introduces the geometric objects and follows
                their path into financial restrictions.
              </p>
              <a href={sitePath("/#film")}>
                Watch the video <ArrowUpRight aria-hidden="true" size={15} />
              </a>
            </article>
            <article className="media-kit-resource">
              <span className="media-kit-resource-index">03</span>
              <h3>Researcher profiles</h3>
              <p>
                Marcus Gawronsky and Chun-Sung Huang are available for conversations about the
                mathematics, evidence, and applications.
              </p>
              <a href={sitePath("/#researchers")}>
                Meet the researchers <ArrowUpRight aria-hidden="true" size={15} />
              </a>
            </article>
          </div>
        </div>

        <div className="media-kit-section media-kit-partnerships">
          <SectionHeading
            eyebrow="For partners"
            title="Build a useful next step."
            description="We welcome carefully scoped work with research organisations, AI and model companies, data platforms, and risk or advisory teams."
          />
          <div className="partnership-grid">
            {collaborationFormats.map((format) => (
              <article key={format.number}>
                <span className="media-kit-partnership-index">{format.number}</span>
                <h3>{format.title}</h3>
                <p>{format.description}</p>
              </article>
            ))}
          </div>
          <div className="media-kit-close">
            <p>
              A good first conversation can be technical or exploratory. Tell us what decision,
              dataset, audience, or research question you have in mind.
            </p>
            <ButtonLink href={contactHref} variant="outline">
              Propose a collaboration <ArrowUpRight aria-hidden="true" size={16} />
            </ButtonLink>
          </div>
        </div>
      </section>
    </>
  );
}
