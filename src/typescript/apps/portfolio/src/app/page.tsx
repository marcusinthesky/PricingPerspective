import { ArrowDown, Play } from "lucide-react";

import AuthorsContent from "@/content/authors.mdx";
import HomeContent from "@/content/home.mdx";
import { AiResearchLinks } from "@/components/ai-research-links";
import { EconomicField } from "@/components/economic-field";
import { AppearingIn } from "@/components/appearing-in";
import { JsonLd } from "@/components/json-ld";
import { PaperCard } from "@/components/paper-card";
import { ResearcherCard } from "@/components/researcher-card";
import { ResearchTimeline } from "@/components/research-timeline";
import { Reveal } from "@/components/reveal";
import { SectionHeading } from "@/components/section-heading";
import { ButtonLink } from "@/components/ui/button";
import { VideoFeature } from "@/components/video-feature";
import { papers, researchers } from "@/content/data";
import { homeJsonLd } from "@/lib/schema";
import { researchVideos } from "@/lib/site";

export const dynamic = "force-static";

export default function HomePage() {
  return (
    <>
      <JsonLd data={homeJsonLd()} />
      <section className="hero site-shell" aria-labelledby="hero-title">
        <div className="hero-copy">
          <p className="eyebrow">Information, geometry, and pricing</p>
          <h1 id="hero-title">Firms are not points. Their relationships have geometry.</h1>
          <p className="hero-deck">
            Probability-valued information links firm representations to financial dependence.
          </p>
          <div className="hero-actions">
            <ButtonLink href="#papers">
              Explore the papers <ArrowDown aria-hidden="true" size={17} />
            </ButtonLink>
            <ButtonLink href={researchVideos[1].href} variant="outline">
              <Play aria-hidden="true" size={16} /> Watch the video
            </ButtonLink>
          </div>
          <AiResearchLinks
            arxivLinks={papers.map((paper) => `https://arxiv.org/abs/${paper.arxivId}`)}
          />
          <dl className="hero-facts">
            <div>
              <dt>Representation</dt>
              <dd>Firm as probability law</dd>
            </div>
            <div>
              <dt>Method</dt>
              <dd>Wasserstein geometry</dd>
            </div>
            <div>
              <dt>Question</dt>
              <dd>How information structures dependence</dd>
            </div>
          </dl>
        </div>
        <div className="hero-visual">
          <EconomicField />
        </div>
      </section>

      <section className="section section-ruled site-shell" id="film" aria-labelledby="film-title">
        <SectionHeading
          id="film-title"
          eyebrow="Research videos"
          title="Explore the four papers."
          description="Video guides to covariance, portfolio risk, economic peers, and information games."
        />
        <VideoFeature />
      </section>

      <AppearingIn />

      <section className="editorial-band">
        <div className="site-shell editorial-grid">
          <div className="editorial-index" aria-hidden="true">
            ∴
          </div>
          <div className="mdx-prose">
            <HomeContent />
          </div>
        </div>
      </section>

      <section className="section site-shell" id="papers" aria-labelledby="papers-title">
        <SectionHeading
          id="papers-title"
          eyebrow="Research"
          title="Firm relationships as geometry."
          description="Explore how observable firm information gives structure to financial dependence."
        />
        <div className="paper-grid">
          {papers.map((paper, index) => (
            <Reveal className="h-full" delay={index * 100} key={paper.slug}>
              <PaperCard paper={paper} />
            </Reveal>
          ))}
        </div>
      </section>

      <section
        className="section section-ruled site-shell"
        id="researchers"
        aria-labelledby="researchers-title"
      >
        <SectionHeading
          id="researchers-title"
          eyebrow="Researchers"
          title="Marcus Gawronsky and Chun-Sung Huang."
        />
        <div className="authors-intro mdx-prose">
          <AuthorsContent />
        </div>
        <Reveal>
          <ResearchTimeline />
        </Reveal>
        <div className="researcher-grid">
          {researchers.map((researcher, index) => (
            <Reveal className="h-full" delay={index * 100} key={researcher.name}>
              <ResearcherCard researcher={researcher} />
            </Reveal>
          ))}
        </div>
      </section>
    </>
  );
}
