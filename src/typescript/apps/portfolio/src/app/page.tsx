import { ArrowDown, Play } from "lucide-react";

import AuthorsContent from "@/content/authors.mdx";
import HomeContent from "@/content/home.mdx";
import { AiResearchLinks } from "@/components/ai-research-links";
import { Constellation } from "@/components/constellation";
import { AppearingIn } from "@/components/appearing-in";
import { AudioFeature } from "@/components/audio-feature";
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

export const dynamic = "force-static";

export default function HomePage() {
  return (
    <>
      <JsonLd data={homeJsonLd()} />
      <section className="hero site-shell" aria-labelledby="hero-title">
        <div className="hero-copy">
          <p className="eyebrow">Information, geometry, and pricing</p>
          <h1 id="hero-title">Three papers. One geometric language for financial dependence.</h1>
          <p className="hero-deck">
            Probability-valued firm information is used to study asset co-movement, construct
            interaction fields, and certify portfolio diversification.
          </p>
          <div className="hero-actions">
            <ButtonLink href="#papers">
              Explore the papers <ArrowDown aria-hidden="true" size={17} />
            </ButtonLink>
            <ButtonLink href="#film" variant="outline">
              <Play aria-hidden="true" size={16} /> Watch the video
            </ButtonLink>
          </div>
          <AiResearchLinks
            arxivLinks={papers.map((paper) => `https://arxiv.org/abs/${paper.arxivId}`)}
          />
          <dl className="hero-facts">
            <div>
              <dt>Papers</dt>
              <dd>03</dd>
            </div>
            <div>
              <dt>Progression</dt>
              <dd>Pair → field → portfolio</dd>
            </div>
            <div>
              <dt>Representation</dt>
              <dd>Firm as probability law</dd>
            </div>
          </dl>
        </div>
        <div className="hero-visual">
          <Constellation />
          <div className="orbit-label orbit-label-a">separation</div>
          <div className="orbit-label orbit-label-b">reconstruction</div>
          <div className="orbit-label orbit-label-c">dispersion</div>
          <p>Observable information → geometric structure → financial restriction</p>
        </div>
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
          eyebrow="The papers"
          title="Separation, reconstruction, dispersion."
          description="Each paper changes which object is held fixed and which object is allowed to vary."
        />
        <div className="paper-grid">
          {papers.map((paper, index) => (
            <Reveal className="h-full" delay={index * 100} key={paper.slug}>
              <PaperCard paper={paper} />
            </Reveal>
          ))}
        </div>
      </section>

      <section className="section section-ruled site-shell" id="film" aria-labelledby="film-title">
        <SectionHeading
          eyebrow="Animated companion"
          title="See the mathematical objects move."
          description="A narrated animation of the programme."
        />
        <Reveal className="video-reveal">
          <VideoFeature />
        </Reveal>
        <SectionHeading
          eyebrow="Podcast explainer"
          title="Hear the research in conversation."
          description="A NotebookLM-generated podcast-style explainer of firms as news and probability distributions."
        />
        <Reveal delay={120}>
          <AudioFeature />
        </Reveal>
      </section>

      <section
        className="section section-ruled site-shell"
        id="researchers"
        aria-labelledby="researchers-title"
      >
        <SectionHeading eyebrow="Researchers" title="Marcus Gawronsky and Chun-Sung Huang." />
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
