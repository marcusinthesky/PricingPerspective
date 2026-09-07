import { ArrowUpRight } from "lucide-react";

const milestones = [
  {
    year: "2019",
    stage: "Origin",
    title: "Word2Risk",
    description: "Initial work on translating words into risk representations.",
    href: "https://github.com/marcusinthesky/Word2Risk",
    linkLabel: "View Word2Risk",
  },
  {
    year: "2022",
    stage: "Proposal",
    title: "Initial research proposal",
    description: "The programme takes shape around firm information, geometry, and pricing.",
    href: undefined,
    linkLabel: undefined,
  },
  {
    year: "2024",
    stage: "Pre-print",
    title: "Initial paper pre-print",
    description: "The first manuscript develops distributional geometry for systematic covariance.",
    href: "https://arxiv.org/abs/2410.23447",
    linkLabel: "Read the pre-print",
  },
  {
    year: "2025",
    stage: "Conference",
    title: "World Finance Conference",
    description: "The work reaches its first conference proceeding in Malta.",
    href: "https://www.world-finance-conference.com/",
    linkLabel: "World Finance Conference",
  },
] as const;

export function ResearchTimeline() {
  return (
    <div className="research-timeline-wrap">
      <div className="research-timeline-heading">
        <p className="eyebrow">Work timeline</p>
        <p>From language models to distributional geometry.</p>
      </div>
      <ol className="research-timeline" aria-label="Research timeline">
        {milestones.map((milestone) => (
          <li className="research-timeline-item" key={milestone.year}>
            <div className="research-timeline-marker">
              <div className="research-timeline-meta">
                <span className="research-timeline-year">{milestone.year}</span>
                <span className="research-timeline-stage">{milestone.stage}</span>
              </div>
              <span className="research-timeline-dot" aria-hidden="true" />
            </div>
            <div className="research-timeline-card">
              <h3>{milestone.title}</h3>
              <p>{milestone.description}</p>
              {milestone.href ? (
                <a href={milestone.href} target="_blank" rel="noreferrer">
                  {milestone.linkLabel} <ArrowUpRight aria-hidden="true" size={15} />
                </a>
              ) : null}
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
