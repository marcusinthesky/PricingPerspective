import { ArrowLeft, ArrowRight } from "lucide-react";
import Link from "next/link";
import type { ReactNode } from "react";

import { PaperLinks } from "@/components/paper-links";
import { Badge } from "@/components/ui/badge";
import { papers } from "@/content/data";
import type { Paper } from "@/lib/types";
import { formatDate } from "@/lib/utils";

export function PaperShell({ paper, children }: { paper: Paper; children: ReactNode }) {
  const currentIndex = papers.findIndex((item) => item.slug === paper.slug);
  const previous = currentIndex > 0 ? papers[currentIndex - 1] : undefined;
  const next = currentIndex < papers.length - 1 ? papers[currentIndex + 1] : undefined;

  return (
    <article className="paper-page">
      <div className="paper-page-header">
        <Link className="back-link" href="/#papers">
          <ArrowLeft aria-hidden="true" size={16} /> All papers
        </Link>
        <div className="paper-page-index">
          <span>Paper {paper.number}</span>
          <span>
            {formatDate(paper.datePublished)}
            {paper.dateModified !== paper.datePublished
              ? ` · revised ${formatDate(paper.dateModified)}`
              : ""}
          </span>
        </div>
        <h1>{paper.title}</h1>
        <p className="paper-page-authors">Marcus Gawronsky · Chun-Sung Huang</p>
        <p className="paper-page-summary">{paper.summary}</p>
        <div className="badge-row">
          {paper.keywords.map((keyword) => (
            <Badge key={keyword}>{keyword}</Badge>
          ))}
        </div>
        <PaperLinks links={paper.links} />
      </div>
      <div className="paper-prose">{children}</div>
      <nav className="paper-pagination" aria-label="Paper sequence">
        {previous ? (
          <Link href={`/papers/${previous.slug}/`}>
            <ArrowLeft aria-hidden="true" size={16} />
            <span>
              <small>Previous</small>
              {previous.shortTitle}
            </span>
          </Link>
        ) : (
          <span />
        )}
        {next ? (
          <Link href={`/papers/${next.slug}/`}>
            <span>
              <small>Next</small>
              {next.shortTitle}
            </span>
            <ArrowRight aria-hidden="true" size={16} />
          </Link>
        ) : (
          <span />
        )}
      </nav>
    </article>
  );
}
