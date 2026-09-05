import { ArrowUpRight } from "lucide-react";
import Link from "next/link";

import { PaperLinks } from "@/components/paper-links";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import type { Paper } from "@/lib/types";

export function PaperCard({ paper }: { paper: Paper }) {
  return (
    <Card className="paper-card">
      <div className="paper-card-topline">
        <span>{paper.number}</span>
        <span>{paper.year}</span>
      </div>
      <div>
        <h3 className="paper-title">
          <Link href={`/papers/${paper.slug}/`}>{paper.title}</Link>
        </h3>
        <p className="paper-summary">{paper.summary}</p>
      </div>
      <div className="badge-row">
        {paper.keywords.map((keyword) => (
          <Badge key={keyword}>{keyword}</Badge>
        ))}
      </div>
      <PaperLinks compact links={paper.links} />
      <Link className="paper-read" href={`/papers/${paper.slug}/`}>
        Read the paper page
        <ArrowUpRight aria-hidden="true" size={17} />
      </Link>
    </Card>
  );
}
