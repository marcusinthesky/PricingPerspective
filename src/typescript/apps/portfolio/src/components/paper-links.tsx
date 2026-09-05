import { Boxes, ExternalLink, FileText, Library, Network, type LucideIcon } from "lucide-react";

import type { PaperLink } from "@/lib/types";

const icons: Record<PaperLink["platform"], LucideIcon> = {
  "Hugging Face": Boxes,
  arXiv: FileText,
  SSRN: Library,
  ResearchGate: Network,
};

function getLinkTitle(link: PaperLink): string {
  if (link.kind === "search") {
    return `No canonical ${link.platform} record was verified; this opens a title search.`;
  }

  return link.note ? `Open on ${link.platform}. ${link.note}` : `Open on ${link.platform}`;
}

export function PaperLinks({ links, compact = false }: { links: PaperLink[]; compact?: boolean }) {
  return (
    <div className={compact ? "paper-links paper-links-compact" : "paper-links"}>
      {links.map((link) => {
        const Icon = icons[link.platform];
        const label = link.kind === "search" ? `Search ${link.platform}` : link.platform;

        return (
          <a
            key={`${link.platform}-${link.href}`}
            className="paper-link"
            href={link.href}
            target="_blank"
            rel="noreferrer"
            data-link-kind={link.kind}
            title={getLinkTitle(link)}
          >
            <Icon aria-hidden="true" size={16} strokeWidth={1.6} />
            <span>{label}</span>
            <ExternalLink className="external-icon" aria-hidden="true" size={13} />
          </a>
        );
      })}
    </div>
  );
}
