import {
  AtSign,
  Boxes,
  Briefcase,
  Code,
  ExternalLink,
  Fingerprint,
  GraduationCap,
  Network,
  type LucideIcon,
} from "lucide-react";

import { Card } from "@/components/ui/card";
import type { ProfileLink, Researcher } from "@/lib/types";

const icons: Record<ProfileLink["platform"], LucideIcon> = {
  LinkedIn: Briefcase,
  GitHub: Code,
  "Google Scholar": GraduationCap,
  ORCID: Fingerprint,
  ResearchGate: Network,
  "Hugging Face": Boxes,
  Twitter: AtSign,
};

export function ResearcherCard({ researcher }: { researcher: Researcher }) {
  return (
    <Card className="researcher-card" id={researcher.initials.toLowerCase()}>
      <div className="researcher-mark" aria-hidden="true">
        <span>{researcher.initials}</span>
        <svg viewBox="0 0 100 100" aria-hidden="true">
          <path d="M14 66 31 22l22 25 31-30 3 58-38 12Z" />
          <circle cx="14" cy="66" r="4" />
          <circle cx="31" cy="22" r="4" />
          <circle cx="53" cy="47" r="4" />
          <circle cx="84" cy="17" r="4" />
          <circle cx="87" cy="75" r="4" />
          <circle cx="49" cy="87" r="4" />
        </svg>
      </div>
      <div className="researcher-copy">
        <p className="eyebrow">{researcher.role}</p>
        <h3>{researcher.name}</h3>
        <p className="researcher-affiliation">{researcher.affiliation}</p>
        <p>{researcher.description}</p>
      </div>
      <nav className="profile-links" aria-label={`${researcher.name} profiles`}>
        {researcher.profiles.map((profile) => {
          const Icon = icons[profile.platform];
          return (
            <a
              key={`${profile.platform}-${profile.href}`}
              href={profile.href}
              target="_blank"
              rel="noreferrer"
              data-link-kind={profile.kind}
              title={`Open ${researcher.name} on ${profile.platform}`}
            >
              <Icon aria-hidden="true" size={17} strokeWidth={1.6} />
              <span>{profile.platform}</span>
              <ExternalLink aria-hidden="true" size={12} />
            </a>
          );
        })}
      </nav>
    </Card>
  );
}
