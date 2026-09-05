import { ArrowUpRight } from "lucide-react";

import { sitePath } from "@/lib/site";

type AiResearchLinksProps = {
  arxivLinks: readonly string[];
};

const aiResearchLinks = [
  { label: "Anthropic", href: "https://claude.ai/new?q=", logo: "/logos/anthropic.svg" },
  {
    label: "OpenAI",
    href: "https://chatgpt.com/?prompt=",
    logo: "/logos/openai.svg",
  },
  { label: "Grok", href: "https://grok.com/?q=", logo: "/logos/grok.svg" },
] as const;

export function AiResearchLinks({ arxivLinks }: AiResearchLinksProps) {
  const prompt = [
    "Request an AI summary of our research using these arXiv papers:",
    ...arxivLinks,
  ].join("\n");

  return (
    <div className="hero-ai">
      <p>Ask AI about our research:</p>
      <nav aria-label="AI research assistants">
        <ul className="hero-ai-links">
          {aiResearchLinks.map(({ label, href, logo }) => (
            <li key={label}>
              <a href={`${href}${encodeURIComponent(prompt)}`} target="_blank" rel="noreferrer">
                <img className="ai-provider-mark" src={sitePath(logo)} alt="" />
                <span>{label}</span>
                <ArrowUpRight aria-hidden="true" size={13} />
              </a>
            </li>
          ))}
        </ul>
      </nav>
    </div>
  );
}
